#include "TD3Trainer.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "AlignedAllocator.h"
#include "src/EigenUtils.h"

TD3Trainer::TD3Trainer(int stateDim, int actionDim, const TD3Config& config)
    : mStateDim(stateDim)
    , mActionDim(actionDim)
    , mConfig(config)
    , mRng(42)
    , mOpponentPool(MAX_POOL_SIZE)
{
    mModel.Init(stateDim, actionDim, config.hiddenDim, config.latentDim, mRng);
    
    int batchSize = config.batchSize;
    mBatchStates.resize(batchSize * stateDim);
    mBatchActions.resize(batchSize * actionDim);
    mBatchRewards.resize(batchSize);
    mBatchNextStates.resize(batchSize * stateDim);
    mBatchDones.resize(batchSize);
    mBatchVectorRewards.resize(batchSize);
    
    mNextActions.resize(batchSize * actionDim);
    mQ1Values.resize(batchSize * 4);
    mQ2Values.resize(batchSize * 4);
    mTargetQ.resize(batchSize);
    mGrads.resize(mModel.GetActor().GetNumWeights());

    int criticInputDim = stateDim + actionDim + mModel.GetLatentDim();
    mCriticInputBuffer.resize(batchSize * criticInputDim);
    mSampledIndices.resize(batchSize);
    
    mActorOutputBuffer.resize(batchSize * actionDim);
    mCriticQBuffer.resize(batchSize * 4);
    mLatentZPos.resize(batchSize * mModel.GetLatentDim());
    mLatentZVel.resize(batchSize * mModel.GetLatentDim());
    
    MuonOptimizer::Config optConfig;
    optConfig.lrMuon = mConfig.muonLR;
    optConfig.betaMuon = mConfig.muonBeta;
    optConfig.nsSteps = mConfig.muonNSSteps;
    optConfig.lrFallback = mConfig.criticLR;
    optConfig.eps = mConfig.muonEpsilon;

    mActorOptimizer = MuonOptimizer(optConfig);
    mCritic1Optimizer = MuonOptimizer(optConfig);
    mCritic2Optimizer = MuonOptimizer(optConfig);
    
    auto registerParams = [&](SpanNetwork& net, MuonOptimizer& opt) {
        for (size_t l = 0; l < net.GetNumLayers(); ++l) {
            auto& layer = net.GetLayer(l);
            auto& cp = layer.GetControlPoints();
            auto& grad = layer.GetControlPointGradients();
            int rows = layer.GetOutputDim();
            int cols = layer.GetNumParams() / rows;
            if (cols > 1) opt.addParameter(cp.data(), grad.data(), rows, cols);
            else opt.addParameter1D(cp.data(), grad.data(), cp.size());
        }
    };
    
    registerParams(mModel.GetActor(), mActorOptimizer);
    registerParams(mModel.GetCritic1(), mCritic1Optimizer);
    registerParams(mModel.GetCritic2(), mCritic2Optimizer);
}

void TD3Trainer::SelectAction(const float* state, float* action) { mModel.SelectAction(state, action, nullptr, true, 0); }
void TD3Trainer::SelectActionEval(const float* state, float* action) { mModel.SelectAction(state, action, nullptr, false, 0); }
void TD3Trainer::SelectActionWithLatent(const float* state, float* action, int envIdx) { mModel.SelectAction(state, action, nullptr, true, envIdx); }
void TD3Trainer::SelectActionBatchWithLatent(const float* states, float* actions, int batchSize, const std::vector<int>& envIndices) { mModel.SelectActionBatchWithLatent(states, actions, batchSize, envIndices, true); }
void TD3Trainer::SelectActionResidual(const float* state, float* residualAction) { SelectAction(state, residualAction); }

void TD3Trainer::Train(ReplayBuffer& buffer)
{
    if (!buffer.IsReady(mConfig.batchSize)) return;

    auto start = std::chrono::high_resolution_clock::now();

    auto sampleStart = std::chrono::high_resolution_clock::now();
    buffer.Sample(mConfig.batchSize, mBatchStates.data(), mBatchActions.data(), mBatchRewards.data(), mBatchNextStates.data(), mBatchDones.data(), mRng);
    float bTime = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - sampleStart).count();

    auto criticStart = std::chrono::high_resolution_clock::now();
    UpdateCritic(buffer);
    float cTime = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - criticStart).count();

    float aTime = 0.0f;
    if (mUpdateCount % mConfig.policyDelay == 0) {
        auto actorStart = std::chrono::high_resolution_clock::now();
        UpdateActor(buffer);
        UpdateTargets();
        aTime = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - actorStart).count();
    }

    float totalTime = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
    mPerfMetrics.Record(0.0f, 0.0f, bTime, totalTime, 0.0f, cTime + aTime, 0.0f, 0.0f);

    mUpdateCount++; mStepCount++;
    if (mStepCount % mConfig.snapshotInterval == 0) SnapshotOpponent();
}

void TD3Trainer::UpdateCritic(ReplayBuffer& buffer)
{
    const int batchSize = mConfig.batchSize;
    const int latentDim = mModel.GetLatentDim();
    const int criticInputDim = mStateDim + mActionDim + latentDim;

    mModel.GetActorTarget().ForwardBatch(mBatchNextStates.data(), mNextActions.data(), batchSize);
    ForwardMoLU_AVX2(mNextActions.data(), mActionDim * batchSize);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937 threadRng(mRng() + tid);
        std::normal_distribution<float> noiseDist(0.0f, mConfig.policyNoise);

        #pragma omp for
        for (int i = 0; i < batchSize; ++i) {
            float* nextAction = mNextActions.data() + i * mActionDim;
            for (int j = 0; j < mActionDim; ++j) {
                float noise = std::clamp(noiseDist(threadRng), -mConfig.noiseClip, mConfig.noiseClip);
                nextAction[j] = std::clamp(nextAction[j] + noise, -1.0f, 1.0f);
            }
            size_t baseIdx = i * criticInputDim;
            std::memcpy(mCriticInputBuffer.data() + baseIdx, mBatchNextStates.data() + i * mStateDim, mStateDim * sizeof(float));
            std::memcpy(mCriticInputBuffer.data() + baseIdx + mStateDim, nextAction, mActionDim * sizeof(float));

            // USE INDIVIDUAL LATENT IF AVAILABLE (Mapping environment indices from buffer)
            // For now, keep using a zeroed latent to maintain speed unless full dynamics are needed per batch item
            std::memset(mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim, 0, latentDim * sizeof(float));
        }
    }

    mModel.GetCritic1Target().ForwardBatch(mCriticInputBuffer.data(), mQ1Values.data(), batchSize);
    mModel.GetCritic2Target().ForwardBatch(mCriticInputBuffer.data(), mQ2Values.data(), batchSize);

    #pragma omp parallel for
    for (int i = 0; i < batchSize; ++i) {
        float minQ = std::min(mQ1Values[i * 4], mQ2Values[i * 4]);
        mTargetQ[i] = mBatchRewards[i] + mConfig.gamma * (1.0f - mBatchDones[i]) * minQ;
    }

    #pragma omp parallel for
    for (int i = 0; i < batchSize; ++i) {
        size_t baseIdx = i * criticInputDim;
        std::memcpy(mCriticInputBuffer.data() + baseIdx, mBatchStates.data() + i * mStateDim, mStateDim * sizeof(float));
        std::memcpy(mCriticInputBuffer.data() + baseIdx + mStateDim, mBatchActions.data() + i * mActionDim, mActionDim * sizeof(float));
        AlignedVector32<float> z(latentDim); mModel.GetLatentMemory().GetLatentStates(z.data(), nullptr, 0);
        std::memcpy(mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim, z.data(), latentDim * sizeof(float));
    }

    if (mUpdateCount % mConfig.muonUpdateInterval == 0) {
        ComputeCriticGradients(mModel.GetCritic1(), buffer, true);
        ComputeCriticGradients(mModel.GetCritic2(), buffer, false);
        mCritic1Optimizer.step(); mCritic2Optimizer.step();
    }
    mModel.GetCritic1Target().SoftUpdate(mModel.GetCritic1(), mConfig.tau);
    mModel.GetCritic2Target().SoftUpdate(mModel.GetCritic2(), mConfig.tau);
}

void TD3Trainer::UpdateActor(ReplayBuffer& buffer)
{
    const int batchSize = mConfig.batchSize;
    const int latentDim = mModel.GetLatentDim();
    auto& actor = mModel.GetActor();

    actor.ForwardBatch(mBatchStates.data(), mActorOutputBuffer.data(), batchSize);
    ForwardMoLU_AVX2(mActorOutputBuffer.data(), mActionDim * batchSize);

    #pragma omp parallel for
    for (int i = 0; i < batchSize; ++i) {
        AlignedVector32<float> zPos(latentDim); mModel.GetLatentMemory().GetLatentStates(zPos.data(), nullptr, 0);
        size_t baseIdx = i * (mStateDim + mActionDim + latentDim);
        std::memcpy(mCriticInputBuffer.data() + baseIdx, mBatchStates.data() + i * mStateDim, mStateDim * sizeof(float));
        std::memcpy(mCriticInputBuffer.data() + baseIdx + mStateDim, mActorOutputBuffer.data() + i * mActionDim, mActionDim * sizeof(float));
        std::memcpy(mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim, zPos.data(), latentDim * sizeof(float));
    }

    mModel.GetCritic1().ForwardBatch(mCriticInputBuffer.data(), mQ1Values.data(), batchSize);
    if (mUpdateCount % mConfig.muonUpdateInterval == 0) {
        ComputeActorGradient(buffer);
        mActorOptimizer.step(); mActorOptimizer.zeroGrad();
    }
    mModel.GetActorTarget().SoftUpdate(mModel.GetActor(), mConfig.tau);
}

void TD3Trainer::ComputeCriticGradients(SpanNetwork& critic, ReplayBuffer& buffer, bool isCritic1)
{
    const int batchSize = mConfig.batchSize;
    const int latentDim = mModel.GetLatentDim();
    const int criticInputDim = mStateDim + mActionDim + latentDim;
    const int numWeights = critic.GetNumWeights();
    critic.ZeroGradients();
    int numThreads = omp_get_max_threads();
    if (mThreadCaches.size() != (size_t)numThreads) {
        mThreadCaches.resize(numThreads);
        mThreadGrads.assign(numThreads, AlignedVector32<float>(numWeights, 0.0f));
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::fill(mThreadGrads[tid].begin(), mThreadGrads[tid].end(), 0.0f);
        #pragma omp for
        for (int i = 0; i < batchSize; ++i) {
            size_t baseIdx = i * criticInputDim;
            AlignedVector32<float> q_pred(4);
            critic.ForwardWithCache(mCriticInputBuffer.data() + baseIdx, q_pred.data(), mThreadCaches[tid]);
            float td_error = mTargetQ[i] - q_pred[0];
            AlignedVector32<float> q_grad(4); q_grad[0] = -2.0f * td_error; q_grad[1] = q_grad[2] = q_grad[3] = 0.0f;
            critic.Backward(mCriticInputBuffer.data() + baseIdx, q_grad.data(), nullptr, mThreadGrads[tid].data(), mThreadCaches[tid]);
        }
    }
    for (int t = 0; t < numThreads; ++t) {
        size_t gradOffset = 0;
        for (size_t l = 0; l < critic.GetNumLayers(); ++l) {
            auto& layer = critic.GetLayer(l);
            AddVectors_AVX2(layer.GetControlPointGradients().data(), mThreadGrads[t].data() + gradOffset, layer.GetNumParams());
            gradOffset += layer.GetNumParams();
        }
    }
    critic.ScaleGradients(1.0f / batchSize);
    (isCritic1 ? mCritic1Optimizer : mCritic2Optimizer).zeroGrad();
}

void TD3Trainer::ComputeActorGradient(ReplayBuffer& buffer)
{
    const int batchSize = mConfig.batchSize;
    const int latentDim = mModel.GetLatentDim();
    const int criticInputDim = mStateDim + mActionDim + latentDim;
    auto& actor = mModel.GetActor(); auto& critic1 = mModel.GetCritic1();
    const int actorWeights = actor.GetNumWeights();
    actor.ZeroGradients();
    int numThreads = omp_get_max_threads();
    if (mActorCaches.size() != (size_t)numThreads) {
        mActorCaches.resize(numThreads); mCriticCaches.resize(numThreads);
        if (mThreadGrads.size() != (size_t)numThreads || mThreadGrads[0].size() < (size_t)actorWeights) mThreadGrads.assign(numThreads, AlignedVector32<float>(actorWeights, 0.0f));
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::fill(mThreadGrads[tid].begin(), mThreadGrads[tid].end(), 0.0f);
        #pragma omp for
        for (int i = 0; i < batchSize; ++i) {
            const float* state = mBatchStates.data() + i * mStateDim;
            AlignedVector32<float> action(mActionDim); actor.ForwardWithCache(state, action.data(), mActorCaches[tid]);
            AlignedVector32<float> critic_input(criticInputDim);
            std::memcpy(critic_input.data(), state, mStateDim * sizeof(float));
            std::memcpy(critic_input.data() + mStateDim, action.data(), mActionDim * sizeof(float));
            AlignedVector32<float> zPos(latentDim); mModel.GetLatentMemory().GetLatentStates(zPos.data(), nullptr, 0);
            std::memcpy(critic_input.data() + mStateDim + mActionDim, zPos.data(), latentDim * sizeof(float));
            AlignedVector32<float> q_value(4); critic1.ForwardWithCache(critic_input.data(), q_value.data(), mCriticCaches[tid]);
            AlignedVector32<float> q_grad(4); q_grad[0] = 1.0f; q_grad[1] = q_grad[2] = q_grad[3] = 0.0f;
            AlignedVector32<float> critic_input_grad(criticInputDim);
            critic1.Backward(critic_input.data(), q_grad.data(), critic_input_grad.data(), nullptr, mCriticCaches[tid]);
            actor.Backward(state, critic_input_grad.data() + mStateDim, nullptr, mThreadGrads[tid].data(), mActorCaches[tid]);
        }
    }
    for (int t = 0; t < numThreads; ++t) {
        size_t gradOffset = 0;
        for (size_t l = 0; l < actor.GetNumLayers(); ++l) {
            auto& layer = actor.GetLayer(l);
            AddVectors_AVX2(layer.GetControlPointGradients().data(), mThreadGrads[t].data() + gradOffset, layer.GetNumParams());
            gradOffset += layer.GetNumParams();
        }
    }
    actor.ScaleGradients(-1.0f / batchSize); mActorOptimizer.zeroGrad();
}

void TD3Trainer::UpdateTargets() { mModel.UpdateTargets(mConfig.tau); }
// EXTERN FROM main_train.cpp
#include <queue>
#include <condition_variable>
#include "VisualState.h" // Needed for IOTask definition visibility if we used it here, but we'll use a simpler extern

struct IOTask {
    enum Type { SAVE_MODEL, SNAPSHOT_OPPONENT };
    Type type;
    std::string path;
    std::string robotPath;
    int numSatellites;
    int obsDim;
    std::vector<float> weights;
};
extern std::queue<IOTask> gIOQueue;
extern std::mutex gIOMutex;
extern std::condition_variable gIOCV;

void TD3Trainer::SnapshotOpponent() {
    IOTask task;
    task.type = IOTask::SNAPSHOT_OPPONENT;
    task.weights = mModel.GetActor().GetAllWeights(); // Copy weights now
    {
        std::lock_guard<std::mutex> lock(gIOMutex);
        gIOQueue.push(std::move(task));
    }
    gIOCV.notify_one();
}

void TD3Trainer::Save(const std::string& path, const std::string& robotPath, int numSatellites, int obsDim) const
{
    IOTask task;
    task.type = IOTask::SAVE_MODEL;
    task.path = path;
    task.robotPath = robotPath;
    task.numSatellites = numSatellites;
    task.obsDim = obsDim;
    // We need to const_cast to call GetAllWeights() or make it const
    task.weights = const_cast<SpanActorCritic&>(mModel).GetActor().GetAllWeights();
    
    {
        std::lock_guard<std::mutex> lock(gIOMutex);
        gIOQueue.push(std::move(task));
    }
    gIOCV.notify_one();
}

void TD3Trainer::Save(const std::string& path) const
{
    Save(path, "robots/combat_bot.json", 0, mStateDim);
}

bool TD3Trainer::SampleOpponent() {
    std::vector<float> weights, biases;
    if (mOpponentPool.SampleOpponentRecent(weights, biases, mRng)) { mModel.GetActor().SetAllWeights(weights); return true; }
    return false;
}

void TD3Trainer::Load(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return;
    int version, stateDim, actionDim;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    file.read(reinterpret_cast<char*>(&stateDim), sizeof(int)); file.read(reinterpret_cast<char*>(&actionDim), sizeof(int));
    if (stateDim != mStateDim || actionDim != mActionDim) return;
    int numWeights; file.read(reinterpret_cast<char*>(&numWeights), sizeof(int));
    std::vector<float> weights(numWeights); file.read(reinterpret_cast<char*>(weights.data()), numWeights * sizeof(float));
    mModel.GetActor().SetAllWeights(weights); mModel.UpdateTargets(1.0f);
}
