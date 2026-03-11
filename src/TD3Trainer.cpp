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
    fprintf(stderr, "[TD3Trainer::TD3Trainer] stateDim=%d, actionDim=%d, hiddenDim=%d, latentDim=%d\n",
            stateDim, actionDim, config.hiddenDim, config.latentDim);
    fflush(stderr);
    
    fprintf(stderr, "[TD3Trainer::TD3Trainer] Initializing model...\n");
    fflush(stderr);
    
    RFFConfig rffConfig;
    rffConfig.num_features = config.rffNumFeatures;
    rffConfig.sigma = config.rffSigma;
    rffConfig.seed = 42;
    
    mModel.Init(stateDim, actionDim, config.hiddenDim, config.latentDim, rffConfig, mRng);
    fprintf(stderr, "[TD3Trainer::TD3Trainer] Model initialized\n");
    fflush(stderr);

    int criticInputDim = stateDim + actionDim + mModel.GetLatentDim();

    int batchSize = config.batchSize;
    fprintf(stderr, "[TD3Trainer::TD3Trainer] Allocating buffers (batchSize=%d)...\n", batchSize);
    fflush(stderr);
    
    try {
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
        
        mCriticInputBuffer.resize(batchSize * criticInputDim);
        mSampledIndices.resize(batchSize);

        mActorOutputBuffer.resize(batchSize * actionDim);
        mCriticQBuffer.resize(batchSize * 4);
        mLatentZPos.resize(batchSize * mModel.GetLatentDim());
        mLatentZVel.resize(batchSize * mModel.GetLatentDim());

        // Resize per-thread caches and buffers
        int numThreads = omp_get_max_threads();
        mCriticCaches.resize(numThreads);
        mActorCaches.resize(numThreads);
        
        // Safety margin for AVX2 and alignment
        size_t safety = 16;
        mThreadGrads.assign(numThreads, AlignedVector32<float>(mModel.GetActor().GetNumWeights() + mModel.GetCritic1().GetNumWeights() + mModel.GetLatentMemory().GetDynamics().GetNumParams() + safety, 0.0f));
        mThreadQValueBuffers.assign(numThreads, AlignedVector32<float>(4 + safety));
        mThreadQGradBuffers.assign(numThreads, AlignedVector32<float>(4 + safety));
        mThreadActorInputBuffers.assign(numThreads, AlignedVector32<float>(stateDim + config.latentDim + safety));
        mThreadActionBuffers.assign(numThreads, AlignedVector32<float>(actionDim + safety));
        mThreadCriticInputBuffers.assign(numThreads, AlignedVector32<float>(criticInputDim + safety));
        mThreadZPosBuffers.assign(numThreads, AlignedVector32<float>(config.latentDim + safety));
        mThreadCriticInputGradBuffers.assign(numThreads, AlignedVector32<float>(criticInputDim + safety));

        fprintf(stderr, "[TD3Trainer::TD3Trainer] Buffers allocated\n");
        fflush(stderr);
    } catch (const std::exception& e) {
        fprintf(stderr, "[TD3Trainer::TD3Trainer] Exception during buffer allocation: %s\n", e.what());
        fflush(stderr);
        throw;
    }

    MuonOptimizer::Config optConfig;
    optConfig.lrMuon = mConfig.muonLR;
    optConfig.betaMuon = mConfig.muonBeta;
    optConfig.nsSteps = mConfig.muonNSSteps;
    optConfig.lrFallback = mConfig.criticLR;
    optConfig.eps = mConfig.muonEpsilon;

    mActorOptimizer = MuonOptimizer(optConfig);
    mCritic1Optimizer = MuonOptimizer(optConfig);
    mCritic2Optimizer = MuonOptimizer(optConfig);
    mDynamicsOptimizer = MuonOptimizer(optConfig);
    
    // Register parameters with optimizers
    auto registerParams = [&](SpanNetwork& net, MuonOptimizer& opt) {
        for (size_t l = 0; l < net.GetNumLayers(); ++l) {
            auto& layer = net.GetLayer(l);
            auto& cp = layer.GetTrainableWeights();
            auto& grad = layer.GetWeightsGradient();
            int rows = layer.GetOutputDim();
            int cols = layer.GetNumFeatures();
            
            if (cols > 1) opt.addParameter(cp.data(), grad.data(), rows, cols);
            else opt.addParameter1D(cp.data(), grad.data(), cp.size());

            auto& bias = layer.GetTrainableBias();
            auto& biasGrad = layer.GetBiasGradient();
            opt.addParameter1D(bias.data(), biasGrad.data(), bias.size());
        }
    };

    registerParams(mModel.GetActor(), mActorOptimizer);
    registerParams(mModel.GetCritic1(), mCritic1Optimizer);
    registerParams(mModel.GetCritic2(), mCritic2Optimizer);

    // Register dynamics
    auto& dynamics = mModel.GetLatentMemory().GetDynamics();
    auto& rffLayer = dynamics.GetRFFLayer();
    {
        auto& cp = rffLayer.GetTrainableWeights();
        auto& grad = rffLayer.GetWeightsGradient();
        int rows = rffLayer.GetOutputDim();
        int cols = rffLayer.GetNumFeatures();
        
        if (cols > 1) mDynamicsOptimizer.addParameter(cp.data(), grad.data(), rows, cols);
        else mDynamicsOptimizer.addParameter1D(cp.data(), grad.data(), cp.size());

        auto& bias = rffLayer.GetTrainableBias();
        auto& biasGrad = rffLayer.GetBiasGradient();
        mDynamicsOptimizer.addParameter1D(bias.data(), biasGrad.data(), bias.size());
    }

    fprintf(stderr, "[TD3Trainer::TD3Trainer] INIT COMPLETE\n");
    fflush(stderr);
}

void TD3Trainer::SelectAction(const float* state, float* action) { 
    std::lock_guard<std::mutex> lock(mMutex);
    mModel.SelectAction(state, action, nullptr, true, 0); 
}
void TD3Trainer::SelectActionEval(const float* state, float* action) { 
    std::lock_guard<std::mutex> lock(mMutex);
    mModel.SelectAction(state, action, nullptr, false, 0); 
}
void TD3Trainer::SelectActionWithLatent(const float* state, float* action, int envIdx) { 
    // Read-only inference - no lock needed
    mModel.SelectAction(state, action, nullptr, true, envIdx); 
}
void TD3Trainer::SelectActionBatchWithLatent(const float* states, float* actions, int batchSize, const std::vector<int>& envIndices) { 
    // Read-only inference - no lock needed (weights are not modified during forward pass)
    mModel.SelectActionBatchWithLatent(states, actions, batchSize, envIndices, true); 
}
void TD3Trainer::SelectActionResidual(const float* state, float* residualAction) { 
    SelectAction(state, residualAction); 
}

std::vector<float> TD3Trainer::GetDynamicsWeights() const
{
    auto& dynamics = mModel.GetLatentMemory().GetDynamics();
    auto& rffLayer = dynamics.GetRFFLayer();
    std::vector<float> weights;
    
    const auto& w = rffLayer.GetTrainableWeights();
    const auto& b = rffLayer.GetTrainableBias();
    
    weights.insert(weights.end(), w.begin(), w.end());
    weights.insert(weights.end(), b.begin(), b.end());
    
    return weights;
}

void TD3Trainer::SetDynamicsWeights(const std::vector<float>& weights)
{
    auto& dynamics = mModel.GetLatentMemory().GetDynamics();
    auto& rffLayer = dynamics.GetRFFLayer();
    
    size_t offset = 0;
    auto& w = rffLayer.GetTrainableWeights();
    auto& b = rffLayer.GetTrainableBias();
    
    std::copy(weights.begin() + offset, weights.begin() + offset + w.size(), w.begin());
    offset += w.size();
    
    std::copy(weights.begin() + offset, weights.begin() + offset + b.size(), b.begin());
}

void TD3Trainer::Train(ReplayBuffer& buffer)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (!buffer.IsReady(mConfig.batchSize)) return;

    if (mStepCount % 10 == 0) {
        fprintf(stderr, "[TD3Trainer::Train] Step %d starting...\n", mStepCount);
        fflush(stderr);
    }

    auto start = std::chrono::high_resolution_clock::now();

    auto sampleStart = std::chrono::high_resolution_clock::now();
    buffer.Sample(mConfig.batchSize, mBatchStates.data(), mBatchActions.data(), mBatchRewards.data(), 
                  mBatchNextStates.data(), mBatchDones.data(), mLatentZPos.data(), mLatentZVel.data(), mRng);
    float bTime = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - sampleStart).count();

    if (mStepCount % 10 == 0) {
        fprintf(stderr, "[TD3Trainer::Train] Sampled buffer\n");
        fflush(stderr);
    }

    auto criticStart = std::chrono::high_resolution_clock::now();
    UpdateCritic(buffer);
    float cTime = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - criticStart).count();

    if (mStepCount % 10 == 0) {
        fprintf(stderr, "[TD3Trainer::Train] Updated critic\n");
        fflush(stderr);
    }

    float aTime = 0.0f;
    if (mUpdateCount % mConfig.policyDelay == 0) {
        auto actorStart = std::chrono::high_resolution_clock::now();
        UpdateActor(buffer);
        UpdateTargets();
        aTime = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - actorStart).count();
        if (mStepCount % 10 == 0) {
            fprintf(stderr, "[TD3Trainer::Train] Updated actor & targets\n");
            fflush(stderr);
        }
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
    int numThreads = omp_get_max_threads();

    // 1. Compute next actions with target actor
    mModel.GetActorTarget().ForwardBatch(mBatchNextStates.data(), mNextActions.data(), batchSize);
    ForwardMoLU_AVX2(mNextActions.data(), mActionDim * batchSize);

    // 2. Add target policy noise
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
        }
    }

    // 3. Step latent dynamics forward to get z_{t+1} for the target Q-value
    // We use the sampled z_t, v_t and step them using the current dynamics
    AlignedVector32<float> nextLatentPos(batchSize * latentDim);
    AlignedVector32<float> nextLatentVel(batchSize * latentDim);
    AlignedVector32<float> accelerations(batchSize * latentDim);

    mModel.GetLatentMemory().GetDynamics().ComputeAccelerationBatch(
        mLatentZPos.data(), mLatentZVel.data(), mBatchStates.data(), 
        accelerations.data(), batchSize);
    
    float dt = mModel.GetLatentMemory().GetMemory().dt;
    #pragma omp parallel for
    for (int i = 0; i < batchSize * latentDim; ++i) {
        float v = mLatentZVel[i] + accelerations[i] * dt;
        nextLatentVel[i] = v;
        nextLatentPos[i] = mLatentZPos[i] + v * dt;
    }

    // 4. Prepare target critic input: [s', a', z_{t+1}]
    #pragma omp parallel for
    for (int i = 0; i < batchSize; ++i) {
        size_t baseIdx = i * criticInputDim;
        std::memcpy(mCriticInputBuffer.data() + baseIdx, mBatchNextStates.data() + i * mStateDim, mStateDim * sizeof(float));
        std::memcpy(mCriticInputBuffer.data() + baseIdx + mStateDim, mNextActions.data() + i * mActionDim, mActionDim * sizeof(float));
        std::memcpy(mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim, nextLatentPos.data() + i * latentDim, latentDim * sizeof(float));
    }

    #pragma omp parallel num_threads(numThreads)
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < batchSize; ++i) {
            size_t baseIdx = i * criticInputDim;
            float q1[4], q2[4];
            mModel.GetCritic1Target().ForwardWithCache(mCriticInputBuffer.data() + baseIdx, q1, mCriticCaches[tid]);
            mModel.GetCritic2Target().ForwardWithCache(mCriticInputBuffer.data() + baseIdx, q2, mCriticCaches[tid]);
            mQ1Values[i * 4] = q1[0];
            mQ2Values[i * 4] = q2[0];
        }
    }

    // 5. Compute target Q-value
    #pragma omp parallel for
    for (int i = 0; i < batchSize; ++i) {
        float minQ = std::min(mQ1Values[i * 4], mQ2Values[i * 4]);
        mTargetQ[i] = mBatchRewards[i] + mConfig.gamma * (1.0f - mBatchDones[i]) * minQ;
    }

    // 6. Prepare current critic input: [s, a, z_t]
    #pragma omp parallel for
    for (int i = 0; i < batchSize; ++i) {
        size_t baseIdx = i * criticInputDim;
        std::memcpy(mCriticInputBuffer.data() + baseIdx, mBatchStates.data() + i * mStateDim, mStateDim * sizeof(float));
        std::memcpy(mCriticInputBuffer.data() + baseIdx + mStateDim, mBatchActions.data() + i * mActionDim, mActionDim * sizeof(float));
        std::memcpy(mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim, mLatentZPos.data() + i * latentDim, latentDim * sizeof(float));
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
    auto& critic1 = mModel.GetCritic1();
    
    int numThreads = omp_get_max_threads();

    // 1. Prepare actor input: [s, z_t]
    size_t actorInputDim = mStateDim + latentDim;
    
    #pragma omp parallel num_threads(numThreads)
    {
        int tid = omp_get_thread_num();
        AlignedVector32<float>& actorInput = mThreadActorInputBuffers[tid];
        AlignedVector32<float>& threadAction = mThreadActionBuffers[tid];
        AlignedVector32<float>& threadCriticInput = mThreadCriticInputBuffers[tid];
        AlignedVector32<float>& threadQValue = mThreadQValueBuffers[tid];

        #pragma omp for
        for (int i = 0; i < batchSize; ++i) {
            const float* state = mBatchStates.data() + i * mStateDim;
            std::memcpy(actorInput.data(), state, mStateDim * sizeof(float));
            std::memcpy(actorInput.data() + mStateDim, mLatentZPos.data() + i * latentDim, latentDim * sizeof(float));

            actor.ForwardWithCache(actorInput.data(), threadAction.data(), mActorCaches[tid]);
            
            // Store predicted action for current batch
            std::memcpy(mActorOutputBuffer.data() + i * mActionDim, threadAction.data(), mActionDim * sizeof(float));

            // Prepare critic input: [s, a_pred, z_t]
            std::memcpy(threadCriticInput.data(), state, mStateDim * sizeof(float));
            std::memcpy(threadCriticInput.data() + mStateDim, threadAction.data(), mActionDim * sizeof(float));
            std::memcpy(threadCriticInput.data() + mStateDim + mActionDim, mLatentZPos.data() + i * latentDim, latentDim * sizeof(float));
            
            critic1.ForwardWithCache(threadCriticInput.data(), threadQValue.data(), mCriticCaches[tid]);
            
            // Note: mQ1Values is updated sequentially or we could store it per thread
            mQ1Values[i * 4] = threadQValue[0];
        }
    }

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
    critic.ZeroGradients();
    int numThreads = omp_get_max_threads();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::fill(mThreadGrads[tid].begin(), mThreadGrads[tid].end(), 0.0f);
        AlignedVector32<float>& q_pred = mThreadQValueBuffers[tid];
        AlignedVector32<float>& q_grad = mThreadQGradBuffers[tid];
        
        #pragma omp for
        for (int i = 0; i < batchSize; ++i) {
            size_t baseIdx = i * criticInputDim;
            critic.ForwardWithCache(mCriticInputBuffer.data() + baseIdx, q_pred.data(), mCriticCaches[tid]);
            float td_error = mTargetQ[i] - q_pred[0];
            
            q_grad[0] = -2.0f * td_error; q_grad[1] = q_grad[2] = q_grad[3] = 0.0f;
            critic.Backward(mCriticInputBuffer.data() + baseIdx, q_grad.data(), nullptr, mThreadGrads[tid].data(), mCriticCaches[tid]);
        }
    }
    for (int t = 0; t < numThreads; ++t) {
        size_t gradOffset = 0;
        for (size_t l = 0; l < critic.GetNumLayers(); ++l) {
            auto& layer = critic.GetLayer(l);
            // Add weight gradients
            auto& wGrad = layer.GetWeightsGradient();
            AddVectors_AVX2(wGrad.data(), mThreadGrads[t].data() + gradOffset, wGrad.size());
            gradOffset += wGrad.size();
            // Add bias gradients
            auto& bGrad = layer.GetBiasGradient();
            AddVectors_AVX2(bGrad.data(), mThreadGrads[t].data() + gradOffset, bGrad.size());
            gradOffset += bGrad.size();
        }
    }
    critic.ScaleGradients(1.0f / static_cast<float>(batchSize));
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

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::fill(mThreadGrads[tid].begin(), mThreadGrads[tid].end(), 0.0f);
        AlignedVector32<float>& actorInput = mThreadActorInputBuffers[tid];
        AlignedVector32<float>& threadAction = mThreadActionBuffers[tid];
        AlignedVector32<float>& threadCriticInput = mThreadCriticInputBuffers[tid];
        AlignedVector32<float>& threadZPos = mThreadZPosBuffers[tid];
        AlignedVector32<float>& threadQValue = mThreadQValueBuffers[tid];
        AlignedVector32<float>& threadQGrad = mThreadQGradBuffers[tid];
        AlignedVector32<float>& threadCriticInputGrad = mThreadCriticInputGradBuffers[tid];

        #pragma omp for
        for (int i = 0; i < batchSize; ++i) {
            const float* state = mBatchStates.data() + i * mStateDim;
            
            // Prepare actor input: [s, z_t]
            std::memcpy(actorInput.data(), state, mStateDim * sizeof(float));
            std::memcpy(actorInput.data() + mStateDim, mLatentZPos.data() + i * latentDim, latentDim * sizeof(float));
            
            actor.ForwardWithCache(actorInput.data(), threadAction.data(), mActorCaches[tid]);
            
            // Prepare critic input: [s, a_pred, z_t]
            std::memcpy(threadCriticInput.data(), state, mStateDim * sizeof(float));
            std::memcpy(threadCriticInput.data() + mStateDim, threadAction.data(), mActionDim * sizeof(float));
            std::memcpy(threadCriticInput.data() + mStateDim + mActionDim, mLatentZPos.data() + i * latentDim, latentDim * sizeof(float));
            
            critic1.ForwardWithCache(threadCriticInput.data(), threadQValue.data(), mCriticCaches[tid]);
            
            threadQGrad[0] = 1.0f; threadQGrad[1] = threadQGrad[2] = threadQGrad[3] = 0.0f;
            critic1.Backward(threadCriticInput.data(), threadQGrad.data(), threadCriticInputGrad.data(), nullptr, mCriticCaches[tid]);
            
            actor.Backward(actorInput.data(), threadCriticInputGrad.data() + mStateDim, nullptr, mThreadGrads[tid].data(), mActorCaches[tid]);
        }
    }
    for (int t = 0; t < numThreads; ++t) {
        size_t gradOffset = 0;
        for (size_t l = 0; l < actor.GetNumLayers(); ++l) {
            auto& layer = actor.GetLayer(l);
            // Add weight gradients
            auto& wGrad = layer.GetWeightsGradient();
            AddVectors_AVX2(wGrad.data(), mThreadGrads[t].data() + gradOffset, wGrad.size());
            gradOffset += wGrad.size();
            // Add bias gradients
            auto& bGrad = layer.GetBiasGradient();
            AddVectors_AVX2(bGrad.data(), mThreadGrads[t].data() + gradOffset, bGrad.size());
            gradOffset += bGrad.size();
        }
    }
    actor.ScaleGradients(-1.0f / static_cast<float>(batchSize)); mActorOptimizer.zeroGrad();
}

void TD3Trainer::UpdateTargets() { mModel.UpdateTargets(mConfig.tau); }

// EXTERN FROM main_train.cpp
#include <queue>
#include <condition_variable>
#include "VisualState.h"

// GLOBAL IO STATE FOR ASYNC SAVING
std::queue<IOTask> gIOQueue;
std::mutex gIOMutex;
std::condition_variable gIOCV;

void TD3Trainer::SnapshotOpponent() {
    std::lock_guard<std::mutex> lock(mMutex);
    IOTask task;
    task.type = IOTask::SNAPSHOT_OPPONENT;
    task.weights = mModel.GetActor().GetAllWeights();
    {
        std::lock_guard<std::mutex> lockIO(gIOMutex);
        gIOQueue.push(std::move(task));
    }
    gIOCV.notify_one();
}

void TD3Trainer::Save(const std::string& path, const std::string& robotPath, int numSatellites, int obsDim) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    IOTask task;
    task.type = IOTask::SAVE_MODEL;
    task.path = path;
    task.robotPath = robotPath;
    task.numSatellites = numSatellites;
    task.obsDim = obsDim;
    task.weights = const_cast<SpanActorCritic&>(mModel).GetActor().GetAllWeights();

    {
        std::lock_guard<std::mutex> lock(gIOMutex);
        gIOQueue.push(std::move(task));
    }
    gIOCV.notify_one();
}

void TD3Trainer::Save(const std::string& path) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    Save(path, "robots/combat_bot.json", 0, mStateDim);
}

void TD3Trainer::SaveToDisk(const std::string& path, const std::string& robotConfigPath, int numSatellites, int observationDim) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[TD3Trainer] Failed to save to " << path << "\n";
        return;
    }

    // Checkpoint format version 3 - with robot config metadata
    int version = 3;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));

    // Core dimensions
    file.write(reinterpret_cast<const char*>(&mStateDim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mActionDim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mStepCount), sizeof(int));

    // Robot configuration metadata
    int robotConfigPathLen = static_cast<int>(robotConfigPath.length());
    file.write(reinterpret_cast<const char*>(&robotConfigPathLen), sizeof(int));
    file.write(robotConfigPath.c_str(), robotConfigPathLen);

    // Robot architecture info (for validation on load)
    file.write(reinterpret_cast<const char*>(&numSatellites), sizeof(int));
    file.write(reinterpret_cast<const char*>(&observationDim), sizeof(int));

    // Model weights
    auto weights = mModel.GetActor().GetAllWeights();
    int numWeights = static_cast<int>(weights.size());
    file.write(reinterpret_cast<const char*>(&numWeights), sizeof(int));
    file.write(reinterpret_cast<const char*>(weights.data()), numWeights * sizeof(float));

    // Preference vector
    file.write(reinterpret_cast<const char*>(mPreferenceVector.data()),
               VECTOR_REWARD_DIM * sizeof(float));

    // Checksum for integrity
    uint32_t checksum = 0;
    for (float w : weights) checksum += static_cast<uint32_t>(w * 1000);
    file.write(reinterpret_cast<const char*>(&checksum), sizeof(uint32_t));

    file.close();
    std::cout << "[TD3Trainer] Checkpoint saved to disk: " << path << "\n";
}

bool TD3Trainer::SampleOpponent() {
    std::lock_guard<std::mutex> lock(mMutex);
    std::vector<float> weights, biases;
    if (mOpponentPool.SampleOpponentRecent(weights, biases, mRng)) { mModel.GetActor().SetAllWeights(weights); return true; }
    return false;
}

void TD3Trainer::Load(const std::string& path)
{
    std::lock_guard<std::mutex> lock(mMutex);
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "[TD3Trainer::Load] FAILED to open %s\n", path.c_str());
        return;
    }
    
    int version, stateDim, actionDim;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    
    // Support older checkpoint versions if needed, but for now expect version 3
    if (version < 3) {
        // Fallback for older versions if necessary
    }
    
    file.read(reinterpret_cast<char*>(&stateDim), sizeof(int));
    file.read(reinterpret_cast<char*>(&actionDim), sizeof(int));
    
    // We can load if action dim matches, even if state dim changed (will be handled by network)
    if (actionDim != mActionDim) {
        fprintf(stderr, "[TD3Trainer::Load] Dimension mismatch: action %d vs %d\n", actionDim, mActionDim);
        return;
    }
    
    int stepCount;
    file.read(reinterpret_cast<char*>(&stepCount), sizeof(int));
    mStepCount = stepCount;
    
    int robotPathLen;
    file.read(reinterpret_cast<char*>(&robotPathLen), sizeof(int));
    std::vector<char> robotPathBuf(robotPathLen + 1, 0);
    file.read(robotPathBuf.data(), robotPathLen);
    
    int numSat, obsDim;
    file.read(reinterpret_cast<char*>(&numSat), sizeof(int));
    file.read(reinterpret_cast<char*>(&obsDim), sizeof(int));
    
    int numWeights;
    file.read(reinterpret_cast<char*>(&numWeights), sizeof(int));
    std::vector<float> weights(numWeights);
    file.read(reinterpret_cast<char*>(weights.data()), numWeights * sizeof(float));
    
    // Set weights to actor and update targets
    try {
        mModel.GetActor().SetAllWeights(weights);
        mModel.UpdateTargets(1.0f); // Fast update to targets
        fprintf(stderr, "[TD3Trainer::Load] Successfully loaded %d weights from %s\n", numWeights, path.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "[TD3Trainer::Load] EXCEPTION setting weights: %s\n", e.what());
    }
}
