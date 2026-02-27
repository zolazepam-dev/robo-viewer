#include "TD3Trainer.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "AlignedAllocator.h"

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

    int criticInputDim = stateDim + actionDim + config.latentDim;
    mCriticInputBuffer.resize(criticInputDim);
}

void TD3Trainer::SelectAction(const float* state, float* action)
{
    float logProb;
    mModel.SelectAction(state, action, &logProb, true);
}

void TD3Trainer::SelectActionWithLatent(const float* state, float* action, int envIdx)
{
    float logProb;
    mModel.SelectAction(state, action, &logProb, true, envIdx); 
}

void TD3Trainer::SelectActionEval(const float* state, float* action)
{
    float logProb;
    mModel.SelectAction(state, action, &logProb, false);
}

void TD3Trainer::SelectActionResidual(const float* state, float* residualAction)
{
    SelectAction(state, residualAction);
    for (int i = 0; i < mActionDim; ++i)
    {
        residualAction[i] = std::clamp(residualAction[i], -1.0f, 1.0f);
    }
}

void TD3Trainer::Train(ReplayBuffer& buffer)
{
    if (!buffer.IsReady(mConfig.batchSize))
    {
        return;
    }
    
    buffer.Sample(mConfig.batchSize,
                  mBatchStates.data(),
                  mBatchActions.data(),
                  mBatchRewards.data(),
                  mBatchNextStates.data(),
                  mBatchDones.data(),
                  mRng);
    
    UpdateCritic(buffer);
    
    if (mUpdateCount % mConfig.policyDelay == 0)
    {
        UpdateActor();
        UpdateTargets();
    }
    
    mUpdateCount++;
    mStepCount++;
    
    if (mStepCount % mConfig.snapshotInterval == 0)
    {
        SnapshotOpponent();
    }
}

void TD3Trainer::TrainWithVectorRewards(ReplayBuffer& buffer)
{
    if (!buffer.IsReady(mConfig.batchSize))
    {
        return;
    }
    
    buffer.SampleVectorRewards(mConfig.batchSize,
                               mBatchStates.data(),
                               mBatchActions.data(),
                               mBatchVectorRewards.data(),
                               mBatchNextStates.data(),
                               mBatchDones.data(),
                               mRng);
    
    UpdateCriticWithVectorRewards(buffer);
    
    if (mUpdateCount % mConfig.policyDelay == 0)
    {
        UpdateActor();
        UpdateTargets();
    }
    
    mUpdateCount++;
    mStepCount++;
    
    if (mStepCount % mConfig.snapshotInterval == 0)
    {
        SnapshotOpponent();
    }
}

void TD3Trainer::UpdateCritic(ReplayBuffer& buffer)
{
    std::normal_distribution<float> noiseDist(0.0f, mConfig.policyNoise);
    
    for (int i = 0; i < mConfig.batchSize; ++i)
    {
        float* nextAction = mNextActions.data() + i * mActionDim;
        mModel.GetActorTarget().Forward(mBatchNextStates.data() + i * mStateDim, nextAction);
        
        for (int j = 0; j < mActionDim; ++j)
        {
            float noise = std::clamp(noiseDist(mRng), -mConfig.noiseClip, mConfig.noiseClip);
            nextAction[j] = std::clamp(nextAction[j] + noise, -1.0f, 1.0f);
        }
    }
    
    // Properly evaluate target Q-values with state+action+latent concatenation
    int latentDim = mModel.GetLatentDim();
    for (int i = 0; i < mConfig.batchSize; ++i)
    {
        // Get latent state (zero-initialized for now - proper latent tracking needed)
        AlignedVector32<float> zPos(LATENT_DIM, 0.0f);
        mModel.GetLatentMemory().GetLatentStates(zPos.data(), nullptr, 0);

        // Concatenate: state + action + latent into mCriticInputBuffer
        const float* nextState = mBatchNextStates.data() + i * mStateDim;
        const float* nextAction = mNextActions.data() + i * mActionDim;

        int idx = 0;
        std::copy(nextState, nextState + mStateDim, mCriticInputBuffer.data());
        idx += mStateDim;
        std::copy(nextAction, nextAction + mActionDim, mCriticInputBuffer.data() + idx);
        idx += mActionDim;
        std::copy(zPos.data(), zPos.data() + latentDim, mCriticInputBuffer.data() + idx);

        float q1[4], q2[4];
        mModel.GetCritic1Target().Forward(mCriticInputBuffer.data(), q1);
        mModel.GetCritic2Target().Forward(mCriticInputBuffer.data(), q2);

        float minQ = std::min(q1[0], q2[0]);
        mTargetQ[i] = mBatchRewards[i] + mConfig.gamma * (1.0f - mBatchDones[i]) * minQ;
    }
}

void TD3Trainer::UpdateCriticWithVectorRewards(ReplayBuffer& buffer)
{
    static std::normal_distribution<float> noiseDist(0.0f, mConfig.policyNoise);
    
    // Use batch forward pass for actor
    mModel.GetActorTarget().ForwardBatch(mBatchNextStates.data(), mNextActions.data(), mConfig.batchSize);
    
    // Apply MoLU activation to all actions
    ForwardMoLU_AVX2(mNextActions.data(), mActionDim * mConfig.batchSize);
    
    // Add noise to all actions
    for (int i = 0; i < mConfig.batchSize; ++i)
    {
        float* nextAction = mNextActions.data() + i * mActionDim;
        for (int j = 0; j < mActionDim; ++j)
        {
            float noise = std::clamp(noiseDist(mRng), -mConfig.noiseClip, mConfig.noiseClip);
            nextAction[j] = std::clamp(nextAction[j] + noise, -1.0f, 1.0f);
        }
    }
    
    // Use batch forward pass for critics
    AlignedVector32<float> q1Batch(mConfig.batchSize * 4);
    AlignedVector32<float> q2Batch(mConfig.batchSize * 4);
    
    // We need to combine states and actions for critic input - this requires a temporary buffer
    AlignedVector32<float> criticInput(mConfig.batchSize * (mStateDim + mActionDim + mModel.GetLatentDim()));
    for (int i = 0; i < mConfig.batchSize; ++i)
    {
        size_t idx = 0;
        const float* state = mBatchNextStates.data() + i * mStateDim;
        const float* action = mNextActions.data() + i * mActionDim;
        
        // Get latent state for this environment
        AlignedVector32<float> zPos(LATENT_DIM);
        mModel.GetLatentMemory().GetLatentStates(zPos.data(), nullptr, 0); // TODO: per-environment latent?
        
        // Copy state
        std::copy(state, state + mStateDim, criticInput.data() + i * (mStateDim + mActionDim + mModel.GetLatentDim()));
        idx += mStateDim;
        // Copy action
        std::copy(action, action + mActionDim, criticInput.data() + i * (mStateDim + mActionDim + mModel.GetLatentDim()) + mStateDim);
        idx += mActionDim;
        // Copy latent
        std::copy(zPos.data(), zPos.data() + mModel.GetLatentDim(), 
                 criticInput.data() + i * (mStateDim + mActionDim + mModel.GetLatentDim()) + mStateDim + mActionDim);
    }
    
    mModel.GetCritic1Target().ForwardBatch(criticInput.data(), q1Batch.data(), mConfig.batchSize);
    mModel.GetCritic2Target().ForwardBatch(criticInput.data(), q2Batch.data(), mConfig.batchSize);
    
    for (int i = 0; i < mConfig.batchSize; ++i)
    {
        float scalarReward = ComputeScalarReward(mBatchVectorRewards[i]);
        float q1 = q1Batch[i * 4];
        float q2 = q2Batch[i * 4];
        float minQ = std::min(q1, q2);
        mTargetQ[i] = scalarReward + mConfig.gamma * (1.0f - mBatchDones[i]) * minQ;
    }
}

void TD3Trainer::UpdateActor()
{
    std::normal_distribution<float> noiseDist(0.0f, mConfig.actorLR * 0.1f);
    auto weights = mModel.GetActor().GetAllWeights();
    
    for (auto& w : weights)
    {
        w -= mConfig.actorLR * noiseDist(mRng);
    }
    
    mModel.GetActor().SetAllWeights(weights);
}

void TD3Trainer::UpdateTargets()
{
    mModel.UpdateTargets(mConfig.tau);
}

void TD3Trainer::SnapshotOpponent()
{
    auto weights = mModel.GetActor().GetAllWeights();
    mOpponentPool.Snapshot(weights, {}, mStepCount);
}

bool TD3Trainer::SampleOpponent()
{
    std::vector<float> weights, biases;
    if (mOpponentPool.SampleOpponentRecent(weights, biases, mRng))
    {
        mModel.GetActor().SetAllWeights(weights);
        return true;
    }
    return false;
}

void TD3Trainer::Save(const std::string& path) const
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "TD3Trainer: Failed to save to " << path << std::endl;
        return;
    }
    
    int version = 2;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mStateDim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mActionDim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mStepCount), sizeof(int));
    
    auto weights = mModel.GetActor().GetAllWeights();
    int numWeights = static_cast<int>(weights.size());
    file.write(reinterpret_cast<const char*>(&numWeights), sizeof(int));
    file.write(reinterpret_cast<const char*>(weights.data()), numWeights * sizeof(float));
    
    file.write(reinterpret_cast<const char*>(mPreferenceVector.data()), 
               VECTOR_REWARD_DIM * sizeof(float));
    
    file.close();
}

void TD3Trainer::Load(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "TD3Trainer: Failed to load from " << path << std::endl;
        return;
    }
    
    int version, stateDim, actionDim;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    file.read(reinterpret_cast<char*>(&stateDim), sizeof(int));
    file.read(reinterpret_cast<char*>(&actionDim), sizeof(int));
    file.read(reinterpret_cast<char*>(&mStepCount), sizeof(int));
    
    if (stateDim != mStateDim || actionDim != mActionDim)
    {
        std::cerr << "TD3Trainer: Dimension mismatch in loaded model" << std::endl;
        return;
    }
    
    int numWeights;
    file.read(reinterpret_cast<char*>(&numWeights), sizeof(int));
    
    std::vector<float> weights(numWeights);
    file.read(reinterpret_cast<char*>(weights.data()), numWeights * sizeof(float));
    mModel.GetActor().SetAllWeights(weights);
    
    if (version >= 2)
    {
        file.read(reinterpret_cast<char*>(mPreferenceVector.data()),
                  VECTOR_REWARD_DIM * sizeof(float));
    }
    
    mModel.UpdateTargets(1.0f);
    
    file.close();
}
