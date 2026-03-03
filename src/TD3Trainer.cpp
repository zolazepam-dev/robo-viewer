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

    int criticInputDim = stateDim + actionDim + mModel.GetLatentDim();
    mCriticInputBuffer.resize(batchSize * criticInputDim);
    mSampledIndices.resize(batchSize);
    
    // Pre-allocate temp buffers for batch operations
    mActorOutputBuffer.resize(batchSize * actionDim);
    mCriticQBuffer.resize(batchSize * 4);
    mLatentZPos.resize(batchSize * mModel.GetLatentDim());
    mLatentZVel.resize(batchSize * mModel.GetLatentDim());
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

void TD3Trainer::SelectActionBatchWithLatent(const float* states, float* actions, int batchSize, const std::vector<int>& envIndices)
{
    mModel.SelectActionBatchWithLatent(states, actions, batchSize, envIndices, true);
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
        UpdateActor(buffer);
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
        UpdateActor(buffer);
        UpdateTargets();
    }
    
    mUpdateCount++;
    mStepCount++;
    
    if (mStepCount % mConfig.snapshotInterval == 0)
    {
        SnapshotOpponent();
    }
}

float TD3Trainer::ComputeCriticLoss(SpanNetwork& critic, const float* criticInput, int batchSize)
{
    critic.ForwardBatch(criticInput, mCriticQBuffer.data(), batchSize);
    
    float loss = 0.0f;
    for (int i = 0; i < batchSize; ++i)
    {
        float tdError = mTargetQ[i] - mCriticQBuffer[i * 4];
        loss += tdError * tdError;
    }
    return loss / batchSize;
}

void TD3Trainer::UpdateCritic(ReplayBuffer& buffer)
{
    const int batchSize = mConfig.batchSize;
    const int latentDim = mModel.GetLatentDim();
    
    // STEP 1: Generate next actions using target actor - BATCH FORWARD
    mModel.GetActorTarget().ForwardBatch(mBatchNextStates.data(), mNextActions.data(), batchSize);
    ForwardMoLU_AVX2(mNextActions.data(), mActionDim * batchSize);
    
    // STEP 2: Add clipped noise to all actions (vectorized)
    std::normal_distribution<float> noiseDist(0.0f, mConfig.policyNoise);
    for (int i = 0; i < batchSize * mActionDim; ++i)
    {
        float noise = std::clamp(noiseDist(mRng), -mConfig.noiseClip, mConfig.noiseClip);
        mNextActions[i] = std::clamp(mNextActions[i] + noise, -1.0f, 1.0f);
    }
    
    // STEP 3: Build critic input buffer (state + action + latent) - BATCH
    // Pre-allocate latent buffers
    AlignedVector32<float> zPos(latentDim);
    AlignedVector32<float> zVel(latentDim);
    
    for (int i = 0; i < batchSize; ++i)
    {
        int envIdx = 0;  // TODO: Use mSampledIndices[i] when ReplayBuffer updated
        mModel.GetLatentMemory().GetLatentStates(zPos.data(), zVel.data(), envIdx);
        
        size_t baseIdx = i * (mStateDim + mActionDim + latentDim);
        std::copy(mBatchNextStates.data() + i * mStateDim, 
                  mBatchNextStates.data() + (i + 1) * mStateDim, 
                  mCriticInputBuffer.data() + baseIdx);
        std::copy(mNextActions.data() + i * mActionDim, 
                  mNextActions.data() + (i + 1) * mActionDim, 
                  mCriticInputBuffer.data() + baseIdx + mStateDim);
        std::copy(zPos.data(), zPos.data() + latentDim, 
                  mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim);
    }
    
    // STEP 4: Target Q evaluation - BATCH FORWARD (both critics at once)
    mModel.GetCritic1Target().ForwardBatch(mCriticInputBuffer.data(), mQ1Values.data(), batchSize);
    mModel.GetCritic2Target().ForwardBatch(mCriticInputBuffer.data(), mQ2Values.data(), batchSize);
    
    // STEP 5: Compute target Q values (vectorized)
    for (int i = 0; i < batchSize; ++i)
    {
        float q1 = mQ1Values[i * 4];
        float q2 = mQ2Values[i * 4];
        float minQ = (q1 < q2) ? q1 : q2;  // Faster than std::min
        mTargetQ[i] = mBatchRewards[i] + mConfig.gamma * (1.0f - mBatchDones[i]) * minQ;
    }
    
    // STEP 6: Update critics via weight perturbation (sampled subset for speed)
    // Build current-state critic input
    for (int i = 0; i < batchSize; ++i)
    {
        int envIdx = 0;
        mModel.GetLatentMemory().GetLatentStates(zPos.data(), zVel.data(), envIdx);
        
        size_t baseIdx = i * (mStateDim + mActionDim + latentDim);
        std::copy(mBatchStates.data() + i * mStateDim, 
                  mBatchStates.data() + (i + 1) * mStateDim, 
                  mCriticInputBuffer.data() + baseIdx);
        std::copy(mBatchActions.data() + i * mActionDim, 
                  mBatchActions.data() + (i + 1) * mActionDim, 
                  mCriticInputBuffer.data() + baseIdx + mStateDim);
        std::copy(zPos.data(), zPos.data() + latentDim, 
                  mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim);
    }
    
    // Update critic 1 with sampled weight perturbation (every 32nd weight for speed)
    auto& critic1 = mModel.GetCritic1();
    auto weights1 = critic1.GetAllWeights();
    const float criticLR = mConfig.criticLR;
    const float epsilon = 0.01f;
    
    for (size_t w = 0; w < weights1.size() && w < mGrads.size(); w += 128)
    {
        float originalLoss = ComputeCriticLoss(critic1, mCriticInputBuffer.data(), batchSize);
        weights1[w] += criticLR;
        critic1.SetAllWeights(weights1);
        float newLoss = ComputeCriticLoss(critic1, mCriticInputBuffer.data(), batchSize);
        
        if (newLoss > originalLoss)
        {
            weights1[w] -= 2.0f * criticLR;
        }
    }
    critic1.SetAllWeights(weights1);
    
    // Update critic 2 similarly
    auto& critic2 = mModel.GetCritic2();
    auto weights2 = critic2.GetAllWeights();
    
    for (size_t w = 0; w < weights2.size() && w < mGrads.size(); w += 128)
    {
        float originalLoss = ComputeCriticLoss(critic2, mCriticInputBuffer.data(), batchSize);
        weights2[w] += criticLR;
        critic2.SetAllWeights(weights2);
        float newLoss = ComputeCriticLoss(critic2, mCriticInputBuffer.data(), batchSize);
        
        if (newLoss > originalLoss)
        {
            weights2[w] -= 2.0f * criticLR;
        }
    }
    critic2.SetAllWeights(weights2);
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

void TD3Trainer::UpdateActor(ReplayBuffer& buffer)
{
    const int batchSize = mConfig.batchSize;
    const int latentDim = mModel.GetLatentDim();
    
    auto& actor = mModel.GetActor();
    auto weights = actor.GetAllWeights();
    const float actorLR = mConfig.actorLR;
    
    AlignedVector32<float> zPos(latentDim);
    AlignedVector32<float> zVel(latentDim);
    
    // STEP 1: Build critic input buffer with current policy actions - BATCH
    for (int i = 0; i < batchSize; ++i)
    {
        const float* state = mBatchStates.data() + i * mStateDim;
        
        // Actor forward - will batch this below
        actor.Forward(state, mActorOutputBuffer.data() + i * mActionDim);
    }
    ForwardMoLU_AVX2(mActorOutputBuffer.data(), mActionDim * batchSize);
    
    // Get latents and build full critic input
    for (int i = 0; i < batchSize; ++i)
    {
        int envIdx = 0;
        mModel.GetLatentMemory().GetLatentStates(zPos.data(), zVel.data(), envIdx);
        
        size_t baseIdx = i * (mStateDim + mActionDim + latentDim);
        std::copy(mBatchStates.data() + i * mStateDim, 
                  mBatchStates.data() + (i + 1) * mStateDim, 
                  mCriticInputBuffer.data() + baseIdx);
        std::copy(mActorOutputBuffer.data() + i * mActionDim, 
                  mActorOutputBuffer.data() + (i + 1) * mActionDim, 
                  mCriticInputBuffer.data() + baseIdx + mStateDim);
        std::copy(zPos.data(), zPos.data() + latentDim, 
                  mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim);
    }
    
    // STEP 2: Compute baseline Q-value - BATCH FORWARD
    mModel.GetCritic1().ForwardBatch(mCriticInputBuffer.data(), mQ1Values.data(), batchSize);
    
    float baselineQ = 0.0f;
    for (int i = 0; i < batchSize; ++i)
    {
        baselineQ += mQ1Values[i * 4];
    }
    baselineQ /= batchSize;
    
    // STEP 3: Directed weight perturbation - sample every 16th weight for speed
    // This is the key optimization: fewer perturbations = faster updates
    for (size_t w = 0; w < weights.size() && w < mGrads.size(); w += 64)
    {
        float originalQ = baselineQ;
        
        // Perturb weight
        weights[w] += actorLR * 2.0f;
        actor.SetAllWeights(weights);
        
        // Re-evaluate actions - BATCH FORWARD
        for (int i = 0; i < batchSize; ++i)
        {
            const float* state = mBatchStates.data() + i * mStateDim;
            actor.Forward(state, mActorOutputBuffer.data() + i * mActionDim);
        }
        ForwardMoLU_AVX2(mActorOutputBuffer.data(), mActionDim * batchSize);
        
        // Rebuild critic input with new actions
        for (int i = 0; i < batchSize; ++i)
        {
            int envIdx = 0;
            mModel.GetLatentMemory().GetLatentStates(zPos.data(), zVel.data(), envIdx);
            
            size_t baseIdx = i * (mStateDim + mActionDim + latentDim);
            std::copy(mBatchStates.data() + i * mStateDim, 
                      mBatchStates.data() + (i + 1) * mStateDim, 
                      mCriticInputBuffer.data() + baseIdx);
            std::copy(mActorOutputBuffer.data() + i * mActionDim, 
                      mActorOutputBuffer.data() + (i + 1) * mActionDim, 
                      mCriticInputBuffer.data() + baseIdx + mStateDim);
            std::copy(zPos.data(), zPos.data() + latentDim, 
                      mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim);
        }
        
        // Evaluate Q - BATCH FORWARD
        mModel.GetCritic1().ForwardBatch(mCriticInputBuffer.data(), mQ1Values.data(), batchSize);
        
        float perturbedQ = 0.0f;
        for (int i = 0; i < batchSize; ++i)
        {
            perturbedQ += mQ1Values[i * 4];
        }
        perturbedQ /= batchSize;
        
        // Keep improvement or revert
        if (perturbedQ <= originalQ)
        {
            weights[w] -= actorLR * 2.0f;
        }
    }
    actor.SetAllWeights(weights);
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
