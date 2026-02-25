#include "TD3Trainer.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

TD3Trainer::TD3Trainer(int stateDim, int actionDim, const TD3Config& config)
    : mStateDim(stateDim)
    , mActionDim(actionDim)
    , mConfig(config)
    , mRng(42)  // Fixed seed for reproducibility
{
    // Create networks
    std::vector<int> actorLayers = {stateDim, config.hiddenDim, config.hiddenDim, actionDim};
    std::vector<int> criticLayers = {stateDim + actionDim, config.hiddenDim, config.hiddenDim, 1};
    
    mActor = std::make_unique<NeuralNetwork>(actorLayers);
    mActorTarget = std::make_unique<NeuralNetwork>(actorLayers);
    
    mCritic1 = std::make_unique<NeuralNetwork>(criticLayers);
    mCritic1Target = std::make_unique<NeuralNetwork>(criticLayers);
    
    mCritic2 = std::make_unique<NeuralNetwork>(criticLayers);
    mCritic2Target = std::make_unique<NeuralNetwork>(criticLayers);
    
    // Initialize weights
    float actorScale = 1.0f / std::sqrt(static_cast<float>(stateDim));
    float criticScale = 1.0f / std::sqrt(static_cast<float>(stateDim + actionDim));
    
    mActor->InitializeWeights(mRng, actorScale);
    mCritic1->InitializeWeights(mRng, criticScale);
    mCritic2->InitializeWeights(mRng, criticScale);
    
    // Copy to targets
    mActorTarget->CopyFrom(*mActor);
    mCritic1Target->CopyFrom(*mCritic1);
    mCritic2Target->CopyFrom(*mCritic2);
    
    // Allocate workspace
    int batchSize = config.batchSize;
    mBatchStates.resize(batchSize * stateDim);
    mBatchActions.resize(batchSize * actionDim);
    mBatchRewards.resize(batchSize);
    mBatchNextStates.resize(batchSize * stateDim);
    mBatchDones.resize(batchSize);
    
    mNextActions.resize(batchSize * actionDim);
    mQ1Values.resize(batchSize);
    mQ2Values.resize(batchSize);
    mTargetQ.resize(batchSize);
    
    mGradsW.resize(mActor->GetNumWeights());
    mGradsB.resize(mActor->GetNumBiases());
}

void TD3Trainer::SelectAction(const float* state, float* action) {
    mActor->Forward(state, action);
    
    // Add exploration noise
    std::normal_distribution<float> noiseDist(0.0f, mConfig.explNoise);
    for (int i = 0; i < mActionDim; ++i) {
        action[i] += noiseDist(mRng);
        action[i] = std::clamp(action[i], -1.0f, 1.0f);  // Clamp to valid range
    }
}

void TD3Trainer::SelectActionEval(const float* state, float* action) {
    mActor->Forward(state, action);
}

void TD3Trainer::Train(ReplayBuffer& buffer) {
    if (!buffer.IsReady(mConfig.batchSize)) {
        return;
    }
    
    // Sample batch
    buffer.Sample(mConfig.batchSize, 
                  mBatchStates.data(), 
                  mBatchActions.data(),
                  mBatchRewards.data(),
                  mBatchNextStates.data(),
                  mBatchDones.data(),
                  mRng);
    
    // Update critic
    UpdateCritic(buffer);
    
    // Delayed policy updates
    if (mUpdateCount % mConfig.policyDelay == 0) {
        UpdateActor();
        UpdateTargets();
    }
    
    mUpdateCount++;
    mStepCount++;
}

void TD3Trainer::UpdateCritic(ReplayBuffer& buffer) {
    // Get next actions from target actor (with target policy smoothing)
    std::normal_distribution<float> noiseDist(0.0f, mConfig.policyNoise);
    
    for (int i = 0; i < mConfig.batchSize; ++i) {
        mActorTarget->Forward(mBatchNextStates.data() + i * mStateDim,
                              mNextActions.data() + i * mActionDim);
        
        // Add clipped noise
        for (int j = 0; j < mActionDim; ++j) {
            float noise = noiseDist(mRng);
            noise = std::clamp(noise, -mConfig.noiseClip, mConfig.noiseClip);
            mNextActions[i * mActionDim + j] = std::clamp(
                mNextActions[i * mActionDim + j] + noise, -1.0f, 1.0f);
        }
    }
    
    // Compute target Q = min(Q1_target, Q2_target)
    for (int i = 0; i < mConfig.batchSize; ++i) {
        // Concatenate state and action for critic input
        std::vector<float> criticInput(mStateDim + mActionDim);
        std::copy(mBatchNextStates.begin() + i * mStateDim,
                  mBatchNextStates.begin() + (i + 1) * mStateDim,
                  criticInput.begin());
        std::copy(mNextActions.begin() + i * mActionDim,
                  mNextActions.begin() + (i + 1) * mActionDim,
                  criticInput.begin() + mStateDim);
        
        float q1, q2;
        mCritic1Target->Forward(criticInput.data(), &q1);
        mCritic2Target->Forward(criticInput.data(), &q2);
        
        float minQ = std::min(q1, q2);
        mTargetQ[i] = mBatchRewards[i] + mConfig.gamma * (1.0f - mBatchDones[i]) * minQ;
    }
    
    // Compute Q values and update critics (simplified - just TD error logging)
    for (int i = 0; i < mConfig.batchSize; ++i) {
        std::vector<float> criticInput(mStateDim + mActionDim);
        std::copy(mBatchStates.begin() + i * mStateDim,
                  mBatchStates.begin() + (i + 1) * mStateDim,
                  criticInput.begin());
        std::copy(mBatchActions.begin() + i * mActionDim,
                  mBatchActions.begin() + (i + 1) * mActionDim,
                  criticInput.begin() + mStateDim);
        
        mCritic1->Forward(criticInput.data(), &mQ1Values[i]);
        mCritic2->Forward(criticInput.data(), &mQ2Values[i]);
    }
    
    // Simple gradient descent update (placeholder for proper backprop)
    // In a real implementation, you'd compute gradients and update weights
    // For now, this is a simplified version
}

void TD3Trainer::UpdateActor() {
    // Compute deterministic policy gradient
    // For simplicity, this is a placeholder
    // Real implementation would compute: grad = E[grad Q(s, pi(s)) * grad pi(s)]
    
    // Placeholder: small random perturbation
    std::normal_distribution<float> noiseDist(0.0f, 0.001f);
    auto& weights = mActor->GetWeights();
    for (auto& w : weights) {
        w -= mConfig.actorLR * noiseDist(mRng);
    }
}

void TD3Trainer::UpdateTargets() {
    mActorTarget->SoftUpdate(*mActor, mConfig.tau);
    mCritic1Target->SoftUpdate(*mCritic1, mConfig.tau);
    mCritic2Target->SoftUpdate(*mCritic2, mConfig.tau);
}

void TD3Trainer::Save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "TD3Trainer: Failed to save to " << path << std::endl;
        return;
    }
    
    // Write header
    int version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mStateDim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mActionDim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mStepCount), sizeof(int));
    
    // Write weights
    int numWeights = mActor->GetNumWeights();
    int numBiases = mActor->GetNumBiases();
    file.write(reinterpret_cast<const char*>(&numWeights), sizeof(int));
    file.write(reinterpret_cast<const char*>(&numBiases), sizeof(int));
    
    const auto& weights = mActor->GetWeights();
    const auto& biases = mActor->GetBiases();
    file.write(reinterpret_cast<const char*>(weights.data()), numWeights * sizeof(float));
    file.write(reinterpret_cast<const char*>(biases.data()), numBiases * sizeof(float));
    
    file.close();
}

void TD3Trainer::Load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "TD3Trainer: Failed to load from " << path << std::endl;
        return;
    }
    
    int version, stateDim, actionDim;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    file.read(reinterpret_cast<char*>(&stateDim), sizeof(int));
    file.read(reinterpret_cast<char*>(&actionDim), sizeof(int));
    file.read(reinterpret_cast<char*>(&mStepCount), sizeof(int));
    
    if (stateDim != mStateDim || actionDim != mActionDim) {
        std::cerr << "TD3Trainer: Dimension mismatch in loaded model" << std::endl;
        return;
    }
    
    int numWeights, numBiases;
    file.read(reinterpret_cast<char*>(&numWeights), sizeof(int));
    file.read(reinterpret_cast<char*>(&numBiases), sizeof(int));
    
    auto& weights = mActor->GetWeights();
    auto& biases = mActor->GetBiases();
    file.read(reinterpret_cast<char*>(weights.data()), numWeights * sizeof(float));
    file.read(reinterpret_cast<char*>(biases.data()), numBiases * sizeof(float));
    
    // Copy to target
    mActorTarget->CopyFrom(*mActor);
    
    file.close();
}
