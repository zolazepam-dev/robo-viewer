#include "IntrinsicMotivation.h"
#include <cmath>
#include <algorithm>

void IntrinsicMotivation::Init(size_t stateDim, size_t actionDim, size_t latentDim, std::mt19937& rng)
{
    // Forward model: (state + action) -> next_state
    // Input: state (512) + action (64) = 576
    // Output: predicted_next_state (512)
    std::vector<SpanLayerConfig> forwardConfigs = {
        {stateDim + actionDim, 1024, 8, 3},
        {1024, 1024, 8, 3},
        {1024, 1024, 8, 3},
        {1024, 1024, 8, 3},
        {1024, stateDim, 8, 3}
    };
    mForwardModel.Init(forwardConfigs, rng);
    
    // Inverse model: (state + next_state) -> action
    // Input: state (512) + next_state (512) = 1024
    // Output: predicted_action (64)
    std::vector<SpanLayerConfig> inverseConfigs = {
        {stateDim * 2, 1024, 8, 3},
        {1024, 1024, 8, 3},
        {1024, 1024, 8, 3},
        {1024, actionDim, 8, 3}
    };
    mInverseModel.Init(inverseConfigs, rng);
    
    // Allocate buffers
    mPredictedNextState.resize(stateDim);
    mPredictedAction.resize(actionDim);
    mForwardInput.resize(stateDim + actionDim);
    
    mTempWeights.resize(mForwardModel.GetNumWeights());
    mOriginalWeights.resize(mForwardModel.GetNumWeights());
}

float IntrinsicMotivation::ComputeIntrinsicReward(const float* state, const float* action,
                                                   const float* nextState)
{
    // Concatenate state + action for forward model input
    std::copy(state, state + 512, mForwardInput.data());
    std::copy(action, action + 64, mForwardInput.data() + 512);
    
    // Forward pass through dynamics model
    mForwardModel.Forward(mForwardInput.data(), mPredictedNextState.data());
    
    // Compute prediction error (L2 norm)
    float squaredError = 0.0f;
    for (size_t i = 0; i < 512; ++i) {
        float diff = mPredictedNextState[i] - nextState[i];
        squaredError += diff * diff;
    }
    float predictionError = std::sqrt(squaredError / 512.0f);
    
    // Update running average for normalization
    mUpdateCount++;
    mRewardEMA = 0.99f * mRewardEMA + 0.01f * predictionError;
    mAvgIntrinsicReward = mRewardEMA / (1.0f - std::pow(0.99f, mUpdateCount));
    
    // Normalize and scale
    float normalizedError = predictionError / (mAvgIntrinsicReward + 1e-6f);
    return normalizedError * mCuriosityScale;
}

void IntrinsicMotivation::UpdateForwardModel(const float* state, const float* action,
                                             const float* nextState, float learningRate)
{
    // Concatenate input
    std::copy(state, state + 512, mForwardInput.data());
    std::copy(action, action + 64, mForwardInput.data() + 512);
    
    // Get current weights
    auto weights = mForwardModel.GetAllWeights();
    mOriginalWeights = weights;
    
    // Compute baseline loss
    mForwardModel.Forward(mForwardInput.data(), mPredictedNextState.data());
    float baselineLoss = 0.0f;
    for (size_t i = 0; i < 512; ++i) {
        float diff = mPredictedNextState[i] - nextState[i];
        baselineLoss += diff * diff;
    }
    
    // Weight perturbation (sample every 64th weight for speed)
    const float epsilon = 0.01f;
    for (size_t w = 0; w < weights.size() && w < mTempWeights.size(); w += 64) {
        // Perturb positively
        weights[w] = mOriginalWeights[w] + epsilon;
        mForwardModel.SetAllWeights(weights);
        
        mForwardModel.Forward(mForwardInput.data(), mPredictedNextState.data());
        float lossPlus = 0.0f;
        for (size_t i = 0; i < 512; ++i) {
            float diff = mPredictedNextState[i] - nextState[i];
            lossPlus += diff * diff;
        }
        
        // Keep improvement or revert
        if (lossPlus > baselineLoss) {
            weights[w] = mOriginalWeights[w] - epsilon;  // Reverse direction
        } else {
            weights[w] = mOriginalWeights[w] + learningRate;  // Keep positive perturbation
        }
    }
    mForwardModel.SetAllWeights(weights);
}

float IntrinsicMotivation::ComputeInverseLoss(const float* state, const float* nextState,
                                              const float* actualAction)
{
    // Concatenate state + next_state for inverse model input
    AlignedVector32<float> inverseInput(1024);
    std::copy(state, state + 512, inverseInput.data());
    std::copy(nextState, nextState + 512, inverseInput.data() + 512);
    
    // Forward pass through inverse model
    mInverseModel.Forward(inverseInput.data(), mPredictedAction.data());
    
    // Compute action prediction error (L2 norm)
    float squaredError = 0.0f;
    for (size_t i = 0; i < 64; ++i) {
        float diff = mPredictedAction[i] - actualAction[i];
        squaredError += diff * diff;
    }
    return std::sqrt(squaredError / 64.0f);
}

void IntrinsicMotivation::UpdateInverseModel(const float* state, const float* nextState,
                                             const float* actualAction, float learningRate)
{
    // Concatenate input
    AlignedVector32<float> inverseInput(1024);
    std::copy(state, state + 512, inverseInput.data());
    std::copy(nextState, nextState + 512, inverseInput.data() + 512);
    
    // Get current weights
    auto weights = mInverseModel.GetAllWeights();
    mOriginalWeights = weights;
    
    // Compute baseline loss
    mInverseModel.Forward(inverseInput.data(), mPredictedAction.data());
    float baselineLoss = 0.0f;
    for (size_t i = 0; i < 64; ++i) {
        float diff = mPredictedAction[i] - actualAction[i];
        baselineLoss += diff * diff;
    }
    
    // Weight perturbation (sample every 64th weight)
    const float epsilon = 0.01f;
    for (size_t w = 0; w < weights.size() && w < mTempWeights.size(); w += 64) {
        weights[w] = mOriginalWeights[w] + epsilon;
        mInverseModel.SetAllWeights(weights);
        
        mInverseModel.Forward(inverseInput.data(), mPredictedAction.data());
        float lossPlus = 0.0f;
        for (size_t i = 0; i < 64; ++i) {
            float diff = mPredictedAction[i] - actualAction[i];
            lossPlus += diff * diff;
        }
        
        if (lossPlus > baselineLoss) {
            weights[w] = mOriginalWeights[w] - epsilon;
        } else {
            weights[w] = mOriginalWeights[w] + learningRate;
        }
    }
    mInverseModel.SetAllWeights(weights);
}
