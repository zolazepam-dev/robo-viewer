#pragma once
// ============================================================================
// PRIMALPHA: Intrinsic Motivation Module (Curiosity-Driven Exploration)
// ============================================================================
// Implements forward and inverse dynamics models for curiosity rewards
// Based on "Curiosity-driven Exploration by Self-Supervised Prediction"
// ============================================================================

#include "SpanNetwork.h"
#include "AlignedAllocator.h"
#include <vector>
#include <random>

class alignas(32) IntrinsicMotivation {
public:
    IntrinsicMotivation() = default;
    
    void Init(size_t stateDim, size_t actionDim, size_t latentDim, std::mt19937& rng);
    
    // Compute curiosity reward: prediction error from forward model
    float ComputeIntrinsicReward(const float* state, const float* action,
                                  const float* nextState);
    
    // Update forward dynamics model using TD error
    void UpdateForwardModel(const float* state, const float* action,
                           const float* nextState, float learningRate);
    
    // Compute inverse loss: how well we predict actions from state transitions
    float ComputeInverseLoss(const float* state, const float* nextState,
                            const float* actualAction);
    
    // Update inverse dynamics model
    void UpdateInverseModel(const float* state, const float* nextState,
                           const float* actualAction, float learningRate);
    
    // Getters
    float GetCuriosityScale() const { return mCuriosityScale; }
    void SetCuriosityScale(float scale) { mCuriosityScale = scale; }
    
    float GetAverageIntrinsicReward() const { return mAvgIntrinsicReward; }
    
private:
    // Forward model: (state, action) -> predicted_next_state
    SpanNetwork mForwardModel;
    
    // Inverse model: (state, next_state) -> predicted_action
    SpanNetwork mInverseModel;
    
    // Hyperparameters
    float mCuriosityScale = 0.1f;      // Weight for intrinsic reward
    float mForwardLR = 1e-4f;          // Forward model learning rate
    float mInverseLR = 1e-4f;          // Inverse model learning rate
    
    // Running average for normalization
    float mAvgIntrinsicReward = 0.0f;
    float mRewardEMA = 0.0f;
    int mUpdateCount = 0;
    
    // Buffers
    AlignedVector32<float> mPredictedNextState;
    AlignedVector32<float> mPredictedAction;
    AlignedVector32<float> mForwardInput;  // state + action concatenation
    
    // Weight perturbation buffers (for gradient-free optimization)
    std::vector<float> mTempWeights;
    std::vector<float> mOriginalWeights;
};
