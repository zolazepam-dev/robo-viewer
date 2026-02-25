#include "NeuralNetwork.h"

#include <algorithm>
#include <immintrin.h>  // AVX2

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes)
    : mLayerSizes(layerSizes)
{
    if (layerSizes.size() < 2) return;
    
    // Calculate total weights and biases
    int totalWeights = 0;
    int totalBiases = 0;
    int totalActivations = 0;
    
    for (size_t i = 1; i < layerSizes.size(); ++i) {
        int inSize = layerSizes[i - 1];
        int outSize = layerSizes[i];
        totalWeights += inSize * outSize;
        totalBiases += outSize;
        totalActivations += outSize;
    }
    
    mWeights.resize(totalWeights);
    mBiases.resize(totalBiases);
    mActivations.resize(totalActivations);
    
    // Compute offsets
    int weightOffset = 0;
    int biasOffset = 0;
    int activationOffset = 0;
    
    for (size_t i = 0; i < layerSizes.size(); ++i) {
        mActivationOffsets.push_back(activationOffset);
        activationOffset += layerSizes[i];
    }
    
    for (size_t i = 1; i < layerSizes.size(); ++i) {
        mWeightOffsets.push_back(weightOffset);
        mBiasOffsets.push_back(biasOffset);
        
        weightOffset += layerSizes[i - 1] * layerSizes[i];
        biasOffset += layerSizes[i];
    }
}

void NeuralNetwork::InitializeWeights(std::mt19937& rng, float scale) {
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (auto& w : mWeights) {
        w = dist(rng);
    }
    
    for (auto& b : mBiases) {
        b = 0.0f;  // Initialize biases to zero
    }
}

void NeuralNetwork::Forward(const float* input, float* output) {
    // Copy input to first activation layer
    int inputSize = mLayerSizes[0];
    std::copy(input, input + inputSize, mActivations.begin());
    
    int actOffset = 0;
    int weightIdx = 0;
    int biasIdx = 0;
    
    for (size_t layer = 1; layer < mLayerSizes.size(); ++layer) {
        int inSize = mLayerSizes[layer - 1];
        int outSize = mLayerSizes[layer];
        
        const float* in = mActivations.data() + actOffset;
        float* out = mActivations.data() + actOffset + inSize;
        
        // Matrix-vector multiply: out = W * in + b
        for (int j = 0; j < outSize; ++j) {
            float sum = mBiases[biasIdx + j];
            
            // Simple loop (can be optimized with AVX2)
            for (int i = 0; i < inSize; ++i) {
                sum += mWeights[weightIdx + j * inSize + i] * in[i];
            }
            
            // Apply activation (ReLU for hidden, Tanh for output)
            if (layer < mLayerSizes.size() - 1) {
                out[j] = ReLU(sum);
            } else {
                out[j] = Tanh(sum);  // Output layer: tanh for bounded actions
            }
        }
        
        actOffset += inSize;
        weightIdx += inSize * outSize;
        biasIdx += outSize;
    }
    
    // Copy output
    int outputSize = mLayerSizes.back();
    std::copy(mActivations.end() - outputSize, mActivations.end(), output);
}

void NeuralNetwork::CopyFrom(const NeuralNetwork& other) {
    mWeights = other.mWeights;
    mBiases = other.mBiases;
}

void NeuralNetwork::SoftUpdate(const NeuralNetwork& other, float tau) {
    for (size_t i = 0; i < mWeights.size(); ++i) {
        mWeights[i] = (1.0f - tau) * mWeights[i] + tau * other.mWeights[i];
    }
    for (size_t i = 0; i < mBiases.size(); ++i) {
        mBiases[i] = (1.0f - tau) * mBiases[i] + tau * other.mBiases[i];
    }
}

// ReplayBuffer implementation
ReplayBuffer::ReplayBuffer(int capacity, int stateDim, int actionDim)
    : mCapacity(capacity)
    , mStateDim(stateDim)
    , mActionDim(actionDim)
{
    mStates.resize(capacity * stateDim);
    mActions.resize(capacity * actionDim);
    mRewards.resize(capacity);
    mNextStates.resize(capacity * stateDim);
    mDones.resize(capacity);
}

void ReplayBuffer::Add(const float* state, const float* action, float reward,
                        const float* nextState, bool done) {
    int idx = mIndex * mStateDim;
    std::copy(state, state + mStateDim, mStates.begin() + idx);
    
    idx = mIndex * mActionDim;
    std::copy(action, action + mActionDim, mActions.begin() + idx);
    
    mRewards[mIndex] = reward;
    
    idx = mIndex * mStateDim;
    std::copy(nextState, nextState + mStateDim, mNextStates.begin() + idx);
    
    mDones[mIndex] = done ? 1.0f : 0.0f;
    
    mIndex = (mIndex + 1) % mCapacity;
    mSize = std::min(mSize + 1, mCapacity);
}

void ReplayBuffer::Sample(int batchSize, float* states, float* actions, float* rewards,
                           float* nextStates, float* dones, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, mSize - 1);
    
    for (int i = 0; i < batchSize; ++i) {
        int idx = dist(rng);
        
        std::copy(mStates.begin() + idx * mStateDim,
                  mStates.begin() + (idx + 1) * mStateDim,
                  states + i * mStateDim);
        
        std::copy(mActions.begin() + idx * mActionDim,
                  mActions.begin() + (idx + 1) * mActionDim,
                  actions + i * mActionDim);
        
        rewards[i] = mRewards[idx];
        
        std::copy(mNextStates.begin() + idx * mStateDim,
                  mNextStates.begin() + (idx + 1) * mStateDim,
                  nextStates + i * mStateDim);
        
        dones[i] = mDones[idx];
    }
}
