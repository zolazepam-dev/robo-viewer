#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cstring>

// Simple feedforward neural network with AVX2 optimization potential
class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layerSizes);
    
    void Forward(const float* input, float* output);
    void Forward(const float* input, float* output, int batchSize);
    
    // Get parameters for training
    std::vector<float>& GetWeights() { return mWeights; }
    std::vector<float>& GetBiases() { return mBiases; }
    const std::vector<float>& GetWeights() const { return mWeights; }
    const std::vector<float>& GetBiases() const { return mBiases; }
    
    // Set parameters (for target network updates)
    void SetWeights(const std::vector<float>& weights) { mWeights = weights; }
    void SetBiases(const std::vector<float>& biases) { mBiases = biases; }
    
    void CopyFrom(const NeuralNetwork& other);
    void SoftUpdate(const NeuralNetwork& other, float tau);
    
    int GetNumWeights() const { return static_cast<int>(mWeights.size()); }
    int GetNumBiases() const { return static_cast<int>(mBiases.size()); }
    int GetInputDim() const { return mLayerSizes.front(); }
    int GetOutputDim() const { return mLayerSizes.back(); }
    
    // Random initialization
    void InitializeWeights(std::mt19937& rng, float scale = 0.1f);
    
private:
    std::vector<int> mLayerSizes;
    std::vector<float> mWeights;
    std::vector<float> mBiases;
    std::vector<float> mActivations;  // Workspace for forward pass
    
    // Precomputed layer offsets
    std::vector<int> mWeightOffsets;
    std::vector<int> mBiasOffsets;
    std::vector<int> mActivationOffsets;
    
    static float ReLU(float x) { return x > 0.0f ? x : 0.0f; }
    static float Tanh(float x) { return std::tanh(x); }
};

// Replay buffer for TD3
struct Transition {
    std::vector<float> state;
    std::vector<float> action;
    float reward;
    std::vector<float> nextState;
    bool done;
};

class ReplayBuffer {
public:
    ReplayBuffer(int capacity, int stateDim, int actionDim);
    
    void Add(const float* state, const float* action, float reward,
             const float* nextState, bool done);
    
    void Sample(int batchSize, float* states, float* actions, float* rewards,
                float* nextStates, float* dones, std::mt19937& rng);
    
    int Size() const { return mSize; }
    bool IsReady(int batchSize) const { return mSize >= batchSize; }
    
private:
    std::vector<float> mStates;
    std::vector<float> mActions;
    std::vector<float> mRewards;
    std::vector<float> mNextStates;
    std::vector<float> mDones;
    
    int mCapacity;
    int mStateDim;
    int mActionDim;
    int mSize = 0;
    int mIndex = 0;
};
