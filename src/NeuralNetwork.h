#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <array>
#include <algorithm>
#include <cstdint>

#include "NeuralMath.h"
#include "LatentMemory.h"
#include "AlignedAllocator.h"

using AlignedVector32f = std::vector<float, AlignedAllocator<float, 32>>;

constexpr int VECTOR_REWARD_DIM = 4;

struct VectorReward
{
    float damage_dealt = 0.0f;
    float damage_taken = 0.0f;
    float airtime = 0.0f;
    float energy_used = 0.0f;

    VectorReward() = default;
    VectorReward(const VectorReward& other) = default;
    VectorReward& operator=(const VectorReward& other) = default;

    float Dot(const std::array<float, 4>& preference) const
    {
        return damage_dealt * preference[0] +
               damage_taken * preference[1] +
               airtime * preference[2] +
               energy_used * preference[3];
    }

    float Scalar() const
    {
        return damage_dealt + damage_taken + airtime + energy_used;
    }
};

static inline float Softplus(float x)
{
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

static inline float MoLU(float x)
{
    return 0.5f * x * (1.0f + tanhf(x));
}

static inline float MoLUDerivative(float x)
{
    float th = tanhf(x);
    float sech2 = 1.0f - th * th;
    return 0.5f * (1.0f + th) + 0.5f * x * sech2;
}

struct ODE2VAENetwork
{
    AlignedVector32f W_encoder;
    AlignedVector32f b_encoder;
    AlignedVector32f W_vel;
    AlignedVector32f b_vel;
    
    int obsDim;
    int latentDim;

    ODE2VAENetwork() = default;
    ODE2VAENetwork(const ODE2VAENetwork& other) = default;
    ODE2VAENetwork& operator=(const ODE2VAENetwork& other) = default;

    void Init(int observationDim, int latentDim, std::mt19937& rng);
    void EncodeObservation(const float* obs, float* z_pos_out, float* z_vel_out);
    void ComputeAcceleration(const float* z_pos, const float* z_vel, const float* obs, float* accel_out);
};

void ForwardMoLU_AVX2(float* data, size_t size);
void ForwardMoLU_Scalar(float* data, size_t size);

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layerSizes);
    ~NeuralNetwork() = default;  // Explicitly define default destructor
    NeuralNetwork(const NeuralNetwork& other) = default;  // Explicitly define default copy constructor
    NeuralNetwork& operator=(const NeuralNetwork& other) = default;  // Explicitly define default assignment operator
    
    void Forward(const float* input, float* output);
    void Forward(const float* input, float* output, int batchSize);
    void ForwardMoLU(const float* input, float* output);
    void ForwardMoLU_AVX2_Batch(const float* input, float* output, int batchSize);

    void ForwardODE2VAE(const float* input, float* output, SecondOrderLatentMemory& memory, int envIdx);
    void ForwardODE2VAEVectorized(const float* inputs, float* outputs, int numEnvs);
    
    AlignedVector32f& GetWeights() { return mWeights; }
    AlignedVector32f& GetBiases() { return mBiases; }
    const AlignedVector32f& GetWeights() const { return mWeights; }
    const AlignedVector32f& GetBiases() const { return mBiases; }
    
    void SetWeights(const AlignedVector32f& weights) { mWeights = weights; }
    void SetBiases(const AlignedVector32f& biases) { mBiases = biases; }
    
    void CopyFrom(const NeuralNetwork& other);
    void SoftUpdate(const NeuralNetwork& other, float tau);
    
    int GetNumWeights() const { return static_cast<int>(mWeights.size()); }
    int GetNumBiases() const { return static_cast<int>(mBiases.size()); }
    int GetInputDim() const { return mLayerSizes.front(); }
    int GetOutputDim() const { return mLayerSizes.back(); }
    
    void InitializeWeights(std::mt19937& rng, float scale = 0.1f);
    void InitODE2VAE(int latentDim, std::mt19937& rng);

    ODE2VAENetwork& GetODE2VAE() { return mODE2VAE; }
    const ODE2VAENetwork& GetODE2VAE() const { return mODE2VAE; }

    SecondOrderLatentMemory& GetLatentMemory() { return mLatentMemory; }
    const SecondOrderLatentMemory& GetLatentMemory() const { return mLatentMemory; }
    
    int GetLatentDim() const { return mODE2VAE.latentDim; }
    bool HasODE2VAE() const { return mHasODE2VAE; }
    
private:
    std::vector<int> mLayerSizes;
    AlignedVector32f mWeights;
    AlignedVector32f mBiases;
    AlignedVector32f mActivations;
    
    std::vector<int> mWeightOffsets;
    std::vector<int> mBiasOffsets;
    std::vector<int> mActivationOffsets;

    ODE2VAENetwork mODE2VAE;
    SecondOrderLatentMemory mLatentMemory;
    bool mHasODE2VAE = false;
    
    static float ReLU(float x) { return x > 0.0f ? x : 0.0f; }
    static float Tanh(float x) { return std::tanh(x); }
};

struct KLPERTransition
{
    AlignedVector32f state;
    AlignedVector32f action;
    AlignedVector32f behaviorLogProb;
    VectorReward reward;
    AlignedVector32f nextState;
    bool done;
    float priority;
    float klDivergence;
    int index;

    KLPERTransition() = default;
    KLPERTransition(const KLPERTransition& other) = default;
    KLPERTransition& operator=(const KLPERTransition& other) = default;
};

class KLPERBuffer {
public:
    KLPERBuffer(int capacity, int stateDim, int actionDim);
    KLPERBuffer(const KLPERBuffer& other) = default;
    KLPERBuffer& operator=(const KLPERBuffer& other) = default;
    
    void Add(const float* state, const float* action, float behaviorLogProb,
             const VectorReward& reward, const float* nextState, bool done);
    
    void Sample(int batchSize, float* states, float* actions, float* logProbs,
                VectorReward* rewards, float* nextStates, float* dones,
                std::vector<int>& indices, std::mt19937& rng);
    
    void UpdatePriorities(const std::vector<int>& indices, const float* targetLogProbs,
                          const float* behaviorLogProbs, int batchSize);
    
    void UpdateKLDivergence(int index, float klDiv);
    float ComputeKLDivergence(float behaviorLogProb, float targetLogProb) const;
    
    int Size() const { return mSize; }
    bool IsReady(int batchSize) const { return mSize >= batchSize; }
    
    void SetAlpha(float alpha) { mAlpha = alpha; }
    void SetBeta(float beta) { mBeta = beta; }

private:
    AlignedVector32f mStates;
    AlignedVector32f mActions;
    AlignedVector32f mBehaviorLogProbs;
    std::vector<VectorReward> mRewards;
    AlignedVector32f mNextStates;
    AlignedVector32f mDones;
    AlignedVector32f mPriorities;
    AlignedVector32f mKLDivergences;
    
    int mCapacity;
    int mStateDim;
    int mActionDim;
    int mSize = 0;
    int mIndex = 0;
    float mAlpha = 0.6f;
    float mBeta = 0.4f;
    float mMaxPriority = 1.0f;
    
    AlignedVector32f mSumTree;
    std::vector<int> mMinTree;
    
    void UpdateTree(int idx, float priority);
    float GetPriorityWeight(int idx) const;
};

class ReplayBuffer {
public:
    ReplayBuffer(int capacity, int stateDim, int actionDim);
    ReplayBuffer(const ReplayBuffer& other) = default;
    ReplayBuffer& operator=(const ReplayBuffer& other) = default;
    
    void Add(const float* state, const float* action, const VectorReward& reward,
             const float* nextState, bool done);
    void Add(const float* state, const float* action, float reward,
             const float* nextState, bool done);
    
    void Sample(int batchSize, float* states, float* actions, float* rewards,
                float* nextStates, float* dones, std::mt19937& rng);
    
    void SampleVectorRewards(int batchSize, float* states, float* actions,
                             VectorReward* rewards, float* nextStates, float* dones,
                             std::mt19937& rng);
    
    int Size() const { return mSize; }
    bool IsReady(int batchSize) const { return mSize >= batchSize; }
    
private:
    AlignedVector32f mStates;
    AlignedVector32f mActions;
    AlignedVector32f mRewards;
    AlignedVector32f mNextStates;
    AlignedVector32f mDones;

    std::vector<VectorReward> mVectorRewards;
    
    int mCapacity;
    int mStateDim;
    int mActionDim;
    int mSize = 0;
    int mIndex = 0;
};