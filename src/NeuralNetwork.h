#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <array>
#include <algorithm>
#include <cstdint>

#include "NeuralMath.h"

struct VectorReward
{
    float damage_dealt = 0.0f;
    float damage_taken = 0.0f;
    float airtime = 0.0f;
    float energy_used = 0.0f;

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

struct alignas(32) SecondOrderLatentMemory
{
    alignas(32) float z_pos[LATENT_DIM * NUM_PARALLEL_ENVS];
    alignas(32) float z_vel[LATENT_DIM * NUM_PARALLEL_ENVS];
    
    int latentDim = LATENT_DIM;
    int numEnvs = NUM_PARALLEL_ENVS;
    float dt = 1.0f / 60.0f;

    void Init()
    {
        std::memset(z_pos, 0, sizeof(z_pos));
        std::memset(z_vel, 0, sizeof(z_vel));
    }

    void ResetEnv(int envIdx)
    {
        int offset = envIdx * latentDim;
        std::memset(z_pos + offset, 0, latentDim * sizeof(float));
        std::memset(z_vel + offset, 0, latentDim * sizeof(float));
    }

    void ResetAll()
    {
        std::memset(z_pos, 0, sizeof(z_pos));
        std::memset(z_vel, 0, sizeof(z_vel));
    }

    float* GetPosition(int envIdx) { return z_pos + envIdx * latentDim; }
    float* GetVelocity(int envIdx) { return z_vel + envIdx * latentDim; }
    const float* GetPosition(int envIdx) const { return z_pos + envIdx * latentDim; }
    const float* GetVelocity(int envIdx) const { return z_vel + envIdx * latentDim; }

    void StepDynamicsScalar(const float* acceleration, int envIdx)
    {
        int offset = envIdx * latentDim;
        for (int i = 0; i < latentDim; ++i)
        {
            z_vel[offset + i] += acceleration[i] * dt;
            z_pos[offset + i] += z_vel[offset + i] * dt;
        }
    }

    void StepDynamicsVectorized(const float* accelerations);
};

struct ODE2VAENetwork
{
    std::vector<float> W_encoder;
    std::vector<float> b_encoder;
    std::vector<float> W_vel;
    std::vector<float> b_vel;
    
    int obsDim;
    int latentDim;

    void Init(int observationDim, int latentDim, std::mt19937& rng);
    void EncodeObservation(const float* obs, float* z_pos_out, float* z_vel_out);
    void ComputeAcceleration(const float* z_pos, const float* z_vel, const float* obs, float* accel_out);
};

void ForwardMoLU_AVX2(float* data, size_t size);
void ForwardMoLU_Scalar(float* data, size_t size);

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layerSizes);
    
    void Forward(const float* input, float* output);
    void Forward(const float* input, float* output, int batchSize);
    void ForwardMoLU(const float* input, float* output);
    void ForwardMoLU_AVX2_Batch(const float* input, float* output, int batchSize);

    void ForwardODE2VAE(const float* input, float* output, SecondOrderLatentMemory& memory, int envIdx);
    void ForwardODE2VAEVectorized(const float* inputs, float* outputs, int numEnvs);
    
    std::vector<float>& GetWeights() { return mWeights; }
    std::vector<float>& GetBiases() { return mBiases; }
    const std::vector<float>& GetWeights() const { return mWeights; }
    const std::vector<float>& GetBiases() const { return mBiases; }
    
    void SetWeights(const std::vector<float>& weights) { mWeights = weights; }
    void SetBiases(const std::vector<float>& biases) { mBiases = biases; }
    
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
    std::vector<float> mWeights;
    std::vector<float> mBiases;
    std::vector<float> mActivations;
    
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
    std::vector<float> state;
    std::vector<float> action;
    std::vector<float> behaviorLogProb;
    VectorReward reward;
    std::vector<float> nextState;
    bool done;
    float priority;
    float klDivergence;
    int index;
};

class KLPERBuffer {
public:
    KLPERBuffer(int capacity, int stateDim, int actionDim);
    
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
    std::vector<float> mStates;
    std::vector<float> mActions;
    std::vector<float> mBehaviorLogProbs;
    std::vector<VectorReward> mRewards;
    std::vector<float> mNextStates;
    std::vector<float> mDones;
    std::vector<float> mPriorities;
    std::vector<float> mKLDivergences;
    
    int mCapacity;
    int mStateDim;
    int mActionDim;
    int mSize = 0;
    int mIndex = 0;
    float mAlpha = 0.6f;
    float mBeta = 0.4f;
    float mMaxPriority = 1.0f;
    
    std::vector<float> mSumTree;
    std::vector<int> mMinTree;
    
    void UpdateTree(int idx, float priority);
    float GetPriorityWeight(int idx) const;
};

class ReplayBuffer {
public:
    ReplayBuffer(int capacity, int stateDim, int actionDim);
    
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
    std::vector<float> mStates;
    std::vector<float> mActions;
    std::vector<float> mRewards;
    std::vector<float> mNextStates;
    std::vector<float> mDones;

    std::vector<VectorReward> mVectorRewards;
    
    int mCapacity;
    int mStateDim;
    int mActionDim;
    int mSize = 0;
    int mIndex = 0;
};