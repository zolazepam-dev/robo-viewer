#include "NeuralNetwork.h"

#include <algorithm>
#include <immintrin.h>
#include <limits>

void ForwardMoLU_Scalar(float* data, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        float x = data[i];
        data[i] = 0.5f * x * (1.0f + tanhf(x));
    }
}

void ForwardMoLU_AVX2(float* data, size_t size)
{
    size_t i = 0;
    const size_t simdEnd = size - (size % 8);
    
    for (; i < simdEnd; i += 8)
    {
        __m256 x = _mm256_loadu_ps(data + i);

        // =====================================================================
        // AVX2 FAST-MATH PADE APPROXIMATION FOR TANH - INTRINSICS PLACEHOLDER
        // =====================================================================
        //
        // The MoLU activation requires: 0.5 * x * (1 + tanh(x))
        //
        // FAST TANH APPROXIMATION via Padé (3,2) rational function:
        // tanh(x) ≈ x * (1 + a*x²) / (1 + b*x²)  where a=0.275, b=0.664
        //
        // This avoids _mm256_div_ps (expensive ~15-20 cycles) by using
        // FMA-based Newton-Raphson reciprocal approximation instead:
        //
        // Step 1: Compute x²
        // __m256 x2 = _mm256_mul_ps(x, x);
        //
        // Step 2: Padé numerator: x * (1 + a*x²)
        // const __m256 a = _mm256_set1_ps(0.275f);
        // __m256 num = _mm256_fmadd_ps(a, x2, _mm256_set1_ps(1.0f));
        // num = _mm256_mul_ps(x, num);
        //
        // Step 3: Padé denominator: (1 + b*x²)
        // const __m256 b = _mm256_set1_ps(0.664f);
        // __m256 den = _mm256_fmadd_ps(b, x2, _mm256_set1_ps(1.0f));
        //
        // Step 4: Fast reciprocal via Newton-Raphson (2 iterations for precision)
        // __m256 rcp = _mm256_rcp_ps(den);                          // Initial approx
        // rcp = _mm256_fmadd_ps(rcp, _mm256_set1_ps(-1.0f),          // Iteration 1
        //              _mm256_mul_ps(rcp, rcp));
        // rcp = _mm256_fmadd_ps(den, _mm256_mul_ps(rcp, rcp), rcp);  // Iteration 2
        //
        // Step 5: tanh approximation
        // __m256 th = _mm256_mul_ps(num, rcp);
        //
        // Step 6: MoLU: 0.5 * x * (1 + tanh(x))
        // __m256 one_plus_th = _mm256_add_ps(_mm256_set1_ps(1.0f), th);
        // __m256 half = _mm256_set1_ps(0.5f);
        // __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_th));
        //
        // NaN SAFETY: Clamp large values before approximation
        // x = _mm256_min_ps(x, _mm256_set1_ps(10.0f));
        // x = _mm256_max_ps(x, _mm256_set1_ps(-10.0f));
        // =====================================================================
        
        __m256 th;
        
        {
            alignas(32) float x_arr[8];
            _mm256_store_ps(x_arr, x);
            alignas(32) float th_arr[8];
            for (int j = 0; j < 8; ++j)
            {
                th_arr[j] = tanhf(x_arr[j]);
            }
            th = _mm256_load_ps(th_arr);
        }
        
        __m256 one_plus_th = _mm256_add_ps(_mm256_set1_ps(1.0f), th);
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_th));
        
        _mm256_storeu_ps(data + i, result);
    }

    for (; i < size; ++i)
    {
        data[i] = 0.5f * data[i] * (1.0f + tanhf(data[i]));
    }
}

void SecondOrderLatentMemory::StepDynamicsVectorized(const float* accelerations)
{
    const int totalSize = latentDim * numEnvs;
    const int simdEnd = totalSize - (totalSize % 8);
    const __m256 dt_vec = _mm256_set1_ps(dt);
    
    int i = 0;
    for (; i < simdEnd; i += 8)
    {
        __m256 vel = _mm256_loadu_ps(z_vel + i);
        __m256 pos = _mm256_loadu_ps(z_pos + i);
        __m256 accel = _mm256_loadu_ps(accelerations + i);
        
        __m256 new_vel = _mm256_fmadd_ps(accel, dt_vec, vel);
        __m256 new_pos = _mm256_fmadd_ps(new_vel, dt_vec, pos);
        
        _mm256_storeu_ps(z_vel + i, new_vel);
        _mm256_storeu_ps(z_pos + i, new_pos);
    }
    
    for (; i < totalSize; ++i)
    {
        z_vel[i] += accelerations[i] * dt;
        z_pos[i] += z_vel[i] * dt;
    }
}

void ODE2VAENetwork::Init(int observationDim, int latentDim, std::mt19937& rng)
{
    obsDim = observationDim;
    this->latentDim = latentDim;
    
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    W_encoder.resize(latentDim * 2 * observationDim);
    b_encoder.resize(latentDim * 2);
    W_vel.resize(latentDim * (latentDim * 2 + observationDim));
    b_vel.resize(latentDim);
    
    for (auto& w : W_encoder) w = dist(rng);
    for (auto& w : W_vel) w = dist(rng);
    
    for (int i = 0; i < latentDim * 2; ++i)
    {
        b_encoder[i] = 0.0f;
    }
    for (int i = 0; i < latentDim; ++i)
    {
        b_vel[i] = 0.0f;
    }
}

void ODE2VAENetwork::EncodeObservation(const float* obs, float* z_pos_out, float* z_vel_out)
{
    for (int i = 0; i < latentDim; ++i)
    {
        float pos_val = b_encoder[i];
        float vel_val = b_encoder[latentDim + i];
        
        for (int j = 0; j < obsDim; ++j)
        {
            float o = obs[j];
            pos_val += W_encoder[i * obsDim + j] * o;
            vel_val += W_encoder[(latentDim + i) * obsDim + j] * o;
        }
        
        z_pos_out[i] = std::tanhf(pos_val);
        z_vel_out[i] = std::tanhf(vel_val);
    }
}

void ODE2VAENetwork::ComputeAcceleration(const float* z_pos, const float* z_vel, 
                                          const float* obs, float* accel_out)
{
    const int inputDim = latentDim * 2 + obsDim;
    std::vector<float> combined(inputDim);
    
    for (int i = 0; i < latentDim; ++i)
    {
        combined[i] = z_pos[i];
        combined[latentDim + i] = z_vel[i];
    }
    for (int i = 0; i < obsDim; ++i)
    {
        combined[latentDim * 2 + i] = obs[i];
    }
    
    for (int i = 0; i < latentDim; ++i)
    {
        float val = b_vel[i];
        for (int j = 0; j < inputDim; ++j)
        {
            val += W_vel[i * inputDim + j] * combined[j];
        }
        accel_out[i] = std::tanhf(val);
    }
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes)
    : mLayerSizes(layerSizes)
{
    if (layerSizes.size() < 2) return;
    
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
        b = 0.0f;
    }
}

void NeuralNetwork::InitODE2VAE(int latentDim, std::mt19937& rng)
{
    mHasODE2VAE = true;
    mODE2VAE.Init(mLayerSizes.front(), latentDim, rng);
    mLatentMemory.Init();
}

void NeuralNetwork::Forward(const float* input, float* output) {
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
        
        for (int j = 0; j < outSize; ++j) {
            float sum = mBiases[biasIdx + j];
            
            for (int i = 0; i < inSize; ++i) {
                sum += mWeights[weightIdx + j * inSize + i] * in[i];
            }
            
            if (layer < mLayerSizes.size() - 1) {
                out[j] = ReLU(sum);
            } else {
                out[j] = Tanh(sum);
            }
        }
        
        actOffset += inSize;
        weightIdx += inSize * outSize;
        biasIdx += outSize;
    }
    
    int outputSize = mLayerSizes.back();
    std::copy(mActivations.end() - outputSize, mActivations.end(), output);
}

void NeuralNetwork::Forward(const float* input, float* output, int batchSize) {
    for (int b = 0; b < batchSize; ++b) {
        Forward(input + b * mLayerSizes[0], output + b * mLayerSizes.back());
    }
}

void NeuralNetwork::ForwardMoLU(const float* input, float* output)
{
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
        
        for (int j = 0; j < outSize; ++j) {
            float sum = mBiases[biasIdx + j];

            for (int i = 0; i < inSize; ++i) {
                sum += mWeights[weightIdx + j * inSize + i] * in[i];
            }
            
            if (layer < mLayerSizes.size() - 1) {
                out[j] = MoLU(sum);
            } else {
                out[j] = Tanh(sum);
            }
        }
        
        actOffset += inSize;
        weightIdx += inSize * outSize;
        biasIdx += outSize;
    }
    
    int outputSize = mLayerSizes.back();
    std::copy(mActivations.end() - outputSize, mActivations.end(), output);
}

void NeuralNetwork::ForwardMoLU_AVX2_Batch(const float* input, float* output, int batchSize)
{
    for (int b = 0; b < batchSize; ++b)
    {
        ForwardMoLU(input + b * mLayerSizes[0], output + b * mLayerSizes.back());
    }
}

void NeuralNetwork::ForwardODE2VAE(const float* input, float* output, 
                                    SecondOrderLatentMemory& memory, int envIdx)
{
    if (!mHasODE2VAE) {
        Forward(input, output);
        return;
    }

    float* z_pos = memory.GetPosition(envIdx);
    float* z_vel = memory.GetVelocity(envIdx);
    
    float accel[LATENT_DIM];
    mODE2VAE.ComputeAcceleration(z_pos, z_vel, input, accel);
    
    memory.StepDynamicsScalar(accel, envIdx);
    
    std::copy(z_pos, z_pos + mODE2VAE.latentDim, output);
}

void NeuralNetwork::ForwardODE2VAEVectorized(const float* inputs, float* outputs, int numEnvs)
{
    if (!mHasODE2VAE) {
        Forward(inputs, outputs, numEnvs);
        return;
    }

    alignas(32) float accelerations[LATENT_DIM * NUM_PARALLEL_ENVS];
    
    for (int env = 0; env < numEnvs; ++env)
    {
        const float* input = inputs + env * mLayerSizes[0];
        float* z_pos = mLatentMemory.GetPosition(env);
        float* z_vel = mLatentMemory.GetVelocity(env);
        float* accel = accelerations + env * mODE2VAE.latentDim;
        
        mODE2VAE.ComputeAcceleration(z_pos, z_vel, input, accel);
    }
    
    mLatentMemory.StepDynamicsVectorized(accelerations);
    
    for (int env = 0; env < numEnvs; ++env)
    {
        float* z_pos = mLatentMemory.GetPosition(env);
        float* output = outputs + env * mODE2VAE.latentDim;
        std::copy(z_pos, z_pos + mODE2VAE.latentDim, output);
    }
}

void NeuralNetwork::CopyFrom(const NeuralNetwork& other) {
    mWeights = other.mWeights;
    mBiases = other.mBiases;
    mODE2VAE = other.mODE2VAE;
    mHasODE2VAE = other.mHasODE2VAE;
}

void NeuralNetwork::SoftUpdate(const NeuralNetwork& other, float tau) {
    for (size_t i = 0; i < mWeights.size(); ++i) {
        mWeights[i] = (1.0f - tau) * mWeights[i] + tau * other.mWeights[i];
    }
    for (size_t i = 0; i < mBiases.size(); ++i) {
        mBiases[i] = (1.0f - tau) * mBiases[i] + tau * other.mBiases[i];
    }
}

KLPERBuffer::KLPERBuffer(int capacity, int stateDim, int actionDim)
    : mCapacity(capacity)
    , mStateDim(stateDim)
    , mActionDim(actionDim)
{
    mStates.resize(capacity * stateDim);
    mActions.resize(capacity * actionDim);
    mBehaviorLogProbs.resize(capacity);
    mRewards.resize(capacity);
    mNextStates.resize(capacity * stateDim);
    mDones.resize(capacity);
    mPriorities.resize(capacity);
    mKLDivergences.resize(capacity);

    int treeSize = 1;
    while (treeSize < capacity) treeSize *= 2;
    mSumTree.resize(2 * treeSize, 0.0f);
    mMinTree.resize(2 * treeSize, std::numeric_limits<int>::max());
}

void KLPERBuffer::Add(const float* state, const float* action, float behaviorLogProb,
                        const VectorReward& reward, const float* nextState, bool done)
{
    int idx = mIndex * mStateDim;
    std::copy(state, state + mStateDim, mStates.begin() + idx);
    
    idx = mIndex * mActionDim;
    std::copy(action, action + mActionDim, mActions.begin() + idx);
    
    mBehaviorLogProbs[mIndex] = behaviorLogProb;
    mRewards[mIndex] = reward;
    
    idx = mIndex * mStateDim;
    std::copy(nextState, nextState + mStateDim, mNextStates.begin() + idx);
    
    mDones[mIndex] = done ? 1.0f : 0.0f;
    mPriorities[mIndex] = mMaxPriority;
    mKLDivergences[mIndex] = 0.0f;
    
    UpdateTree(mIndex, static_cast<int>(mMaxPriority));
    
    mIndex = (mIndex + 1) % mCapacity;
    mSize = std::min(mSize + 1, mCapacity);
}

void KLPERBuffer::Sample(int batchSize, float* states, float* actions, float* logProbs,
                          VectorReward* rewards, float* nextStates, float* dones,
                          std::vector<int>& indices, std::mt19937& rng)
{
    indices.resize(batchSize);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float totalPriority = mSumTree[1];
    float segment = totalPriority / batchSize;

    for (int i = 0; i < batchSize; ++i) {
        float val = segment * (dist(rng) + i);
        int idx = 0;
        
        for (int j = mCapacity; j > 0; j /= 2) {
            if (mSumTree[2 * idx] >= val) {
                idx = 2 * idx;
            } else {
                val -= mSumTree[2 * idx];
                idx = 2 * idx + 1;
            }
        }
        
        indices[i] = std::min(idx, mSize - 1);
        
        std::copy(mStates.begin() + indices[i] * mStateDim,
                  mStates.begin() + (indices[i] + 1) * mStateDim,
                  states + i * mStateDim);
        
        std::copy(mActions.begin() + indices[i] * mActionDim,
                  mActions.begin() + (indices[i] + 1) * mActionDim,
                  actions + i * mActionDim);
        
        logProbs[i] = mBehaviorLogProbs[indices[i]];
        rewards[i] = mRewards[indices[i]];
        
        std::copy(mNextStates.begin() + indices[i] * mStateDim,
                  mNextStates.begin() + (indices[i] + 1) * mStateDim,
                  nextStates + i * mStateDim);
        
        dones[i] = mDones[indices[i]];
    }
}

void KLPERBuffer::UpdatePriorities(const std::vector<int>& indices, const float* targetLogProbs,
                                    const float* behaviorLogProbs, int batchSize)
{
    for (int i = 0; i < batchSize; ++i) {
        float kl = ComputeKLDivergence(behaviorLogProbs[i], targetLogProbs[i]);
        mKLDivergences[indices[i]] = kl;
        
        float priority = std::pow(kl + 1e-6f, mAlpha);
        mPriorities[indices[i]] = priority;
        mMaxPriority = std::max(mMaxPriority, priority);
        
        UpdateTree(indices[i], static_cast<int>(priority));
    }
}

void KLPERBuffer::UpdateKLDivergence(int index, float klDiv)
{
    mKLDivergences[index] = klDiv;
    float priority = std::pow(klDiv + 1e-6f, mAlpha);
    mPriorities[index] = priority;
    mMaxPriority = std::max(mMaxPriority, priority);
    UpdateTree(index, static_cast<int>(priority));
}

float KLPERBuffer::ComputeKLDivergence(float behaviorLogProb, float targetLogProb) const
{
    float p = std::exp(targetLogProb);
    float q = std::exp(behaviorLogProb);
    return p * (targetLogProb - behaviorLogProb);
}

void KLPERBuffer::UpdateTree(int idx, float priority)
{
    idx += mCapacity;
    mSumTree[idx] = priority;
    mMinTree[idx] = static_cast<int>(priority);
    
    while (idx > 1) {
        idx /= 2;
        mSumTree[idx] = mSumTree[2 * idx] + mSumTree[2 * idx + 1];
        mMinTree[idx] = std::min(mMinTree[2 * idx], mMinTree[2 * idx + 1]);
    }
}

float KLPERBuffer::GetPriorityWeight(int idx) const
{
    float minProb = static_cast<float>(mMinTree[1]) / mSumTree[1];
    float maxWeight = std::pow(mSize * minProb, -mBeta);
    float prob = mSumTree[idx + mCapacity] / mSumTree[1];
    float weight = std::pow(mSize * prob, -mBeta);
    return weight / maxWeight;
}

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
    mVectorRewards.resize(capacity);
}

void ReplayBuffer::Add(const float* state, const float* action, const VectorReward& reward,
                        const float* nextState, bool done)
{
    int idx = mIndex * mStateDim;
    std::copy(state, state + mStateDim, mStates.begin() + idx);
    
    idx = mIndex * mActionDim;
    std::copy(action, action + mActionDim, mActions.begin() + idx);
    
    mVectorRewards[mIndex] = reward;
    mRewards[mIndex] = reward.Scalar();
    
    idx = mIndex * mStateDim;
    std::copy(nextState, nextState + mStateDim, mNextStates.begin() + idx);
    
    mDones[mIndex] = done ? 1.0f : 0.0f;
    
    mIndex = (mIndex + 1) % mCapacity;
    mSize = std::min(mSize + 1, mCapacity);
}

void ReplayBuffer::Add(const float* state, const float* action, float reward,
                        const float* nextState, bool done)
{
    VectorReward vr;
    vr.damage_dealt = reward;
    Add(state, action, vr, nextState, done);
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

void ReplayBuffer::SampleVectorRewards(int batchSize, float* states, float* actions,
                                        VectorReward* rewards, float* nextStates, float* dones,
                                        std::mt19937& rng)
{
    std::uniform_int_distribution<int> dist(0, mSize - 1);
    
    for (int i = 0; i < batchSize; ++i) {
        int idx = dist(rng);
        
        std::copy(mStates.begin() + idx * mStateDim,
                  mStates.begin() + (idx + 1) * mStateDim,
                  states + i * mStateDim);
        
        std::copy(mActions.begin() + idx * mActionDim,
                  mActions.begin() + (idx + 1) * mActionDim,
                  actions + i * mActionDim);
        
        rewards[i] = mVectorRewards[idx];
        
        std::copy(mNextStates.begin() + idx * mStateDim,
                  mNextStates.begin() + (idx + 1) * mStateDim,
                  nextStates + i * mStateDim);
        
        dones[i] = mDones[idx];
    }
}