#include "LatentMemory.h"
#include "NeuralMath.h"

#include <cstring>
#include <cmath>
#include <immintrin.h>

void SecondOrderLatentMemory::StepDynamicsScalar(const float* accelerations, size_t envIdx)
{
    size_t offset = envIdx * mLatentDimAligned;
    float* zPos = z_pos.data() + offset;
    float* zVel = z_vel.data() + offset;

    for (size_t i = 0; i < mLatentDim; ++i)
    {
        // Simple Euler integration (second order)
        // v = v + a * dt
        // p = p + v * dt
        zVel[i] += accelerations[i] * dt;
        zPos[i] += zVel[i] * dt;
    }
}

void SecondOrderLatentMemory::StepDynamicsVectorized(const float* accelerations)
{
    const size_t simdWidth = 8;
    __m256 dtVec = _mm256_set1_ps(dt);

    for (size_t i = 0; i < z_pos.size(); i += simdWidth)
    {
        __m256 p = _mm256_load_ps(z_pos.data() + i);
        __m256 v = _mm256_load_ps(z_vel.data() + i);
        __m256 a = _mm256_load_ps(accelerations + i);

        // v = v + a * dt
        v = _mm256_add_ps(v, _mm256_mul_ps(a, dtVec));
        // p = p + v * dt
        p = _mm256_add_ps(p, _mm256_mul_ps(v, dtVec));

        _mm256_store_ps(z_vel.data() + i, v);
        _mm256_store_ps(z_pos.data() + i, p);
    }
}

void SecondOrderLatentMemory::StepDynamicsVectorizedBatch(const float* accelerations, int numEnvs)
{
    const size_t simdWidth = 8;
    __m256 dtVec = _mm256_set1_ps(dt);
    size_t totalElements = mLatentDimAligned * static_cast<size_t>(numEnvs);

    for (size_t i = 0; i < totalElements; i += simdWidth)
    {
        __m256 p = _mm256_load_ps(z_pos.data() + i);
        __m256 v = _mm256_load_ps(z_vel.data() + i);
        __m256 a = _mm256_load_ps(accelerations + i);

        v = _mm256_add_ps(v, _mm256_mul_ps(a, dtVec));
        p = _mm256_add_ps(p, _mm256_mul_ps(v, dtVec));

        _mm256_store_ps(z_vel.data() + i, v);
        _mm256_store_ps(z_pos.data() + i, p);
    }
}

void ODE2VAEEncoder::Init(size_t obsDim, size_t latentDim, std::mt19937& rng)
{
    mObsDim = obsDim;
    mLatentDim = latentDim;

    mWeightsPos.resize(latentDim * obsDim);
    mWeightsVel.resize(latentDim * obsDim);
    mBiasPos.resize(latentDim);
    mBiasVel.resize(latentDim);

    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (auto& w : mWeightsPos) w = dist(rng);
    for (auto& w : mWeightsVel) w = dist(rng);
    for (auto& b : mBiasPos) b = 0.0f;
    for (auto& b : mBiasVel) b = 0.0f;
}

void ODE2VAEEncoder::Encode(const float* observation, float* z_pos_out, float* z_vel_out)
{
    for (size_t i = 0; i < mLatentDim; ++i)
    {
        float p = mBiasPos[i];
        float v = mBiasVel[i];
        const float* wp = mWeightsPos.data() + i * mObsDim;
        const float* wv = mWeightsVel.data() + i * mObsDim;

        for (size_t j = 0; j < mObsDim; ++j)
        {
            p += wp[j] * observation[j];
            v += wv[j] * observation[j];
        }

        z_pos_out[i] = p;
        z_vel_out[i] = v;
    }
}

void ODE2VAEEncoder::EncodeBatch(const float* observations, float* z_pos_out, float* z_vel_out, int batchSize)
{
    for (int b = 0; b < batchSize; ++b)
    {
        Encode(
            observations + static_cast<size_t>(b) * mObsDim,
            z_pos_out + static_cast<size_t>(b) * mLatentDim,
            z_vel_out + static_cast<size_t>(b) * mLatentDim
        );
    }
}

void ODE2VAEDynamics::Init(size_t latentDim, size_t obsDim, std::mt19937& rng)
{
    mLatentDim = latentDim;
    mObsDim = obsDim;

    size_t inputDim = latentDim * 2 + obsDim;
    mWeights.resize(latentDim * inputDim);
    mBias.resize(latentDim);

    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (auto& w : mWeights) w = dist(rng);
    for (auto& b : mBias) b = 0.0f;

    mCombinedInput.resize(inputDim);
}

void ODE2VAEDynamics::ComputeAcceleration(const float* z_pos, const float* z_vel, const float* obs, float* accel_out)
{
    size_t inputIdx = 0;
    for (size_t i = 0; i < mLatentDim; ++i)
    {
        mCombinedInput[inputIdx++] = z_pos[i];
    }
    for (size_t i = 0; i < mLatentDim; ++i)
    {
        mCombinedInput[inputIdx++] = z_vel[i];
    }
    for (size_t i = 0; i < mObsDim; ++i)
    {
        mCombinedInput[inputIdx++] = obs[i];
    }

    size_t inputDim = mLatentDim * 2 + mObsDim;

    for (size_t i = 0; i < mLatentDim; ++i)
    {
        float val = mBias[i];
        const float* w = mWeights.data() + i * inputDim;

        for (size_t j = 0; j < inputDim; ++j)
        {
            val += w[j] * mCombinedInput[j];
        }

        accel_out[i] = tanhf(val);
    }
}

void ODE2VAEDynamics::ComputeAccelerationBatch(const float* z_pos, const float* z_vel, const float* obs,
                                                float* accel_out, int batchSize)
{
    for (int b = 0; b < batchSize; ++b) {
        ComputeAcceleration(
            z_pos + static_cast<size_t>(b) * mLatentDim,
            z_vel + static_cast<size_t>(b) * mLatentDim,
            obs + static_cast<size_t>(b) * mObsDim,
            accel_out + static_cast<size_t>(b) * mLatentDim
        );
    }
}

// LatentMemoryManager implementation
void LatentMemoryManager::Init(size_t obsDim, size_t latentDim, const RFFConfig& config, std::mt19937& rng)
{
    mObsDim = obsDim;
    mLatentDim = latentDim;

    mMemory.Init(latentDim, NUM_PARALLEL_ROBOTS);

    mEncoder.Init(obsDim, latentDim, rng);

    // Initialize RFF dynamics with provided config
    mDynamics.Init(latentDim, obsDim, config, rng);

    // Initialize legacy dynamics (for compatibility, not used)
    mLegacyDynamics.Init(latentDim, obsDim, rng);

    mAccelerationBuffer.resize(GetAVX2PaddedSize(latentDim) * NUM_PARALLEL_ROBOTS);
}

void LatentMemoryManager::EncodeObservations(const float* observations, int numEnvs)
{
    AlignedVector32<float> tempPos(mLatentDim);
    AlignedVector32<float> tempVel(mLatentDim);

    for (int env = 0; env < numEnvs; ++env)
    {
        mEncoder.Encode(observations + static_cast<size_t>(env) * mObsDim, tempPos.data(), tempVel.data());

        float* zPos = mMemory.GetPosition(static_cast<size_t>(env));
        float* zVel = mMemory.GetVelocity(static_cast<size_t>(env));

        std::memcpy(zPos, tempPos.data(), mLatentDim * sizeof(float));
        std::memcpy(zVel, tempVel.data(), mLatentDim * sizeof(float));
    }
}

void LatentMemoryManager::StepLatentDynamics(const float* observations, int numEnvs)
{
    // Use RFF dynamics for acceleration computation
    mDynamics.ComputeAccelerationBatch(
        mMemory.z_pos.data(),
        mMemory.z_vel.data(),
        observations,
        mAccelerationBuffer.data(),
        numEnvs
    );

    // Apply tanh activation to accelerations
    ForwardTanh_AVX2(mAccelerationBuffer.data(), GetAVX2PaddedSize(mLatentDim) * static_cast<size_t>(numEnvs));

    // Step the latent dynamics
    mMemory.StepDynamicsVectorizedBatch(mAccelerationBuffer.data(), numEnvs);
}

void LatentMemoryManager::StepLatentDynamics(const float* observations, const std::vector<int>& envIndices)
{
    int numEnvs = static_cast<int>(envIndices.size());
    
    // Process each environment individually with its specific index
    for (int i = 0; i < numEnvs; ++i) {
        int envIdx = envIndices[i];
        const float* obs = observations + static_cast<size_t>(i) * mObsDim;
        
        // Get pointers to this environment's latent state
        float* zPos = mMemory.GetPosition(static_cast<size_t>(envIdx));
        float* zVel = mMemory.GetVelocity(static_cast<size_t>(envIdx));
        
        // Compute acceleration for this environment
        mDynamics.ComputeAcceleration(zPos, zVel, obs, mAccelerationBuffer.data());
        
        // Apply tanh activation
        ForwardTanh_AVX2(mAccelerationBuffer.data(), mLatentDim);
        
        // Step this environment's latent dynamics
        mMemory.StepDynamicsScalar(mAccelerationBuffer.data(), static_cast<size_t>(envIdx));
    }
}

void LatentMemoryManager::GetLatentStates(float* z_pos_out, float* z_vel_out, size_t envIdx) const
{
    const float* zPos = mMemory.GetPosition(envIdx);
    const float* zVel = mMemory.GetVelocity(envIdx);

    if (z_pos_out) std::memcpy(z_pos_out, zPos, mLatentDim * sizeof(float));
    if (z_vel_out) std::memcpy(z_vel_out, zVel, mLatentDim * sizeof(float));
}

void LatentMemoryManager::GetLatentStatesBatch(float* z_pos_out, float* z_vel_out, int numEnvs) const
{
    size_t elements = mMemory.mLatentDimAligned * static_cast<size_t>(numEnvs);
    if (z_pos_out) std::memcpy(z_pos_out, mMemory.z_pos.data(), elements * sizeof(float));
    if (z_vel_out) std::memcpy(z_vel_out, mMemory.z_vel.data(), elements * sizeof(float));
}

void LatentMemoryManager::ResetEnv(int envIdx)
{
    mMemory.ResetEnv(static_cast<size_t>(envIdx));
}

void LatentMemoryManager::ResetAll()
{
    mMemory.ResetAll();
}
