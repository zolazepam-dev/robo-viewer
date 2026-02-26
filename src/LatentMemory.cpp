#include "LatentMemory.h"
#include "NeuralMath.h"

#include <cstring>
#include <cmath>
#include <immintrin.h>

void SecondOrderLatentMemory::StepDynamicsScalar(const float* accelerations, size_t envIdx)
{
    size_t offset = envIdx * LATENT_DIM_ALIGNED;
    for (size_t i = 0; i < LATENT_DIM; ++i)
    {
        z_vel[offset + i] += accelerations[i] * dt;
        z_pos[offset + i] += z_vel[offset + i] * dt;
    }
}

void SecondOrderLatentMemory::StepDynamicsVectorized(const float* accelerations)
{
    const size_t totalSize = LATENT_DIM_ALIGNED * NUM_PARALLEL_ENVS;
    const size_t simdWidth = 8;
    const size_t simdEnd = totalSize - (totalSize % simdWidth);
    const __m256 dtVec = _mm256_set1_ps(dt);
    
    size_t i = 0;
    for (; i < simdEnd; i += simdWidth)
    {
        __m256 vel = _mm256_loadu_ps(z_vel + i);
        __m256 pos = _mm256_loadu_ps(z_pos + i);
        __m256 accel = _mm256_loadu_ps(accelerations + i);
        
        __m256 newVel = _mm256_fmadd_ps(accel, dtVec, vel);
        __m256 newPos = _mm256_fmadd_ps(newVel, dtVec, pos);
        
        _mm256_storeu_ps(z_vel + i, newVel);
        _mm256_storeu_ps(z_pos + i, newPos);
    }
    
    for (; i < totalSize; ++i)
    {
        z_vel[i] += accelerations[i] * dt;
        z_pos[i] += z_vel[i] * dt;
    }
}

void SecondOrderLatentMemory::StepDynamicsVectorizedBatch(const float* accelerations, int numEnvs)
{
    const size_t totalSize = LATENT_DIM_ALIGNED * static_cast<size_t>(numEnvs);
    const size_t simdWidth = 8;
    const size_t simdEnd = totalSize - (totalSize % simdWidth);
    const __m256 dtVec = _mm256_set1_ps(dt);
    
    size_t i = 0;
    for (; i < simdEnd; i += simdWidth)
    {
        __m256 vel = _mm256_loadu_ps(z_vel + i);
        __m256 pos = _mm256_loadu_ps(z_pos + i);
        __m256 accel = _mm256_loadu_ps(accelerations + i);
        
        __m256 newVel = _mm256_fmadd_ps(accel, dtVec, vel);
        __m256 newPos = _mm256_fmadd_ps(newVel, dtVec, pos);
        
        _mm256_storeu_ps(z_vel + i, newVel);
        _mm256_storeu_ps(z_pos + i, newPos);
    }
    
    for (; i < totalSize; ++i)
    {
        z_vel[i] += accelerations[i] * dt;
        z_pos[i] += z_vel[i] * dt;
    }
}

void ODE2VAEEncoder::Init(size_t obsDim, size_t latentDim, std::mt19937& rng)
{
    mObsDim = obsDim;
    mLatentDim = latentDim;
    
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    mWeightsPos.resize(latentDim * obsDim);
    mWeightsVel.resize(latentDim * obsDim);
    mBiasPos.resize(latentDim);
    mBiasVel.resize(latentDim);
    mTempBuffer.resize(latentDim * 2);
    
    for (auto& w : mWeightsPos) w = dist(rng);
    for (auto& w : mWeightsVel) w = dist(rng);
    for (size_t i = 0; i < latentDim; ++i)
    {
        mBiasPos[i] = 0.0f;
        mBiasVel[i] = 0.0f;
    }
}

void ODE2VAEEncoder::Encode(const float* observation, float* z_pos_out, float* z_vel_out)
{
    const size_t simdWidth = 8;
    
    for (size_t i = 0; i < mLatentDim; ++i)
    {
        float posVal = mBiasPos[i];
        float velVal = mBiasVel[i];
        
        const float* wPos = mWeightsPos.data() + i * mObsDim;
        const float* wVel = mWeightsVel.data() + i * mObsDim;
        
        size_t j = 0;
        for (; j + simdWidth <= mObsDim; j += simdWidth)
        {
            __m256 obs = _mm256_loadu_ps(observation + j);
            __m256 wp = _mm256_loadu_ps(wPos + j);
            __m256 wv = _mm256_loadu_ps(wVel + j);
            
            __m256 prodP = _mm256_mul_ps(obs, wp);
            __m256 prodV = _mm256_mul_ps(obs, wv);
            
            alignas(32) float tempP[8], tempV[8];
            _mm256_store_ps(tempP, prodP);
            _mm256_store_ps(tempV, prodV);
            
            for (int k = 0; k < 8; ++k)
            {
                posVal += tempP[k];
                velVal += tempV[k];
            }
        }
        
        for (; j < mObsDim; ++j)
        {
            posVal += wPos[j] * observation[j];
            velVal += wVel[j] * observation[j];
        }
        
        z_pos_out[i] = tanhf(posVal);
        z_vel_out[i] = tanhf(velVal);
    }
}

void ODE2VAEEncoder::EncodeBatch(const float* observations, float* z_pos_out, float* z_vel_out, int batchSize)
{
    for (int b = 0; b < batchSize; ++b)
    {
        Encode(observations + static_cast<size_t>(b) * mObsDim, 
               z_pos_out + static_cast<size_t>(b) * mLatentDim, 
               z_vel_out + static_cast<size_t>(b) * mLatentDim);
    }
}

void ODE2VAEDynamics::Init(size_t latentDim, size_t obsDim, std::mt19937& rng)
{
    mLatentDim = latentDim;
    mObsDim = obsDim;
    
    size_t inputDim = latentDim * 2 + obsDim;
    
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    mWeights.resize(latentDim * inputDim);
    mBias.resize(latentDim);
    mCombinedInput.resize(inputDim);
    
    for (auto& w : mWeights) w = dist(rng);
    for (size_t i = 0; i < latentDim; ++i)
    {
        mBias[i] = 0.0f;
    }
}

void ODE2VAEDynamics::ComputeAcceleration(const float* z_pos, const float* z_vel, 
                                           const float* obs, float* accel_out)
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
    for (int b = 0; b < batchSize; ++b)
    {
        ComputeAcceleration(
            z_pos + static_cast<size_t>(b) * mLatentDim,
            z_vel + static_cast<size_t>(b) * mLatentDim,
            obs + static_cast<size_t>(b) * mObsDim,
            accel_out + static_cast<size_t>(b) * mLatentDim
        );
    }
}

void LatentMemoryManager::Init(size_t obsDim, size_t latentDim, std::mt19937& rng)
{
    mObsDim = obsDim;
    mLatentDim = latentDim;
    
    mMemory.latentDim = latentDim;
    mMemory.numEnvs = NUM_PARALLEL_ENVS;
    mMemory.Init();
    
    mEncoder.Init(obsDim, latentDim, rng);
    mDynamics.Init(latentDim, obsDim, rng);
    
    mAccelerationBuffer.resize(latentDim * NUM_PARALLEL_ENVS);
}

void LatentMemoryManager::EncodeObservations(const float* observations, int numEnvs)
{
    AlignedVector32<float> tempPos(LATENT_DIM);
    AlignedVector32<float> tempVel(LATENT_DIM);

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
    mDynamics.ComputeAccelerationBatch(
        mMemory.z_pos,
        mMemory.z_vel,
        observations,
        mAccelerationBuffer.data(),
        numEnvs
    );
    
    ForwardTanh_AVX2(mAccelerationBuffer.data(), mLatentDim * static_cast<size_t>(numEnvs));
    
    mMemory.StepDynamicsVectorizedBatch(mAccelerationBuffer.data(), numEnvs);
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
    std::memcpy(z_pos_out, mMemory.z_pos, mLatentDim * static_cast<size_t>(numEnvs) * sizeof(float));
    std::memcpy(z_vel_out, mMemory.z_vel, mLatentDim * static_cast<size_t>(numEnvs) * sizeof(float));
}

void LatentMemoryManager::ResetEnv(int envIdx)
{
    mMemory.ResetEnv(static_cast<size_t>(envIdx));
}

void LatentMemoryManager::ResetAll()
{
    mMemory.ResetAll();
}
