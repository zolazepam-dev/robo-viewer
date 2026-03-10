#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <random>
#include <cstring>
#include <memory>

#include "NeuralMath.h"
#include "AlignedAllocator.h"
#include "RFFLayer.h"

struct alignas(32) SecondOrderLatentMemory
{
    AlignedVector32<float> z_pos;
    AlignedVector32<float> z_vel;

    size_t mLatentDim = 0;
    size_t mLatentDimAligned = 0;
    size_t mNumEnvs = 0;
    float dt = 1.0f / 60.0f;

    void Init(size_t latentDim, size_t numEnvs)
    {
        mLatentDim = latentDim;
        mLatentDimAligned = GetAVX2PaddedSize(latentDim);
        mNumEnvs = numEnvs;
        
        z_pos.assign(mLatentDimAligned * numEnvs, 0.0f);
        z_vel.assign(mLatentDimAligned * numEnvs, 0.0f);
    }

    void ResetEnv(size_t envIdx)
    {
        if (envIdx >= mNumEnvs) return;
        size_t offset = envIdx * mLatentDimAligned;
        std::memset(z_pos.data() + offset, 0, mLatentDimAligned * sizeof(float));
        std::memset(z_vel.data() + offset, 0, mLatentDimAligned * sizeof(float));
    }

    void ResetAll()
    {
        std::fill(z_pos.begin(), z_pos.end(), 0.0f);
        std::fill(z_vel.begin(), z_vel.end(), 0.0f);
    }

    float* GetPosition(size_t envIdx)
    {
        return z_pos.data() + envIdx * mLatentDimAligned;
    }

    float* GetVelocity(size_t envIdx)
    {
        return z_vel.data() + envIdx * mLatentDimAligned;
    }

    const float* GetPosition(size_t envIdx) const
    {
        return z_pos.data() + envIdx * mLatentDimAligned;
    }

    const float* GetVelocity(size_t envIdx) const
    {
        return z_vel.data() + envIdx * mLatentDimAligned;
    }

    void StepDynamicsScalar(const float* accelerations, size_t envIdx);
    void StepDynamicsVectorized(const float* accelerations);
    void StepDynamicsVectorizedBatch(const float* accelerations, int numEnvs);
};

class alignas(32) ODE2VAEEncoder
{
public:
    ODE2VAEEncoder() = default;
    ODE2VAEEncoder(const ODE2VAEEncoder& other) = default;
    ODE2VAEEncoder& operator=(const ODE2VAEEncoder& other) = default;

    void Init(size_t obsDim, size_t latentDim, std::mt19937& rng);

    void Encode(const float* observation, float* z_pos_out, float* z_vel_out);
    void EncodeBatch(const float* observations, float* z_pos_out, float* z_vel_out, int batchSize);

    size_t GetObsDim() const { return mObsDim; }
    size_t GetLatentDim() const { return mLatentDim; }

    AlignedVector32<float>& GetWeightsPos() { return mWeightsPos; }
    AlignedVector32<float>& GetWeightsVel() { return mWeightsVel; }
    AlignedVector32<float>& GetBiasPos() { return mBiasPos; }
    AlignedVector32<float>& GetBiasVel() { return mBiasVel; }

    const AlignedVector32<float>& GetWeightsPos() const { return mWeightsPos; }
    const AlignedVector32<float>& GetWeightsVel() const { return mWeightsVel; }
    const AlignedVector32<float>& GetBiasPos() const { return mBiasPos; }
    const AlignedVector32<float>& GetBiasVel() const { return mBiasVel; }

private:
    size_t mObsDim = 0;
    size_t mLatentDim = 0;

    AlignedVector32<float> mWeightsPos;
    AlignedVector32<float> mWeightsVel;
    AlignedVector32<float> mBiasPos;
    AlignedVector32<float> mBiasVel;
};

class alignas(32) ODE2VAEDynamics
{
public:
    ODE2VAEDynamics() = default;
    ODE2VAEDynamics(const ODE2VAEDynamics& other) = default;
    ODE2VAEDynamics& operator=(const ODE2VAEDynamics& other) = default;

    void Init(size_t latentDim, size_t obsDim, std::mt19937& rng);

    void ComputeAcceleration(const float* z_pos, const float* z_vel, const float* obs, float* accel_out);
    void ComputeAccelerationBatch(const float* z_pos, const float* z_vel, const float* obs,
                                   float* accel_out, int batchSize);

    size_t GetLatentDim() const { return mLatentDim; }
    size_t GetObsDim() const { return mObsDim; }

    AlignedVector32<float>& GetWeights() { return mWeights; }
    AlignedVector32<float>& GetBias() { return mBias; }

    const AlignedVector32<float>& GetWeights() const { return mWeights; }
    const AlignedVector32<float>& GetBias() const { return mBias; }

private:
    size_t mLatentDim = 0;
    size_t mObsDim = 0;

    AlignedVector32<float> mWeights;
    AlignedVector32<float> mBias;

    AlignedVector32<float> mCombinedInput;
};

#include "RFFLatentDynamics.h"

class alignas(32) LatentMemoryManager
{
public:
    LatentMemoryManager() = default;
    LatentMemoryManager(const LatentMemoryManager& other) = default;
    LatentMemoryManager& operator=(const LatentMemoryManager& other) = default;

    void Init(size_t obsDim, size_t latentDim, const RFFConfig& config, std::mt19937& rng);

    void EncodeObservations(const float* observations, int numEnvs);
    void StepLatentDynamics(const float* observations, int numEnvs);
    void StepLatentDynamics(const float* observations, const std::vector<int>& envIndices);
    void GetLatentStates(float* z_pos_out, float* z_vel_out, size_t envIdx) const;
    void GetLatentStatesBatch(float* z_pos_out, float* z_vel_out, int numEnvs) const;

    void ResetEnv(int envIdx);
    void ResetAll();

    SecondOrderLatentMemory& GetMemory() { return mMemory; }
    const SecondOrderLatentMemory& GetMemory() const { return mMemory; }

    ODE2VAEEncoder& GetEncoder() { return mEncoder; }
    
    // Use RFF latent dynamics
    RFFLatentDynamics& GetDynamics() { return mDynamics; }
    const RFFLatentDynamics& GetDynamics() const { return mDynamics; }

    // Legacy compatibility (deprecated)
    ODE2VAEDynamics& GetLegacyDynamics() { return mLegacyDynamics; }

    size_t GetLatentDim() const { return mLatentDim; }
    size_t GetObsDim() const { return mObsDim; }

private:
    size_t mObsDim = 0;
    size_t mLatentDim = 0;

    SecondOrderLatentMemory mMemory;
    ODE2VAEEncoder mEncoder;
    
    // Primary RFF-based dynamics
    RFFLatentDynamics mDynamics;
    
    // Legacy ODE dynamics (kept for compatibility, not used)
    ODE2VAEDynamics mLegacyDynamics;

    AlignedVector32<float> mAccelerationBuffer;
};

struct alignas(32) VectorizedLatentBatch
{
    AlignedVector32<float> batch_z_pos;
    AlignedVector32<float> batch_z_vel;
    AlignedVector32<float> batch_accel;

    size_t mLatentDimAligned = 0;
    size_t mNumEnvs = 0;

    void Init(size_t latentDim, size_t numEnvs)
    {
        mLatentDimAligned = GetAVX2PaddedSize(latentDim);
        mNumEnvs = numEnvs;
        
        batch_z_pos.assign(mLatentDimAligned * numEnvs, 0.0f);
        batch_z_vel.assign(mLatentDimAligned * numEnvs, 0.0f);
        batch_accel.assign(mLatentDimAligned * numEnvs, 0.0f);
    }

    void Clear()
    {
        std::fill(batch_z_pos.begin(), batch_z_pos.end(), 0.0f);
        std::fill(batch_z_vel.begin(), batch_z_vel.end(), 0.0f);
        std::fill(batch_accel.begin(), batch_accel.end(), 0.0f);
    }
};
