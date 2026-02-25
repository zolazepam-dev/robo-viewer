#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <random>
#include <cstring>

#include "NeuralMath.h"

struct alignas(32) SecondOrderLatentMemory
{
    alignas(32) float z_pos[LATENT_DIM_ALIGNED * NUM_PARALLEL_ENVS];
    alignas(32) float z_vel[LATENT_DIM_ALIGNED * NUM_PARALLEL_ENVS];
    
    size_t latentDim = LATENT_DIM;
    size_t numEnvs = NUM_PARALLEL_ENVS;
    float dt = 1.0f / 60.0f;

    void Init()
    {
        std::memset(z_pos, 0, sizeof(z_pos));
        std::memset(z_vel, 0, sizeof(z_vel));
    }

    void ResetEnv(size_t envIdx)
    {
        size_t offset = envIdx * LATENT_DIM_ALIGNED;
        std::memset(z_pos + offset, 0, LATENT_DIM_ALIGNED * sizeof(float));
        std::memset(z_vel + offset, 0, LATENT_DIM_ALIGNED * sizeof(float));
    }

    void ResetAll()
    {
        std::memset(z_pos, 0, sizeof(z_pos));
        std::memset(z_vel, 0, sizeof(z_vel));
    }

    float* GetPosition(size_t envIdx) 
    { 
        return z_pos + envIdx * LATENT_DIM_ALIGNED; 
    }
    
    float* GetVelocity(size_t envIdx) 
    { 
        return z_vel + envIdx * LATENT_DIM_ALIGNED; 
    }
    
    const float* GetPosition(size_t envIdx) const 
    { 
        return z_pos + envIdx * LATENT_DIM_ALIGNED; 
    }
    
    const float* GetVelocity(size_t envIdx) const 
    { 
        return z_vel + envIdx * LATENT_DIM_ALIGNED; 
    }

    void StepDynamicsScalar(const float* accelerations, size_t envIdx);
    void StepDynamicsVectorized(const float* accelerations);
    void StepDynamicsVectorizedBatch(const float* accelerations, int numEnvs);
};

class alignas(32) ODE2VAEEncoder
{
public:
    ODE2VAEEncoder() = default;
    
    void Init(size_t obsDim, size_t latentDim, std::mt19937& rng);
    
    void Encode(const float* observation, float* z_pos_out, float* z_vel_out);
    void EncodeBatch(const float* observations, float* z_pos_out, float* z_vel_out, int batchSize);
    
    size_t GetObsDim() const { return mObsDim; }
    size_t GetLatentDim() const { return mLatentDim; }
    
    std::vector<float>& GetWeightsPos() { return mWeightsPos; }
    std::vector<float>& GetWeightsVel() { return mWeightsVel; }
    std::vector<float>& GetBiasPos() { return mBiasPos; }
    std::vector<float>& GetBiasVel() { return mBiasVel; }
    
    const std::vector<float>& GetWeightsPos() const { return mWeightsPos; }
    const std::vector<float>& GetWeightsVel() const { return mWeightsVel; }
    const std::vector<float>& GetBiasPos() const { return mBiasPos; }
    const std::vector<float>& GetBiasVel() const { return mBiasVel; }

private:
    size_t mObsDim = 0;
    size_t mLatentDim = 0;
    
    alignas(32) std::vector<float> mWeightsPos;
    alignas(32) std::vector<float> mWeightsVel;
    alignas(32) std::vector<float> mBiasPos;
    alignas(32) std::vector<float> mBiasVel;
    
    alignas(32) std::vector<float> mTempBuffer;
};

class alignas(32) ODE2VAEDynamics
{
public:
    ODE2VAEDynamics() = default;
    
    void Init(size_t latentDim, size_t obsDim, std::mt19937& rng);
    
    void ComputeAcceleration(const float* z_pos, const float* z_vel, const float* obs, float* accel_out);
    void ComputeAccelerationBatch(const float* z_pos, const float* z_vel, const float* obs, 
                                   float* accel_out, int batchSize);
    
    size_t GetLatentDim() const { return mLatentDim; }
    size_t GetObsDim() const { return mObsDim; }
    
    std::vector<float>& GetWeights() { return mWeights; }
    std::vector<float>& GetBias() { return mBias; }
    
    const std::vector<float>& GetWeights() const { return mWeights; }
    const std::vector<float>& GetBias() const { return mBias; }

private:
    size_t mLatentDim = 0;
    size_t mObsDim = 0;
    
    alignas(32) std::vector<float> mWeights;
    alignas(32) std::vector<float> mBias;
    
    alignas(32) std::vector<float> mCombinedInput;
};

class alignas(32) LatentMemoryManager
{
public:
    LatentMemoryManager() = default;
    
    void Init(size_t obsDim, size_t latentDim, std::mt19937& rng);
    
    void EncodeObservations(const float* observations, int numEnvs);
    void StepLatentDynamics(const float* observations, int numEnvs);
    void GetLatentStates(float* z_pos_out, float* z_vel_out, size_t envIdx) const;
    void GetLatentStatesBatch(float* z_pos_out, float* z_vel_out, int numEnvs) const;
    
    void ResetEnv(int envIdx);
    void ResetAll();
    
    SecondOrderLatentMemory& GetMemory() { return mMemory; }
    const SecondOrderLatentMemory& GetMemory() const { return mMemory; }
    
    ODE2VAEEncoder& GetEncoder() { return mEncoder; }
    ODE2VAEDynamics& GetDynamics() { return mDynamics; }
    
    size_t GetLatentDim() const { return mLatentDim; }
    size_t GetObsDim() const { return mObsDim; }

private:
    size_t mObsDim = 0;
    size_t mLatentDim = 0;
    
    SecondOrderLatentMemory mMemory;
    ODE2VAEEncoder mEncoder;
    ODE2VAEDynamics mDynamics;
    
    alignas(32) std::vector<float> mAccelerationBuffer;
};

struct alignas(32) VectorizedLatentBatch
{
    static constexpr size_t TOTAL_LATENT_FLOATS = LATENT_DIM_ALIGNED * NUM_PARALLEL_ENVS * 2;
    
    alignas(32) float batch_z_pos[LATENT_DIM_ALIGNED * NUM_PARALLEL_ENVS];
    alignas(32) float batch_z_vel[LATENT_DIM_ALIGNED * NUM_PARALLEL_ENVS];
    alignas(32) float batch_accel[LATENT_DIM_ALIGNED * NUM_PARALLEL_ENVS];
    
    void Clear()
    {
        std::memset(batch_z_pos, 0, sizeof(batch_z_pos));
        std::memset(batch_z_vel, 0, sizeof(batch_z_vel));
        std::memset(batch_accel, 0, sizeof(batch_accel));
    }
};
