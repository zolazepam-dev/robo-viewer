#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <random>
#include <cstring>

#include "RFFLayer.h"
#include "AlignedAllocator.h"

/**
 * RFF-based Latent Dynamics
 * 
 * Replaces the hardcoded ODE2VAEDynamics with an RFF-based dynamics model.
 * Computes acceleration from latent state and observation:
 *   accel = RFF([obs; zPos; zVel])
 * 
 * Input: [obs; zPos; zVel] (concatenated)
 * Output: acceleration (same dim as latent)
 */
class alignas(32) RFFLatentDynamics
{
public:
    RFFLatentDynamics() = default;
    RFFLatentDynamics(const RFFLatentDynamics& other) = default;
    RFFLatentDynamics& operator=(const RFFLatentDynamics& other) = default;

    /**
     * Initialize the RFF latent dynamics
     * @param latentDim Dimension of latent space
     * @param obsDim Dimension of observation space
     * @param config RFF configuration
     * @param rng Random number generator
     */
    void Init(size_t latentDim, size_t obsDim, const RFFConfig& config, std::mt19937& rng);

    /**
     * Compute acceleration for single sample
     * @param z_pos Latent position [latentDim]
     * @param z_vel Latent velocity [latentDim]
     * @param obs Observation [obsDim]
     * @param accel_out Output acceleration [latentDim]
     */
    void ComputeAcceleration(const float* z_pos, const float* z_vel, 
                             const float* obs, float* accel_out);

    /**
     * Compute acceleration for batch
     * @param z_pos Latent positions [batchSize * latentDim]
     * @param z_vel Latent velocities [batchSize * latentDim]
     * @param obs Observations [batchSize * obsDim]
     * @param accel_out Output accelerations [batchSize * latentDim]
     * @param batchSize Number of samples in batch
     */
    void ComputeAccelerationBatch(const float* z_pos, const float* z_vel, 
                                   const float* obs, float* accel_out, int batchSize);

    /**
     * Get the underlying RFF layer
     * @return Reference to RFF layer
     */
    RFFLayer& GetRFFLayer() { return mRFFLayer; }
    const RFFLayer& GetRFFLayer() const { return mRFFLayer; }

    /**
     * Get trainable weights (compatibility methods)
     */
    AlignedVector32<float>& GetTrainableWeights() { return mRFFLayer.GetTrainableWeights(); }
    const AlignedVector32<float>& GetTrainableWeights() const { return mRFFLayer.GetTrainableWeights(); }
    AlignedVector32<float>& GetTrainableBias() { return mRFFLayer.GetTrainableBias(); }
    const AlignedVector32<float>& GetTrainableBias() const { return mRFFLayer.GetTrainableBias(); }
    
    // Compatibility with ODE2VAEDynamics interface
    AlignedVector32<float>& GetWeights() { return mRFFLayer.GetTrainableWeights(); }
    AlignedVector32<float>& GetBias() { return mRFFLayer.GetTrainableBias(); }
    const AlignedVector32<float>& GetWeights() const { return mRFFLayer.GetTrainableWeights(); }
    const AlignedVector32<float>& GetBias() const { return mRFFLayer.GetTrainableBias(); }

    /**
     * Get gradients
     */
    AlignedVector32<float>& GetWeightsGradient() { return mRFFLayer.GetWeightsGradient(); }
    const AlignedVector32<float>& GetWeightsGradient() const { return mRFFLayer.GetWeightsGradient(); }
    AlignedVector32<float>& GetBiasGradient() { return mRFFLayer.GetBiasGradient(); }
    const AlignedVector32<float>& GetBiasGradient() const { return mRFFLayer.GetBiasGradient(); }

    /**
     * Get number of trainable parameters
     */
    size_t GetNumParams() const { return mRFFLayer.GetNumParams(); }
    size_t GetNumFeatures() const { return mRFFLayer.GetNumFeatures(); }

    size_t GetLatentDim() const { return mLatentDim; }
    size_t GetObsDim() const { return mObsDim; }
    size_t GetInputDim() const { return mInputDim; }

    /**
     * Gradient management
     */
    void ZeroGradients() { mRFFLayer.ZeroGradients(); }
    void ScaleGradients(float scale) { mRFFLayer.ScaleGradients(scale); }

private:
    size_t mLatentDim = 0;
    size_t mObsDim = 0;
    size_t mInputDim = 0;  // latentDim * 2 + obsDim

    RFFLayer mRFFLayer;

    AlignedVector32<float> mCombinedInput;
    
    // Batch buffers (reused across calls)
    AlignedVector32<float> mCombinedInputBatch;
    AlignedVector32<float> mAccelOutBatch;
};
