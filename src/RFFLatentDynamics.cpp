#include "RFFLatentDynamics.h"

#include <cstring>
#include <cmath>
#include <immintrin.h>
#include "NeuralMath.h"

void RFFLatentDynamics::Init(size_t latentDim, size_t obsDim, 
                              const RFFConfig& config, std::mt19937& rng)
{
    mLatentDim = latentDim;
    mObsDim = obsDim;
    mInputDim = latentDim * 2 + obsDim;

    // Initialize RFF layer
    // Input: [obs; zPos; zVel] -> Output: acceleration (latentDim)
    mRFFLayer.Init(mInputDim, latentDim, config, rng);

    // Allocate combined input buffer
    mCombinedInput.resize(mInputDim);
}

void RFFLatentDynamics::ComputeAcceleration(const float* z_pos, const float* z_vel,
                                             const float* obs, float* accel_out)
{
    // Combine inputs: [z_pos; z_vel; obs]
    size_t inputIdx = 0;
    
    // Copy latent position
    for (size_t i = 0; i < mLatentDim; ++i) {
        mCombinedInput[inputIdx++] = z_pos[i];
    }
    
    // Copy latent velocity
    for (size_t i = 0; i < mLatentDim; ++i) {
        mCombinedInput[inputIdx++] = z_vel[i];
    }
    
    // Copy observation
    for (size_t i = 0; i < mObsDim; ++i) {
        mCombinedInput[inputIdx++] = obs[i];
    }

    // Forward pass through RFF layer
    AlignedVector32<float> featureBuffer(mRFFLayer.GetNumFeatures());
    mRFFLayer.Forward(mCombinedInput.data(), accel_out, featureBuffer.data());
}

void RFFLatentDynamics::ComputeAccelerationBatch(const float* z_pos, const float* z_vel,
                                                  const float* obs, float* accel_out, 
                                                  int batchSize)
{
    // Allocate batch buffers
    size_t batch_size = static_cast<size_t>(batchSize);
    size_t combined_size = batch_size * mInputDim;
    size_t output_size = batch_size * mLatentDim;
    
    AlignedVector32<float> combinedInputBatch(combined_size);
    AlignedVector32<float> accelOutBatch(output_size);
    
    // Pack all environments into single batch matrix: [batch_size × input_dim]
    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        size_t src_pos = b * mLatentDim;
        size_t src_vel = b * mLatentDim;
        size_t src_obs = b * mObsDim;
        size_t dst = b * mInputDim;
        
        // Copy z_pos
        for (size_t i = 0; i < mLatentDim; ++i) {
            combinedInputBatch[dst + i] = z_pos[src_pos + i];
        }
        // Copy z_vel
        for (size_t i = 0; i < mLatentDim; ++i) {
            combinedInputBatch[dst + mLatentDim + i] = z_vel[src_vel + i];
        }
        // Copy obs
        for (size_t i = 0; i < mObsDim; ++i) {
            combinedInputBatch[dst + mLatentDim * 2 + i] = obs[src_obs + i];
        }
    }
    
    // Single batch forward pass through RFF
    AlignedVector32<float> batchFeatureBuffer(batch_size * mRFFLayer.GetNumFeatures());
    mRFFLayer.ForwardBatch(combinedInputBatch.data(), accelOutBatch.data(), batchSize, batchFeatureBuffer.data());
    
    // Copy results back
    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < mLatentDim; ++i) {
            accel_out[b * mLatentDim + i] = accelOutBatch[b * mLatentDim + i];
        }
    }
}
