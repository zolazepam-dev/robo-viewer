#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <random>
#include <cstring>

#include <Eigen/Core>
#include "NeuralMath.h"
#include "AlignedAllocator.h"

struct RFFConfig {
    int num_features = 256;   // RFF features (reduced for speed)
    float sigma = 1.0f;        // Kernel bandwidth
    int seed = 42;             // Fixed seed for reproducibility
};

/**
 * Random Fourier Features Layer
 * 
 * Implements RFF approximation of RBF kernel:
 *   z(x) = cos(W_fixed * x + b_fixed)
 *   output = W_train * z(x) + bias_train
 * 
 * W_fixed and b_fixed are FIXED (non-trainable), initialized from:
 *   W_fixed ~ N(0, 1/sigma^2)
 *   b_fixed ~ U[0, 2*pi]
 * 
 * W_train and bias_train are TRAINABLE parameters.
 */
class alignas(32) RFFLayer
{
public:
    RFFLayer() = default;
    RFFLayer(const RFFLayer& other) = default;
    RFFLayer& operator=(const RFFLayer& other) = default;

    /**
     * Initialize the RFF layer
     * @param inputDim Dimension of input vectors
     * @param outputDim Dimension of output vectors
     * @param config RFF configuration (num_features, sigma, seed)
     * @param rng Random number generator
     */
    void Init(size_t inputDim, size_t outputDim, const RFFConfig& config, std::mt19937& rng);

    /**
     * Forward pass for single sample
     * @param input Input vector [inputDim]
     * @param output Output vector [outputDim]
     * @param featureBuffer External buffer for RFF features (cos) [numFeatures]
     */
    void Forward(const float* input, float* output, float* featureBuffer);

    /**
     * Forward pass for batch
     * @param input Input batch [batchSize * inputDim]
     * @param output Output batch [batchSize * outputDim]
     * @param batchSize Number of samples in batch
     * @param featureBuffer External buffer for RFF features (cos) [batchSize * numFeatures]
     */
    void ForwardBatch(const float* input, float* output, int batchSize, float* featureBuffer);

    /**
     * Forward pass for batch using Eigen optimization
     * @param input Input batch [batchSize * inputDim]
     * @param output Output batch [batchSize * outputDim]
     * @param batchSize Number of samples in batch
     * @param featureBuffer External buffer for RFF features (cos) [batchSize * numFeatures]
     */
    void ForwardBatchEigen(const float* input, float* output, int batchSize, float* featureBuffer);

    /**
     * Forward pass for batch using Eigen (returns MatrixXf)
     * @param input Input batch [batchSize × inputDim]
     * @return Output batch [batchSize × outputDim]
     */
    Eigen::MatrixXf ForwardEigen(const Eigen::MatrixXf& input) const;

    /**
     * Backward pass for gradient computation
     * @param input Original input to forward pass [inputDim]
     * @param output_grad Gradient of loss w.r.t. output [outputDim]
     * @param input_grad Gradient of loss w.r.t. input [inputDim] (can be nullptr)
     * @param weights_grad Gradient of loss w.r.t. trainable weights [outputDim * numFeatures] (accumulated)
     * @param bias_grad Gradient of loss w.r.t. trainable bias [outputDim] (accumulated)
     * @param featureBuffer External buffer for RFF features (cos) [numFeatures]
     * @param sinFeatureBuffer External buffer for RFF sin features [numFeatures]
     * @param featureGradBuffer External buffer for RFF feature gradients [numFeatures]
     */
    void Backward(const float* input, const float* output_grad, float* input_grad, 
                  float* weights_grad, float* bias_grad,
                  const float* featureBuffer, const float* sinFeatureBuffer, float* featureGradBuffer);

    /**
     * Get trainable weights (W_train)
     * @return Reference to trainable weights matrix [outputDim * numFeatures]
     */
    AlignedVector32<float>& GetTrainableWeights() { return mTrainableWeights; }
    const AlignedVector32<float>& GetTrainableWeights() const { return mTrainableWeights; }

    /**
     * Get trainable bias
     * @return Reference to trainable bias vector [outputDim]
     */
    AlignedVector32<float>& GetTrainableBias() { return mTrainableBias; }
    const AlignedVector32<float>& GetTrainableBias() const { return mTrainableBias; }

    /**
     * Get gradient of trainable weights
     * @return Reference to weights gradient [outputDim * numFeatures]
     */
    AlignedVector32<float>& GetWeightsGradient() { return mWeightsGradient; }
    const AlignedVector32<float>& GetWeightsGradient() const { return mWeightsGradient; }

    /**
     * Get gradient of trainable bias
     * @return Reference to bias gradient [outputDim]
     */
    AlignedVector32<float>& GetBiasGradient() { return mBiasGradient; }
    const AlignedVector32<float>& GetBiasGradient() const { return mBiasGradient; }

    /**
     * Get total number of trainable parameters
     * @return Number of floats in weights + bias (padded for alignment)
     */
    size_t GetNumParams() const { 
        return mTrainableWeights.size() + mTrainableBias.size(); 
    }

    size_t GetInputDim() const { return mInputDim; }
    size_t GetOutputDim() const { return mOutputDim; }
    int GetNumFeatures() const { return mNumFeatures; }
    float GetSigma() const { return mSigma; }

    /**
     * Zero out all gradients
     */
    void ZeroGradients();

    /**
     * Scale all gradients by a factor
     * @param scale Scale factor
     */
    void ScaleGradients(float scale);

    /**
     * Compute RFF features: cos(W_fixed * x + b_fixed)
     * @param input Input vector [inputDim]
     * @param features Output features [numFeatures]
     * @param sin_features Optional output for sin(W_fixed * x + b_fixed) [numFeatures]
     */
    void ComputeFeatures(const float* input, float* features, float* sin_features = nullptr);

    void SetFastMode(bool fast) { mFastMode = fast; }

private:
    size_t mInputDim = 0;
    size_t mOutputDim = 0;
    int mNumFeatures = 1024;
    float mSigma = 1.0f;
    bool mFastMode = false;

    // FIXED (non-trainable) RFF parameters
    AlignedVector32<float> mFixedWeights;    // [numFeatures * inputDim] ~ N(0, 1/sigma^2)
    AlignedVector32<float> mFixedBiases;     // [numFeatures] ~ U[0, 2*pi]

    // TRAINABLE parameters
    AlignedVector32<float> mTrainableWeights; // [outputDim * numFeatures]
    AlignedVector32<float> mTrainableBias;    // [outputDim]

    // Gradients for trainable parameters
    AlignedVector32<float> mWeightsGradient;  // [outputDim * numFeatures]
    AlignedVector32<float> mBiasGradient;     // [outputDim]

    // Fast approximation parameters
    AlignedVector32<float> mFastWeights;      // [outputDim * inputDim]
    AlignedVector32<float> mFastBias;         // [outputDim]
};
