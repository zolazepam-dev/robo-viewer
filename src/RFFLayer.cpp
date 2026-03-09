#include "RFFLayer.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "src/EigenUtils.h"

void RFFLayer::Init(size_t inputDim, size_t outputDim, const RFFConfig& config, std::mt19937& rng)
{
    mInputDim = inputDim;
    mOutputDim = outputDim;
    mNumFeatures = config.num_features;
    mSigma = config.sigma;

    // Initialize FIXED weights from N(0, 1/sigma^2)
    // Variance = 1/sigma^2, so std = 1/sigma
    float fixedWeightStd = 1.0f / mSigma;
    std::normal_distribution<float> fixedDist(0.0f, fixedWeightStd);
    
    size_t fixedWeightsSize = static_cast<size_t>(mNumFeatures) * mInputDim;
    mFixedWeights.resize(fixedWeightsSize);
    for (size_t i = 0; i < fixedWeightsSize; ++i) {
        mFixedWeights[i] = fixedDist(rng);
    }

    // Initialize FIXED biases from U[0, 2*pi]
    std::uniform_real_distribution<float> biasDist(0.0f, 2.0f * static_cast<float>(M_PI));
    mFixedBiases.resize(mNumFeatures);
    for (int i = 0; i < mNumFeatures; ++i) {
        mFixedBiases[i] = biasDist(rng);
    }

    // Initialize TRAINABLE weights with small values
    std::normal_distribution<float> trainDist(0.0f, 0.1f);
    size_t trainWeightsSize = mOutputDim * static_cast<size_t>(mNumFeatures);
    mTrainableWeights.resize(trainWeightsSize);
    for (size_t i = 0; i < trainWeightsSize; ++i) {
        mTrainableWeights[i] = trainDist(rng);
    }

    // Initialize trainable bias to zero
    mTrainableBias.resize(mOutputDim, 0.0f);

    // Initialize gradients to zero
    mWeightsGradient.resize(trainWeightsSize, 0.0f);
    mBiasGradient.resize(mOutputDim, 0.0f);

    // Allocate temporary buffers
    mFeatureBuffer.resize(mNumFeatures);
    mFeatureGradBuffer.resize(mNumFeatures);
}

void RFFLayer::ComputeFeatures(const float* input, float* features)
{
    // Compute z(x) = cos(W_fixed * x + b_fixed)
    // For each feature i: z_i = cos(sum_j(W_ij * x_j) + b_i)
    
    const size_t simdWidth = 8;
    
    for (int i = 0; i < mNumFeatures; ++i) {
        float dotProduct = mFixedBiases[i];
        const float* weights = mFixedWeights.data() + i * mInputDim;
        
        size_t j = 0;
        // Vectorized dot product
        for (; j + simdWidth <= mInputDim; j += simdWidth) {
            __m256 w = _mm256_loadu_ps(weights + j);
            __m256 x = _mm256_loadu_ps(input + j);
            __m256 prod = _mm256_mul_ps(w, x);
            
            // Horizontal sum
            alignas(32) float temp[8];
            _mm256_store_ps(temp, prod);
            for (int k = 0; k < 8; ++k) {
                dotProduct += temp[k];
            }
        }
        
        // Remainder
        for (; j < mInputDim; ++j) {
            dotProduct += weights[j] * input[j];
        }
        
        features[i] = std::cos(dotProduct);
    }
}

void RFFLayer::Forward(const float* input, float* output)
{
    // Step 1: Compute RFF features
    ComputeFeatures(input, mFeatureBuffer.data());
    
    // Step 2: Compute output = W_train * features + bias
    // output[i] = sum_j(W_train[i,j] * features[j]) + bias[i]
    
    const size_t simdWidth = 8;
    
    for (size_t i = 0; i < mOutputDim; ++i) {
        float val = mTrainableBias[i];
        const float* weights = mTrainableWeights.data() + i * mNumFeatures;
        
        size_t j = 0;
        for (; j + simdWidth <= static_cast<size_t>(mNumFeatures); j += simdWidth) {
            __m256 w = _mm256_loadu_ps(weights + j);
            __m256 f = _mm256_loadu_ps(mFeatureBuffer.data() + j);
            __m256 prod = _mm256_mul_ps(w, f);
            
            alignas(32) float temp[8];
            _mm256_store_ps(temp, prod);
            for (int k = 0; k < 8; ++k) {
                val += temp[k];
            }
        }
        
        for (; j < static_cast<size_t>(mNumFeatures); ++j) {
            val += weights[j] * mFeatureBuffer.data()[j];
        }
        
        output[i] = val;
    }
}

void RFFLayer::ForwardBatch(const float* input, float* output, int batchSize)
{
    // Use Eigen-optimized batch implementation
    ForwardBatchEigen(input, output, batchSize);
}

void RFFLayer::ForwardBatchEigen(const float* input, float* output, int batchSize)
{
    // Use Eigen for batched matrix multiplication
    // features = cos(input * W_fixed^T + b_fixed)
    // output = features * W_train^T + bias
    
    Eigen::Map<const Eigen::MatrixXf> inputMap(input, batchSize, mInputDim);
    Eigen::Map<const Eigen::MatrixXf> W_fixedMap(mFixedWeights.data(), mNumFeatures, mInputDim);
    Eigen::Map<const Eigen::VectorXf> b_fixedMap(mFixedBiases.data(), mNumFeatures);
    
    // Compute features: (batchSize x numFeatures)
    Eigen::MatrixXf features = (inputMap * W_fixedMap.transpose()).rowwise() + b_fixedMap.transpose();
    features = features.array().cos();
    
    // Compute output: (batchSize x outputDim)
    Eigen::Map<const Eigen::MatrixXf> W_trainMap(mTrainableWeights.data(), mOutputDim, mNumFeatures);
    Eigen::Map<const Eigen::VectorXf> biasMap(mTrainableBias.data(), mOutputDim);
    
    Eigen::MatrixXf outputMap = (features * W_trainMap.transpose()).rowwise() + biasMap.transpose();
    
    // Copy back
    std::memcpy(output, outputMap.data(), batchSize * mOutputDim * sizeof(float));
}

Eigen::MatrixXf RFFLayer::ForwardEigen(const Eigen::MatrixXf& input)
{
    // Map fixed weights to Eigen
    Eigen::Map<const Eigen::MatrixXf> W_fixedMap(mFixedWeights.data(), mNumFeatures, mInputDim);
    Eigen::Map<const Eigen::VectorXf> b_fixedMap(mFixedBiases.data(), mNumFeatures);
    
    // Compute features: cos(input * W_fixed^T + b_fixed)
    Eigen::MatrixXf features = (input * W_fixedMap.transpose()).rowwise() + b_fixedMap.transpose();
    features = features.array().cos();
    
    // Map trainable weights to Eigen
    Eigen::Map<const Eigen::MatrixXf> W_trainMap(mTrainableWeights.data(), mOutputDim, mNumFeatures);
    Eigen::Map<const Eigen::VectorXf> biasMap(mTrainableBias.data(), mOutputDim);
    
    // Compute output: features * W_train^T + bias
    Eigen::MatrixXf output = (features * W_trainMap.transpose()).rowwise() + biasMap.transpose();
    
    return output;
}

void RFFLayer::Backward(const float* input, const float* output_grad, float* input_grad,
                        float* weights_grad, float* bias_grad)
{
    // Recompute features (needed for gradient computation)
    ComputeFeatures(input, mFeatureBuffer.data());
    
    // Gradient w.r.t. trainable weights: dL/dW_train = output_grad^T * features
    // For each output dim i and feature j: dL/dW[i,j] = output_grad[i] * features[j]
    for (size_t i = 0; i < mOutputDim; ++i) {
        float grad = output_grad[i];
        float* w_grad = weights_grad + i * mNumFeatures;
        for (int j = 0; j < mNumFeatures; ++j) {
            w_grad[j] += grad * mFeatureBuffer.data()[j];
        }
    }
    
    // Gradient w.r.t. trainable bias: dL/dbias = output_grad
    for (size_t i = 0; i < mOutputDim; ++i) {
        bias_grad[i] += output_grad[i];
    }
    
    // Gradient w.r.t. input (if needed)
    // dL/dx = sum_i(output_grad[i] * sum_j(W_train[i,j] * d(features[j])/dx))
    // d(features[j])/dx = -sin(W_fixed[j] * x + b_fixed[j]) * W_fixed[j]
    if (input_grad != nullptr) {
        // Compute feature gradients: dL/d(features[j]) = sum_i(output_grad[i] * W_train[i,j])
        std::fill(mFeatureGradBuffer.begin(), mFeatureGradBuffer.end(), 0.0f);
        for (int j = 0; j < mNumFeatures; ++j) {
            for (size_t i = 0; i < mOutputDim; ++i) {
                mFeatureGradBuffer.data()[j] += output_grad[i] * mTrainableWeights.data()[i * mNumFeatures + j];
            }
        }
        
        // Compute input gradient
        std::fill(input_grad, input_grad + mInputDim, 0.0f);
        for (int j = 0; j < mNumFeatures; ++j) {
            // d(features[j])/dx = -sin(dot_j) * W_fixed[j]
            float sinVal = std::sin(mFeatureBuffer.data()[j]);  // sin(W_fixed[j] * x + b_fixed[j])
            const float* w_fixed = mFixedWeights.data() + j * mInputDim;
            float featureGrad = mFeatureGradBuffer.data()[j];
            
            for (size_t k = 0; k < mInputDim; ++k) {
                input_grad[k] -= featureGrad * sinVal * w_fixed[k];
            }
        }
    }
}

void RFFLayer::ZeroGradients()
{
    std::fill(mWeightsGradient.begin(), mWeightsGradient.end(), 0.0f);
    std::fill(mBiasGradient.begin(), mBiasGradient.end(), 0.0f);
}

void RFFLayer::ScaleGradients(float scale)
{
    ScaleVector_AVX2(mWeightsGradient.data(), scale, mWeightsGradient.size());
    ScaleVector_AVX2(mBiasGradient.data(), scale, mBiasGradient.size());
}
