#include "RFFLayer.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "src/EigenUtils.h"

void RFFLayer::Init(size_t inputDim, size_t outputDim, const RFFConfig& config, std::mt19937& rng)
{
    fprintf(stderr, "[RFFLayer::Init] %zu -> %zu, features=%d, sigma=%.2f\n",
            inputDim, outputDim, config.num_features, config.sigma);
    fflush(stderr);

    mInputDim = inputDim;
    mOutputDim = outputDim;
    mNumFeatures = config.num_features;
    mSigma = config.sigma;

    // Initialize FIXED weights from N(0, 1/sigma^2)
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

    // Initialize TRAINABLE parameters (Xavier/He initialization)
    size_t trainWeightsSize = mOutputDim * static_cast<size_t>(mNumFeatures);
    mTrainableWeights.resize(trainWeightsSize);
    
    // Padded bias for AVX2 alignment
    size_t paddedOutputDim = GetAVX2PaddedSize(mOutputDim);
    mTrainableBias.resize(paddedOutputDim, 0.0f);
    
    float limit = std::sqrt(6.0f / (static_cast<float>(mNumFeatures) + static_cast<float>(mOutputDim)));
    std::uniform_real_distribution<float> weightDist(-limit, limit);
    for (auto& w : mTrainableWeights) w = weightDist(rng);

    // Initialize gradients to zero
    mWeightsGradient.resize(trainWeightsSize, 0.0f);
    mBiasGradient.resize(paddedOutputDim, 0.0f);

    fprintf(stderr, "[RFFLayer::Init] Complete. Bias size: %zu (padded from %zu)\n", paddedOutputDim, mOutputDim);
    fflush(stderr);
}

void RFFLayer::ComputeFeatures(const float* input, float* features, float* sin_features)
{
    const size_t simdWidth = 8;
    
    for (int i = 0; i < mNumFeatures; ++i) {
        float dotProduct = mFixedBiases[i];
        const float* weights = mFixedWeights.data() + i * mInputDim;
        
        size_t j = 0;
        __m256 sum_vec = _mm256_setzero_ps();
        for (; j + simdWidth <= mInputDim; j += simdWidth) {
            __m256 w = _mm256_loadu_ps(weights + j);
            __m256 x = _mm256_loadu_ps(input + j);
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(w, x));
        }
        
        alignas(32) float temp[8];
        _mm256_store_ps(temp, sum_vec);
        for (int k = 0; k < 8; ++k) dotProduct += temp[k];
        
        for (; j < mInputDim; ++j) dotProduct += weights[j] * input[j];
        
        features[i] = std::cos(dotProduct);
        if (sin_features) sin_features[i] = std::sin(dotProduct);
    }
}

void RFFLayer::Forward(const float* input, float* output, float* featureBuffer)
{
    ComputeFeatures(input, featureBuffer);
    
    const size_t simdWidth = 8;
    for (size_t i = 0; i < mOutputDim; ++i) {
        float val = mTrainableBias[i];
        const float* weights = mTrainableWeights.data() + i * mNumFeatures;
        
        size_t j = 0;
        __m256 sum_vec = _mm256_setzero_ps();
        for (; j + simdWidth <= static_cast<size_t>(mNumFeatures); j += simdWidth) {
            __m256 w = _mm256_loadu_ps(weights + j);
            __m256 f = _mm256_loadu_ps(featureBuffer + j);
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(w, f));
        }
        
        alignas(32) float temp[8];
        _mm256_store_ps(temp, sum_vec);
        for (int k = 0; k < 8; ++k) val += temp[k];
        
        for (; j < static_cast<size_t>(mNumFeatures); ++j) val += weights[j] * featureBuffer[j];
        
        output[i] = val;
    }
}

void RFFLayer::ForwardBatch(const float* input, float* output, int batchSize, float* featureBuffer)
{
    ForwardBatchEigen(input, output, batchSize, featureBuffer);
}

void RFFLayer::ForwardBatchEigen(const float* input, float* output, int batchSize, float* featureBuffer)
{
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrixXf;
    
    Eigen::Map<const RowMajorMatrixXf> inputMap(input, batchSize, mInputDim);
    Eigen::Map<const Eigen::MatrixXf> W_fixedMap(mFixedWeights.data(), mNumFeatures, mInputDim);
    Eigen::Map<const Eigen::VectorXf> b_fixedMap(mFixedBiases.data(), mNumFeatures);
    
    Eigen::MatrixXf features = (inputMap * W_fixedMap.transpose()).rowwise() + b_fixedMap.transpose();
    features = features.array().cos();
    
    if (featureBuffer) {
        std::memcpy(featureBuffer, features.data(), batchSize * mNumFeatures * sizeof(float));
    }
    
    Eigen::Map<const Eigen::MatrixXf> W_trainMap(mTrainableWeights.data(), mOutputDim, mNumFeatures);
    Eigen::Map<const Eigen::VectorXf> biasMap(mTrainableBias.data(), mOutputDim);
    
    RowMajorMatrixXf outputMap = (features * W_trainMap.transpose()).rowwise() + biasMap.transpose();
    std::memcpy(output, outputMap.data(), batchSize * mOutputDim * sizeof(float));
}

Eigen::MatrixXf RFFLayer::ForwardEigen(const Eigen::MatrixXf& input) const
{
    int batchSize = static_cast<int>(input.rows());
    Eigen::Map<const Eigen::MatrixXf> W_fixedMap(mFixedWeights.data(), mNumFeatures, mInputDim);
    Eigen::Map<const Eigen::VectorXf> b_fixedMap(mFixedBiases.data(), mNumFeatures);
    
    Eigen::MatrixXf features = (input * W_fixedMap.transpose()).rowwise() + b_fixedMap.transpose();
    features = features.array().cos();
    
    Eigen::Map<const Eigen::MatrixXf> W_trainMap(mTrainableWeights.data(), mOutputDim, mNumFeatures);
    Eigen::Map<const Eigen::VectorXf> biasMap(mTrainableBias.data(), mOutputDim);
    
    return (features * W_trainMap.transpose()).rowwise() + biasMap.transpose();
}

void RFFLayer::Backward(const float* input, const float* output_grad, float* input_grad,
                        float* weights_grad, float* bias_grad,
                        const float* featureBuffer, const float* sinFeatureBuffer, float* featureGradBuffer)
{
    // SAFETY: weights_grad can be nullptr when backpropagating through critic to actor
    if (weights_grad) {
        for (size_t i = 0; i < mOutputDim; ++i) {
            float grad = output_grad[i];
            float* w_grad = weights_grad + i * mNumFeatures;
            for (int j = 0; j < mNumFeatures; ++j) {
                w_grad[j] += grad * featureBuffer[j];
            }
        }
    }
    
    if (bias_grad) {
        for (size_t i = 0; i < mOutputDim; ++i) {
            bias_grad[i] += output_grad[i];
        }
    }
    
    if (input_grad != nullptr) {
        std::fill(featureGradBuffer, featureGradBuffer + mNumFeatures, 0.0f);
        for (int j = 0; j < mNumFeatures; ++j) {
            for (size_t i = 0; i < mOutputDim; ++i) {
                featureGradBuffer[j] += output_grad[i] * mTrainableWeights.data()[i * mNumFeatures + j];
            }
        }
        
        std::fill(input_grad, input_grad + mInputDim, 0.0f);
        for (int j = 0; j < mNumFeatures; ++j) {
            float sinVal = sinFeatureBuffer[j];
            const float* w_fixed = mFixedWeights.data() + j * mInputDim;
            float featureGrad = featureGradBuffer[j];
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
