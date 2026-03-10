#pragma once

/**
 * @file EigenUtils.h
 * @brief Optimized linear algebra utilities using Eigen for RL training
 * 
 * Provides zero-copy Eigen views over aligned memory for maximum performance.
 * Uses Eigen's GEMM operations for batched neural network forward passes.
 */

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cstdint>
#include <cstring>

#include "AlignedAllocator.h"
#include "NeuralMath.h"

// Enable AVX2/FMA vectorization in Eigen
#ifndef EIGEN_ENABLE_AVX2
#define EIGEN_ENABLE_AVX2
#endif

// Eigen configuration for maximum performance
#define EIGEN_DONT_VECTORIZE 0
#define EIGEN_FAST_MATH 1

namespace EigenUtils {

/**
 * @brief Create a const Eigen map over aligned float data
 * Zero-copy view - data must remain valid during use
 */
inline Eigen::Map<const Eigen::MatrixXf> MapConst(const float* data, int rows, int cols) {
    return Eigen::Map<const Eigen::MatrixXf>(data, rows, cols);
}

/**
 * @brief Create a mutable Eigen map over aligned float data
 * Zero-copy view - data must be 32-byte aligned
 */
inline Eigen::Map<Eigen::MatrixXf> Map(float* data, int rows, int cols) {
    return Eigen::Map<Eigen::MatrixXf>(data, rows, cols);
}

/**
 * @brief Batched matrix multiplication: C = A * B
 * Uses Eigen's optimized GEMM with AVX2/FMA
 * 
 * @param A Input matrix [M x K]
 * @param B Input matrix [K x N]
 * @param C Output matrix [M x N]
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B
 * @param N Number of columns in B and C
 */
inline void MatMul(const float* A, const float* B, float* C, int M, int K, int N) {
    auto a = MapConst(A, M, K);
    auto b = MapConst(B, K, N);
    auto c = Map(C, M, N);
    
    // Eigen's GEMM is highly optimized with AVX2/FMA
    c.noalias() = a * b;
}

/**
 * @brief Batched matrix-vector multiplication for multiple inputs
 * Computes Y = X * W^T + b for batch of inputs
 * 
 * @param X Input batch [batchSize x inputDim]
 * @param W Weight matrix [outputDim x inputDim]
 * @param b Bias vector [outputDim]
 * @param Y Output batch [batchSize x outputDim]
 * @param batchSize Number of samples in batch
 * @param inputDim Input dimension
 * @param outputDim Output dimension
 */
inline void FullyConnectedBatch(const float* X, const float* W, const float* b,
                                 float* Y, int batchSize, int inputDim, int outputDim) {
    auto x = MapConst(X, batchSize, inputDim);
    auto w = MapConst(W, outputDim, inputDim);
    auto y = Map(Y, batchSize, outputDim);
    
    // Y = X * W^T
    y.noalias() = x * w.transpose();
    
    // Add bias to each row
    if (b) {
        auto biasVec = Eigen::Map<const Eigen::VectorXf>(b, outputDim);
        y.rowwise() += biasVec.transpose();
    }
}

/**
 * @brief Batched element-wise activation: MoLU (Modified Leaky ReLU)
 * MoLU(x) = x if x > 0 else alpha * x (alpha = 0.01)
 * 
 * @param data Input/output data [batchSize * dim]
 * @param batchSize Number of samples
 * @param dim Dimension per sample
 */
inline void MoLUActivationBatch(float* data, int batchSize, int dim) {
    auto mat = Map(data, batchSize, dim);
    const float alpha = 0.01f;
    
    // Vectorized MoLU using Eigen
    mat = (mat.array() > 0).select(mat, alpha * mat);
}

/**
 * @brief Batched element-wise Tanh activation
 * 
 * @param data Input/output data [batchSize * dim]
 * @param batchSize Number of samples
 * @param dim Dimension per sample
 */
inline void TanhActivationBatch(float* data, int batchSize, int dim) {
    auto mat = Map(data, batchSize, dim);
    mat = mat.array().tanh();
}

/**
 * @brief Batched element-wise ReLU activation
 * 
 * @param data Input/output data [batchSize * dim]
 * @param batchSize Number of samples
 * @param dim Dimension per sample
 */
inline void ReLUActivationBatch(float* data, int batchSize, int dim) {
    auto mat = Map(data, batchSize, dim);
    mat = mat.cwiseMax(0.0f);
}

/**
 * @brief Batched element-wise Sigmoid activation
 * 
 * @param data Input/output data [batchSize * dim]
 * @param batchSize Number of samples
 * @param dim Dimension per sample
 */
inline void SigmoidActivationBatch(float* data, int batchSize, int dim) {
    auto mat = Map(data, batchSize, dim);
    mat = 1.0f / (1.0f + (-mat.array()).exp());
}

/**
 * @brief Add Gaussian noise to batch with clipping
 * 
 * @param data Input/output data [batchSize * dim]
 * @param batchSize Number of samples
 * @param dim Dimension per sample
 * @param noiseStd Standard deviation of noise
 * @param clipMin Minimum clip value
 * @param clipMax Maximum clip value
 * @param rng Random number generator
 */
template<typename Rng>
inline void AddGaussianNoiseClipped(float* data, int batchSize, int dim,
                                     float noiseStd, float clipMin, float clipMax, Rng& rng) {
    auto mat = Map(data, batchSize, dim);
    
    std::normal_distribution<float> dist(0.0f, noiseStd);
    
    for (int i = 0; i < batchSize * dim; ++i) {
        float noise = dist(rng);
        data[i] = std::clamp(data[i] + noise, clipMin, clipMax);
    }
}

/**
 * @brief Vectorized noise generation for entire batch
 * More efficient than element-wise generation
 * 
 * @param noise Output noise array [batchSize * dim]
 * @param batchSize Number of samples
 * @param dim Dimension per sample
 * @param std Standard deviation
 * @param rng Random number generator
 */
template<typename Rng>
inline void GenerateGaussianNoise(float* noise, int batchSize, int dim, float std, Rng& rng) {
    int totalSize = batchSize * dim;
    
    // Generate all noise at once using vectorized approach
    std::normal_distribution<float> dist(0.0f, std);
    for (int i = 0; i < totalSize; ++i) {
        noise[i] = dist(rng);
    }
}

/**
 * @brief Batched element-wise minimum of two Q-value predictions
 * Implements clipped double Q-learning
 * 
 * @param Q1 First Q-value predictions [batchSize x numOutputs]
 * @param Q2 Second Q-value predictions [batchSize x numOutputs]
 * @param minQ Output: minimum Q values [batchSize]
 * @param batchSize Number of samples
 * @param numOutputs Number of Q outputs per sample
 */
inline void MinQValues(const float* Q1, const float* Q2, float* minQ, int batchSize, int numOutputs) {
    auto q1 = MapConst(Q1, batchSize, numOutputs);
    auto q2 = MapConst(Q2, batchSize, numOutputs);
    auto out = Map(minQ, batchSize, 1);
    
    // Take minimum of first output (main Q value)
    for (int i = 0; i < batchSize; ++i) {
        minQ[i] = std::min(q1(i, 0), q2(i, 0));
    }
}

/**
 * @brief Compute TD targets: targetQ = reward + gamma * (1 - done) * minQ
 * Fully vectorized computation
 * 
 * @param rewards Reward array [batchSize]
 * @param minQ Minimum Q values [batchSize]
 * @param dones Done flags [batchSize]
 * @param targetQ Output: target Q values [batchSize]
 * @param batchSize Number of samples
 * @param gamma Discount factor
 */
inline void ComputeTDTargets(const float* rewards, const float* minQ, const float* dones,
                              float* targetQ, int batchSize, float gamma) {
    auto r = MapConst(rewards, batchSize, 1);
    auto mq = MapConst(minQ, batchSize, 1);
    auto d = MapConst(dones, batchSize, 1);
    auto out = Map(targetQ, batchSize, 1);
    
    // Vectorized: targetQ = reward + gamma * (1 - done) * minQ
    out = r.array() + gamma * (1.0f - d.array()) * mq.array();
}

/**
 * @brief Batched memory copy with SIMD optimization
 * Uses aligned copy when possible
 * 
 * @param dst Destination buffer
 * @param src Source buffer
 * @param size Number of floats to copy
 */
inline void BatchedMemcpy(float* dst, const float* src, int size) {
    // For large copies, Eigen's vectorized operations help
    if (size >= 8) {
        auto d = Map(dst, 1, size);
        auto s = MapConst(src, 1, size);
        d = s;
    } else {
        std::memcpy(dst, src, size * sizeof(float));
    }
}

/**
 * @brief Batched gather operation: output[i] = input[indices[i]]
 * For replay buffer sampling
 * 
 * @param input Input data [maxSize x dim]
 * @param indices Index array [batchSize]
 * @param output Output data [batchSize x dim]
 * @param batchSize Number of samples to gather
 * @param dim Dimension per sample
 * @param maxSize Size of input buffer
 */
inline void BatchedGather(const float* input, const size_t* indices, float* output,
                          int batchSize, int dim, size_t maxSize) {
    for (int i = 0; i < batchSize; ++i) {
        size_t idx = indices[i] % maxSize;
        std::memcpy(output + i * dim, input + idx * dim, dim * sizeof(float));
    }
}

/**
 * @brief Soft update of target network weights: target = (1 - tau) * target + tau * source
 * Vectorized interpolation
 * 
 * @param target Target weights (modified in place)
 * @param source Source weights
 * @param size Number of weights
 * @param tau Interpolation factor (typically 0.005)
 */
inline void SoftUpdate(float* target, const float* source, int size, float tau) {
    auto t = Map(target, size, 1);
    auto s = MapConst(source, size, 1);
    
    // t = (1 - tau) * t + tau * s
    t = (1.0f - tau) * t + tau * s;
}

/**
 * @brief Compute MSE loss: loss = mean((pred - target)^2)
 * 
 * @param pred Predictions [batchSize x dim]
 * @param target Targets [batchSize x dim]
 * @param batchSize Number of samples
 * @param dim Dimension per sample
 * @return float MSE loss value
 */
inline float ComputeMSELoss(const float* pred, const float* target, int batchSize, int dim) {
    auto p = MapConst(pred, batchSize, dim);
    auto t = MapConst(target, batchSize, dim);
    
    auto diff = p - t;
    return (diff.array().square().sum()) / (batchSize * dim);
}

/**
 * @brief Row-major to column-major transpose for batched data
 * Optimizes memory layout for Eigen operations
 * 
 * @param input Input in row-major [batchSize x dim]
 * @param output Output in column-major [dim x batchSize]
 * @param batchSize Number of samples
 * @param dim Dimension per sample
 */
inline void TransposeBatch(const float* input, float* output, int batchSize, int dim) {
    auto in = MapConst(input, batchSize, dim);
    auto out = Map(output, dim, batchSize);
    
    out = in.transpose();
}

/**
 * @brief Aligned buffer for Eigen operations
 * Ensures 32-byte alignment for AVX2
 */
template<typename T>
class AlignedBuffer {
public:
    AlignedBuffer() : mData(nullptr), mSize(0) {}
    
    explicit AlignedBuffer(size_t size) : mSize(size) {
        mData = static_cast<T*>(_mm_malloc(size * sizeof(T), 32));
    }
    
    ~AlignedBuffer() {
        if (mData) _mm_free(mData);
    }
    
    // Move semantics
    AlignedBuffer(AlignedBuffer&& other) noexcept 
        : mData(other.mData), mSize(other.mSize) {
        other.mData = nullptr;
        other.mSize = 0;
    }
    
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (mData) _mm_free(mData);
            mData = other.mData;
            mSize = other.mSize;
            other.mData = nullptr;
            other.mSize = 0;
        }
        return *this;
    }
    
    // Disable copy
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    
    void resize(size_t newSize) {
        if (mData) _mm_free(mData);
        mSize = newSize;
        mData = static_cast<T*>(_mm_malloc(newSize * sizeof(T), 32));
    }
    
    T* data() { return mData; }
    const T* data() const { return mData; }
    size_t size() const { return mSize; }
    
    T& operator[](size_t idx) { return mData[idx]; }
    const T& operator[](size_t idx) const { return mData[idx]; }
    
private:
    T* mData;
    size_t mSize;
};

/**
 * @brief Pre-allocated workspace for batched operations
 * Reduces allocation overhead in training loop
 */
struct BatchWorkspace {
    AlignedBuffer<float> tempBuffer1;
    AlignedBuffer<float> tempBuffer2;
    AlignedBuffer<float> noiseBuffer;
    AlignedBuffer<size_t> indexBuffer;
    
    void resize(int batchSize, int maxDim) {
        tempBuffer1.resize(batchSize * maxDim);
        tempBuffer2.resize(batchSize * maxDim);
        noiseBuffer.resize(batchSize * maxDim);
        indexBuffer.resize(batchSize);
    }
};

} // namespace EigenUtils
