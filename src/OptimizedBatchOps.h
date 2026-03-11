#pragma once

#include <immintrin.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cstring>
#include <omp.h>

#include "AlignedAllocator.h"

// ============================================================================
// HIGH-PERFORMANCE BATCHED NEURAL NETWORK OPERATIONS
// Optimized for SPS-critical RL training pipeline
// ============================================================================

namespace opt {

// Cache line size for alignment
constexpr size_t CACHE_LINE_SIZE = 64;

// Template for cache-aligned storage
template<typename T>
struct alignas(64) CacheAligned {
    T data;
    char padding[CACHE_LINE_SIZE - (sizeof(T) % CACHE_LINE_SIZE)];
};

// ============================================================================
// BATCHED MATRIX MULTIPLICATION (Eigen-optimized with AVX2)
// ============================================================================

/**
 * Batched GEMM: Y = X * W + b
 * Uses Eigen with explicit vectorization and cache-friendly access patterns
 * 
 * @param X Input matrix [batch_size, input_dim]
 * @param W Weight matrix [input_dim, output_dim]
 * @param b Bias vector [output_dim]
 * @param Y Output matrix [batch_size, output_dim]
 * @param batch_size Number of samples in batch
 * @param input_dim Input feature dimension
 * @param output_dim Output feature dimension
 */
inline void BatchedGEMM_Eigen(const float* X, const float* W, const float* b,
                               float* Y, int batch_size, int input_dim, int output_dim) {
    // Map matrices with explicit alignment hints for AVX2
    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
    Eigen::Map<const MatrixXfRow> X_map(X, batch_size, input_dim);
    Eigen::Map<const MatrixXfRow> W_map(W, input_dim, output_dim);
    Eigen::Map<MatrixXfRow> Y_map(Y, batch_size, output_dim);
    
    // Eigen auto-vectorizes with AVX2 when compiled with -mavx2
    Y_map.noalias() = X_map * W_map;
    
    // Add bias (vectorized)
    #pragma omp simd aligned(Y: 32)
    for (int i = 0; i < batch_size * output_dim; i++) {
        Y[i] += b[i % output_dim];
    }
}

/**
 * Batched GEMM without bias (for pre-allocated output)
 */
inline void BatchedGEMM_NoBias(const float* X, const float* W, float* Y,
                                int batch_size, int input_dim, int output_dim) {
    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
    Eigen::Map<const MatrixXfRow> X_map(X, batch_size, input_dim);
    Eigen::Map<const MatrixXfRow> W_map(W, input_dim, output_dim);
    Eigen::Map<MatrixXfRow> Y_map(Y, batch_size, output_dim);
    
    Y_map.noalias() = X_map * W_map;
}

// ============================================================================
// BATCHED ACTIVATION FUNCTIONS (AVX2-optimized)
// ============================================================================

/**
 * Batched MoLU (Modulated Leaky Unit) with forward caching
 * f(x) = beta * x if x > 0, else alpha * (exp(x) - 1)
 *
 * @param X Input/output array
 * @param cachedInput Cache for backward pass (can be same as X)
 * @param size Number of elements
 */
inline void BatchedMoLU_WithCache(float* X, float* cachedInput, int size) {
    constexpr float alpha = 0.1f;
    constexpr float beta = 1.0f;

    // Copy input to cache first (needed for backward pass)
    std::memcpy(cachedInput, X, size * sizeof(float));

    // Use scalar implementation for all elements since exp requires it
    for (int i = 0; i < size; i++) {
        float x = X[i];
        X[i] = (x > 0) ? (beta * x) : (alpha * (std::exp(x) - 1.0f));
    }
}

/**
 * Batched MoLU without caching (faster, use when backward not needed)
 * Uses scalar fallback for exp since AVX2 doesn't have native exp instruction
 */
inline void BatchedMoLU(float* X, int size) {
    constexpr float alpha = 0.1f;
    constexpr float beta = 1.0f;

    // Use scalar implementation for all elements since exp requires it
    for (int i = 0; i < size; i++) {
        float x = X[i];
        X[i] = (x > 0) ? (beta * x) : (alpha * (std::exp(x) - 1.0f));
    }
}

/**
 * Batched backward MoLU
 * dL/dx = beta * dL/dy if x > 0, else alpha * exp(x) * dL/dy
 */
inline void BackwardMoLU_Batched(float* grad, const float* cachedInput, int size) {
    constexpr float alpha = 0.1f;
    constexpr float beta = 1.0f;

    // Use scalar implementation for all elements since exp requires it
    for (int i = 0; i < size; i++) {
        float x = cachedInput[i];
        if (x > 0) {
            grad[i] *= beta;
        } else {
            grad[i] *= alpha * std::exp(x);
        }
    }
}

/**
 * Batched Tanh with AVX2
 * Uses higher-order Pade approximation for numerical stability
 * tanh(x) ≈ x * (105 + 10*x^2) / (105 + 45*x^2 + x^4) for |x| < 4
 * Saturates to ±1 for |x| >= 4
 */
inline void BatchedTanh_AVX2(float* X, size_t size) {
    const size_t simd_size = size - (size % 8);

    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 minusOne = _mm256_set1_ps(-1.0f);
    const __m256 four = _mm256_set1_ps(4.0f);
    const __m256 p0 = _mm256_set1_ps(105.0f);
    const __m256 p1 = _mm256_set1_ps(10.0f);
    const __m256 q0 = _mm256_set1_ps(105.0f);
    const __m256 q1 = _mm256_set1_ps(45.0f);

    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 x = _mm256_loadu_ps(X + i);
        
        // xc = clamp(x, -4, 4) for the rational part
        __m256 xc = _mm256_min_ps(x, four);
        xc = _mm256_max_ps(xc, _mm256_sub_ps(_mm256_setzero_ps(), four));

        __m256 xc2 = _mm256_mul_ps(xc, xc);
        __m256 xc4 = _mm256_mul_ps(xc2, xc2);

        // num = xc * (105 + 10 * xc^2)
        __m256 num = _mm256_fmadd_ps(p1, xc2, p0);
        num = _mm256_mul_ps(xc, num);

        // den = 105 + 45 * xc^2 + xc^4
        __m256 den = _mm256_fmadd_ps(q1, xc2, q0);
        den = _mm256_add_ps(den, xc4);

        // Fast reciprocal with Newton-Raphson refinement
        __m256 rcp = _mm256_rcp_ps(den);
        rcp = _mm256_mul_ps(rcp, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(den, rcp)));

        __m256 th = _mm256_mul_ps(num, rcp);

        // For |x| > 4, use ±1
        __m256 isHi = _mm256_cmp_ps(x, four, _CMP_GT_OQ);
        __m256 isLo = _mm256_cmp_ps(x, _mm256_sub_ps(_mm256_setzero_ps(), four), _CMP_LT_OQ);
        th = _mm256_blendv_ps(th, one, isHi);
        th = _mm256_blendv_ps(th, minusOne, isLo);

        _mm256_storeu_ps(X + i, th);
    }

    for (size_t i = simd_size; i < size; i++) {
        X[i] = std::tanh(X[i]);
    }
}

/**
 * Batched ReLU
 */
inline void BatchedReLU_AVX2(float* X, int size) {
    const int simd_size = size - (size % 8);
    
    __m256 v_zero = _mm256_setzero_ps();
    
    for (int i = 0; i < simd_size; i += 8) {
        __m256 x = _mm256_loadu_ps(X + i);
        __m256 result = _mm256_max_ps(x, v_zero);
        _mm256_storeu_ps(X + i, result);
    }
    
    for (int i = simd_size; i < size; i++) {
        X[i] = std::max(0.0f, X[i]);
    }
}

/**
 * Batched Sigmoid
 * Uses scalar fallback for exp since AVX2 doesn't have native exp instruction
 */
inline void BatchedSigmoid_AVX2(float* X, int size) {
    // Use scalar implementation for all elements since exp requires it
    for (int i = 0; i < size; i++) {
        X[i] = 1.0f / (1.0f + std::exp(-X[i]));
    }
}

// ============================================================================
// BATCHED LAYER NORMALIZATION
// ============================================================================

/**
 * Batched Layer Normalization with gamma/beta
 * Optimized for SoA layout
 */
inline void BatchedLayerNorm_GammaBeta(const float* X, float* Y, const float* gamma,
                                        const float* beta, int batch_size, int dim,
                                        float epsilon = 1e-5f) {
    const float inv_dim = 1.0f / dim;
    
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        const float* x_row = X + i * dim;
        float* y_row = Y + i * dim;
        
        // Compute mean
        float mean = 0.0f;
        #pragma omp simd reduction(+:mean) aligned(x_row: 32)
        for (int j = 0; j < dim; j++) {
            mean += x_row[j];
        }
        mean *= inv_dim;
        
        // Compute variance
        float var = 0.0f;
        #pragma omp simd reduction(+:var) aligned(x_row: 32)
        for (int j = 0; j < dim; j++) {
            float diff = x_row[j] - mean;
            var += diff * diff;
        }
        var *= inv_dim;
        
        // Compute inverse std
        float inv_std = 1.0f / std::sqrt(var + epsilon);
        
        // Normalize and apply gamma/beta
        #pragma omp simd aligned(x_row, y_row, gamma, beta: 32)
        for (int j = 0; j < dim; j++) {
            y_row[j] = gamma[j] * (x_row[j] - mean) * inv_std + beta[j];
        }
    }
}

/**
 * Simplified batched layer norm (no gamma/beta, for inference)
 */
inline void BatchedLayerNorm_Simple(const float* X, float* Y, int batch_size, int dim,
                                     float epsilon = 1e-5f) {
    const float inv_dim = 1.0f / dim;
    
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        const float* x_row = X + i * dim;
        float* y_row = Y + i * dim;
        
        // Compute mean
        float mean = 0.0f;
        for (int j = 0; j < dim; j++) mean += x_row[j];
        mean *= inv_dim;
        
        // Compute variance
        float var = 0.0f;
        for (int j = 0; j < dim; j++) {
            float diff = x_row[j] - mean;
            var += diff * diff;
        }
        var *= inv_dim;
        
        // Normalize
        float inv_std = 1.0f / std::sqrt(var + epsilon);
        for (int j = 0; j < dim; j++) {
            y_row[j] = (x_row[j] - mean) * inv_std;
        }
    }
}

// ============================================================================
// VECTOR OPERATIONS (AVX2-optimized)
// ============================================================================

/**
 * Fused multiply-add: dst = dst + a * b
 */
inline void FMA_Batched(float* dst, const float* a, const float* b, int size) {
    const int simd_size = size - (size % 8);
    
    for (int i = 0; i < simd_size; i += 8) {
        __m256 v_dst = _mm256_loadu_ps(dst + i);
        __m256 v_a = _mm256_loadu_ps(a + i);
        __m256 v_b = _mm256_loadu_ps(b + i);
        __m256 v_fma = _mm256_fmadd_ps(v_a, v_b, v_dst);
        _mm256_storeu_ps(dst + i, v_fma);
    }
    
    for (int i = simd_size; i < size; i++) {
        dst[i] += a[i] * b[i];
    }
}

/**
 * Scale vector: dst = src * scale
 */
inline void ScaleVector_AVX2(const float* src, float* dst, float scale, int size) {
    const int simd_size = size - (size % 8);
    
    __m256 v_scale = _mm256_set1_ps(scale);
    
    for (int i = 0; i < simd_size; i += 8) {
        __m256 v_src = _mm256_loadu_ps(src + i);
        __m256 v_dst = _mm256_mul_ps(v_src, v_scale);
        _mm256_storeu_ps(dst + i, v_dst);
    }
    
    for (int i = simd_size; i < size; i++) {
        dst[i] = src[i] * scale;
    }
}

/**
 * Add vectors: dst = a + b
 */
inline void AddVectors_AVX2(const float* a, const float* b, float* dst, int size) {
    const int simd_size = size - (size % 8);
    
    for (int i = 0; i < simd_size; i += 8) {
        __m256 v_a = _mm256_loadu_ps(a + i);
        __m256 v_b = _mm256_loadu_ps(b + i);
        __m256 v_dst = _mm256_add_ps(v_a, v_b);
        _mm256_storeu_ps(dst + i, v_dst);
    }
    
    for (int i = simd_size; i < size; i++) {
        dst[i] = a[i] + b[i];
    }
}

/**
 * Subtract vectors: dst = a - b
 */
inline void SubVectors_AVX2(const float* a, const float* b, float* dst, int size) {
    const int simd_size = size - (size % 8);
    
    for (int i = 0; i < simd_size; i += 8) {
        __m256 v_a = _mm256_loadu_ps(a + i);
        __m256 v_b = _mm256_loadu_ps(b + i);
        __m256 v_dst = _mm256_sub_ps(v_a, v_b);
        _mm256_storeu_ps(dst + i, v_dst);
    }
    
    for (int i = simd_size; i < size; i++) {
        dst[i] = a[i] - b[i];
    }
}

// ============================================================================
// SOA (STRUCTURE OF ARRAYS) UTILITIES
// ============================================================================

/**
 * Convert AoS to SoA layout (batch transformation)
 * Input: [batch][features] (row-major)
 * Output: [features][batch] (column-major, SoA)
 */
inline void AoS_to_SoA(const float* aos_input, float* soa_output,
                       int batch_size, int feature_dim) {
    #pragma omp parallel for
    for (int f = 0; f < feature_dim; f++) {
        for (int b = 0; b < batch_size; b++) {
            soa_output[f * batch_size + b] = aos_input[b * feature_dim + f];
        }
    }
}

/**
 * Convert SoA to AoS layout
 */
inline void SoA_to_AoS(const float* soa_input, float* aos_output,
                       int batch_size, int feature_dim) {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < feature_dim; f++) {
            aos_output[b * feature_dim + f] = soa_input[f * batch_size + b];
        }
    }
}

// ============================================================================
// PREFETCHING UTILITIES
// ============================================================================

/**
 * Software prefetching for sequential access
 * Prefetch data ahead of current position to hide memory latency
 */
inline void PrefetchAhead(const float* data, size_t ahead_elements, size_t count) {
    constexpr size_t PREFETCH_DISTANCE = 256;  // ~1000 cycles at 4GHz
    
    for (size_t i = 0; i < count; i += 8) {
        if (i + PREFETCH_DISTANCE < count) {
            _mm_prefetch(data + i + PREFETCH_DISTANCE, _MM_HINT_T0);
        }
    }
}

/**
 * Prefetch for matrix row access
 */
inline void PrefetchMatrixRow(const float* row, size_t next_row_offset) {
    _mm_prefetch(row, _MM_HINT_T0);
    _mm_prefetch(row + 64, _MM_HINT_T0);
    _mm_prefetch(row + 128, _MM_HINT_T0);
}

} // namespace opt
