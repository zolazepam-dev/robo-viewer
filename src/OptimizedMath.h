#pragma once

#include <armadillo>
#include <cmath>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>

// Optimized batched neural network operations using Armadillo + Eigen
// Zero-copy operations where possible, AVX2-friendly memory layout

// Initialize Eigen for optimal performance - call once at startup
inline void InitEigenOptimized() {
    // Force Eigen to use single-threading (we manage parallelism ourselves)
    Eigen::setNbThreads(1);
    
    // Enable AVX2/FMA optimizations
    #if defined(__AVX2__) && defined(__FMA__)
        fprintf(stderr, "[OptimizedMath] AVX2/FMA enabled for Eigen\n");
    #endif
    
    #if defined(EIGEN_ENABLE_AVX2)
        fprintf(stderr, "[OptimizedMath] EIGEN_ENABLE_AVX2 defined\n");
    #endif
}

namespace opt {

// Optimized batched matrix multiplication using Eigen with RowMajor layout
// Y = X * W^T + b (for neural network forward pass)
// X: [batch_size, input_dim] - RowMajor
// W: [output_dim, input_dim] - we transpose internally
// b: [output_dim]
// Y: [batch_size, output_dim] - RowMajor
inline void BatchedGEMM_Optimized(const float* X, const float* W, const float* b,
                                   float* Y, int batch_size, int input_dim, int output_dim) {
    // Use RowMajor for better cache locality when processing batches row-wise
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrixXf;
    
    // Zero-copy mapping of input matrices
    Eigen::Map<const RowMajorMatrixXf> X_mat(X, batch_size, input_dim);
    Eigen::Map<const RowMajorMatrixXf> W_mat(W, output_dim, input_dim);
    
    // Compute Y = X * W^T + b
    // RowMajor * RowMajor^T is cache-friendly for batch operations
    RowMajorMatrixXf Y_mat = X_mat * W_mat.transpose();
    
    // Add bias (broadcasting)
    Eigen::Map<const Eigen::VectorXf> b_vec(b, output_dim);
    Y_mat.rowwise() += b_vec.transpose();
    
    // Copy result to output (already in RowMajor layout)
    std::memcpy(Y, Y_mat.data(), batch_size * output_dim * sizeof(float));
}

// In-place batched cos activation using Eigen's vectorized operations
inline void BatchedCos_Inplace(float* X, int size) {
    Eigen::Map<Eigen::VectorXf> X_vec(X, size);
    X_vec = X_vec.array().cos();
}

// Batched matrix multiplication with pre-allocated workspace (zero allocation)
inline void BatchedGEMM_Workspace(
    const float* X, const float* W, const float* b,
    float* Y, float* workspace,
    int batch_size, int input_dim, int output_dim) {
    
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrixXf;
    
    // Use pre-allocated workspace for temporary matrix
    Eigen::Map<RowMajorMatrixXf> Y_mat(workspace, batch_size, output_dim);
    
    Eigen::Map<const RowMajorMatrixXf> X_mat(X, batch_size, input_dim);
    Eigen::Map<const RowMajorMatrixXf> W_mat(W, output_dim, input_dim);
    
    Y_mat.noalias() = X_mat * W_mat.transpose();
    
    // Add bias
    Eigen::Map<const Eigen::VectorXf> b_vec(b, output_dim);
    Y_mat.rowwise() += b_vec.transpose();
    
    // Copy to output
    std::memcpy(Y, workspace, batch_size * output_dim * sizeof(float));
}

// Batched matrix multiplication: Y = X * W + b
// X: [batch_size, input_dim]
// W: [input_dim, output_dim]
// b: [output_dim]
// Y: [batch_size, output_dim]
inline void BatchedGEMM(const float* X, const float* W, const float* b,
                        float* Y, int batch_size, int input_dim, int output_dim) {
    // Wrap input matrices (zero-copy) - use fmat for float
    arma::fmat X_mat(const_cast<float*>(X), input_dim, batch_size, false, true);  // Column-major
    arma::fmat W_mat(const_cast<float*>(W), output_dim, input_dim, false, true);
    
    // Compute Y = W * X (result is [output_dim, batch_size])
    arma::fmat Y_mat = W_mat * X_mat;
    
    // Add bias and transpose to row-major output
    #pragma omp simd
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            Y[i * output_dim + j] = Y_mat(j, i) + b[j];
        }
    }
}

// Batched MoLU activation (Modulated ReLU)
// Optimized with AVX2-friendly access patterns
inline void BatchedMoLU(float* X, int batch_size, int dim) {
    const float alpha = 0.1f;
    const float beta = 1.0f;
    
    #pragma omp simd
    for (int i = 0; i < batch_size * dim; ++i) {
        float x = X[i];
        if (x > 0) {
            X[i] = beta * x;
        } else {
            float exp_x = std::exp(x);
            X[i] = alpha * (exp_x - 1.0f);
        }
    }
}

// Vectorized sigmoid
inline void BatchedSigmoid(const float* X, float* Y, int size) {
    #pragma omp simd
    for (int i = 0; i < size; ++i) {
        Y[i] = 1.0f / (1.0f + std::exp(-X[i]));
    }
}

// Vectorized tanh
inline void BatchedTanh(const float* X, float* Y, int size) {
    #pragma omp simd
    for (int i = 0; i < size; ++i) {
        Y[i] = std::tanh(X[i]);
    }
}

// Batched layer normalization
inline void BatchedLayerNorm(const float* X, float* Y, float* mean, float* var,
                             int batch_size, int dim, float epsilon = 1e-5f) {
    for (int i = 0; i < batch_size; ++i) {
        const float* x_row = X + i * dim;
        float* y_row = Y + i * dim;
        
        // Compute mean
        float m = 0.0f;
        #pragma omp simd reduction(+:m)
        for (int j = 0; j < dim; ++j) {
            m += x_row[j];
        }
        m /= dim;
        mean[i] = m;
        
        // Compute variance
        float v = 0.0f;
        #pragma omp simd reduction(+:v)
        for (int j = 0; j < dim; ++j) {
            float diff = x_row[j] - m;
            v += diff * diff;
        }
        v /= dim;
        var[i] = v;
        
        // Normalize
        float inv_std = 1.0f / std::sqrt(v + epsilon);
        #pragma omp simd
        for (int j = 0; j < dim; ++j) {
            y_row[j] = (x_row[j] - m) * inv_std;
        }
    }
}

// Batched dropout (training mode)
inline void BatchedDropout(float* X, int batch_size, int dim, float drop_prob, float* rng_state) {
    float scale = 1.0f / (1.0f - drop_prob);
    
    #pragma omp simd
    for (int i = 0; i < batch_size * dim; ++i) {
        // Simple LCG random number generator
        *rng_state = std::fmod(*rng_state * 1664525.0f + 1013904223.0f, 4294967296.0f);
        float r = *rng_state / 4294967296.0f;
        
        if (r < drop_prob) {
            X[i] = 0.0f;
        } else {
            X[i] *= scale;
        }
    }
}

// Optimized weight perturbation with Armadillo
inline void PerturbWeights(float* weights, int size, float noise_scale, float* rng_state) {
    arma::fvec w_vec(weights, size, false, true);
    arma::fvec noise = arma::randn<arma::fvec>(size) * noise_scale;
    w_vec += noise;
}

// Batched dot product
inline void BatchedDot(const float* A, const float* B, float* C, int batch_size, int dim) {
    #pragma omp simd
    for (int i = 0; i < batch_size; ++i) {
        float sum = 0.0f;
        const float* a_row = A + i * dim;
        const float* b_row = B + i * dim;
        
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < dim; ++j) {
            sum += a_row[j] * b_row[j];
        }
        C[i] = sum;
    }
}

// Matrix-vector product for single inference
inline void FastFC(const float* X, const float* W, const float* b, float* Y,
                   int input_dim, int output_dim) {
    for (int i = 0; i < output_dim; ++i) {
        float sum = b[i];
        const float* w_row = W + i * input_dim;
        
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < input_dim; ++j) {
            sum += w_row[j] * X[j];
        }
        Y[i] = sum;
    }
}

} // namespace opt