#pragma once

#include <cmath>
#include <random>
#include <immintrin.h>
#include <cpuid.h>

namespace cput::rff {

/**
 * Random Fourier Features Network
 * 
 * Approximates RBF kernel: k(x,y) = exp(-||x-y||²/2σ²)
 * Using: φ(x) = √(2/D) * cos(Wx + b)
 * where W ~ N(0, 1/σ²), b ~ Uniform[0, 2π]
 */

// ============= Initialization =============

/**
 * Initialize RFF weight matrix W ~ N(0, 1/σ²)
 * @param W Output weight matrix [rff_dim x in_dim]
 * @param in_dim Input dimension
 * @param rff_dim RFF feature dimension
 * @param sigma RBF kernel bandwidth parameter
 */
inline void initWeights(float* W, size_t in_dim, size_t rff_dim, float sigma = 1.0f) {
    std::mt19937 gen(42);
    float std_dev = 1.0f / sigma;
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (size_t i = 0; i < rff_dim * in_dim; ++i) {
        W[i] = dist(gen);
    }
}

/**
 * Initialize RFF bias vector b ~ Uniform[0, 2π]
 * @param b Output bias vector [rff_dim]
 * @param rff_dim RFF feature dimension
 */
inline void initBias(float* b, size_t rff_dim) {
    std::mt19937 gen(43);
    std::uniform_real_distribution<float> dist(0.0f, 2.0f * 3.14159265359f);
    
    for (size_t i = 0; i < rff_dim; ++i) {
        b[i] = dist(gen);
    }
}

// ============= SIMD Helper Functions =============

namespace simd {

// AVX2 cosine - use standard lib for accuracy
inline void cos_std(const float* a, float* c, size_t n) {
    size_t i = 0;
    for (; i < n; ++i) {
        c[i] = std::cos(a[i]);
    }
}

// AVX2 sine - use standard lib for accuracy
inline void sin_std(const float* a, float* c, size_t n) {
    size_t i = 0;
    for (; i < n; ++i) {
        c[i] = std::sin(a[i]);
    }
}

// Auto-dispatch
inline void cos(const float* a, float* c, size_t n) {
    // For accuracy, use std::cos (compiler will auto-vectorize)
    cos_std(a, c, n);
}

inline void sin(const float* a, float* c, size_t n) {
    // For accuracy, use std::sin (compiler will auto-vectorize)
    sin_std(a, c, n);
}

} // namespace simd

// ============= Forward Pass =============

/**
 * RFF Forward Pass: φ(x) = √(2/D) * cos(Wx + b)
 * @param x Input vector [in_dim]
 * @param W Weight matrix [rff_dim x in_dim]
 * @param b Bias vector [rff_dim]
 * @param output Output features [rff_dim]
 * @param in_dim Input dimension
 * @param rff_dim RFF feature dimension
 * @param workspace Temporary buffer [rff_dim] for intermediate results
 */
inline void forward(const float* x, const float* W, const float* b, 
                    float* output, size_t in_dim, size_t rff_dim, float* workspace) {
    
    // Step 1: Linear projection z = Wx + b
    for (size_t i = 0; i < rff_dim; ++i) {
        workspace[i] = 0.0f;
        for (size_t j = 0; j < in_dim; ++j) {
            workspace[i] += W[i * in_dim + j] * x[j];
        }
        workspace[i] += b[i];
    }
    
    // Step 2: Cosine activation φ = cos(z)
    simd::cos(workspace, output, rff_dim);
    
    // Step 3: Scale φ *= √(2/D)
    float scale = std::sqrt(2.0f / static_cast<float>(rff_dim));
    
    size_t i = 0;
    for (; i + 7 < rff_dim; i += 8) {
        __m256 vout = _mm256_loadu_ps(output + i);
        __m256 vscale = _mm256_set1_ps(scale);
        vout = _mm256_mul_ps(vout, vscale);
        _mm256_storeu_ps(output + i, vout);
    }
    for (; i < rff_dim; ++i) {
        output[i] *= scale;
    }
}

/**
 * Batch RFF Forward Pass
 * @param X Input matrix [batch_size x in_dim]
 * @param W Weight matrix [rff_dim x in_dim]
 * @param b Bias vector [rff_dim]
 * @param output Output features [batch_size x rff_dim]
 * @param batch_size Number of samples
 * @param in_dim Input dimension
 * @param rff_dim RFF feature dimension
 * @param workspace Temporary buffer [batch_size x rff_dim]
 */
inline void forwardBatch(const float* X, const float* W, const float* b,
                         float* output, size_t batch_size, size_t in_dim, 
                         size_t rff_dim, float* workspace) {
    
    for (size_t batch = 0; batch < batch_size; ++batch) {
        forward(X + batch * in_dim, W, b, 
                output + batch * rff_dim, 
                in_dim, rff_dim, 
                workspace + batch * rff_dim);
    }
}

// ============= Backward Pass =============

/**
 * RFF Backward Pass (compute gradients)
 * 
 * Given dL/dφ (grad_output), compute:
 * - dL/dx (grad_input)
 * - dL/dW (grad_W)
 * - dL/db (grad_b)
 * 
 * @param x Input vector [in_dim]
 * @param W Weight matrix [rff_dim x in_dim]
 * @param b Bias vector [rff_dim]
 * @param grad_output Gradient dL/dφ [rff_dim]
 * @param grad_input Output gradient dL/dx [in_dim]
 * @param grad_W Output gradient dL/dW [rff_dim x in_dim]
 * @param grad_b Output gradient dL/db [rff_dim]
 * @param in_dim Input dimension
 * @param rff_dim RFF feature dimension
 * @param workspace Temporary buffer [2 * rff_dim]
 */
inline void backward(const float* x, const float* W, const float* b,
                     const float* grad_output, float* grad_input, 
                     float* grad_W, float* grad_b,
                     size_t in_dim, size_t rff_dim, float* workspace) {
    
    float* z = workspace;           // [rff_dim] - pre-activation
    float* sin_z = workspace + rff_dim;  // [rff_dim] - sin(z)
    
    // Step 1: Recompute z = Wx + b
    for (size_t i = 0; i < rff_dim; ++i) {
        z[i] = b[i];
        for (size_t j = 0; j < in_dim; ++j) {
            z[i] += W[i * in_dim + j] * x[j];
        }
    }
    
    // Step 2: Compute sin(z) for gradient
    simd::sin(z, sin_z, rff_dim);
    
    // Step 3: Compute dL/dz = dL/dφ * dφ/dz = dL/dφ * (-sin(z)) * √(2/D)
    float scale = std::sqrt(2.0f / static_cast<float>(rff_dim));
    
    size_t i = 0;
    for (; i + 7 < rff_dim; i += 8) {
        __m256 vgrad = _mm256_loadu_ps(grad_output + i);
        __m256 vsin = _mm256_loadu_ps(sin_z + i);
        __m256 vscale = _mm256_set1_ps(scale);
        __m256 vdz = _mm256_mul_ps(_mm256_mul_ps(vgrad, vsin), vscale);
        vdz = _mm256_sub_ps(_mm256_setzero_ps(), vdz);  // negate
        _mm256_storeu_ps(z + i, vdz);  // reuse z for dL/dz
    }
    for (; i < rff_dim; ++i) {
        z[i] = -grad_output[i] * std::sin(z[i]) * scale;
    }
    
    // Step 4: Gradient w.r.t. bias: dL/db = dL/dz
    for (size_t i = 0; i < rff_dim; ++i) {
        grad_b[i] = z[i];
    }
    
    // Step 5: Gradient w.r.t. weights: dL/dW = dL/dz * xᵀ
    for (size_t i = 0; i < rff_dim; ++i) {
        for (size_t j = 0; j < in_dim; ++j) {
            grad_W[i * in_dim + j] = z[i] * x[j];
        }
    }
    
    // Step 6: Gradient w.r.t. input: dL/dx = Wᵀ * dL/dz
    for (size_t j = 0; j < in_dim; ++j) {
        grad_input[j] = 0.0f;
        for (size_t i = 0; i < rff_dim; ++i) {
            grad_input[j] += W[i * in_dim + j] * z[i];
        }
    }
}

/**
 * Batch RFF Backward Pass
 * @param X Input matrix [batch_size x in_dim]
 * @param W Weight matrix [rff_dim x in_dim]
 * @param b Bias vector [rff_dim]
 * @param grad_output Gradient dL/dφ [batch_size x rff_dim]
 * @param grad_input Output gradient dL/dx [batch_size x in_dim]
 * @param grad_W Output gradient dL/dW [rff_dim x in_dim] (accumulated)
 * @param grad_b Output gradient dL/db [rff_dim] (accumulated)
 * @param batch_size Number of samples
 * @param in_dim Input dimension
 * @param rff_dim RFF feature dimension
 * @param workspace Temporary buffer [batch_size * (2 * rff_dim + in_dim)]
 */
inline void backwardBatch(const float* X, const float* W, const float* b,
                          const float* grad_output, float* grad_input,
                          float* grad_W, float* grad_b,
                          size_t batch_size, size_t in_dim, size_t rff_dim,
                          float* workspace) {
    
    // Initialize gradients to zero
    for (size_t i = 0; i < batch_size * in_dim; ++i) grad_input[i] = 0.0f;
    for (size_t i = 0; i < rff_dim * in_dim; ++i) grad_W[i] = 0.0f;
    for (size_t i = 0; i < rff_dim; ++i) grad_b[i] = 0.0f;
    
    // Process each sample
    for (size_t batch = 0; batch < batch_size; ++batch) {
        float* grad_input_b = grad_input + batch * in_dim;
        const float* grad_out_b = grad_output + batch * rff_dim;
        float* ws_b = workspace + batch * (2 * rff_dim + in_dim);
        
        backward(X + batch * in_dim, W, b, grad_out_b,
                 grad_input_b, ws_b, ws_b + rff_dim,
                 in_dim, rff_dim, ws_b + rff_dim * 2);
        
        // Accumulate gradients
        for (size_t i = 0; i < rff_dim * in_dim; ++i) {
            grad_W[i] += ws_b[i];
        }
        for (size_t i = 0; i < rff_dim; ++i) {
            grad_b[i] += (ws_b + rff_dim)[i];
        }
    }
}

// ============= Kernel Approximation =============

/**
 * Compute RBF kernel approximation: k(x,y) ≈ φ(x)ᵀφ(y)
 * @param phi_x RFF features of x [rff_dim]
 * @param phi_y RFF features of y [rff_dim]
 * @param rff_dim RFF feature dimension
 * @return Approximate kernel value
 */
inline float kernelApprox(const float* phi_x, const float* phi_y, size_t rff_dim) {
    float result = 0.0f;
    
    size_t i = 0;
    for (; i + 7 < rff_dim; i += 8) {
        __m256 vpx = _mm256_loadu_ps(phi_x + i);
        __m256 vpy = _mm256_loadu_ps(phi_y + i);
        __m256 vprod = _mm256_mul_ps(vpx, vpy);
        
        float prod[8];
        _mm256_storeu_ps(prod, vprod);
        for (int j = 0; j < 8; ++j) result += prod[j];
    }
    for (; i < rff_dim; ++i) {
        result += phi_x[i] * phi_y[i];
    }
    
    return result;
}

/**
 * Compute kernel matrix for batch of inputs
 * @param Phi RFF features [batch_size x rff_dim]
 * @param K Output kernel matrix [batch_size x batch_size]
 * @param batch_size Number of samples
 * @param rff_dim RFF feature dimension
 */
inline void kernelMatrix(const float* Phi, float* K, size_t batch_size, size_t rff_dim) {
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            float k = kernelApprox(Phi + i * rff_dim, Phi + j * rff_dim, rff_dim);
            K[i * batch_size + j] = k;
            K[j * batch_size + i] = k;  // Symmetric
        }
    }
}

} // namespace cput::rff
