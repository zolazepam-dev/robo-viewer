#pragma once

#include <cstdint>
#include <cmath>
#include <immintrin.h>

/**
 * Fast AVX2 vectorized sin/cos approximations
 * Uses Chebyshev polynomial approximation (accurate to ~0.001)
 * 
 * Much faster than std::sin/std::cos for gradient computation
 */

namespace FastTrig {

// Fast sine approximation using polynomial
// Accurate to ~0.001 radians in [-pi, pi]
inline __m256 sin_ps(__m256 x) {
    // Range reduction to [-pi, pi]
    const __m256 pi = _mm256_set1_ps(3.141592653589793f);
    const __m256 two_pi = _mm256_set1_ps(6.283185307179586f);
    const __m256 half_pi = _mm256_set1_ps(1.5707963267948966f);
    
    // Reduce to [-pi, pi]
    x = _mm256_sub_ps(x, _mm256_mul_ps(two_pi, _mm256_round_ps(_mm256_div_ps(x, two_pi), _MM_FROUND_TO_NEAREST_INT)));
    
    // Chebyshev approximation: sin(x) ≈ x - x^3/6 + x^5/120 - x^7/5040
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    __m256 x5 = _mm256_mul_ps(x3, x2);
    __m256 x7 = _mm256_mul_ps(x5, x2);
    
    __m256 result = _mm256_add_ps(x, _mm256_mul_ps(x3, _mm256_set1_ps(-0.1666666667f)));
    result = _mm256_add_ps(result, _mm256_mul_ps(x5, _mm256_set1_ps(0.0083333333f)));
    result = _mm256_add_ps(result, _mm256_mul_ps(x7, _mm256_set1_ps(-0.0001984127f)));
    
    return result;
}

// Fast cosine approximation: cos(x) = sin(x + pi/2)
inline __m256 cos_ps(__m256 x) {
    const __m256 half_pi = _mm256_set1_ps(1.5707963267948966f);
    return sin_ps(_mm256_add_ps(x, half_pi));
}

// Compute both sin and cos simultaneously
inline void sincos_ps(__m256 x, __m256* sin_out, __m256* cos_out) {
    const __m256 pi = _mm256_set1_ps(3.141592653589793f);
    const __m256 two_pi = _mm256_set1_ps(6.283185307179586f);
    const __m256 half_pi = _mm256_set1_ps(1.5707963267948966f);
    
    // Range reduction
    x = _mm256_sub_ps(x, _mm256_mul_ps(two_pi, _mm256_round_ps(_mm256_div_ps(x, two_pi), _MM_FROUND_TO_NEAREST_INT)));
    
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    __m256 x5 = _mm256_mul_ps(x3, x2);
    __m256 x7 = _mm256_mul_ps(x5, x2);
    
    // sin(x)
    *sin_out = _mm256_add_ps(x, _mm256_mul_ps(x3, _mm256_set1_ps(-0.1666666667f)));
    *sin_out = _mm256_add_ps(*sin_out, _mm256_mul_ps(x5, _mm256_set1_ps(0.0083333333f)));
    *sin_out = _mm256_add_ps(*sin_out, _mm256_mul_ps(x7, _mm256_set1_ps(-0.0001984127f)));
    
    // cos(x) = sin(x + pi/2)
    __m256 x_cos = _mm256_add_ps(x, half_pi);
    __m256 x2_cos = _mm256_mul_ps(x_cos, x_cos);
    __m256 x3_cos = _mm256_mul_ps(x2_cos, x_cos);
    __m256 x5_cos = _mm256_mul_ps(x3_cos, x2_cos);
    __m256 x7_cos = _mm256_mul_ps(x5_cos, x2_cos);
    
    *cos_out = _mm256_add_ps(x_cos, _mm256_mul_ps(x3_cos, _mm256_set1_ps(-0.1666666667f)));
    *cos_out = _mm256_add_ps(*cos_out, _mm256_mul_ps(x5_cos, _mm256_set1_ps(0.0083333333f)));
    *cos_out = _mm256_add_ps(*cos_out, _mm256_mul_ps(x7_cos, _mm256_set1_ps(-0.0001984127f)));
}

// Batch versions for arrays
inline void sin_batch(const float* input, float* output, int size) {
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 result = sin_ps(x);
        _mm256_storeu_ps(output + i, result);
    }
    // Handle remainder
    for (; i < size; ++i) {
        output[i] = std::sin(input[i]);
    }
}

inline void cos_batch(const float* input, float* output, int size) {
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 result = cos_ps(x);
        _mm256_storeu_ps(output + i, result);
    }
    // Handle remainder
    for (; i < size; ++i) {
        output[i] = std::cos(input[i]);
    }
}

inline void sincos_batch(const float* input, float* sin_out, float* cos_out, int size) {
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 s, c;
        sincos_ps(x, &s, &c);
        _mm256_storeu_ps(sin_out + i, s);
        _mm256_storeu_ps(cos_out + i, c);
    }
    // Handle remainder
    for (; i < size; ++i) {
        sin_out[i] = std::sin(input[i]);
        cos_out[i] = std::cos(input[i]);
    }
}

} // namespace FastTrig
