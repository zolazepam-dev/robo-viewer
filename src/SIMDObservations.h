#pragma once

#include <immintrin.h>  // AVX2 intrinsics
#include <cmath>

// SIMD-optimized vector operations for observation computation
// Uses AVX2 for 8-way parallel float operations

namespace simd {

// Store 8 floats using AVX2
inline void Store8(float* dst, __m256 v) {
    _mm256_storeu_ps(dst, v);
}

// Load 8 floats using AVX2
inline __m256 Load8(const float* src) {
    return _mm256_loadu_ps(src);
}

// Set all 8 lanes to same value
inline __m256 Set8(float v) {
    return _mm256_set1_ps(v);
}

// Compute 8 inverse values: 1.0f / (x + epsilon)
inline __m256 Inv8(__m256 x, float epsilon = 0.1f) {
    return _mm256_div_ps(Set8(1.0f), _mm256_add_ps(x, Set8(epsilon)));
}

// Compute 8 square values: x * x
inline __m256 Square8(__m256 x) {
    return _mm256_mul_ps(x, x);
}

// Compute 8 sqrt values
inline __m256 Sqrt8(__m256 x) {
    return _mm256_sqrt_ps(x);
}

// Compute 8 normalized values: x / scale
inline __m256 Normalize8(__m256 x, float scale) {
    return _mm256_div_ps(x, Set8(scale));
}

// Compute absolute values for 8 floats
inline __m256 Abs8(__m256 x) {
    return _mm256_andnot_ps(Set8(-0.0f), x);  // Clear sign bit
}

// SIMD-optimized observation packing for 9 floats (3x Vec3)
inline void Pack9Floats(float* obs, int& idx, 
                        float x1, float y1, float z1,
                        float x2, float y2, float z2,
                        float x3, float y3, float z3) {
    alignas(32) float temp[8];
    temp[0] = x1; temp[1] = y1; temp[2] = z1;
    temp[3] = x2; temp[4] = y2; temp[5] = z2;
    temp[6] = x3; temp[7] = y3;
    
    __m256 v = _mm256_load_ps(temp);
    _mm256_storeu_ps(obs + idx, v);
    obs[idx + 8] = z3;  // 9th float
}

// SIMD-optimized compound feature computation (8 features at once)
inline void ComputeCompoundFeatures(float* obs, int idx, 
                                    float distance, float speed, float hp) {
    alignas(32) float features[8];
    const float ARENA_RADIUS = 30.0f;
    features[0] = distance / ARENA_RADIUS;
    features[1] = 1.0f / (distance + 0.1f);
    features[2] = distance * distance;
    features[3] = std::sqrt(distance);
    features[4] = speed / 20.0f;
    features[5] = hp / 100.0f;
    features[6] = distance * speed;
    features[7] = speed * hp;
    
    __m256 v0 = _mm256_load_ps(features);
    _mm256_storeu_ps(obs + idx, v0);
}

} // namespace simd
