#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <immintrin.h>
#include <cpuid.h>

namespace cput::simd {

// Fast exp approximation using polynomial
inline __m256 exp_approx(__m256 x) {
    // exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120
    // For better accuracy, use: exp(x) = 2^x * exp(x - x*ln(2))
    __m256 vln2 = _mm256_set1_ps(0.69314718056f);
    __m256 vinvln2 = _mm256_set1_ps(1.44269504089f);
    
    // Reduce to [-ln2/2, ln2/2]
    __m256 k_f = _mm256_floor_ps(_mm256_mul_ps(x, vinvln2) + _mm256_set1_ps(0.5f));
    __m256 x_reduced = _mm256_sub_ps(x, _mm256_mul_ps(k_f, vln2));
    
    // Polynomial approximation for exp(x) in [-ln2/2, ln2/2]
    __m256 x2 = _mm256_mul_ps(x_reduced, x_reduced);
    __m256 x3 = _mm256_mul_ps(x2, x_reduced);
    __m256 x4 = _mm256_mul_ps(x2, x2);
    
    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 c1 = _mm256_set1_ps(1.0f);
    __m256 c2 = _mm256_set1_ps(0.5f);
    __m256 c3 = _mm256_set1_ps(0.1666666667f);
    __m256 c4 = _mm256_set1_ps(0.04166666667f);
    
    __m256 result = _mm256_add_ps(c0, 
        _mm256_add_ps(
            _mm256_mul_ps(c1, x_reduced),
            _mm256_add_ps(
                _mm256_mul_ps(c2, x2),
                _mm256_add_ps(
                    _mm256_mul_ps(c3, x3),
                    _mm256_mul_ps(c4, x4)
                )
            )
        )
    );
    
    // Multiply by 2^k using bit manipulation
    int32_t k_int[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(k_int), _mm256_cvtps_epi32(k_f));
    
    for (int i = 0; i < 8; ++i) {
        float* result_ptr = reinterpret_cast<float*>(&result);
        int32_t exp_bits = (k_int[i] + 127) << 23;
        float scale;
        std::memcpy(&scale, &exp_bits, sizeof(float));
        result_ptr[i] *= scale;
    }
    
    return result;
}

// AVX2-optimized operations for float32
namespace avx2 {

// Vector add
inline void add(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
}

// Vector subtract
inline void sub(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) c[i] = a[i] - b[i];
}

// Vector multiply
inline void mul(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) c[i] = a[i] * b[i];
}

// Vector multiply-add: c = a * b + c
inline void fma(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        vc = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) c[i] += a[i] * b[i];
}

// Scalar multiply
inline void mulScalar(const float* a, float scalar, float* c, size_t n) {
    __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) c[i] = a[i] * scalar;
}

// ReLU activation
inline void relu(const float* a, float* c, size_t n) {
    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = _mm256_max_ps(va, vzero);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) c[i] = a[i] > 0 ? a[i] : 0;
}

// Leaky ReLU activation
inline void leakyRelu(const float* a, float* c, size_t n, float alpha = 0.01f) {
    __m256 valpha = _mm256_set1_ps(alpha);
    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vneg = _mm256_mul_ps(va, valpha);
        __m256 vpos = _mm256_max_ps(va, vzero);
        __m256 vneg_check = _mm256_cmp_ps(va, vzero, _CMP_LT_OQ);
        __m256 vc = _mm256_blendv_ps(vpos, vneg, vneg_check);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) c[i] = a[i] > 0 ? a[i] : a[i] * alpha;
}

// Fast tanh approximation using polynomial
inline __m256 tanh_approx(__m256 x) {
    __m256 v1 = _mm256_set1_ps(1.0f);
    __m256 v27 = _mm256_set1_ps(27.0f);
    __m256 v9 = _mm256_set1_ps(9.0f);
    
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 num = _mm256_mul_ps(x, _mm256_add_ps(v27, x2));
    __m256 denom = _mm256_add_ps(v27, _mm256_mul_ps(v9, x2));
    __m256 result = _mm256_div_ps(num, denom);
    
    __m256 vneg1 = _mm256_set1_ps(-1.0f);
    result = _mm256_max_ps(result, vneg1);
    result = _mm256_min_ps(result, v1);
    
    return result;
}

// Sigmoid approximation (fast)
inline void sigmoid(const float* a, float* c, size_t n) {
    __m256 vone = _mm256_set1_ps(1.0f);
    __m256 vhalf = _mm256_set1_ps(0.5f);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vhalf_x = _mm256_mul_ps(vhalf, va);
        __m256 vtanh = tanh_approx(vhalf_x);
        __m256 vc = _mm256_mul_ps(vhalf, _mm256_add_ps(vone, vtanh));
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) c[i] = 1.0f / (1.0f + std::exp(-a[i]));
}

// Tanh using AVX2 approximation
inline void tanh(const float* a, float* c, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = tanh_approx(va);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) c[i] = std::tanh(a[i]);
}

// GELU activation (approximation)
inline void gelu(const float* a, float* c, size_t n) {
    __m256 v0797885f = _mm256_set1_ps(0.7978845608f);
    __m256 v0044715f = _mm256_set1_ps(0.044715f);
    __m256 v05f = _mm256_set1_ps(0.5f);
    __m256 v1f = _mm256_set1_ps(1.0f);
    
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 va3 = _mm256_mul_ps(_mm256_mul_ps(va, va), va);
        __m256 vinner = _mm256_add_ps(va, _mm256_mul_ps(v0044715f, va3));
        __m256 vscaled = _mm256_mul_ps(v0797885f, vinner);
        __m256 vtanh = tanh_approx(vscaled);
        __m256 vadd = _mm256_add_ps(v1f, vtanh);
        __m256 vc = _mm256_mul_ps(_mm256_mul_ps(v05f, va), vadd);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) {
        float x = a[i];
        c[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Softmax (single pass with max finding)
inline void softmax(float* a, size_t n) {
    float max_val = a[0];
    for (size_t i = 1; i < n; ++i) {
        if (a[i] > max_val) max_val = a[i];
    }
    
    float sum = 0;
    size_t i = 0;
    __m256 vmax = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();
    
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        va = _mm256_sub_ps(va, vmax);
        va = exp_approx(va);
        _mm256_storeu_ps(a + i, va);
        vsum = _mm256_add_ps(vsum, va);
    }
    
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, vsum);
    for (int j = 0; j < 8; ++j) sum += sum_arr[j];
    
    for (; i < n; ++i) {
        a[i] = std::exp(a[i] - max_val);
        sum += a[i];
    }
    
    __m256 vinvsum = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        va = _mm256_mul_ps(va, vinvsum);
        _mm256_storeu_ps(a + i, va);
    }
    for (; i < n; ++i) a[i] /= sum;
}

// Dot product
inline float dot(const float* a, const float* b, size_t n) {
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, vsum);
    float sum = 0;
    for (size_t j = 0; j < 8 && i + j < n; ++j) sum += sum_arr[j];
    
    for (; i < n; ++i) sum += a[i] * b[i];
    
    return sum;
}

// Sum reduction
inline float sum(const float* a, size_t n) {
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        vsum = _mm256_add_ps(vsum, va);
    }
    
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, vsum);
    float sum = 0;
    for (size_t j = 0; j < 8 && i + j < n; ++j) sum += sum_arr[j];
    
    for (; i < n; ++i) sum += a[i];
    
    return sum;
}

// Mean
inline float mean(const float* a, size_t n) {
    return sum(a, n) / static_cast<float>(n);
}

// L2 norm
inline float l2norm(const float* a, size_t n) {
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        vsum = _mm256_fmadd_ps(va, va, vsum);
    }
    
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, vsum);
    float sum = 0;
    for (size_t j = 0; j < 8 && i + j < n; ++j) sum += sum_arr[j];
    
    for (; i < n; ++i) sum += a[i] * a[i];
    
    return std::sqrt(sum);
}

// Copy
inline void copy(const float* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < n; ++i) dst[i] = src[i];
}

// Set all to value
inline void set(float* a, float value, size_t n) {
    __m256 v = _mm256_set1_ps(value);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(a + i, v);
    }
    for (; i < n; ++i) a[i] = value;
}

} // namespace avx2

// CPU feature detection
inline bool hasAVX2() {
    unsigned int eax, ebx, ecx, edx;
    // First check if XSAVE is supported (leaf 1)
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    // Check OS uses XSAVE/XRSTORE (ECX bit 27)
    if (!(ecx & (1 << 27))) return false;
    // Check AVX support (ECX bit 28)
    if (!(ecx & (1 << 28))) return false;
    // Now check AVX2 (leaf 7, subleaf 0)
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & (1 << 5)) != 0;
}

// Auto-dispatch
inline void add(const float* a, const float* b, float* c, size_t n) {
    if (hasAVX2()) avx2::add(a, b, c, n);
    else for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

inline void mul(const float* a, const float* b, float* c, size_t n) {
    if (hasAVX2()) avx2::mul(a, b, c, n);
    else for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
}

inline float dot(const float* a, const float* b, size_t n) {
    if (hasAVX2()) return avx2::dot(a, b, n);
    float sum = 0;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

inline void relu(const float* a, float* c, size_t n) {
    if (hasAVX2()) avx2::relu(a, c, n);
    else for (size_t i = 0; i < n; ++i) c[i] = a[i] > 0 ? a[i] : 0;
}

inline void tanh(const float* a, float* c, size_t n) {
    if (hasAVX2()) avx2::tanh(a, c, n);
    else for (size_t i = 0; i < n; ++i) c[i] = std::tanh(a[i]);
}

inline void gelu(const float* a, float* c, size_t n) {
    if (hasAVX2()) avx2::gelu(a, c, n);
    else for (size_t i = 0; i < n; ++i) {
        float x = a[i];
        c[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

inline void softmax(float* a, size_t n) {
    if (hasAVX2()) avx2::softmax(a, n);
    else {
        float max_val = a[0];
        for (size_t i = 1; i < n; ++i) if (a[i] > max_val) max_val = a[i];
        float sum = 0;
        for (size_t i = 0; i < n; ++i) { a[i] = std::exp(a[i] - max_val); sum += a[i]; }
        for (size_t i = 0; i < n; ++i) a[i] /= sum;
    }
}

inline void sigmoid(const float* a, float* c, size_t n) {
    if (hasAVX2()) avx2::sigmoid(a, c, n);
    else for (size_t i = 0; i < n; ++i) c[i] = 1.0f / (1.0f + std::exp(-a[i]));
}

inline void leakyRelu(const float* a, float* c, size_t n, float alpha = 0.01f) {
    if (hasAVX2()) avx2::leakyRelu(a, c, n, alpha);
    else for (size_t i = 0; i < n; ++i) c[i] = a[i] > 0 ? a[i] : a[i] * alpha;
}

inline void sub(const float* a, const float* b, float* c, size_t n) {
    if (hasAVX2()) avx2::sub(a, b, c, n);
    else for (size_t i = 0; i < n; ++i) c[i] = a[i] - b[i];
}

inline void fma(const float* a, const float* b, float* c, size_t n) {
    if (hasAVX2()) avx2::fma(a, b, c, n);
    else for (size_t i = 0; i < n; ++i) c[i] += a[i] * b[i];
}

inline void mulScalar(const float* a, float scalar, float* c, size_t n) {
    if (hasAVX2()) avx2::mulScalar(a, scalar, c, n);
    else for (size_t i = 0; i < n; ++i) c[i] = a[i] * scalar;
}

inline float sum(const float* a, size_t n) {
    if (hasAVX2()) return avx2::sum(a, n);
    float s = 0;
    for (size_t i = 0; i < n; ++i) s += a[i];
    return s;
}

inline float mean(const float* a, size_t n) {
    return sum(a, n) / static_cast<float>(n);
}

inline float l2norm(const float* a, size_t n) {
    if (hasAVX2()) return avx2::l2norm(a, n);
    float s = 0;
    for (size_t i = 0; i < n; ++i) s += a[i] * a[i];
    return std::sqrt(s);
}

inline void copy(const float* src, float* dst, size_t n) {
    if (hasAVX2()) avx2::copy(src, dst, n);
    else for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}

inline void set(float* a, float value, size_t n) {
    if (hasAVX2()) avx2::set(a, value, n);
    else for (size_t i = 0; i < n; ++i) a[i] = value;
}

} // namespace cput::simd
