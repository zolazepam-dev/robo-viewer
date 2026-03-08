#include "NeuralMath.h"

#include <cmath>
#include <immintrin.h>
#include <algorithm>
#include <cstring>

void ForwardMoLU_Scalar(float* data, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        float x = data[i];
        data[i] = 0.5f * x * (1.0f + tanhf(x));
    }
}

void ForwardMoLU_AVX2(float* data, size_t size)
{
    AssertAligned32(data);
    
    const size_t simdWidth = 8;
    const size_t simdEnd = size - (size % simdWidth);
    
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 clampHi = _mm256_set1_ps(10.0f);
    const __m256 clampLo = _mm256_set1_ps(-10.0f);
    
    // Higher-order rational approximation for tanh(x)
    const __m256 p0 = _mm256_set1_ps(105.0f);
    const __m256 p1 = _mm256_set1_ps(10.0f);
    const __m256 q0 = _mm256_set1_ps(105.0f);
    const __m256 q1 = _mm256_set1_ps(45.0f);
    const __m256 minusOne = _mm256_set1_ps(-1.0f);
    
    size_t i = 0;
    for (; i < simdEnd; i += simdWidth)
    {
        __m256 x = _mm256_load_ps(data + i);
        
        // xc = clamp(x, -4, 4) for the rational part to avoid overflow/instability
        __m256 xc = _mm256_min_ps(x, _mm256_set1_ps(4.0f));
        xc = _mm256_max_ps(xc, _mm256_set1_ps(-4.0f));
        
        __m256 xc2 = _mm256_mul_ps(xc, xc);
        __m256 xc4 = _mm256_mul_ps(xc2, xc2);
        
        // num = xc * (105 + 10 * xc^2)
        __m256 num = _mm256_fmadd_ps(p1, xc2, p0);
        num = _mm256_mul_ps(xc, num);
        
        // den = 105 + 45 * xc^2 + xc^4
        __m256 den = _mm256_fmadd_ps(q1, xc2, q0);
        den = _mm256_add_ps(den, xc4);
        
        __m256 rcp = _mm256_rcp_ps(den);
        rcp = _mm256_mul_ps(rcp, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(den, rcp)));
        
        __m256 th = _mm256_mul_ps(num, rcp);
        
        // For values outside [-4, 4], use +/- 1
        __m256 isHi = _mm256_cmp_ps(x, _mm256_set1_ps(4.0f), _CMP_GT_OQ);
        __m256 isLo = _mm256_cmp_ps(x, _mm256_set1_ps(-4.0f), _CMP_LT_OQ);
        th = _mm256_blendv_ps(th, one, isHi);
        th = _mm256_blendv_ps(th, minusOne, isLo);
        
        __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, th)));
        _mm256_store_ps(data + i, result);
    }
    
    for (; i < size; ++i)
    {
        float x = data[i];
        float xc = std::clamp(x, -10.0f, 10.0f);
        data[i] = 0.5f * x * (1.0f + tanhf(xc));
    }
}

void ForwardTanh_AVX2(float* data, size_t size)
{
    AssertAligned32(data);
    const size_t simdWidth = 8;
    const size_t simdEnd = size - (size % simdWidth);
    
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 clampHi = _mm256_set1_ps(10.0f);
    const __m256 clampLo = _mm256_set1_ps(-10.0f);
    const __m256 pade_a = _mm256_set1_ps(0.275f);
    const __m256 pade_b = _mm256_set1_ps(0.664f);
    
    size_t i = 0;
    for (; i < simdEnd; i += simdWidth)
    {
        __m256 x = _mm256_load_ps(data + i);
        x = _mm256_min_ps(x, clampHi);
        x = _mm256_max_ps(x, clampLo);
        
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 num = _mm256_fmadd_ps(pade_a, x2, one);
        num = _mm256_mul_ps(x, num);
        __m256 den = _mm256_fmadd_ps(pade_b, x2, one);
        
        __m256 rcp = _mm256_rcp_ps(den);
        rcp = _mm256_mul_ps(rcp, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(den, rcp)));
        
        _mm256_store_ps(data + i, _mm256_mul_ps(num, rcp));
    }
    
    for (; i < size; ++i)
    {
        data[i] = tanhf(std::clamp(data[i], -10.0f, 10.0f));
    }
}

void ForwardSigmoid_AVX2(float* data, size_t size)
{
    AssertAligned32(data);
    const size_t simdWidth = 8;
    const size_t simdEnd = size - (size % simdWidth);
    
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 pade_a = _mm256_set1_ps(0.275f);
    const __m256 pade_b = _mm256_set1_ps(0.664f);
    
    size_t i = 0;
    for (; i < simdEnd; i += simdWidth)
    {
        // sigmoid(x) = 0.5 * tanh(0.5 * x) + 0.5
        __m256 x = _mm256_load_ps(data + i);
        __m256 hx = _mm256_mul_ps(x, half);
        
        __m256 hx2 = _mm256_mul_ps(hx, hx);
        __m256 num = _mm256_fmadd_ps(pade_a, hx2, one);
        num = _mm256_mul_ps(hx, num);
        __m256 den = _mm256_fmadd_ps(pade_b, hx2, one);
        
        __m256 rcp = _mm256_rcp_ps(den);
        rcp = _mm256_mul_ps(rcp, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(den, rcp)));
        
        __m256 th = _mm256_mul_ps(num, rcp);
        _mm256_store_ps(data + i, _mm256_fmadd_ps(half, th, half));
    }
    
    for (; i < size; ++i)
    {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
}

void AddVectors_AVX2(float* dst, const float* src, size_t size)
{
    const size_t simdWidth = 8;
    const size_t simdEnd = size - (size % simdWidth);
    
    size_t i = 0;
    for (; i < simdEnd; i += simdWidth)
    {
        __m256 a = _mm256_loadu_ps(dst + i);
        __m256 b = _mm256_loadu_ps(src + i);
        __m256 result = _mm256_add_ps(a, b);
        _mm256_storeu_ps(dst + i, result);
    }
    
    for (; i < size; ++i)
    {
        dst[i] += src[i];
    }
}

void ScaleVector_AVX2(float* dst, float scale, size_t size)
{
    const size_t simdWidth = 8;
    const size_t simdEnd = size - (size % simdWidth);
    const __m256 s = _mm256_set1_ps(scale);
    
    size_t i = 0;
    for (; i < simdEnd; i += simdWidth)
    {
        __m256 x = _mm256_loadu_ps(dst + i);
        __m256 result = _mm256_mul_ps(x, s);
        _mm256_storeu_ps(dst + i, result);
    }
    
    for (; i < size; ++i)
    {
        dst[i] *= scale;
    }
}

void FMAVector_AVX2(float* dst, const float* a, const float* b, size_t size)
{
    const size_t simdWidth = 8;
    const size_t simdEnd = size - (size % simdWidth);
    
    size_t i = 0;
    for (; i < simdEnd; i += simdWidth)
    {
        __m256 x = _mm256_loadu_ps(dst + i);
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        __m256 result = _mm256_fmadd_ps(av, bv, x);
        _mm256_storeu_ps(dst + i, result);
    }
    
    for (; i < size; ++i)
    {
        dst[i] += a[i] * b[i];
    }
}

void Softmax_AVX2(float* data, size_t size)
{
    AssertAligned32(data);
    const size_t simdWidth = 8;
    const size_t simdEnd = size - (size % simdWidth);
    
    // 1. Find Max
    __m256 maxVec = _mm256_set1_ps(-1e30f);
    for (size_t i = 0; i < simdEnd; i += simdWidth)
    {
        maxVec = _mm256_max_ps(maxVec, _mm256_load_ps(data + i));
    }
    
    float maxVal = -1e30f;
    alignas(32) float maxBuf[8];
    _mm256_store_ps(maxBuf, maxVec);
    for (int i = 0; i < 8; ++i) maxVal = std::max(maxVal, maxBuf[i]);
    for (size_t i = simdEnd; i < size; ++i) maxVal = std::max(maxVal, data[i]);
    
    // 2. Exp and Sum
    __m256 sumVec = _mm256_setzero_ps();
    __m256 mVec = _mm256_set1_ps(maxVal);
    
    for (size_t i = 0; i < simdEnd; i += simdWidth)
    {
        __m256 x = _mm256_load_ps(data + i);
        // Exponential approximation (very rough but fast) or scalar fallback for expf
        // For accuracy, we'll use scalar expf in this part or a vectorized exp if available
        for (int j = 0; j < 8; ++j) {
            data[i + j] = expf(data[i + j] - maxVal);
        }
        sumVec = _mm256_add_ps(sumVec, _mm256_load_ps(data + i));
    }
    
    for (size_t i = simdEnd; i < size; ++i)
    {
        data[i] = expf(data[i] - maxVal);
    }
    
    float totalSum = 0.0f;
    alignas(32) float sumBuf[8];
    _mm256_store_ps(sumBuf, sumVec);
    for (int i = 0; i < 8; ++i) totalSum += sumBuf[i];
    for (size_t i = simdEnd; i < size; ++i) totalSum += data[i];
    
    // 3. Normalize
    ScaleVector_AVX2(data, 1.0f / totalSum, size);
}

void LayerNorm_AVX2(float* data, size_t size, const float* gamma, const float* beta)
{
    AssertAligned32(data);
    const size_t simdWidth = 8;
    const size_t simdEnd = size - (size % simdWidth);
    
    // 1. Mean
    __m256 sumVec = _mm256_setzero_ps();
    for (size_t i = 0; i < simdEnd; i += simdWidth)
    {
        sumVec = _mm256_add_ps(sumVec, _mm256_load_ps(data + i));
    }
    
    float sum = 0.0f;
    alignas(32) float sumBuf[8];
    _mm256_store_ps(sumBuf, sumVec);
    for (int i = 0; i < 8; ++i) sum += sumBuf[i];
    for (size_t i = simdEnd; i < size; ++i) sum += data[i];
    
    float mean = sum / static_cast<float>(size);
    __m256 meanVec = _mm256_set1_ps(mean);
    
    // 2. Variance
    __m256 varVec = _mm256_setzero_ps();
    for (size_t i = 0; i < simdEnd; i += simdWidth)
    {
        __m256 diff = _mm256_sub_ps(_mm256_load_ps(data + i), meanVec);
        varVec = _mm256_add_ps(varVec, _mm256_mul_ps(diff, diff));
    }
    
    float varSum = 0.0f;
    _mm256_store_ps(sumBuf, varVec);
    for (int i = 0; i < 8; ++i) varSum += sumBuf[i];
    for (size_t i = simdEnd; i < size; ++i) {
        float diff = data[i] - mean;
        varSum += diff * diff;
    }
    
    float variance = varSum / static_cast<float>(size);
    float invStd = 1.0f / sqrtf(variance + 1e-5f);
    __m256 invStdVec = _mm256_set1_ps(invStd);
    
    // 3. Normalize and scale
    size_t i = 0;
    for (; i < simdEnd; i += simdWidth)
    {
        __m256 x = _mm256_load_ps(data + i);
        __m256 g = _mm256_loadu_ps(gamma + i);
        __m256 b = _mm256_loadu_ps(beta + i);
        
        __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x, meanVec), invStdVec);
        _mm256_store_ps(data + i, _mm256_fmadd_ps(norm, g, b));
    }
    
    for (; i < size; ++i)
    {
        data[i] = (data[i] - mean) * invStd * gamma[i] + beta[i];
    }
}

void MatMul_AVX2(const float* A, const float* B, float* C, 
                 size_t M, size_t K, size_t N)
{
    
    std::memset(C, 0, M * N * sizeof(float));
    
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t k = 0; k < K; ++k)
        {
            const float a_ik = A[i * K + k];
            const __m256 a_vec = _mm256_set1_ps(a_ik);
            
            size_t j = 0;
            for (; j + 8 <= N; j += 8)
            {
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                __m256 result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C[i * N + j], result);
            }
            
            for (; j < N; ++j)
            {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

void MatVec_FMA_AVX2(const float* weights, const float* inputs, float* outputs,
                      size_t input_dim, size_t output_dim)
{
    const size_t simdWidth = 8;
    
    for (size_t outIdx = 0; outIdx < output_dim; ++outIdx)
    {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        
        size_t inIdx = 0;
        for (; inIdx + 32 <= input_dim; inIdx += 32)
        {
            const float* wPtr = weights + outIdx * input_dim + inIdx;
            
            PrefetchL1(wPtr + 64);
            PrefetchL1(inputs + inIdx + 64);
            
            __m256 w0 = _mm256_loadu_ps(wPtr);
            __m256 w1 = _mm256_loadu_ps(wPtr + 8);
            __m256 w2 = _mm256_loadu_ps(wPtr + 16);
            __m256 w3 = _mm256_loadu_ps(wPtr + 24);
            
            __m256 i0 = _mm256_loadu_ps(inputs + inIdx);
            __m256 i1 = _mm256_loadu_ps(inputs + inIdx + 8);
            __m256 i2 = _mm256_loadu_ps(inputs + inIdx + 16);
            __m256 i3 = _mm256_loadu_ps(inputs + inIdx + 24);
            
            sum0 = _mm256_fmadd_ps(w0, i0, sum0);
            sum1 = _mm256_fmadd_ps(w1, i1, sum1);
            sum2 = _mm256_fmadd_ps(w2, i2, sum2);
            sum3 = _mm256_fmadd_ps(w3, i3, sum3);
        }
        
        for (; inIdx + 8 <= input_dim; inIdx += 8)
        {
            __m256 w = _mm256_loadu_ps(weights + outIdx * input_dim + inIdx);
            __m256 i = _mm256_loadu_ps(inputs + inIdx);
            sum0 = _mm256_fmadd_ps(w, i, sum0);
        }
        
        __m256 sum01 = _mm256_add_ps(sum0, sum1);
        __m256 sum23 = _mm256_add_ps(sum2, sum3);
        __m256 total = _mm256_add_ps(sum01, sum23);
        
        __m128 hi = _mm256_extractf128_ps(total, 1);
        __m128 lo = _mm256_castps256_ps128(total);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        
        float finalSum = _mm_cvtss_f32(sum128);
        
        for (; inIdx < input_dim; ++inIdx)
        {
            finalSum += weights[outIdx * input_dim + inIdx] * inputs[inIdx];
        }
        
        outputs[outIdx] = finalSum;
    }
}

void MatVec_FMA_AVX2_Prefetch(const float* weights, const float* inputs, float* outputs,
                               size_t input_dim, size_t output_dim)
{
    const size_t simdWidth = 8;
    const size_t prefetchDistance = 4 * CACHE_LINE_SIZE / sizeof(float);
    
    for (size_t outIdx = 0; outIdx < output_dim; ++outIdx)
    {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        
        size_t inIdx = 0;
        for (; inIdx + 32 <= input_dim; inIdx += 32)
        {
            const float* wPtr = weights + outIdx * input_dim + inIdx;
            
            _mm_prefetch(reinterpret_cast<const char*>(wPtr + prefetchDistance), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(inputs + inIdx + prefetchDistance), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(wPtr + prefetchDistance + 8), _MM_HINT_T0);
            
            __m256 w0 = _mm256_loadu_ps(wPtr);
            __m256 w1 = _mm256_loadu_ps(wPtr + 8);
            __m256 w2 = _mm256_loadu_ps(wPtr + 16);
            __m256 w3 = _mm256_loadu_ps(wPtr + 24);
            
            __m256 i0 = _mm256_loadu_ps(inputs + inIdx);
            __m256 i1 = _mm256_loadu_ps(inputs + inIdx + 8);
            __m256 i2 = _mm256_loadu_ps(inputs + inIdx + 16);
            __m256 i3 = _mm256_loadu_ps(inputs + inIdx + 24);
            
            sum0 = _mm256_fmadd_ps(w0, i0, sum0);
            sum1 = _mm256_fmadd_ps(w1, i1, sum1);
            sum2 = _mm256_fmadd_ps(w2, i2, sum2);
            sum3 = _mm256_fmadd_ps(w3, i3, sum3);
        }
        
        for (; inIdx + 8 <= input_dim; inIdx += 8)
        {
            _mm_prefetch(reinterpret_cast<const char*>(weights + outIdx * input_dim + inIdx + 8), _MM_HINT_T0);
            
            __m256 w = _mm256_loadu_ps(weights + outIdx * input_dim + inIdx);
            __m256 i = _mm256_loadu_ps(inputs + inIdx);
            sum0 = _mm256_fmadd_ps(w, i, sum0);
        }
        
        __m256 sum01 = _mm256_add_ps(sum0, sum1);
        __m256 sum23 = _mm256_add_ps(sum2, sum3);
        __m256 total = _mm256_add_ps(sum01, sum23);
        
        __m128 hi = _mm256_extractf128_ps(total, 1);
        __m128 lo = _mm256_castps256_ps128(total);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        
        float finalSum = _mm_cvtss_f32(sum128);
        
        for (; inIdx < input_dim; ++inIdx)
        {
            finalSum += weights[outIdx * input_dim + inIdx] * inputs[inIdx];
        }
        
        outputs[outIdx] = finalSum;
    }
}

void Transpose8x8_AVX2(const float* src, float* dst, size_t srcStride, size_t dstStride)
{
    __m256 row0 = _mm256_loadu_ps(src + 0 * srcStride);
    __m256 row1 = _mm256_loadu_ps(src + 1 * srcStride);
    __m256 row2 = _mm256_loadu_ps(src + 2 * srcStride);
    __m256 row3 = _mm256_loadu_ps(src + 3 * srcStride);
    __m256 row4 = _mm256_loadu_ps(src + 4 * srcStride);
    __m256 row5 = _mm256_loadu_ps(src + 5 * srcStride);
    __m256 row6 = _mm256_loadu_ps(src + 6 * srcStride);
    __m256 row7 = _mm256_loadu_ps(src + 7 * srcStride);
    
    __m256 t0 = _mm256_unpacklo_ps(row0, row1);
    __m256 t1 = _mm256_unpackhi_ps(row0, row1);
    __m256 t2 = _mm256_unpacklo_ps(row2, row3);
    __m256 t3 = _mm256_unpackhi_ps(row2, row3);
    __m256 t4 = _mm256_unpacklo_ps(row4, row5);
    __m256 t5 = _mm256_unpackhi_ps(row4, row5);
    __m256 t6 = _mm256_unpacklo_ps(row6, row7);
    __m256 t7 = _mm256_unpackhi_ps(row6, row7);
    
    __m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    
    __m256 col0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
    __m256 col1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
    __m256 col2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
    __m256 col3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
    __m256 col4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
    __m256 col5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
    __m256 col6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
    __m256 col7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
    
    _mm256_storeu_ps(dst + 0 * dstStride, col0);
    _mm256_storeu_ps(dst + 1 * dstStride, col1);
    _mm256_storeu_ps(dst + 2 * dstStride, col2);
    _mm256_storeu_ps(dst + 3 * dstStride, col3);
    _mm256_storeu_ps(dst + 4 * dstStride, col4);
    _mm256_storeu_ps(dst + 5 * dstStride, col5);
    _mm256_storeu_ps(dst + 6 * dstStride, col6);
    _mm256_storeu_ps(dst + 7 * dstStride, col7);
}

void TransposeBatch_AoS_to_SoA(const float* aos_input, float* soa_output,
                                size_t batch_size, size_t feature_dim)
{
    AssertAligned32(aos_input);
    AssertAligned32(soa_output);
    
    const size_t simdWidth = 8;
    
    if (feature_dim >= simdWidth && batch_size >= simdWidth)
    {
        for (size_t f = 0; f + simdWidth <= feature_dim; f += simdWidth)
        {
            for (size_t b = 0; b + simdWidth <= batch_size; b += simdWidth)
            {
                float tempBlock[64];
                
                for (size_t bb = 0; bb < simdWidth; ++bb)
                {
                    for (size_t ff = 0; ff < simdWidth; ++ff)
                    {
                        tempBlock[bb * simdWidth + ff] = aos_input[(b + bb) * feature_dim + (f + ff)];
                    }
                }
                
                Transpose8x8_AVX2(tempBlock, tempBlock, simdWidth, simdWidth);
                
                for (size_t ff = 0; ff < simdWidth; ++ff)
                {
                    for (size_t bb = 0; bb < simdWidth; ++bb)
                    {
                        soa_output[(f + ff) * batch_size + (b + bb)] = tempBlock[ff * simdWidth + bb];
                    }
                }
            }
        }
    }
    else
    {
        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t f = 0; f < feature_dim; ++f)
            {
                soa_output[f * batch_size + b] = aos_input[b * feature_dim + f];
            }
        }
    }
}

void MatVec_Vertical_AVX2(const float* weights, const float* inputs, float* outputs,
                            size_t input_dim, size_t output_dim, size_t batch_size)
{
    AssertAligned32(weights);
    AssertAligned32(inputs);
    AssertAligned32(outputs);
    
    const size_t simdWidth = 8;
    
    for (size_t outIdx = 0; outIdx < output_dim; ++outIdx)
    {
        const float* weightRow = weights + outIdx * input_dim;
        float* outCol = outputs + outIdx * batch_size;
        
        size_t batchIdx = 0;
        for (; batchIdx + simdWidth <= batch_size; batchIdx += simdWidth)
        {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();
            
            size_t inIdx = 0;
            for (; inIdx + 32 <= input_dim; inIdx += 32)
            {
                _mm_prefetch(reinterpret_cast<const char*>(weightRow + inIdx + 64), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(inputs + (inIdx + 32) * batch_size + batchIdx), _MM_HINT_T0);
                
                __m256 w0 = _mm256_set1_ps(weightRow[inIdx]);
                __m256 w1 = _mm256_set1_ps(weightRow[inIdx + 1]);
                __m256 w2 = _mm256_set1_ps(weightRow[inIdx + 2]);
                __m256 w3 = _mm256_set1_ps(weightRow[inIdx + 3]);
                __m256 w4 = _mm256_set1_ps(weightRow[inIdx + 4]);
                __m256 w5 = _mm256_set1_ps(weightRow[inIdx + 5]);
                __m256 w6 = _mm256_set1_ps(weightRow[inIdx + 6]);
                __m256 w7 = _mm256_set1_ps(weightRow[inIdx + 7]);
                
                const float* i0 = inputs + inIdx * batch_size + batchIdx;
                const float* i1 = inputs + (inIdx + 1) * batch_size + batchIdx;
                const float* i2 = inputs + (inIdx + 2) * batch_size + batchIdx;
                const float* i3 = inputs + (inIdx + 3) * batch_size + batchIdx;
                const float* i4 = inputs + (inIdx + 4) * batch_size + batchIdx;
                const float* i5 = inputs + (inIdx + 5) * batch_size + batchIdx;
                const float* i6 = inputs + (inIdx + 6) * batch_size + batchIdx;
                const float* i7 = inputs + (inIdx + 7) * batch_size + batchIdx;
                
                __m256 in0 = _mm256_load_ps(i0);
                __m256 in1 = _mm256_load_ps(i1);
                __m256 in2 = _mm256_load_ps(i2);
                __m256 in3 = _mm256_load_ps(i3);
                __m256 in4 = _mm256_load_ps(i4);
                __m256 in5 = _mm256_load_ps(i5);
                __m256 in6 = _mm256_load_ps(i6);
                __m256 in7 = _mm256_load_ps(i7);
                
                sum0 = _mm256_fmadd_ps(w0, in0, sum0);
                sum1 = _mm256_fmadd_ps(w1, in1, sum1);
                sum2 = _mm256_fmadd_ps(w2, in2, sum2);
                sum3 = _mm256_fmadd_ps(w3, in3, sum3);
                sum0 = _mm256_fmadd_ps(w4, in4, sum0);
                sum1 = _mm256_fmadd_ps(w5, in5, sum1);
                sum2 = _mm256_fmadd_ps(w6, in6, sum2);
                sum3 = _mm256_fmadd_ps(w7, in7, sum3);
                
                w0 = _mm256_set1_ps(weightRow[inIdx + 8]);
                w1 = _mm256_set1_ps(weightRow[inIdx + 9]);
                w2 = _mm256_set1_ps(weightRow[inIdx + 10]);
                w3 = _mm256_set1_ps(weightRow[inIdx + 11]);
                w4 = _mm256_set1_ps(weightRow[inIdx + 12]);
                w5 = _mm256_set1_ps(weightRow[inIdx + 13]);
                w6 = _mm256_set1_ps(weightRow[inIdx + 14]);
                w7 = _mm256_set1_ps(weightRow[inIdx + 15]);
                
                i0 = inputs + (inIdx + 8) * batch_size + batchIdx;
                i1 = inputs + (inIdx + 9) * batch_size + batchIdx;
                i2 = inputs + (inIdx + 10) * batch_size + batchIdx;
                i3 = inputs + (inIdx + 11) * batch_size + batchIdx;
                i4 = inputs + (inIdx + 12) * batch_size + batchIdx;
                i5 = inputs + (inIdx + 13) * batch_size + batchIdx;
                i6 = inputs + (inIdx + 14) * batch_size + batchIdx;
                i7 = inputs + (inIdx + 15) * batch_size + batchIdx;
                
                in0 = _mm256_load_ps(i0);
                in1 = _mm256_load_ps(i1);
                in2 = _mm256_load_ps(i2);
                in3 = _mm256_load_ps(i3);
                in4 = _mm256_load_ps(i4);
                in5 = _mm256_load_ps(i5);
                in6 = _mm256_load_ps(i6);
                in7 = _mm256_load_ps(i7);
                
                sum0 = _mm256_fmadd_ps(w0, in0, sum0);
                sum1 = _mm256_fmadd_ps(w1, in1, sum1);
                sum2 = _mm256_fmadd_ps(w2, in2, sum2);
                sum3 = _mm256_fmadd_ps(w3, in3, sum3);
                sum0 = _mm256_fmadd_ps(w4, in4, sum0);
                sum1 = _mm256_fmadd_ps(w5, in5, sum1);
                sum2 = _mm256_fmadd_ps(w6, in6, sum2);
                sum3 = _mm256_fmadd_ps(w7, in7, sum3);
                
                w0 = _mm256_set1_ps(weightRow[inIdx + 16]);
                w1 = _mm256_set1_ps(weightRow[inIdx + 17]);
                w2 = _mm256_set1_ps(weightRow[inIdx + 18]);
                w3 = _mm256_set1_ps(weightRow[inIdx + 19]);
                w4 = _mm256_set1_ps(weightRow[inIdx + 20]);
                w5 = _mm256_set1_ps(weightRow[inIdx + 21]);
                w6 = _mm256_set1_ps(weightRow[inIdx + 22]);
                w7 = _mm256_set1_ps(weightRow[inIdx + 23]);
                
                i0 = inputs + (inIdx + 16) * batch_size + batchIdx;
                i1 = inputs + (inIdx + 17) * batch_size + batchIdx;
                i2 = inputs + (inIdx + 18) * batch_size + batchIdx;
                i3 = inputs + (inIdx + 19) * batch_size + batchIdx;
                i4 = inputs + (inIdx + 20) * batch_size + batchIdx;
                i5 = inputs + (inIdx + 21) * batch_size + batchIdx;
                i6 = inputs + (inIdx + 22) * batch_size + batchIdx;
                i7 = inputs + (inIdx + 23) * batch_size + batchIdx;
                
                in0 = _mm256_load_ps(i0);
                in1 = _mm256_load_ps(i1);
                in2 = _mm256_load_ps(i2);
                in3 = _mm256_load_ps(i3);
                in4 = _mm256_load_ps(i4);
                in5 = _mm256_load_ps(i5);
                in6 = _mm256_load_ps(i6);
                in7 = _mm256_load_ps(i7);
                
                sum0 = _mm256_fmadd_ps(w0, in0, sum0);
                sum1 = _mm256_fmadd_ps(w1, in1, sum1);
                sum2 = _mm256_fmadd_ps(w2, in2, sum2);
                sum3 = _mm256_fmadd_ps(w3, in3, sum3);
                sum0 = _mm256_fmadd_ps(w4, in4, sum0);
                sum1 = _mm256_fmadd_ps(w5, in5, sum1);
                sum2 = _mm256_fmadd_ps(w6, in6, sum2);
                sum3 = _mm256_fmadd_ps(w7, in7, sum3);
                
                w0 = _mm256_set1_ps(weightRow[inIdx + 24]);
                w1 = _mm256_set1_ps(weightRow[inIdx + 25]);
                w2 = _mm256_set1_ps(weightRow[inIdx + 26]);
                w3 = _mm256_set1_ps(weightRow[inIdx + 27]);
                w4 = _mm256_set1_ps(weightRow[inIdx + 28]);
                w5 = _mm256_set1_ps(weightRow[inIdx + 29]);
                w6 = _mm256_set1_ps(weightRow[inIdx + 30]);
                w7 = _mm256_set1_ps(weightRow[inIdx + 31]);
                
                i0 = inputs + (inIdx + 24) * batch_size + batchIdx;
                i1 = inputs + (inIdx + 25) * batch_size + batchIdx;
                i2 = inputs + (inIdx + 26) * batch_size + batchIdx;
                i3 = inputs + (inIdx + 27) * batch_size + batchIdx;
                i4 = inputs + (inIdx + 28) * batch_size + batchIdx;
                i5 = inputs + (inIdx + 29) * batch_size + batchIdx;
                i6 = inputs + (inIdx + 30) * batch_size + batchIdx;
                i7 = inputs + (inIdx + 31) * batch_size + batchIdx;
                
                in0 = _mm256_load_ps(i0);
                in1 = _mm256_load_ps(i1);
                in2 = _mm256_load_ps(i2);
                in3 = _mm256_load_ps(i3);
                in4 = _mm256_load_ps(i4);
                in5 = _mm256_load_ps(i5);
                in6 = _mm256_load_ps(i6);
                in7 = _mm256_load_ps(i7);
                
                sum0 = _mm256_fmadd_ps(w0, in0, sum0);
                sum1 = _mm256_fmadd_ps(w1, in1, sum1);
                sum2 = _mm256_fmadd_ps(w2, in2, sum2);
                sum3 = _mm256_fmadd_ps(w3, in3, sum3);
                sum0 = _mm256_fmadd_ps(w4, in4, sum0);
                sum1 = _mm256_fmadd_ps(w5, in5, sum1);
                sum2 = _mm256_fmadd_ps(w6, in6, sum2);
                sum3 = _mm256_fmadd_ps(w7, in7, sum3);
            }
            
            for (; inIdx + 8 <= input_dim; inIdx += 8)
            {
                __m256 w0 = _mm256_set1_ps(weightRow[inIdx]);
                __m256 w1 = _mm256_set1_ps(weightRow[inIdx + 1]);
                __m256 w2 = _mm256_set1_ps(weightRow[inIdx + 2]);
                __m256 w3 = _mm256_set1_ps(weightRow[inIdx + 3]);
                __m256 w4 = _mm256_set1_ps(weightRow[inIdx + 4]);
                __m256 w5 = _mm256_set1_ps(weightRow[inIdx + 5]);
                __m256 w6 = _mm256_set1_ps(weightRow[inIdx + 6]);
                __m256 w7 = _mm256_set1_ps(weightRow[inIdx + 7]);
                
                __m256 in0 = _mm256_load_ps(inputs + (inIdx) * batch_size + batchIdx);
                __m256 in1 = _mm256_load_ps(inputs + (inIdx + 1) * batch_size + batchIdx);
                __m256 in2 = _mm256_load_ps(inputs + (inIdx + 2) * batch_size + batchIdx);
                __m256 in3 = _mm256_load_ps(inputs + (inIdx + 3) * batch_size + batchIdx);
                __m256 in4 = _mm256_load_ps(inputs + (inIdx + 4) * batch_size + batchIdx);
                __m256 in5 = _mm256_load_ps(inputs + (inIdx + 5) * batch_size + batchIdx);
                __m256 in6 = _mm256_load_ps(inputs + (inIdx + 6) * batch_size + batchIdx);
                __m256 in7 = _mm256_load_ps(inputs + (inIdx + 7) * batch_size + batchIdx);
                
                sum0 = _mm256_fmadd_ps(w0, in0, sum0);
                sum1 = _mm256_fmadd_ps(w1, in1, sum1);
                sum2 = _mm256_fmadd_ps(w2, in2, sum2);
                sum3 = _mm256_fmadd_ps(w3, in3, sum3);
                sum0 = _mm256_fmadd_ps(w4, in4, sum0);
                sum1 = _mm256_fmadd_ps(w5, in5, sum1);
                sum2 = _mm256_fmadd_ps(w6, in6, sum2);
                sum3 = _mm256_fmadd_ps(w7, in7, sum3);
            }
            
            __m256 sum01 = _mm256_add_ps(sum0, sum1);
            __m256 sum23 = _mm256_add_ps(sum2, sum3);
            __m256 total = _mm256_add_ps(sum01, sum23);
            
            for (; inIdx < input_dim; ++inIdx)
            {
                __m256 w = _mm256_set1_ps(weightRow[inIdx]);
                __m256 in = _mm256_load_ps(inputs + inIdx * batch_size + batchIdx);
                total = _mm256_fmadd_ps(w, in, total);
            }
            
            _mm256_store_ps(outCol + batchIdx, total);
        }
        
        for (; batchIdx < batch_size; ++batchIdx)
        {
            float sum = 0.0f;
            for (size_t inIdx = 0; inIdx < input_dim; ++inIdx)
            {
                sum += weightRow[inIdx] * inputs[inIdx * batch_size + batchIdx];
            }
            outCol[batchIdx] = sum;
        }
    }
}

// ============================================================================
// MoLU BACKWARD PASS
// ============================================================================

void BackwardMoLU_Scalar(float* grad, const float* cached_input, size_t size)
{
    // MoLU forward: y = 0.5 * x * (1 + tanh(x))
    // MoLU derivative: dy/dx = 0.5 * (1 + tanh(x)) + 0.5 * x * (1 - tanh(x)^2)
    // Backward: grad_out *= dy/dx
    for (size_t i = 0; i < size; ++i)
    {
        float x = cached_input[i];
        float tanh_x = tanhf(x);
        float tanh_sq = tanh_x * tanh_x;
        float derivative = 0.5f * (1.0f + tanh_x) + 0.5f * x * (1.0f - tanh_sq);
        grad[i] *= derivative;
    }
}

void BackwardMoLU_AVX2(float* grad, const float* cached_input, size_t size)
{
    AssertAligned32(grad);
    AssertAligned32(cached_input);

    const size_t simdWidth = 8;
    const size_t simdEnd = size - (size % simdWidth);

    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 clampHi = _mm256_set1_ps(10.0f);
    const __m256 clampLo = _mm256_set1_ps(-10.0f);

    // Pade constants for tanh(x)
    const __m256 pade_a = _mm256_set1_ps(0.275f);
    const __m256 pade_b = _mm256_set1_ps(0.664f);

    size_t i = 0;
    for (; i < simdEnd; i += simdWidth)
    {
        // Load cached input and gradient
        __m256 x = _mm256_load_ps(cached_input + i);
        __m256 g = _mm256_load_ps(grad + i);

        // Clamp x for numerical stability
        __m256 xc = _mm256_min_ps(x, clampHi);
        xc = _mm256_max_ps(xc, clampLo);

        // Compute tanh(x) using Pade approximation
        __m256 x2 = _mm256_mul_ps(xc, xc);
        __m256 num = _mm256_fmadd_ps(pade_a, x2, one);
        num = _mm256_mul_ps(xc, num);
        __m256 den = _mm256_fmadd_ps(pade_b, x2, one);
        __m256 rcp = _mm256_rcp_ps(den);
        rcp = _mm256_mul_ps(rcp, _mm256_sub_ps(two, _mm256_mul_ps(den, rcp)));
        __m256 tanh_x = _mm256_mul_ps(num, rcp);

        // Compute derivative: dy/dx = 0.5 * (1 + tanh(x)) + 0.5 * x * (1 - tanh(x)^2)
        __m256 one_plus_tanh = _mm256_add_ps(one, tanh_x);
        __m256 tanh_sq = _mm256_mul_ps(tanh_x, tanh_x);
        __m256 one_minus_tanh_sq = _mm256_sub_ps(one, tanh_sq);
        __m256 x_times = _mm256_mul_ps(x, one_minus_tanh_sq);
        __m256 deriv = _mm256_mul_ps(half, one_plus_tanh);
        deriv = _mm256_fmadd_ps(half, x_times, deriv);

        // Multiply gradient by derivative
        __m256 result = _mm256_mul_ps(g, deriv);
        _mm256_store_ps(grad + i, result);
    }

    // Handle remainder
    for (; i < size; ++i)
    {
        float x = cached_input[i];
        float tanh_x = tanhf(std::clamp(x, -10.0f, 10.0f));
        float tanh_sq = tanh_x * tanh_x;
        float derivative = 0.5f * (1.0f + tanh_x) + 0.5f * x * (1.0f - tanh_sq);
        grad[i] *= derivative;
    }
}
