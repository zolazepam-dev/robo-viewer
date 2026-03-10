#pragma once

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <immintrin.h>

// Maximum number of parallel environments - increased for high-SPS training
// Each environment runs 2 robots, so 1024 envs = 2048 robots simulated in parallel
constexpr size_t NUM_PARALLEL_ENVS = 4096;
constexpr size_t NUM_PARALLEL_ROBOTS = NUM_PARALLEL_ENVS * 2;

// Buffer safety limits
constexpr size_t MAX_LATENT_DIM = 128; 

constexpr size_t AVX2_WIDTH = 8;
constexpr size_t AVX2_ALIGNMENT = 32;
constexpr size_t CACHE_LINE_SIZE = 64;

constexpr size_t PAD_TO_AVX2(size_t dim) {
    return ((dim + AVX2_WIDTH - 1) / AVX2_WIDTH) * AVX2_WIDTH;
}

inline size_t GetAVX2PaddedSize(size_t dim) {
    return ((dim + AVX2_WIDTH - 1) / AVX2_WIDTH) * AVX2_WIDTH;
}

// AVX2 alignment requirements - enforce at compile time for fixed-size buffers if any remain
// (These static asserts may need to be removed if all buffers become dynamic)
// static_assert(OBS_DIM % AVX2_WIDTH == 0, "OBS_DIM must be multiple of 8 for AVX2");
// static_assert(ACTION_DIM % AVX2_WIDTH == 0, "ACTION_DIM must be multiple of 8 for AVX2");


inline void AssertAligned32(const void* ptr) {
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 32 == 0 && "Memory must be 32-byte aligned for AVX2");
}

void ForwardMoLU_AVX2(float* data, size_t size);
void ForwardMoLU_Scalar(float* data, size_t size);

// Backward pass for MoLU activation
// Requires cached forward input values for proper gradient computation
void BackwardMoLU_AVX2(float* grad, const float* cached_input, size_t size);
void BackwardMoLU_Scalar(float* grad, const float* cached_input, size_t size);

void ForwardTanh_AVX2(float* data, size_t size);
void ForwardReLU_AVX2(float* data, size_t size);
void ForwardSigmoid_AVX2(float* data, size_t size);

void AddVectors_AVX2(float* dst, const float* src, size_t size);
void ScaleVector_AVX2(float* dst, float scale, size_t size);
void FMAVector_AVX2(float* dst, const float* a, const float* b, size_t size);

void Softmax_AVX2(float* data, size_t size);
void LayerNorm_AVX2(float* data, size_t size, const float* gamma, const float* beta);

// Memory safety utilities
size_t GetAvailableMemoryBytes();

void MatMul_AVX2(const float* A, const float* B, float* C, 
                 size_t M, size_t K, size_t N);

void MatVec_FMA_AVX2(const float* weights, const float* inputs, float* outputs,
                      size_t input_dim, size_t output_dim);

void MatVec_FMA_AVX2_Prefetch(const float* weights, const float* inputs, float* outputs,
                               size_t input_dim, size_t output_dim);
// AVX2 transformations
void Transpose8x8_AVX2(const float* src, float* dst, size_t srcStride, size_t dstStride);

void TransposeBatch_AoS_to_SoA(const float* aos_input, float* soa_output,
                                size_t batch_size, size_t feature_dim);

void MatVec_Vertical_AVX2(const float* weights, const float* inputs, float* outputs,
                            size_t input_dim, size_t output_dim, size_t batch_size);

inline void PrefetchL1(const void* ptr) {
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
}

inline void PrefetchL2(const void* ptr) {
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T1);
}

inline void PrefetchNTA(const void* ptr) {
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_NTA);
}
