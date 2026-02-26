#pragma once

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <immintrin.h>

constexpr size_t OBS_DIM = 240;  // Updated: 236 actual + 4 AVX2 padding
constexpr size_t LATENT_DIM = 64;
constexpr size_t ACTOR_INPUT_DIM = OBS_DIM + LATENT_DIM;  // 240 + 64 = 304
constexpr size_t ACTOR_HIDDEN_DIM = 512;
constexpr size_t ACTOR_LAYERS = 3;
constexpr size_t ACTION_DIM = 56;  // 13 satellites * 4 + 4 reaction wheels
constexpr size_t CRITIC_INPUT_DIM = OBS_DIM + ACTION_DIM + LATENT_DIM;  // 240 + 56 + 64 = 360
constexpr size_t CRITIC_HIDDEN_DIM = 1024;
constexpr size_t CRITIC_LAYERS = 3;
constexpr size_t REWARD_VECTOR_DIM = 4;
constexpr size_t NUM_PARALLEL_ENVS = 128;

constexpr size_t AVX2_WIDTH = 8;
constexpr size_t AVX2_ALIGNMENT = 32;
constexpr size_t CACHE_LINE_SIZE = 64;

constexpr size_t PAD_TO_AVX2(size_t dim) {
    return ((dim + AVX2_WIDTH - 1) / AVX2_WIDTH) * AVX2_WIDTH;
}

constexpr size_t OBS_DIM_ALIGNED = PAD_TO_AVX2(OBS_DIM);
constexpr size_t LATENT_DIM_ALIGNED = PAD_TO_AVX2(LATENT_DIM);
constexpr size_t ACTOR_HIDDEN_DIM_ALIGNED = PAD_TO_AVX2(ACTOR_HIDDEN_DIM);
constexpr size_t CRITIC_HIDDEN_DIM_ALIGNED = PAD_TO_AVX2(CRITIC_HIDDEN_DIM);

// AVX2 alignment requirements - enforce at compile time
static_assert(OBS_DIM % AVX2_WIDTH == 0, "OBS_DIM must be multiple of 8 for AVX2");
static_assert(ACTION_DIM % AVX2_WIDTH == 0, "ACTION_DIM must be multiple of 8 for AVX2");
static_assert(ACTOR_INPUT_DIM % AVX2_WIDTH == 0, "ACTOR_INPUT_DIM must be multiple of 8 for AVX2");
static_assert(CRITIC_INPUT_DIM % AVX2_WIDTH == 0, "CRITIC_INPUT_DIM must be multiple of 8 for AVX2");

inline void AssertAligned32(const void* ptr) {
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 32 == 0 && "Memory must be 32-byte aligned for AVX2");
}

void ForwardMoLU_AVX2(float* data, size_t size);
void ForwardMoLU_Scalar(float* data, size_t size);

void ForwardTanh_AVX2(float* data, size_t size);
void ForwardReLU_AVX2(float* data, size_t size);
void ForwardSigmoid_AVX2(float* data, size_t size);

void AddVectors_AVX2(float* dst, const float* src, size_t size);
void ScaleVector_AVX2(float* dst, float scale, size_t size);
void FMAVector_AVX2(float* dst, const float* a, const float* b, size_t size);

void Softmax_AVX2(float* data, size_t size);
void LayerNorm_AVX2(float* data, size_t size, const float* gamma, const float* beta);

void MatMul_AVX2(const float* A, const float* B, float* C, 
                 size_t M, size_t K, size_t N);

void MatVec_FMA_AVX2(const float* weights, const float* inputs, float* outputs,
                      size_t input_dim, size_t output_dim);

void MatVec_FMA_AVX2_Prefetch(const float* weights, const float* inputs, float* outputs,
                               size_t input_dim, size_t output_dim);

void Transpose8x8_AVX2(const float* src, float* dst, size_t srcStride, size_t dstStride);

void TransposeBatch_AoS_to_SoA(const float* aos_input, float* soa_output,
                                size_t batch_size, size_t feature_dim);

void MatVec_Vertical_AVX2(const float* weights, const float* inputs, float* outputs,
                            size_t input_dim, size_t output_dim, size_t batch_size);

inline size_t GetAVX2PaddedSize(size_t dim) {
    return ((dim + AVX2_WIDTH - 1) / AVX2_WIDTH) * AVX2_WIDTH;
}

inline void PrefetchL1(const void* ptr) {
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
}

inline void PrefetchL2(const void* ptr) {
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T1);
}

inline void PrefetchNTA(const void* ptr) {
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_NTA);
}
