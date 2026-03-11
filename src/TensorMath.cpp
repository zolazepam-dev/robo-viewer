// TensorMath.cpp - Math operations for tensors
#include <cmath>
#include <cstring>

namespace cput {

// Matrix multiplication: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
void matMul(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
    std::memset(C, 0, M * N * sizeof(float));
    
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            float a_val = A[m * K + k];
            for (size_t n = 0; n < N; ++n) {
                C[m * N + n] += a_val * B[k * N + n];
            }
        }
    }
}

// Matrix multiplication with bias: C = A * B + bias
void matMulBias(const float* A, const float* B, float* C, float* bias, 
                size_t M, size_t K, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        std::memcpy(C + m * N, bias, N * sizeof(float));
        for (size_t k = 0; k < K; ++k) {
            float a_val = A[m * K + k];
            for (size_t n = 0; n < N; ++n) {
                C[m * N + n] += a_val * B[k * N + n];
            }
        }
    }
}

// Transpose matrix: B = A^T
void transpose(const float* A, float* B, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

} // namespace cput
