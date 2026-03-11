#pragma once

#include "Tensor.hpp"
#include <cmath>
#include <random>
#include <immintrin.h>
#include <algorithm>

namespace cput::rff {

/**
 * RFF Tensor Layer
 * 
 * Uses cput::Tensor for all storage and operations.
 * Implements high-performance RFF projection.
 */
class RFFTensorLayer {
public:
    RFFTensorLayer() = default;

    /**
     * Initialize the layer
     * @param in_dim Input dimension
     * @param rff_dim RFF feature dimension
     * @param sigma RBF bandwidth
     */
    void init(size_t in_dim, size_t rff_dim, float sigma = 1.0f) {
        m_in_dim = in_dim;
        m_rff_dim = rff_dim;
        m_sigma = sigma;

        // W_fixed: [rff_dim, in_dim]
        m_W_fixed.resize({rff_dim, in_dim});
        
        // b_fixed: [rff_dim]
        m_b_fixed.resize({rff_dim});

        // Initialize W_fixed ~ N(0, 1/sigma^2)
        std::mt19937 gen(42);
        float std_dev = 1.0f / sigma;
        std::normal_distribution<float> w_dist(0.0f, std_dev);
        for (size_t i = 0; i < m_W_fixed.size(); ++i) {
            m_W_fixed.data()[i] = w_dist(gen);
        }

        // Initialize b_fixed ~ U[0, 2π]
        std::uniform_real_distribution<float> b_dist(0.0f, 2.0f * 3.14159265359f);
        for (size_t i = 0; i < m_b_fixed.size(); ++i) {
            m_b_fixed.data()[i] = b_dist(gen);
        }

        // Trainable parameters
        m_W_train.resize({1, rff_dim}); // Output dim = 1 for now (expandable)
        m_b_train.resize({1, 1});

        // Initialize W_train ~ N(0, 1/sqrt(rff_dim))
        std::normal_distribution<float> train_dist(0.0f, 1.0f / std::sqrt(static_cast<float>(rff_dim)));
        for (size_t i = 0; i < m_W_train.size(); ++i) {
            m_W_train.data()[i] = train_dist(gen);
        }
        m_b_train.fill(0.0f);
    }

    /**
     * Forward Pass: y = W_train * cos(W_fixed * x + b_fixed) * scale + b_train
     * @param input Input tensor [batch_size, in_dim]
     * @param output Output tensor [batch_size, 1]
     */
    void forward(const TensorF32& input, TensorF32& output) {
        size_t batch_size = input.shape(0);
        
        // 1. Linear projection: z = W_fixed * x + b_fixed
        // Workspace: [batch_size, rff_dim]
        m_workspace.resize({batch_size, m_rff_dim});
        
        const float* x_ptr = input.data();
        const float* W_f_ptr = m_W_fixed.data();
        const float* b_f_ptr = m_b_fixed.data();
        float* z_ptr = m_workspace.data();

        // Optimized matrix-vector product for each sample in batch
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < m_rff_dim; ++i) {
                float sum = b_f_ptr[i];
                for (size_t j = 0; j < m_in_dim; ++j) {
                    sum += W_f_ptr[i * m_in_dim + j] * x_ptr[b * m_in_dim + j];
                }
                z_ptr[b * m_rff_dim + i] = sum;
            }
        }

        // 2. Cosine activation & Scaling
        float scale = std::sqrt(2.0f / static_cast<float>(m_rff_dim));
        for (size_t i = 0; i < m_workspace.size(); ++i) {
            m_workspace.data()[i] = std::cos(m_workspace.data()[i]) * scale;
        }

        // 3. Trainable linear layer
        // output = m_workspace * m_W_train^T + m_b_train
        output.resize({batch_size, 1});
        const float* phi_ptr = m_workspace.data();
        const float* W_t_ptr = m_W_train.data();
        float b_t = m_b_train.data()[0];
        float* out_ptr = output.data();

        for (size_t b = 0; b < batch_size; ++b) {
            float sum = b_t;
            for (size_t i = 0; i < m_rff_dim; ++i) {
                sum += W_t_ptr[i] * phi_ptr[b * m_rff_dim + i];
            }
            out_ptr[b] = sum;
        }
    }

    // Accessors
    TensorF32& getWeights() { return m_W_train; }
    TensorF32& getBias() { return m_b_train; }

private:
    size_t m_in_dim = 0;
    size_t m_rff_dim = 0;
    float m_sigma = 1.0f;

    TensorF32 m_W_fixed; // [rff_dim, in_dim]
    VectorF32 m_b_fixed; // [rff_dim]

    TensorF32 m_W_train; // [out_dim, rff_dim]
    TensorF32 m_b_train; // [out_dim, 1]

    TensorF32 m_workspace;
};

} // namespace cput::rff
