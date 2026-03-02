#pragma once
// ============================================================================
// PRIMALPHA: Multi-Head Attention State Encoder
// ============================================================================
// Implements self-attention over observation dimensions to focus on
// relevant features in different combat contexts
// ============================================================================

#include "SpanNetwork.h"
#include "AlignedAllocator.h"
#include <vector>
#include <random>

class alignas(32) MultiHeadAttention {
public:
    MultiHeadAttention() = default;
    
    void Init(size_t inputDim, size_t embedDim, size_t numHeads, std::mt19937& rng);
    
    // Attend to input features with multi-head attention
    void Attend(const float* input, float* output, int batchSize);
    
    // Get attention weights (for debugging/visualization)
    const AlignedVector32<float>& GetAttentionWeights() const { return mAttentionWeights; }
    
private:
    size_t mNumHeads = 8;
    size_t mInputDim = 0;
    size_t mEmbedDim = 0;
    size_t mHeadDim = 0;  // embedDim / numHeads
    
    // Learnable projections
    AlignedVector32<float> mQueryWeights;    // [inputDim, embedDim]
    AlignedVector32<float> mKeyWeights;      // [inputDim, embedDim]
    AlignedVector32<float> mValueWeights;    // [inputDim, embedDim]
    AlignedVector32<float> mOutputWeights;   // [embedDim, inputDim]
    
    // Buffers
    AlignedVector32<float> mQueries;         // [batchSize, embedDim]
    AlignedVector32<float> mKeys;            // [batchSize, embedDim]
    AlignedVector32<float> mValues;          // [batchSize, embedDim]
    AlignedVector32<float> mAttentionWeights; // [batchSize, numHeads, seqLen]
    AlignedVector32<float> mAttentionOutput; // [batchSize, embedDim]
    
    // Compute scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V
    void ComputeScaledDotProductAttention(const float* Q, const float* K, const float* V,
                                          float* output, float* weights, int batchSize);
};

class alignas(32) AttentionStateEncoder {
public:
    AttentionStateEncoder() = default;
    
    void Init(size_t obsDim, size_t latentDim, std::mt19937& rng);
    
    // Encode observations with attention mechanism
    void Encode(const float* observations, float* attendedState, int batchSize);
    
    // Get attention weights for visualization
    const AlignedVector32<float>& GetAttentionWeights() const {
        return mAttention.GetAttentionWeights();
    }
    
private:
    MultiHeadAttention mAttention;
    SpanNetwork mOutputMLP;  // MLP after attention
    
    size_t mObsDim = 0;
    size_t mLatentDim = 0;
    size_t mAttendDim = 256;  // Attention embedding dimension
    
    AlignedVector32<float> mAttendedFeatures;
};
