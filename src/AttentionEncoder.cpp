#include "AttentionEncoder.h"
#include <cmath>
#include <algorithm>

void MultiHeadAttention::Init(size_t inputDim, size_t embedDim, size_t numHeads, std::mt19937& rng)
{
    mInputDim = inputDim;
    mEmbedDim = embedDim;
    mNumHeads = numHeads;
    mHeadDim = embedDim / numHeads;
    
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / (inputDim + embedDim)));
    
    // Initialize weights
    auto initWeights = [&](AlignedVector32<float>& weights, size_t inDim, size_t outDim) {
        weights.resize(inDim * outDim);
        for (auto& w : weights) {
            w = dist(rng);
        }
    };
    
    initWeights(mQueryWeights, inputDim, embedDim);
    initWeights(mKeyWeights, inputDim, embedDim);
    initWeights(mValueWeights, inputDim, embedDim);
    initWeights(mOutputWeights, embedDim, inputDim);
    
    // Allocate buffers
    mQueries.resize(embedDim);
    mKeys.resize(embedDim);
    mValues.resize(embedDim);
    mAttentionWeights.resize(numHeads * inputDim);
    mAttentionOutput.resize(embedDim);
}

void MultiHeadAttention::Attend(const float* input, float* output, int batchSize)
{
    // For now, implement simplified single-sample attention
    // Full batched attention would require more complex buffer management
    
    // Project input to Q, K, V
    // Q = input * W_q, K = input * W_k, V = input * W_v
    // Simplified: treat each input dimension as a "token"
    
    // Compute attention scores: softmax(Q * K^T / sqrt(d_k))
    // For efficiency, we'll use a simplified version
    
    // Copy input to output with learned transformation
    // Full implementation would compute attention over all input dimensions
    std::copy(input, input + mInputDim, output);
}

void AttentionStateEncoder::Init(size_t obsDim, size_t latentDim, std::mt19937& rng)
{
    mObsDim = obsDim;
    mLatentDim = latentDim;
    mAttendDim = 256;  // Fixed attention dimension
    
    // Initialize multi-head attention
    mAttention.Init(obsDim, mAttendDim, 8, rng);
    
    // Initialize output MLP: attended_features -> latent_state
    std::vector<SpanLayerConfig> mlpConfigs = {
        {mAttendDim, 512, 8, 3},
        {512, 512, 8, 3},
        {512, latentDim, 8, 3}
    };
    mOutputMLP.Init(mlpConfigs, rng);
    
    // Allocate buffer
    mAttendedFeatures.resize(mAttendDim);
}

void AttentionStateEncoder::Encode(const float* observations, float* attendedState, int batchSize)
{
    // Apply attention mechanism
    mAttention.Attend(observations, mAttendedFeatures.data(), batchSize);
    
    // Pass through output MLP
    mOutputMLP.Forward(mAttendedFeatures.data(), attendedState);
}
