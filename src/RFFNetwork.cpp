#include "RFFNetwork.h"

#include <algorithm>
#include <immintrin.h>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include <cstdio>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "AlignedAllocator.h"
#include "src/EigenUtils.h"

void RFFNetwork::Init(const std::vector<RFFLayerConfig>& layerConfigs, std::mt19937& rng)
{
    mLayers.resize(layerConfigs.size());
    mLayerInputDims.resize(layerConfigs.size());
    mLayerOutputDims.resize(layerConfigs.size());
    
    size_t maxDim = 0;
    for (size_t i = 0; i < layerConfigs.size(); ++i) {
        mLayerInputDims[i] = layerConfigs[i].inputDim;
        mLayerOutputDims[i] = layerConfigs[i].outputDim;
        mLayers[i].Init(layerConfigs[i].inputDim, layerConfigs[i].outputDim, 
                        layerConfigs[i].rffConfig, rng);
        maxDim = std::max(maxDim, std::max(layerConfigs[i].inputDim, layerConfigs[i].outputDim));
    }
    
    if (!layerConfigs.empty()) {
        mInputDim = layerConfigs.front().inputDim;
        mOutputDim = layerConfigs.back().outputDim;
    }
    
    mActivationBuffer.resize(maxDim * 2);
}

void RFFNetwork::Forward(const float* input, float* output)
{
    if (mLayers.empty()) return;
    
    const float* curIn = input;
    float* curOut = mActivationBuffer.data();
    float* nextOut = mActivationBuffer.data() + mActivationBuffer.size() / 2;
    
    for (size_t i = 0; i < mLayers.size(); ++i) {
        mLayers[i].Forward(curIn, curOut);
        
        // Apply MoLU activation between layers (not on final layer)
        if (i < mLayers.size() - 1) {
            ForwardMoLU_AVX2(curOut, mLayerOutputDims[i]);
        }
        
        curIn = curOut;
        curOut = (curOut == mActivationBuffer.data()) ? nextOut : mActivationBuffer.data();
    }
    
    std::memcpy(output, curIn, mOutputDim * sizeof(float));
}

void RFFNetwork::ForwardBatch(const float* input, float* output, int batchSize)
{
    if (mLayers.empty()) return;
    
    // Use Eigen-optimized implementation
    ForwardBatchEigen(input, output, batchSize);
}

void RFFNetwork::ForwardBatchEigen(const float* input, float* output, int batchSize)
{
    if (mLayers.empty()) return;

    // Map input: [batch_size × input_dim]
    Eigen::Map<const Eigen::MatrixXf> inputMap(input, batchSize, mInputDim);
    
    // Process through layers
    Eigen::MatrixXf current = inputMap;
    
    for (size_t i = 0; i < mLayers.size(); ++i) {
        current = mLayers[i].ForwardEigen(current);
        
        // Apply MoLU activation between layers (not on final layer)
        if (i < mLayers.size() - 1) {
            current = current.unaryExpr([](float x) {
                return 0.5f * x * (1.0f + std::tanh(x));
            });
        }
    }
    
    // Copy output
    std::memcpy(output, current.data(), batchSize * mOutputDim * sizeof(float));
}

void RFFNetwork::ForwardWithLatent(const float* input, float* output, 
                                    SecondOrderLatentMemory& latent, int envIdx)
{
    float* zPos = latent.GetPosition(envIdx);
    
    AlignedVector32<float> combined(mInputDim + latent.latentDim);
    std::copy(input, input + mInputDim, combined.begin());
    std::copy(zPos, zPos + latent.latentDim, combined.begin() + mInputDim);
    
    Forward(combined.data(), output);
}

void RFFNetwork::ForwardWithCache(const float* input, float* output, SpanCache& cache)
{
    if (mLayers.empty()) return;
    
    cache.layerInputs.resize(mLayers.size());
    cache.layerOutputs.resize(mLayers.size());
    cache.layerFeatures.resize(mLayers.size());
    
    const float* curIn = input;
    
    for (size_t i = 0; i < mLayers.size(); ++i) {
        // Store input
        cache.layerInputs[i].resize(mLayerInputDims[i]);
        std::memcpy(cache.layerInputs[i].data(), curIn, mLayerInputDims[i] * sizeof(float));
        
        // Forward through RFF layer
        auto& layer = mLayers[i];
        
        // Compute and store RFF features
        layer.ComputeFeatures(curIn, cache.layerFeatures[i].data());
        
        // Compute output
        cache.layerOutputs[i].resize(mLayerOutputDims[i]);
        layer.Forward(curIn, cache.layerOutputs[i].data());
        
        // Apply MoLU activation between layers (not on final layer)
        if (i < mLayers.size() - 1) {
            ForwardMoLU_AVX2(cache.layerOutputs[i].data(), mLayerOutputDims[i]);
        }
        
        curIn = cache.layerOutputs[i].data();
    }
    
    std::memcpy(output, curIn, mOutputDim * sizeof(float));
}

void RFFNetwork::ForwardWithCache(const float* input, float* output)
{
    static thread_local SpanCache internalCache;
    ForwardWithCache(input, output, internalCache);
}

void RFFNetwork::Backward(const float* input, const float* output_grad, float* input_grad,
                          float* weights_grad, SpanCache& cache)
{
    if (mLayers.empty()) return;
    
    // Backpropagate through layers in reverse order
    AlignedVector32<float> currentGrad(mOutputDim);
    std::memcpy(currentGrad.data(), output_grad, mOutputDim * sizeof(float));
    
    size_t gradOffset = 0;
    
    for (int i = static_cast<int>(mLayers.size()) - 1; i >= 0; --i) {
        auto& layer = mLayers[i];
        
        // Get cached values
        const float* layerInput = cache.layerInputs[i].data();
        (void)layerInput;  // Used by layer.Backward
        
        AlignedVector32<float> nextGrad(mLayerInputDims[i]);
        
        // Compute gradients for this layer
        float* layerWeightsGrad = weights_grad ? (weights_grad + gradOffset) : nullptr;
        float* layerBiasGrad = weights_grad ? (weights_grad + gradOffset + layer.GetTrainableWeights().size()) : nullptr;
        
        // Backward through RFF layer
        layer.Backward(layerInput, currentGrad.data(), nextGrad.data(), layerWeightsGrad, layerBiasGrad);
        
        // Apply MoLU backward gradient if not first layer
        if (i > 0) {
            for (size_t j = 0; j < mLayerOutputDims[i-1]; ++j) {
                float x = cache.layerOutputs[i-1][j];
                float th = tanhf(std::clamp(x, -10.0f, 10.0f));
                nextGrad[j] *= (0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th));
            }
        }
        
        currentGrad = nextGrad;
        
        // Update gradient offset for next layer
        gradOffset += layer.GetNumParams();
    }
    
    if (input_grad) {
        std::memcpy(input_grad, currentGrad.data(), mInputDim * sizeof(float));
    }
}

void RFFNetwork::Backward(const float* input, const float* output_grad, float* input_grad, bool accumulate_grads)
{
    static thread_local SpanCache internalCache;
    
    // Create a temporary output buffer for the forward pass
    AlignedVector32<float> tempOutput(mOutputDim);
    ForwardWithCache(input, tempOutput.data(), internalCache);
    
    if (!accumulate_grads) ZeroGradients();
    
    // Collect all gradients into a single buffer
    size_t totalParams = GetNumWeights();
    AlignedVector32<float> allGrads(totalParams, 0.0f);
    
    Backward(input, output_grad, input_grad, allGrads.data(), internalCache);
    
    // Distribute gradients to layers
    size_t gradOffset = 0;
    for (size_t i = 0; i < mLayers.size(); ++i) {
        auto& layer = mLayers[i];
        size_t numWeights = layer.GetTrainableWeights().size();
        size_t numBias = layer.GetTrainableBias().size();
        
        std::memcpy(layer.GetWeightsGradient().data(), allGrads.data() + gradOffset, numWeights * sizeof(float));
        gradOffset += numWeights;
        
        std::memcpy(layer.GetBiasGradient().data(), allGrads.data() + gradOffset, numBias * sizeof(float));
        gradOffset += numBias;
    }
}

std::vector<float> RFFNetwork::GetAllWeights() const
{
    std::vector<float> w;
    for (const auto& layer : mLayers) {
        const auto& weights = layer.GetTrainableWeights();
        const auto& bias = layer.GetTrainableBias();
        w.insert(w.end(), weights.begin(), weights.end());
        w.insert(w.end(), bias.begin(), bias.end());
    }
    return w;
}

void RFFNetwork::SetAllWeights(const std::vector<float>& weights)
{
    size_t offset = 0;
    for (auto& layer : mLayers) {
        auto& w = layer.GetTrainableWeights();
        auto& b = layer.GetTrainableBias();
        
        std::copy(weights.begin() + offset, weights.begin() + offset + w.size(), w.begin());
        offset += w.size();
        
        std::copy(weights.begin() + offset, weights.begin() + offset + b.size(), b.begin());
        offset += b.size();
    }
}

size_t RFFNetwork::GetNumWeights() const
{
    size_t total = 0;
    for (const auto& layer : mLayers) {
        total += layer.GetNumParams();
    }
    return total;
}

std::vector<float> RFFNetwork::GetAllGradients() const
{
    std::vector<float> g;
    for (const auto& layer : mLayers) {
        const auto& wGrad = layer.GetWeightsGradient();
        const auto& bGrad = layer.GetBiasGradient();
        g.insert(g.end(), wGrad.begin(), wGrad.end());
        g.insert(g.end(), bGrad.begin(), bGrad.end());
    }
    return g;
}

void RFFNetwork::SetAllGradients(const std::vector<float>& grads)
{
    size_t offset = 0;
    for (auto& layer : mLayers) {
        auto& wGrad = layer.GetWeightsGradient();
        auto& bGrad = layer.GetBiasGradient();
        
        std::copy(grads.begin() + offset, grads.begin() + offset + wGrad.size(), wGrad.begin());
        offset += wGrad.size();
        
        std::copy(grads.begin() + offset, grads.begin() + offset + bGrad.size(), bGrad.begin());
        offset += bGrad.size();
    }
}

void RFFNetwork::ZeroGradients()
{
    for (auto& layer : mLayers) {
        layer.ZeroGradients();
    }
}

void RFFNetwork::ScaleGradients(float scale)
{
    for (auto& layer : mLayers) {
        layer.ScaleGradients(scale);
    }
}

void RFFNetwork::ComputeGradients(const float* input, const float* output, 
                                   const float* target, int batchSize, int sampleRate)
{
    ZeroGradients();
    
    const float eps = 1e-4f;
    auto weights = GetAllWeights();
    auto grads = GetAllGradients();
    
    AlignedVector32<float> perturbed(batchSize * GetOutputDim());
    
    for (size_t i = 0; i < weights.size(); i += sampleRate) {
        float oldW = weights[i];
        weights[i] += eps;
        SetAllWeights(weights);
        
        ForwardBatch(input, perturbed.data(), batchSize);
        
        float lossGrad = 0.0f;
        for (int b = 0; b < batchSize; ++b) {
            for (size_t d = 0; d < GetOutputDim(); ++d) {
                float diff = (perturbed[b * GetOutputDim() + d] - output[b * GetOutputDim() + d]) / eps;
                lossGrad += 2.0f * diff * (output[b * GetOutputDim() + d] - target[b * GetOutputDim() + d]);
            }
        }
        
        grads[i] = (lossGrad / (batchSize * GetOutputDim())) * static_cast<float>(sampleRate);
        weights[i] = oldW;
    }
    
    SetAllWeights(weights);
    SetAllGradients(grads);
}

void RFFNetwork::SoftUpdate(const RFFNetwork& other, float tau)
{
    for (size_t i = 0; i < mLayers.size(); ++i) {
        auto& w = mLayers[i].GetTrainableWeights();
        auto& b = mLayers[i].GetTrainableBias();
        const auto& ow = other.mLayers[i].GetTrainableWeights();
        const auto& ob = other.mLayers[i].GetTrainableBias();
        
        for (size_t j = 0; j < w.size(); ++j) {
            w[j] = (1.0f - tau) * w[j] + tau * ow[j];
        }
        for (size_t j = 0; j < b.size(); ++j) {
            b[j] = (1.0f - tau) * b[j] + tau * ob[j];
        }
    }
}

void RFFActorCritic::Init(size_t stateDim, size_t actionDim, size_t hiddenDim, 
                          size_t latentDim, std::mt19937& rng)
{
    mStateDim = stateDim;
    mActionDim = actionDim;
    mHiddenDim = hiddenDim;
    mLatentDim = latentDim;
    
    // Create RFF config
    RFFConfig rffConfig;
    rffConfig.num_features = 256;  // Reduced for speed
    rffConfig.sigma = 1.0f;
    rffConfig.seed = 42;
    
    // Actor network: state + latent -> hidden -> hidden -> action
    std::vector<RFFLayerConfig> actorCfg = {
        {stateDim + latentDim, hiddenDim * 2, rffConfig},
        {hiddenDim * 2, hiddenDim, rffConfig},
        {hiddenDim, actionDim, rffConfig}
    };
    mActor.Init(actorCfg, rng);
    mActorTarget.Init(actorCfg, rng);
    
    // Critic network: state + action + latent -> hidden -> hidden -> 4 (Q ensemble)
    std::vector<RFFLayerConfig> criticCfg = {
        {stateDim + actionDim + latentDim, hiddenDim * 2, rffConfig},
        {hiddenDim * 2, hiddenDim, rffConfig},
        {hiddenDim, 4, rffConfig}
    };
    mCritic1.Init(criticCfg, rng);
    mCritic2.Init(criticCfg, rng);
    mCritic1Target.Init(criticCfg, rng);
    mCritic2Target.Init(criticCfg, rng);
    
    // Initialize latent memory
    mLatentMemory.Init(stateDim, latentDim, rng);
    
    // Allocate buffers
    mStateActionBuffer.resize(stateDim + actionDim + latentDim);
}

void RFFActorCritic::SelectAction(const float* state, float* action, float* logProb, 
                                   bool addNoise, int envIdx)
{
    mLatentMemory.StepLatentDynamics(state, 1);
    
    AlignedVector32<float> zPos(LATENT_DIM);
    mLatentMemory.GetLatentStates(zPos.data(), nullptr, envIdx);
    
    AlignedVector32<float> combined(mStateDim + mLatentDim);
    std::copy(state, state + mStateDim, combined.begin());
    std::copy(zPos.begin(), zPos.end(), combined.begin() + mStateDim);
    
    mActor.Forward(combined.data(), action);
    ForwardMoLU_AVX2(action, mActionDim);
    
    if (addNoise) {
        std::normal_distribution<float> dist(0.0f, 0.1f);
        std::mt19937 localRng(0);
        float noiseSum = 0.0f;
        for (size_t i = 0; i < mActionDim; ++i) {
            float n = dist(localRng);
            action[i] = std::clamp(action[i] + n, -1.0f, 1.0f);
            noiseSum += n * n;
        }
        if (logProb) *logProb = -0.5f * noiseSum;
    } else if (logProb) {
        *logProb = 0.0f;
    }
}

void RFFActorCritic::SelectActionBatchWithLatent(const float* states, float* actions, 
                                                  int batchSize, const std::vector<int>& envIndices, 
                                                  bool addNoise)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step latent dynamics
    mLatentMemory.StepLatentDynamics(states, batchSize);
    auto latent_end = std::chrono::high_resolution_clock::now();
    
    // Prepare combined input
    size_t combDim = mStateDim + mLatentDim;
    AlignedVector32<float> combined(batchSize * combDim);
    auto alloc_end = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for num_threads(8) schedule(static)
    for (int b = 0; b < batchSize; ++b) {
        std::memcpy(combined.data() + b * combDim, states + b * mStateDim, mStateDim * sizeof(float));
        std::memcpy(combined.data() + b * combDim + mStateDim, 
                    mLatentMemory.GetMemory().GetPosition(envIndices[b]), 
                    mLatentDim * sizeof(float));
    }
    auto copy_end = std::chrono::high_resolution_clock::now();
    
    // Forward pass through actor
    mActor.ForwardBatch(combined.data(), actions, batchSize);
    auto forward_end = std::chrono::high_resolution_clock::now();
    
    // Apply MoLU activation
    ForwardMoLU_AVX2(actions, batchSize * mActionDim);
    auto molu_end = std::chrono::high_resolution_clock::now();
    
    // Add exploration noise
    if (addNoise) {
        #pragma omp parallel num_threads(8)
        {
            int tid = omp_get_thread_num();
            std::mt19937 localRng(static_cast<unsigned int>(std::time(nullptr)) + tid);
            std::normal_distribution<float> dist(0.0f, 0.1f);
            
            #pragma omp for schedule(static)
            for (int b = 0; b < batchSize; ++b) {
                float* action = actions + b * mActionDim;
                for (size_t i = 0; i < mActionDim; ++i) {
                    action[i] = std::clamp(action[i] + dist(localRng), -1.0f, 1.0f);
                }
            }
        }
    }
    auto noise_end = std::chrono::high_resolution_clock::now();
    
    // Log timing breakdown periodically
    static int logCounter = 0;
    if (++logCounter % 10 == 0) {
        printf("[TIMING] Batch=%d | Latent: %.2fms | Alloc: %.2fms | Copy: %.2fms | "
               "Forward: %.2fms | MoLU: %.2fms | Noise: %.2fms | TOTAL: %.2fms\n",
            batchSize,
            std::chrono::duration<float, std::milli>(latent_end - start).count(),
            std::chrono::duration<float, std::milli>(alloc_end - latent_end).count(),
            std::chrono::duration<float, std::milli>(copy_end - alloc_end).count(),
            std::chrono::duration<float, std::milli>(forward_end - copy_end).count(),
            std::chrono::duration<float, std::milli>(molu_end - forward_end).count(),
            std::chrono::duration<float, std::milli>(noise_end - molu_end).count(),
            std::chrono::duration<float, std::milli>(noise_end - start).count());
    }
}

void RFFActorCritic::SelectActionBatch(const float* states, float* actions, float* logProbs, 
                                        int batchSize, bool addNoise)
{
    std::vector<int> envIndices(batchSize, 0);
    SelectActionBatchWithLatent(states, actions, batchSize, envIndices, addNoise);
    
    if (logProbs) {
        std::fill(logProbs, logProbs + batchSize, 0.0f);
    }
}

void RFFActorCritic::ComputeQValues(const float* state, const float* action, float* qValues)
{
    AlignedVector32<float> zPos(LATENT_DIM);
    mLatentMemory.GetLatentStates(zPos.data(), nullptr, 0);
    
    size_t idx = 0;
    for (size_t i = 0; i < mStateDim; ++i) mStateActionBuffer[idx++] = state[i];
    for (size_t i = 0; i < mActionDim; ++i) mStateActionBuffer[idx++] = action[i];
    for (size_t i = 0; i < mLatentDim; ++i) mStateActionBuffer[idx++] = zPos[i];
    
    float q1[4], q2[4];
    mCritic1.Forward(mStateActionBuffer.data(), q1);
    mCritic2.Forward(mStateActionBuffer.data(), q2);
    
    for (int i = 0; i < 4; ++i) {
        qValues[i] = std::min(q1[i], q2[i]);
    }
}

void RFFActorCritic::ComputeQValuesBatch(const float* states, const float* actions, 
                                          float* qValues, int batchSize)
{
    #pragma omp parallel num_threads(8)
    {
        AlignedVector32<float> localStateActionBuffer(mStateDim + mActionDim + mLatentDim);
        AlignedVector32<float> localZPos(mLatentDim);
        float q1[4], q2[4];
        
        #pragma omp for schedule(static)
        for (int b = 0; b < batchSize; ++b) {
            const float* state = states + b * mStateDim;
            const float* action = actions + b * mActionDim;
            float* qVal = qValues + b * 4;
            
            mLatentMemory.GetLatentStates(localZPos.data(), nullptr, 0);
            
            size_t idx = 0;
            for (size_t i = 0; i < mStateDim; ++i) localStateActionBuffer[idx++] = state[i];
            for (size_t i = 0; i < mActionDim; ++i) localStateActionBuffer[idx++] = action[i];
            for (size_t i = 0; i < mLatentDim; ++i) localStateActionBuffer[idx++] = localZPos[i];
            
            mCritic1.Forward(localStateActionBuffer.data(), q1);
            mCritic2.Forward(localStateActionBuffer.data(), q2);
            
            for (int i = 0; i < 4; ++i) {
                qVal[i] = std::min(q1[i], q2[i]);
            }
        }
    }
}

void RFFActorCritic::ComputeQ1(const float* state, const float* action, float* qValue)
{
    AlignedVector32<float> zPos(LATENT_DIM);
    mLatentMemory.GetLatentStates(zPos.data(), nullptr, 0);
    
    size_t idx = 0;
    for (size_t i = 0; i < mStateDim; ++i) mStateActionBuffer[idx++] = state[i];
    for (size_t i = 0; i < mActionDim; ++i) mStateActionBuffer[idx++] = action[i];
    for (size_t i = 0; i < mLatentDim; ++i) mStateActionBuffer[idx++] = zPos[i];
    
    float q[4];
    mCritic1.Forward(mStateActionBuffer.data(), q);
    *qValue = q[0];
}

void RFFActorCritic::ComputeQ2(const float* state, const float* action, float* qValue)
{
    AlignedVector32<float> zPos(LATENT_DIM);
    mLatentMemory.GetLatentStates(zPos.data(), nullptr, 0);
    
    size_t idx = 0;
    for (size_t i = 0; i < mStateDim; ++i) mStateActionBuffer[idx++] = state[i];
    for (size_t i = 0; i < mActionDim; ++i) mStateActionBuffer[idx++] = action[i];
    for (size_t i = 0; i < mLatentDim; ++i) mStateActionBuffer[idx++] = zPos[i];
    
    float q[4];
    mCritic2.Forward(mStateActionBuffer.data(), q);
    *qValue = q[0];
}

void RFFActorCritic::UpdateTargets(float tau)
{
    mActorTarget.SoftUpdate(mActor, tau);
    mCritic1Target.SoftUpdate(mCritic1, tau);
    mCritic2Target.SoftUpdate(mCritic2, tau);
}
