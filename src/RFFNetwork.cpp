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

void RFFNetwork::Init(const std::vector<RFFLayerConfig>& configs, std::mt19937& rng)
{
    mLayers.clear();
    mLayerInputDims.clear();
    mLayerOutputDims.clear();
    
    mInputDim = configs[0].inputDim;
    mOutputDim = configs.back().outputDim;
    
    for (const auto& cfg : configs) {
        RFFLayer layer;
        layer.Init(cfg.inputDim, cfg.outputDim, cfg.rffConfig, rng);
        mLayers.push_back(std::move(layer));
        
        mLayerInputDims.push_back(cfg.inputDim);
        mLayerOutputDims.push_back(cfg.outputDim);
    }
    
    fprintf(stderr, "[RFFNetwork::Init] Complete. %zu layers.\n", mLayers.size());
    fflush(stderr);
}

void RFFNetwork::Forward(const float* input, float* output)
{
    if (mLayers.empty()) return;
    
    // Determine maximum activation dimension for local ping-pong buffers
    size_t maxDim = mInputDim;
    for (size_t d : mLayerOutputDims) if (d > maxDim) maxDim = d;
    
    // Allocate local buffers to ensure thread safety (no shared member buffers)
    AlignedVector32<float> buffer1(maxDim);
    AlignedVector32<float> buffer2(maxDim);
    
    const float* curIn = input;
    float* curOut = buffer1.data();
    float* nextOut = buffer2.data();
    
    AlignedVector32<float> layerFeatureBuffer;

    for (size_t i = 0; i < mLayers.size(); ++i) {
        layerFeatureBuffer.resize(mLayers[i].GetNumFeatures());
        mLayers[i].Forward(curIn, curOut, layerFeatureBuffer.data());
        
        // Apply MoLU activation between layers (not on final layer)
        if (i < mLayers.size() - 1) {
            ForwardMoLU_AVX2(curOut, mLayerOutputDims[i]);
        }
        
        curIn = curOut;
        // Swap pointers for ping-pong
        float* tmp = curOut;
        curOut = nextOut;
        nextOut = tmp;
    }
    
    // Final result is in the buffer pointed to by curIn
    std::memcpy(output, curIn, mOutputDim * sizeof(float));
}

void RFFNetwork::ForwardBatch(const float* input, float* output, int batchSize)
{
    ForwardBatchEigen(input, output, batchSize);
}

void RFFNetwork::ForwardBatchEigen(const float* input, float* output, int batchSize)
{
    if (mLayers.empty()) return;

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrixXf;
    Eigen::Map<const RowMajorMatrixXf> inputMap(input, batchSize, mInputDim);
    
    Eigen::MatrixXf current = inputMap;
    
    for (size_t i = 0; i < mLayers.size(); ++i) {
        current = mLayers[i].ForwardEigen(current);
        
        if (i < mLayers.size() - 1) {
            current = current.unaryExpr([](float x) {
                float xc = std::clamp(x, -10.0f, 10.0f);
                return 0.5f * x * (1.0f + std::tanh(xc));
            });
        }
    }
    
    RowMajorMatrixXf outputMap = current;
    std::memcpy(output, outputMap.data(), batchSize * mOutputDim * sizeof(float));
}

void RFFNetwork::ForwardWithLatent(const float* input, float* output, 
                                    SecondOrderLatentMemory& latent, int envIdx)
{
    float* zPos = latent.GetPosition(envIdx);
    
    AlignedVector32<float> combined(mInputDim + latent.mLatentDim);
    std::copy(input, input + mInputDim, combined.begin());
    std::copy(zPos, zPos + latent.mLatentDim, combined.begin() + mInputDim);
    
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
        cache.layerInputs[i].resize(mLayerInputDims[i]);
        std::memcpy(cache.layerInputs[i].data(), curIn, mLayerInputDims[i] * sizeof(float));
        
        auto& layer = mLayers[i];
        cache.layerFeatures[i].resize(layer.GetNumFeatures());
        
        cache.layerOutputs[i].resize(mLayerOutputDims[i]);
        layer.Forward(curIn, cache.layerOutputs[i].data(), cache.layerFeatures[i].data());
        
        if (i < mLayers.size() - 1) {
            ForwardMoLU_AVX2(cache.layerOutputs[i].data(), mLayerOutputDims[i]);
        }
        
        curIn = cache.layerOutputs[i].data();
    }
    
    std::memcpy(output, curIn, mOutputDim * sizeof(float));
}

void RFFNetwork::Backward(const float* input, const float* output_grad, float* input_grad,
                          float* weights_grad, SpanCache& cache)
{
    if (mLayers.empty()) return;
    
    cache.layerSinFeatures.resize(mLayers.size());
    cache.layerFeatureGrads.resize(mLayers.size());

    // Backpropagate through layers in reverse order
    AlignedVector32<float> currentGrad(mLayerOutputDims.back());
    std::memcpy(currentGrad.data(), output_grad, mLayerOutputDims.back() * sizeof(float));
    
    // Find maximum dimension among all layers for temporary gradient buffer
    size_t maxDim = mInputDim;
    for (size_t d : mLayerInputDims) if (d > maxDim) maxDim = d;
    for (size_t d : mLayerOutputDims) if (d > maxDim) maxDim = d;
    AlignedVector32<float> tempGradBuffer(maxDim);

    // Calculate initial parameter offset
    size_t totalParams = 0;
    for (const auto& layer : mLayers) totalParams += layer.GetNumParams();
    size_t currentOffset = totalParams;

    for (int i = static_cast<int>(mLayers.size()) - 1; i >= 0; --i) {
        auto& layer = mLayers[i];
        currentOffset -= layer.GetNumParams();
        
        const float* layerInput = (i == 0) ? input : cache.layerOutputs[i-1].data();
        float* layerInputGrad = (i == 0) ? input_grad : tempGradBuffer.data();

        cache.layerSinFeatures[i].resize(layer.GetNumFeatures());
        cache.layerFeatureGrads[i].resize(layer.GetNumFeatures());
        
        float* layerWeightsGrad = weights_grad ? (weights_grad + currentOffset) : nullptr;
        float* layerBiasGrad = weights_grad ? (layerWeightsGrad + layer.GetTrainableWeights().size()) : nullptr;

        layer.Backward(layerInput, currentGrad.data(), layerInputGrad,
                       layerWeightsGrad, 
                       layerBiasGrad,
                       cache.layerFeatures[i].data(), 
                       cache.layerSinFeatures[i].data(), 
                       cache.layerFeatureGrads[i].data());
        
        if (i > 0) {
            BackwardMoLU_AVX2(cache.layerOutputs[i-1].data(), layerInputGrad, mLayerInputDims[i]);
            currentGrad.resize(mLayerInputDims[i]);
            std::memcpy(currentGrad.data(), layerInputGrad, mLayerInputDims[i] * sizeof(float));
        }
    }
}

void RFFNetwork::Backward(const float* input, const float* output_grad, float* input_grad, bool accumulate_grads)
{
    // Use local cache to ensure complete thread safety
    SpanCache cache;
    
    AlignedVector32<float> tempOutput(mOutputDim);
    ForwardWithCache(input, tempOutput.data(), cache);
    
    if (!accumulate_grads) ZeroGradients();
    
    size_t totalParams = GetNumWeights();
    AlignedVector32<float> allGrads(totalParams, 0.0f);
    
    Backward(input, output_grad, input_grad, allGrads.data(), cache);
    
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
    w.reserve(GetNumWeights());
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
        
        if (offset + w.size() + b.size() > weights.size()) break;
        
        std::copy(weights.begin() + offset, weights.begin() + offset + w.size(), w.begin());
        offset += w.size();
        
        std::copy(weights.begin() + offset, weights.begin() + offset + b.size(), b.begin());
        offset += b.size();
    }
}

size_t RFFNetwork::GetNumWeights() const
{
    size_t total = 0;
    for (const auto& layer : mLayers) total += layer.GetNumParams();
    return total;
}

std::vector<float> RFFNetwork::GetAllGradients() const
{
    std::vector<float> g;
    g.reserve(GetNumWeights());
    for (const auto& layer : mLayers) {
        const auto& weightsGrad = layer.GetWeightsGradient();
        const auto& biasGrad = layer.GetBiasGradient();
        g.insert(g.end(), weightsGrad.begin(), weightsGrad.end());
        g.insert(g.end(), biasGrad.begin(), biasGrad.end());
    }
    return g;
}

void RFFNetwork::SetAllGradients(const std::vector<float>& grads)
{
    size_t offset = 0;
    for (auto& layer : mLayers) {
        auto& wg = layer.GetWeightsGradient();
        auto& bg = layer.GetBiasGradient();
        
        if (offset + wg.size() + bg.size() > grads.size()) break;
        
        std::copy(grads.begin() + offset, grads.begin() + offset + wg.size(), wg.begin());
        offset += wg.size();
        
        std::copy(grads.begin() + offset, grads.begin() + offset + bg.size(), bg.begin());
        offset += bg.size();
    }
}

void RFFNetwork::ZeroGradients()
{
    for (auto& layer : mLayers) layer.ZeroGradients();
}

void RFFNetwork::ScaleGradients(float scale)
{
    for (auto& layer : mLayers) layer.ScaleGradients(scale);
}

void RFFNetwork::SoftUpdate(const RFFNetwork& other, float tau)
{
    if (mLayers.size() != other.mLayers.size()) return;
    
    for (size_t i = 0; i < mLayers.size(); ++i) {
        auto& targetWeights = mLayers[i].GetTrainableWeights();
        auto& targetBias = mLayers[i].GetTrainableBias();
        const auto& sourceWeights = other.mLayers[i].GetTrainableWeights();
        const auto& sourceBias = other.mLayers[i].GetTrainableBias();
        
        for (size_t j = 0; j < targetWeights.size(); ++j) {
            targetWeights[j] = (1.0f - tau) * targetWeights[j] + tau * sourceWeights[j];
        }
        for (size_t j = 0; j < targetBias.size(); ++j) {
            targetBias[j] = (1.0f - tau) * targetBias[j] + tau * sourceBias[j];
        }
    }
}

void RFFActorCritic::Init(size_t stateDim, size_t actionDim, size_t hiddenDim,
                          size_t latentDim, const RFFConfig& rffConfig, std::mt19937& rng)
{
    mStateDim = stateDim;
    mActionDim = actionDim;
    mHiddenDim = hiddenDim;
    mLatentDim = latentDim;

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

    // Initialize latent memory with the same RFF config for dynamics
    mLatentMemory.Init(stateDim, latentDim, rffConfig, rng);
}

void RFFActorCritic::SelectAction(const float* state, float* action, float* logProb, 
                                   bool addNoise, int envIdx)
{
    std::vector<int> indices = {envIdx};
    mLatentMemory.StepLatentDynamics(state, indices);
    
    AlignedVector32<float> zPos(mLatentDim);
    mLatentMemory.GetLatentStates(zPos.data(), nullptr, envIdx);
    
    AlignedVector32<float> combined(mStateDim + mLatentDim);
    std::copy(state, state + mStateDim, combined.begin());
    std::copy(zPos.begin(), zPos.end(), combined.begin() + mStateDim);
    
    mActor.Forward(combined.data(), action);
    ForwardMoLU_AVX2(action, mActionDim);
    
    if (addNoise) {
        std::normal_distribution<float> dist(0.0f, 0.1f);
        std::mt19937 localRng(static_cast<unsigned int>(std::time(nullptr)));
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

    mLatentMemory.StepLatentDynamics(states, envIndices);
    auto latent_end = std::chrono::high_resolution_clock::now();

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

    mActor.ForwardBatch(combined.data(), actions, batchSize);
    auto forward_end = std::chrono::high_resolution_clock::now();

    ForwardMoLU_AVX2(actions, batchSize * mActionDim);
    auto molu_end = std::chrono::high_resolution_clock::now();

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

    static int logCounter = 0;
    if (++logCounter % 100 == 0) {
        printf("[TIMING] Batch=%d | Latent: %.2fms | Forward: %.2fms | TOTAL: %.2fms\n",
            batchSize,
            std::chrono::duration<float, std::milli>(latent_end - start).count(),
            std::chrono::duration<float, std::milli>(forward_end - copy_end).count(),
            std::chrono::duration<float, std::milli>(noise_end - start).count());
    }
}

void RFFActorCritic::SelectActionBatch(const float* states, float* actions, float* logProbs, 
                                        int batchSize, bool addNoise)
{
    std::vector<int> envIndices(batchSize, 0);
    SelectActionBatchWithLatent(states, actions, batchSize, envIndices, addNoise);
}

void RFFActorCritic::ComputeQValue(const float* state, const float* action, float* qValue)
{
    float q[4];
    ComputeQValues(state, action, q, 0);
    *qValue = q[0];
}

void RFFActorCritic::ComputeQValues(const float* state, const float* action, float* qValues, int envIdx)
{
    AlignedVector32<float> zPos(mLatentDim);
    mLatentMemory.GetLatentStates(zPos.data(), nullptr, envIdx);
    
    AlignedVector32<float> combined(mStateDim + mActionDim + mLatentDim);
    size_t idx = 0;
    for (size_t i = 0; i < mStateDim; ++i) combined[idx++] = state[i];
    for (size_t i = 0; i < mActionDim; ++i) combined[idx++] = action[i];
    for (size_t i = 0; i < mLatentDim; ++i) combined[idx++] = zPos[i];
    
    mCritic1.Forward(combined.data(), qValues);
}

void RFFActorCritic::ComputeQValuesBatch(const float* states, const float* actions, 
                                          float* qValues, int batchSize,
                                          const std::vector<int>* envIndices)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        AlignedVector32<float> localStateActionBuffer(mStateDim + mActionDim + mLatentDim);
        AlignedVector32<float> localZPos(mLatentDim);
        float q1[4], q2[4];
        
        #pragma omp for schedule(static)
        for (int b = 0; b < batchSize; ++b) {
            int envIdx = envIndices ? (*envIndices)[b] : 0;
            mLatentMemory.GetLatentStates(localZPos.data(), nullptr, envIdx);
            
            size_t idx = 0;
            for (size_t i = 0; i < mStateDim; ++i) localStateActionBuffer[idx++] = states[b * mStateDim + i];
            for (size_t i = 0; i < mActionDim; ++i) localStateActionBuffer[idx++] = actions[b * mActionDim + i];
            for (size_t i = 0; i < mLatentDim; ++i) localStateActionBuffer[idx++] = localZPos[i];
            
            mCritic1.Forward(localStateActionBuffer.data(), q1);
            mCritic2.Forward(localStateActionBuffer.data(), q2);
            
            qValues[b * 2] = q1[0];
            qValues[b * 2 + 1] = q2[0];
        }
    }
}

void RFFActorCritic::ComputeQ1(const float* state, const float* action, float* qValue, int envIdx)
{
    float q[4];
    ComputeQValues(state, action, q, envIdx);
    *qValue = q[0];
}

void RFFActorCritic::ComputeQ2(const float* state, const float* action, float* qValue, int envIdx)
{
    AlignedVector32<float> zPos(mLatentDim);
    mLatentMemory.GetLatentStates(zPos.data(), nullptr, envIdx);
    
    AlignedVector32<float> combined(mStateDim + mActionDim + mLatentDim);
    size_t idx = 0;
    for (size_t i = 0; i < mStateDim; ++i) combined[idx++] = state[i];
    for (size_t i = 0; i < mActionDim; ++i) combined[idx++] = action[i];
    for (size_t i = 0; i < mLatentDim; ++i) combined[idx++] = zPos[i];
    
    float q[4];
    mCritic2.Forward(combined.data(), q);
    *qValue = q[0];
}

void RFFActorCritic::UpdateTargets(float tau)
{
    mActorTarget.SoftUpdate(mActor, tau);
    mCritic1Target.SoftUpdate(mCritic1, tau);
    mCritic2Target.SoftUpdate(mCritic2, tau);
}
