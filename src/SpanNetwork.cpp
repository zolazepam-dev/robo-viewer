#include "SpanNetwork.h"

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

void TensorProductBSpline::Init(size_t inputDim, size_t outputDim, int numKnots, int splineDegree, std::mt19937& rng)
{
    mInputDim = inputDim;
    mOutputDim = outputDim;
    mNumKnots = numKnots;
    mSplineDegree = splineDegree;
    
    ComputeKnotVector();
    
    size_t numBasis = static_cast<size_t>(numKnots + splineDegree + 1);
    size_t outputDimAligned = PAD_TO_AVX2(outputDim);
    size_t totalControlPoints = numBasis * outputDimAligned;
    
    mControlPoints.resize(totalControlPoints, 0.0f);
    mControlPointGradients.resize(totalControlPoints, 0.0f);
    
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (size_t b = 0; b < numBasis; ++b) {
        for (size_t o = 0; o < outputDim; ++o) {
            mControlPoints[b * outputDimAligned + o] = dist(rng);
        }
    }
    
    mBasisBuffer.resize(numBasis);
    mBasisFunctionsBuffer.resize(inputDim * (mSplineDegree + 1));
    mSpanIndicesBuffer.resize(inputDim);
    mTempOutput.resize(outputDim);
    
    mBasisLookupTable.resize(BASIS_LOOKUP_SIZE * (mSplineDegree + 1));
    for (int i = 0; i < BASIS_LOOKUP_SIZE; ++i) {
        float x = (static_cast<float>(i) / static_cast<float>(BASIS_LOOKUP_SIZE - 1));
        int spanIdx;
        ComputeBasisFunctions(x, &mBasisLookupTable[i * (mSplineDegree + 1)], spanIdx);
    }
    mUseLookupTable = true;
}

void TensorProductBSpline::ComputeKnotVector()
{
    int numKnots = mNumKnots + mSplineDegree + 1;
    mKnots.resize(numKnots);
    int numInternal = mNumKnots - mSplineDegree - 1;
    float step = 1.0f / static_cast<float>(numInternal + 1);
    for (int i = 0; i <= mSplineDegree; ++i) mKnots[i] = 0.0f;
    for (int i = 0; i < numInternal; ++i) mKnots[mSplineDegree + 1 + i] = (i + 1) * step;
    for (int i = mKnots.size() - mSplineDegree - 1; i < static_cast<int>(mKnots.size()); ++i) mKnots[i] = 1.0f;
}

void TensorProductBSpline::ComputeBasisFunctions(float x, float* basis, int& spanIdx)
{
    x = std::clamp(x, 0.0f, 1.0f);
    spanIdx = mSplineDegree;
    for (int i = mSplineDegree; i < static_cast<int>(mKnots.size()) - mSplineDegree - 1; ++i) {
        if (x >= mKnots[i] && x < mKnots[i + 1]) { spanIdx = i; break; }
    }
    if (x >= 1.0f - 1e-6f) spanIdx = static_cast<int>(mKnots.size()) - mSplineDegree - 2;
    for (int i = 0; i <= mSplineDegree; ++i) basis[i] = 0.0f;
    basis[0] = 1.0f;
    for (int j = 1; j <= mSplineDegree; ++j) {
        float saved = 0.0f;
        for (int r = j; r >= 0; --r) {
            int idx = spanIdx - j + r + 1;
            float knotDiff = mKnots[idx + mSplineDegree - j] - mKnots[idx];
            float temp = (std::abs(knotDiff) > 1e-8f) ? basis[r] / knotDiff : 0.0f;
            basis[r + 1] = basis[r + 1] + temp * (mKnots[spanIdx + j + 1] - mKnots[idx + mSplineDegree - j] > 1e-8f ? 
                         (mKnots[spanIdx + j + 1] - mKnots[idx]) / mKnots[spanIdx + j + 1] : 0.0f);
            if (r > 0) basis[r] = saved + temp * (mKnots[idx] - mKnots[spanIdx] > 1e-8f ? 
                          (mKnots[idx] - mKnots[spanIdx]) / (mKnots[idx] - mKnots[spanIdx]) : 0.0f);
            saved = temp * (mKnots[spanIdx + j + 1] - x);
        }
    }
    for (int i = 0; i <= mSplineDegree; ++i) basis[i] = std::max(0.0f, basis[i]);
}

void TensorProductBSpline::Forward(const float* input, float* output)
{
    const size_t numBasis = static_cast<size_t>(mNumKnots + mSplineDegree + 1);
    const int degreePlus1 = mSplineDegree + 1;
    const size_t outputDimAligned = PAD_TO_AVX2(mOutputDim);
    for (size_t inIdx = 0; inIdx < mInputDim; ++inIdx) {
        float x = tanhf(input[inIdx]) * 0.5f + 0.5f;
        x = std::clamp(x, 0.0f, 1.0f);
        int idx = (int)(x * (BASIS_LOOKUP_SIZE - 1));
        if (idx >= BASIS_LOOKUP_SIZE) idx = BASIS_LOOKUP_SIZE - 1;
        const float* lookupBasis = &mBasisLookupTable[idx * degreePlus1];
        float* destBasis = &mBasisFunctionsBuffer[inIdx * degreePlus1];
        for (int i = 0; i < degreePlus1; ++i) destBasis[i] = lookupBasis[i];
    }
    std::fill(output, output + mOutputDim, 0.0f);
    for (size_t inIdx = 0; inIdx < mInputDim; ++inIdx) {
        const float* basisFuncs = &mBasisFunctionsBuffer[inIdx * degreePlus1];
        int spanIdx = mSplineDegree;
        for (int b = 0; b <= mSplineDegree; ++b) {
            int basisIdx = spanIdx - mSplineDegree + b;
            if (basisIdx >= 0 && static_cast<size_t>(basisIdx) < numBasis) {
                float bVal = basisFuncs[b];
                const float* cpRow = mControlPoints.data() + basisIdx * outputDimAligned;
                for (size_t outIdx = 0; outIdx < mOutputDim; ++outIdx) output[outIdx] += bVal * cpRow[outIdx];
            }
        }
    }
    ScaleVector_AVX2(output, 1.0f / static_cast<float>(mInputDim), mOutputDim);
}

void TensorProductBSpline::ForwardAVX2(const float* input, float* output)
{
    const size_t numBasis = static_cast<size_t>(mNumKnots + mSplineDegree + 1);
    const int degreePlus1 = mSplineDegree + 1;
    const size_t outputDimAligned = PAD_TO_AVX2(mOutputDim);
    for (size_t inIdx = 0; inIdx < mInputDim; ++inIdx) {
        float x = tanhf(input[inIdx]) * 0.5f + 0.5f;
        x = std::clamp(x, 0.0f, 1.0f);
        int idx = (int)(x * (BASIS_LOOKUP_SIZE - 1));
        if (idx >= BASIS_LOOKUP_SIZE) idx = BASIS_LOOKUP_SIZE - 1;
        const float* lookupBasis = &mBasisLookupTable[idx * degreePlus1];
        float* destBasis = &mBasisFunctionsBuffer[inIdx * degreePlus1];
        for (int i = 0; i < degreePlus1; ++i) destBasis[i] = lookupBasis[i];
    }
    std::memset(output, 0, mOutputDim * sizeof(float));
    for (size_t inIdx = 0; inIdx < mInputDim; ++inIdx) {
        const float* basisFuncs = &mBasisFunctionsBuffer[inIdx * degreePlus1];
        int spanIdx = mSplineDegree;
        for (int b = 0; b <= mSplineDegree; ++b) {
            int basisIdx = spanIdx - mSplineDegree + b;
            if (basisIdx >= 0 && static_cast<size_t>(basisIdx) < numBasis) {
                __m256 bVec = _mm256_set1_ps(basisFuncs[b]);
                const float* cpRow = mControlPoints.data() + basisIdx * outputDimAligned;
                size_t outIdx = 0;
                for (; outIdx + 8 <= mOutputDim; outIdx += 8) {
                    __m256 sum = _mm256_loadu_ps(output + outIdx);
                    __m256 cp = _mm256_load_ps(cpRow + outIdx); 
                    sum = _mm256_fmadd_ps(bVec, cp, sum);
                    _mm256_storeu_ps(output + outIdx, sum);
                }
                for (; outIdx < mOutputDim; ++outIdx) output[outIdx] += basisFuncs[b] * cpRow[outIdx];
            }
        }
    }
    ScaleVector_AVX2(output, 1.0f / static_cast<float>(mInputDim), mOutputDim);
}

void TensorProductBSpline::ForwardBatch(const float* input, float* output, int batchSize)
{
    for (int b = 0; b < batchSize; ++b) ForwardAVX2(input + b * mInputDim, output + b * mOutputDim);
}

void TensorProductBSpline::ForwardBatchAVX2(const float* input, float* output, int batchSize)
{
    for (int b = 0; b < batchSize; ++b) ForwardAVX2(input + b * mInputDim, output + b * mOutputDim);
}

void TensorProductBSpline::ForwardBatchEigen(const float* input, float* output, int batchSize)
{
    ForwardBatchAVX2(input, output, batchSize);
}

void TensorProductBSpline::Backward(const float* input, const float* output_grad, float* input_grad, float* control_points_grad)
{
    const size_t numBasis = static_cast<size_t>(mNumKnots + mSplineDegree + 1);
    const int degreePlus1 = mSplineDegree + 1;
    const size_t outputDimAligned = PAD_TO_AVX2(mOutputDim);
    const float invInputDim = 1.0f / static_cast<float>(mInputDim);
    for (size_t inIdx = 0; inIdx < mInputDim; ++inIdx) {
        float x = tanhf(input[inIdx]) * 0.5f + 0.5f;
        x = std::clamp(x, 0.0f, 1.0f);
        int idx = (int)(x * (BASIS_LOOKUP_SIZE - 1));
        if (idx >= BASIS_LOOKUP_SIZE) idx = BASIS_LOOKUP_SIZE - 1;
        const float* lookupBasis = &mBasisLookupTable[idx * degreePlus1];
        float* destBasis = &mBasisFunctionsBuffer[inIdx * degreePlus1];
        for (int i = 0; i < degreePlus1; ++i) destBasis[i] = lookupBasis[i];
    }
    for (size_t inIdx = 0; inIdx < mInputDim; ++inIdx) {
        const float* basisFuncs = &mBasisFunctionsBuffer[inIdx * degreePlus1];
        int spanIdx = mSplineDegree;
        for (int b = 0; b <= mSplineDegree; ++b) {
            int basisIdx = spanIdx - mSplineDegree + b;
            if (basisIdx >= 0 && static_cast<size_t>(basisIdx) < numBasis) {
                float bVal = basisFuncs[b];
                float* cpGradRow = control_points_grad + basisIdx * outputDimAligned;
                for (size_t outIdx = 0; outIdx < mOutputDim; ++outIdx) {
                    cpGradRow[outIdx] += bVal * output_grad[outIdx] * invInputDim;
                }
            }
        }
    }
    if (input_grad != nullptr) {
        const float delta = 1e-4f;
        for (size_t inIdx = 0; inIdx < mInputDim; ++inIdx) {
            float inVal = input[inIdx];
            float tanhVal = tanhf(inVal);
            float normDeriv = 0.5f * (1.0f - tanhVal * tanhVal);
            float x = tanhVal * 0.5f + 0.5f;
            float x_plus = std::clamp(x + delta, 0.0f, 1.0f);
            float x_minus = std::clamp(x - delta, 0.0f, 1.0f);
            int idx_p = (int)(x_plus * (BASIS_LOOKUP_SIZE - 1));
            int idx_m = (int)(x_minus * (BASIS_LOOKUP_SIZE - 1));
            const float* b_p = &mBasisLookupTable[idx_p * degreePlus1];
            const float* b_m = &mBasisLookupTable[idx_m * degreePlus1];
            float inGradAccum = 0.0f;
            int spanIdx = mSplineDegree;
            for (int b = 0; b <= mSplineDegree; ++b) {
                int basisIdx = spanIdx - mSplineDegree + b;
                if (basisIdx >= 0 && static_cast<size_t>(basisIdx) < numBasis) {
                    float bDeriv = (b_p[b] - b_m[b]) / (2.0f * delta);
                    const float* cpRow = mControlPoints.data() + basisIdx * outputDimAligned;
                    for (size_t outIdx = 0; outIdx < mOutputDim; ++outIdx) {
                        inGradAccum += output_grad[outIdx] * bDeriv * cpRow[outIdx] * invInputDim;
                    }
                }
            }
            input_grad[inIdx] = inGradAccum * normDeriv;
        }
    }
}

void SpanNetwork::Init(const std::vector<SpanLayerConfig>& layerConfigs, std::mt19937& rng)
{
    mLayers.resize(layerConfigs.size());
    mLayerInputDims.resize(layerConfigs.size());
    mLayerOutputDims.resize(layerConfigs.size());
    size_t maxDim = 0;
    for (size_t i = 0; i < layerConfigs.size(); ++i) {
        mLayerInputDims[i] = layerConfigs[i].inputDim;
        mLayerOutputDims[i] = layerConfigs[i].outputDim;
        mLayers[i].Init(layerConfigs[i].inputDim, layerConfigs[i].outputDim, layerConfigs[i].numKnots, layerConfigs[i].splineDegree, rng);
        maxDim = std::max(maxDim, std::max(layerConfigs[i].inputDim, layerConfigs[i].outputDim));
    }
    if (!layerConfigs.empty()) { mInputDim = layerConfigs.front().inputDim; mOutputDim = layerConfigs.back().outputDim; }
    mActivationBuffer.resize(maxDim * 2);
}

void SpanNetwork::Forward(const float* input, float* output)
{
    if (mLayers.empty()) return;
    const float* curIn = input;
    float* curOut = mActivationBuffer.data();
    float* nextOut = mActivationBuffer.data() + mActivationBuffer.size() / 2;
    for (size_t i = 0; i < mLayers.size(); ++i) {
        mLayers[i].ForwardAVX2(curIn, curOut);
        if (i < mLayers.size() - 1) ForwardMoLU_AVX2(curOut, mLayerOutputDims[i]);
        curIn = curOut;
        curOut = (curOut == mActivationBuffer.data()) ? nextOut : mActivationBuffer.data();
    }
    std::memcpy(output, curIn, mOutputDim * sizeof(float));
}

void SpanNetwork::ForwardBatch(const float* input, float* output, int batchSize)
{
    if (mLayers.empty()) return;
    for (int b = 0; b < batchSize; ++b) Forward(input + b * mInputDim, output + b * mOutputDim);
}

void SpanNetwork::ForwardWithLatent(const float* input, float* output, SecondOrderLatentMemory& latent, int envIdx)
{
    float* zPos = latent.GetPosition(envIdx);
    AlignedVector32<float> combined(mInputDim + latent.latentDim);
    std::copy(input, input + mInputDim, combined.begin());
    std::copy(zPos, zPos + latent.latentDim, combined.begin() + mInputDim);
    Forward(combined.data(), output);
}

void SpanNetwork::ForwardWithCache(const float* input, float* output, SpanCache& cache)
{
    if (mLayers.empty()) return;
    cache.layerInputs.resize(mLayers.size());
    cache.layerOutputs.resize(mLayers.size());
    const float* curIn = input;
    for (size_t i = 0; i < mLayers.size(); ++i) {
        cache.layerInputs[i].resize(mLayerInputDims[i]);
        std::memcpy(cache.layerInputs[i].data(), curIn, mLayerInputDims[i] * sizeof(float));
        cache.layerOutputs[i].resize(mLayerOutputDims[i]);
        mLayers[i].ForwardAVX2(curIn, cache.layerOutputs[i].data());
        if (i < mLayers.size() - 1) ForwardMoLU_AVX2(cache.layerOutputs[i].data(), mLayerOutputDims[i]);
        curIn = cache.layerOutputs[i].data();
    }
    std::memcpy(output, curIn, mOutputDim * sizeof(float));
}

void SpanNetwork::ForwardWithCache(const float* input, float* output)
{
    ForwardWithCache(input, output, mInternalCache);
}

void SpanNetwork::ForwardBatchWithCache(const float* input, float* output, int batchSize, std::vector<SpanCache>& caches)
{
    caches.resize(batchSize);
    for (int b = 0; b < batchSize; ++b) ForwardWithCache(input + b * mInputDim, output + b * mOutputDim, caches[b]);
}

void SpanNetwork::Backward(const float* input, const float* output_grad, float* input_grad, float* cp_grad_base, SpanCache& cache)
{
    if (mLayers.empty()) return;
    AlignedVector32<float> currentGrad(mOutputDim);
    std::memcpy(currentGrad.data(), output_grad, mOutputDim * sizeof(float));
    size_t gradOffset = 0;
    for (int i = static_cast<int>(mLayers.size()) - 1; i >= 0; --i) {
        AlignedVector32<float> nextGrad(mLayerInputDims[i]);
        float* layer_cp_grad = cp_grad_base ? (cp_grad_base + gradOffset) : mLayers[i].GetControlPointGradients().data();
        mLayers[i].Backward(cache.layerInputs[i].data(), currentGrad.data(), nextGrad.data(), layer_cp_grad);
        if (i > 0) {
            for (size_t j = 0; j < mLayerOutputDims[i-1]; ++j) {
                float x = cache.layerOutputs[i-1][j];
                float th = tanhf(std::clamp(x, -10.0f, 10.0f));
                nextGrad[j] *= (0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th));
            }
        }
        currentGrad = nextGrad;
    }
    if (input_grad) std::memcpy(input_grad, currentGrad.data(), mInputDim * sizeof(float));
}

void SpanNetwork::Backward(const float* input, const float* output_grad, float* input_grad, bool accumulate_grads)
{
    if (!accumulate_grads) ZeroGradients();
    Backward(input, output_grad, input_grad, nullptr, mInternalCache);
}

std::vector<float> SpanNetwork::GetAllWeights() const
{
    std::vector<float> w;
    for (const auto& l : mLayers) { const auto& cp = l.GetControlPoints(); w.insert(w.end(), cp.begin(), cp.end()); }
    return w;
}

void SpanNetwork::SetAllWeights(const std::vector<float>& weights)
{
    size_t off = 0;
    for (auto& l : mLayers) { auto& cp = l.GetControlPoints(); std::copy(weights.begin() + off, weights.begin() + off + cp.size(), cp.begin()); off += cp.size(); }
}

size_t SpanNetwork::GetNumWeights() const { size_t t = 0; for (const auto& l : mLayers) t += l.GetNumParams(); return t; }

void SpanNetwork::SoftUpdate(const SpanNetwork& other, float tau)
{
    for (size_t i = 0; i < mLayers.size(); ++i) {
        auto& cp = mLayers[i].GetControlPoints(); const auto& ocp = other.mLayers[i].GetControlPoints();
        for (size_t j = 0; j < cp.size(); ++j) cp[j] = (1.0f - tau) * cp[j] + tau * ocp[j];
    }
}

std::vector<float> SpanNetwork::GetAllGradients() const
{
    std::vector<float> g;
    for (const auto& l : mLayers) { const auto& grad = l.GetControlPointGradients(); g.insert(g.end(), grad.begin(), grad.end()); }
    return g;
}

void SpanNetwork::SetAllGradients(const std::vector<float>& grads)
{
    size_t off = 0;
    for (auto& l : mLayers) { auto& g = l.GetControlPointGradients(); std::copy(grads.begin() + off, grads.begin() + off + g.size(), g.begin()); off += g.size(); }
}

void SpanNetwork::ZeroGradients() { for (auto& l : mLayers) std::fill(l.GetControlPointGradients().begin(), l.GetControlPointGradients().end(), 0.0f); }

void SpanNetwork::ScaleGradients(float scale) { for (auto& l : mLayers) ScaleVector_AVX2(l.GetControlPointGradients().data(), scale, l.GetControlPointGradients().size()); }

void SpanNetwork::ComputeGradients(const float* input, const float* output, const float* target, int batchSize, int sampleRate)
{
    ZeroGradients();
    const float eps = 1e-4f;
    auto weights = GetAllWeights(); auto grads = GetAllGradients();
    AlignedVector32<float> perturbed(batchSize * GetOutputDim());
    for (size_t i = 0; i < weights.size(); i += sampleRate) {
        float oldW = weights[i]; weights[i] += eps; SetAllWeights(weights);
        ForwardBatch(input, perturbed.data(), batchSize);
        float lossGrad = 0.0f;
        for (int b = 0; b < batchSize; ++b) {
            for (size_t d = 0; d < GetOutputDim(); ++d) {
                float diff = (perturbed[b * GetOutputDim() + d] - output[b * GetOutputDim() + d]) / eps;
                lossGrad += 2.0f * diff * (output[b * GetOutputDim() + d] - target[b * GetOutputDim() + d]);
            }
        }
        grads[i] = (lossGrad / (batchSize * GetOutputDim())) * sampleRate;
        weights[i] = oldW;
    }
    SetAllWeights(weights); SetAllGradients(grads);
}

void SpanActorCritic::Init(size_t stateDim, size_t actionDim, size_t hiddenDim, size_t latentDim, std::mt19937& rng)
{
    mStateDim = stateDim; mActionDim = actionDim; mHiddenDim = hiddenDim; mLatentDim = latentDim;
    std::vector<SpanLayerConfig> actorCfg = {{stateDim + latentDim, hiddenDim * 2, 8, 3}, {hiddenDim * 2, hiddenDim, 8, 3}, {hiddenDim, actionDim, 8, 3}};
    mActor.Init(actorCfg, rng); mActorTarget.Init(actorCfg, rng);
    std::vector<SpanLayerConfig> criticCfg = {{stateDim + actionDim + latentDim, hiddenDim * 2, 8, 3}, {hiddenDim * 2, hiddenDim, 8, 3}, {hiddenDim, 4, 8, 3}};
    mCritic1.Init(criticCfg, rng); mCritic2.Init(criticCfg, rng);
    mCritic1Target.Init(criticCfg, rng); mCritic2Target.Init(criticCfg, rng);
    mLatentMemory.Init(stateDim, latentDim, rng);
    mStateActionBuffer.resize(stateDim + actionDim + latentDim);
}

void SpanActorCritic::SelectAction(const float* state, float* action, float* logProb, bool addNoise, int envIdx)
{
    mLatentMemory.StepLatentDynamics(state, 1);
    AlignedVector32<float> zPos(LATENT_DIM); mLatentMemory.GetLatentStates(zPos.data(), nullptr, envIdx);
    AlignedVector32<float> combined(mStateDim + mLatentDim);
    std::copy(state, state + mStateDim, combined.begin());
    std::copy(zPos.begin(), zPos.end(), combined.begin() + mStateDim);
    mActor.Forward(combined.data(), action);
    ForwardMoLU_AVX2(action, mActionDim);
    if (addNoise) {
        std::normal_distribution<float> dist(0.0f, 0.1f); std::mt19937 localRng(0);
        float noiseSum = 0.0f;
        for (size_t i = 0; i < mActionDim; ++i) {
            float n = dist(localRng); action[i] = std::clamp(action[i] + n, -1.0f, 1.0f); noiseSum += n * n;
        }
        if (logProb) *logProb = -0.5f * noiseSum;
    } else if (logProb) *logProb = 0.0f;
}

void SpanActorCritic::SelectActionBatchWithLatent(const float* states, float* actions, int batchSize, const std::vector<int>& envIndices, bool addNoise)
{
    auto start = std::chrono::high_resolution_clock::now();
    mLatentMemory.StepLatentDynamics(states, batchSize);
    auto latent_end = std::chrono::high_resolution_clock::now();
    
    size_t combDim = mStateDim + mLatentDim;
    AlignedVector32<float> combined(batchSize * combDim);
    auto alloc_end = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(8) schedule(static)
    for (int b = 0; b < batchSize; ++b) {
        std::memcpy(combined.data() + b * combDim, states + b * mStateDim, mStateDim * sizeof(float));
        std::memcpy(combined.data() + b * combDim + mStateDim, mLatentMemory.GetMemory().GetPosition(envIndices[b]), mLatentDim * sizeof(float));
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
    
    // Log timing breakdown
    static int logCounter = 0;
    if (++logCounter % 10 == 0) {
        fflush(stdout);
        printf("[TIMING] Batch=%d | Latent: %.2fms | Alloc: %.2fms | Copy: %.2fms | Forward: %.2fms | MoLU: %.2fms | Noise: %.2fms | TOTAL: %.2fms\n",
            batchSize,
            std::chrono::duration<float, std::milli>(latent_end - start).count(),
            std::chrono::duration<float, std::milli>(alloc_end - latent_end).count(),
            std::chrono::duration<float, std::milli>(copy_end - alloc_end).count(),
            std::chrono::duration<float, std::milli>(forward_end - copy_end).count(),
            std::chrono::duration<float, std::milli>(molu_end - forward_end).count(),
            std::chrono::duration<float, std::milli>(noise_end - molu_end).count(),
            std::chrono::duration<float, std::milli>(noise_end - start).count());
        fflush(stdout);
    }
}

void SpanActorCritic::ComputeQValues(const float* state, const float* action, float* qValues)
{
    AlignedVector32<float> zPos(LATENT_DIM); mLatentMemory.GetLatentStates(zPos.data(), nullptr, 0);
    size_t idx = 0;
    for (size_t i = 0; i < mStateDim; ++i) mStateActionBuffer[idx++] = state[i];
    for (size_t i = 0; i < mActionDim; ++i) mStateActionBuffer[idx++] = action[i];
    for (size_t i = 0; i < mLatentDim; ++i) mStateActionBuffer[idx++] = zPos[i];
    float q1[4], q2[4]; mCritic1.Forward(mStateActionBuffer.data(), q1); mCritic2.Forward(mStateActionBuffer.data(), q2);
    for (int i = 0; i < 4; ++i) qValues[i] = std::min(q1[i], q2[i]);
}

void SpanActorCritic::ComputeQValuesBatch(const float* states, const float* actions, float* qValues, int batchSize)
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
            
            for (int i = 0; i < 4; ++i) qVal[i] = std::min(q1[i], q2[i]);
        }
    }
}

void SpanActorCritic::ComputeQ1(const float* state, const float* action, float* qValue)
{
    AlignedVector32<float> zPos(LATENT_DIM); mLatentMemory.GetLatentStates(zPos.data(), nullptr, 0);
    size_t idx = 0;
    for (size_t i = 0; i < mStateDim; ++i) mStateActionBuffer[idx++] = state[i];
    for (size_t i = 0; i < mActionDim; ++i) mStateActionBuffer[idx++] = action[i];
    for (size_t i = 0; i < mLatentDim; ++i) mStateActionBuffer[idx++] = zPos[i];
    float q[4]; mCritic1.Forward(mStateActionBuffer.data(), q); *qValue = q[0];
}

void SpanActorCritic::ComputeQ2(const float* state, const float* action, float* qValue)
{
    AlignedVector32<float> zPos(LATENT_DIM); mLatentMemory.GetLatentStates(zPos.data(), nullptr, 0);
    size_t idx = 0;
    for (size_t i = 0; i < mStateDim; ++i) mStateActionBuffer[idx++] = state[i];
    for (size_t i = 0; i < mActionDim; ++i) mStateActionBuffer[idx++] = action[i];
    for (size_t i = 0; i < mLatentDim; ++i) mStateActionBuffer[idx++] = zPos[i];
    float q[4]; mCritic2.Forward(mStateActionBuffer.data(), q); *qValue = q[0];
}

void SpanActorCritic::UpdateTargets(float tau) { mActorTarget.SoftUpdate(mActor, tau); mCritic1Target.SoftUpdate(mCritic1, tau); mCritic2Target.SoftUpdate(mCritic2, tau); }
