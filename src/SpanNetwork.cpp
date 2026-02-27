#include "SpanNetwork.h"

#include <algorithm>
#include <immintrin.h>
#include <vector>
#include <random>
#include "AlignedAllocator.h"

void TensorProductBSpline::Init(size_t inputDim, size_t outputDim, int numKnots, int splineDegree, std::mt19937& rng)
{
    mInputDim = inputDim;
    mOutputDim = outputDim;
    mNumKnots = numKnots;
    mSplineDegree = splineDegree;
    
    ComputeKnotVector();
    
    size_t numBasis = static_cast<size_t>(numKnots + splineDegree + 1);
    size_t controlPointsPerOutput = numBasis;
    size_t totalControlPoints = outputDim * controlPointsPerOutput;
    
    mControlPoints.resize(totalControlPoints);
    
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (auto& cp : mControlPoints)
    {
        cp = dist(rng);
    }
    
    mBasisBuffer.resize(numBasis);
    mBasisFunctionsBuffer.resize(inputDim * (mSplineDegree + 1));
    mSpanIndicesBuffer.resize(inputDim);
    mTempOutput.resize(outputDim);
}

void TensorProductBSpline::ComputeKnotVector()
{
    int numKnots = mNumKnots + mSplineDegree + 1;
    mKnots.resize(numKnots);
    
    int numInternal = mNumKnots - mSplineDegree - 1;
    float step = 1.0f / static_cast<float>(numInternal + 1);
    
    for (int i = 0; i <= mSplineDegree; ++i)
    {
        mKnots[i] = 0.0f;
    }
    
    for (int i = 0; i < numInternal; ++i)
    {
        mKnots[mSplineDegree + 1 + i] = (i + 1) * step;
    }
    
    for (int i = mKnots.size() - mSplineDegree - 1; i < static_cast<int>(mKnots.size()); ++i)
    {
        mKnots[i] = 1.0f;
    }
}

void TensorProductBSpline::ComputeBasisFunctions(float x, float* basis, int& spanIdx)
{
    x = std::clamp(x, 0.0f, 1.0f);
    
    spanIdx = mSplineDegree;
    for (int i = mSplineDegree; i < static_cast<int>(mKnots.size()) - mSplineDegree - 1; ++i)
    {
        if (x >= mKnots[i] && x < mKnots[i + 1])
        {
            spanIdx = i;
            break;
        }
    }
    if (x >= 1.0f - 1e-6f) spanIdx = static_cast<int>(mKnots.size()) - mSplineDegree - 2;
    
    for (int i = 0; i <= mSplineDegree; ++i)
    {
        basis[i] = 0.0f;
    }
    basis[0] = 1.0f;
    
    for (int j = 1; j <= mSplineDegree; ++j)
    {
        float saved = 0.0f;
        for (int r = j; r >= 0; --r)
        {
            int idx = spanIdx - j + r + 1;
            float knotDiff = mKnots[idx + mSplineDegree - j] - mKnots[idx];
            float temp = 0.0f;
            
            if (std::abs(knotDiff) > 1e-8f)
            {
                temp = basis[r] / knotDiff;
            }
            
            basis[r + 1] = basis[r + 1] + temp * (mKnots[spanIdx + j + 1] - mKnots[idx + mSplineDegree - j] > 1e-8f ? 
                         (mKnots[spanIdx + j + 1] - mKnots[idx]) / mKnots[spanIdx + j + 1] : 0.0f);
            if (r > 0)
            {
                basis[r] = saved + temp * (mKnots[idx] - mKnots[spanIdx] > 1e-8f ? 
                          (mKnots[idx] - mKnots[spanIdx]) / (mKnots[idx] - mKnots[spanIdx]) : 0.0f);
            }
            saved = temp * (mKnots[spanIdx + j + 1] - x);
        }
    }
    
    for (int i = 0; i <= mSplineDegree; ++i)
    {
        basis[i] = std::max(0.0f, basis[i]);
    }
}

void TensorProductBSpline::Forward(const float* input, float* output)
{
    const size_t numBasis = static_cast<size_t>(mNumKnots + mSplineDegree + 1);
    const int degreePlus1 = mSplineDegree + 1;

    // 1. Precompute basis functions for the input vector once
    for (size_t inIdx = 0; inIdx < mInputDim; ++inIdx)
    {
        float x = std::tanh(input[inIdx]) * 0.5f + 0.5f;
        int spanIdx;
        ComputeBasisFunctions(x, &mBasisFunctionsBuffer[inIdx * degreePlus1], spanIdx);
        mSpanIndicesBuffer[inIdx] = spanIdx;
    }

    // 2. Optimized Accumulation with precomputed cp pointers
    for (size_t outIdx = 0; outIdx < mOutputDim; ++outIdx)
    {
        const float* cpBase = mControlPoints.data() + outIdx * numBasis;
        float sum = 0.0f;
        for (size_t inIdx = 0; inIdx < mInputDim; ++inIdx)
        {
            const float* basisFuncs = &mBasisFunctionsBuffer[inIdx * degreePlus1];
            int spanIdx = mSpanIndicesBuffer[inIdx];
            
            for (int b = 0; b <= mSplineDegree; ++b)
            {
                int basisIdx = spanIdx - mSplineDegree + b;
                if (basisIdx >= 0 && static_cast<size_t>(basisIdx) < numBasis)
                {
                    sum += basisFuncs[b] * cpBase[basisIdx];
                }
            }
        }
        output[outIdx] = sum / static_cast<float>(mInputDim);
    }
}

void TensorProductBSpline::ForwardBatch(const float* input, float* output, int batchSize)
{
    const size_t numBasis = static_cast<size_t>(mNumKnots + mSplineDegree + 1);
    const int degreePlus1 = mSplineDegree + 1;
    
    // Precompute basis functions for all inputs in the batch
    std::vector<float> batchBasis(batchSize * mInputDim * degreePlus1);
    std::vector<int> batchSpanIndices(batchSize * mInputDim);
    
    for (int b = 0; b < batchSize; ++b) {
        for (size_t inIdx = 0; inIdx < mInputDim; ++inIdx) {
            float x = std::tanh(input[b * mInputDim + inIdx]) * 0.5f + 0.5f;
            int spanIdx;
            ComputeBasisFunctions(x, &batchBasis[b * mInputDim * degreePlus1 + inIdx * degreePlus1], spanIdx);
            batchSpanIndices[b * mInputDim + inIdx] = spanIdx;
        }
    }
    
    // Optimized accumulation for all outputs in the batch
    for (int b = 0; b < batchSize; ++b) {
        for (size_t outIdx = 0; outIdx < mOutputDim; ++outIdx) {
            const float* cpBase = mControlPoints.data() + outIdx * numBasis;
            float sum = 0.0f;
            for (size_t inIdx = 0; inIdx < mInputDim; ++inIdx) {
                const float* basisFuncs = &batchBasis[b * mInputDim * degreePlus1 + inIdx * degreePlus1];
                int spanIdx = batchSpanIndices[b * mInputDim + inIdx];
                
                for (int k = 0; k <= mSplineDegree; ++k) {
                    int basisIdx = spanIdx - mSplineDegree + k;
                    if (basisIdx >= 0 && static_cast<size_t>(basisIdx) < numBasis) {
                        sum += basisFuncs[k] * cpBase[basisIdx];
                    }
                }
            }
            output[b * mOutputDim + outIdx] = sum / static_cast<float>(mInputDim);
        }
    }
}

void TensorProductBSpline::ForwardAVX2(const float* input, float* output)
{
    Forward(input, output);
}

void SpanNetwork::Init(const std::vector<SpanLayerConfig>& layerConfigs, std::mt19937& rng)
{
    mLayers.resize(layerConfigs.size());
    mLayerInputDims.resize(layerConfigs.size());
    mLayerOutputDims.resize(layerConfigs.size());
    
    size_t maxDim = 0;
    for (size_t i = 0; i < layerConfigs.size(); ++i)
    {
        mLayerInputDims[i] = layerConfigs[i].inputDim;
        mLayerOutputDims[i] = layerConfigs[i].outputDim;
        mLayers[i].Init(layerConfigs[i].inputDim, layerConfigs[i].outputDim,
                        layerConfigs[i].numKnots, layerConfigs[i].splineDegree, rng);
        
        maxDim = std::max(maxDim, std::max(layerConfigs[i].inputDim, layerConfigs[i].outputDim));
    }
    
    if (!layerConfigs.empty())
    {
        mInputDim = layerConfigs.front().inputDim;
        mOutputDim = layerConfigs.back().outputDim;
    }
    
    mActivationBuffer.resize(maxDim * 2);
}

void SpanNetwork::Forward(const float* input, float* output)
{
    if (mLayers.empty()) return;
    
    const float* currentInput = input;
    float* currentOutput = mActivationBuffer.data();
    
    for (size_t i = 0; i < mLayers.size(); ++i)
    {
        mLayers[i].Forward(currentInput, currentOutput);
        
        if (i < mLayers.size() - 1)
        {
            ForwardMoLU_AVX2(currentOutput, mLayerOutputDims[i]);
        }
        
        float* temp = const_cast<float*>(currentInput);
        currentInput = currentOutput;
        currentOutput = temp;
    }
    
    std::copy(currentOutput, currentOutput + mOutputDim, output);
}

void SpanNetwork::ForwardBatch(const float* input, float* output, int batchSize)
{
    if (mLayers.empty()) return;
    
    int maxDim = 0;
    for (size_t i = 0; i < mLayers.size(); ++i) {
        maxDim = std::max(maxDim, (int)std::max(mLayerInputDims[i], mLayerOutputDims[i]));
    }
    
    AlignedVector32<float> activationBuffer(maxDim * batchSize * 2);
    AlignedVector32<float> tempBuffer(maxDim * batchSize);
    
    // First layer
    mLayers[0].ForwardBatch(input, tempBuffer.data(), batchSize);
    if (mLayers.size() > 1) {
        ForwardMoLU_AVX2(tempBuffer.data(), mLayerOutputDims[0] * batchSize);
    }
    
    // Hidden layers
    for (size_t i = 1; i < mLayers.size() - 1; ++i) {
        mLayers[i].ForwardBatch(tempBuffer.data(), activationBuffer.data(), batchSize);
        ForwardMoLU_AVX2(activationBuffer.data(), mLayerOutputDims[i] * batchSize);
        std::swap(tempBuffer, activationBuffer);
    }
    
    // Last layer
    if (mLayers.size() > 1) {
        mLayers.back().ForwardBatch(tempBuffer.data(), output, batchSize);
    } else {
        std::copy(tempBuffer.begin(), tempBuffer.begin() + batchSize * mOutputDim, output);
    }
}

void SpanNetwork::ForwardWithLatent(const float* input, float* output, SecondOrderLatentMemory& latent, int envIdx)
{
    float* zPos = latent.GetPosition(envIdx);
    float* zVel = latent.GetVelocity(envIdx);
    
    int combinedDim = mInputDim + latent.latentDim;
    alignas(32) AlignedVector32<float> combinedInput(combinedDim);
    
    std::copy(input, input + mInputDim, combinedInput.begin());
    std::copy(zPos, zPos + latent.latentDim, combinedInput.begin() + mInputDim);
    
    Forward(combinedInput.data(), output);
}

std::vector<float> SpanNetwork::GetAllWeights() const
{
    std::vector<float> weights;
    for (const auto& layer : mLayers)
    {
        const auto& cp = layer.GetControlPoints();
        weights.insert(weights.end(), cp.begin(), cp.end());
    }
    return weights;
}

void SpanNetwork::SetAllWeights(const std::vector<float>& weights)
{
    size_t offset = 0;
    for (auto& layer : mLayers)
    {
        auto& cp = layer.GetControlPoints();
        size_t n = cp.size();
        std::copy(weights.begin() + offset, weights.begin() + offset + n, cp.begin());
        offset += n;
    }
}

size_t SpanNetwork::GetNumWeights() const
{
    int total = 0;
    for (const auto& layer : mLayers)
    {
        total += layer.GetNumParams();
    }
    return total;
}

void SpanNetwork::SoftUpdate(const SpanNetwork& other, float tau)
{
    auto myWeights = GetAllWeights();
    auto otherWeights = other.GetAllWeights();
    
    for (size_t i = 0; i < myWeights.size(); ++i)
    {
        myWeights[i] = (1.0f - tau) * myWeights[i] + tau * otherWeights[i];
    }
    
    SetAllWeights(myWeights);
}

void SpanActorCritic::Init(size_t stateDim, size_t actionDim, size_t hiddenDim, size_t latentDim, std::mt19937& rng)
{
    mStateDim = stateDim;
     mActionDim = actionDim;
     mHiddenDim = hiddenDim;
     mLatentDim = latentDim;
     
     size_t actorInputDim = stateDim + latentDim;
    std::vector<SpanLayerConfig> actorConfig = {
        {actorInputDim, hiddenDim, 4, 2},
        {hiddenDim, actionDim, 4, 2}
    };
     mActor.Init(actorConfig, rng);
     mActorTarget.Init(actorConfig, rng);
     
     size_t criticInputDim = stateDim + actionDim + latentDim;
    std::vector<SpanLayerConfig> criticConfig = {
        {criticInputDim, hiddenDim, 4, 2},
        {hiddenDim, 4, 4, 2}
    };
    mCritic1.Init(criticConfig, rng);
    mCritic2.Init(criticConfig, rng);
    mCritic1Target.Init(criticConfig, rng);
    mCritic2Target.Init(criticConfig, rng);
    
    mLatentMemory.Init(stateDim, latentDim, rng);
    
    mStateActionBuffer.resize(criticInputDim);
    mLatentBuffer.resize(latentDim);
    mNoiseBuffer.resize(actionDim);
}

void SpanActorCritic::SelectAction(const float* state, float* action, float* logProb, bool addNoise, int envIdx)
{
    mLatentMemory.StepLatentDynamics(state, 1);
    
    AlignedVector32<float> zPos(LATENT_DIM);
    AlignedVector32<float> zVel(LATENT_DIM);
    mLatentMemory.GetLatentStates(zPos.data(), zVel.data(), envIdx);
    
    size_t combinedDim = mStateDim + mLatentDim;
    alignas(32) AlignedVector32<float> combined(combinedDim);
    std::copy(state, state + mStateDim, combined.begin());
    std::copy(zPos.data(), zPos.data() + mLatentDim, combined.begin() + mStateDim);
    
    mActor.Forward(combined.data(), action);
    
    ForwardMoLU_AVX2(action, mActionDim);
    
    if (addNoise)
    {
        std::normal_distribution<float> noiseDist(0.0f, 0.1f);
        std::mt19937 localRng(0);
        
        float noiseSum = 0.0f;
        for (size_t i = 0; i < mActionDim; ++i)
        {
            float noise = noiseDist(localRng);
            action[i] = std::clamp(action[i] + noise, -1.0f, 1.0f);
            noiseSum += noise * noise;
        }
        
        if (logProb)
        {
            *logProb = -0.5f * noiseSum;
        }
    }
    else if (logProb)
    {
        *logProb = 0.0f;
    }
}

void SpanActorCritic::SelectActionBatch(const float* states, float* actions, float* logProbs, int batchSize, bool addNoise)
{
    for (int b = 0; b < batchSize; ++b)
    {
        SelectAction(states + b * mStateDim, actions + b * mActionDim,
                     logProbs ? logProbs + b : nullptr, addNoise);
    }
}

void SpanActorCritic::ComputeQValues(const float* state, const float* action, float* qValues)
{
    AlignedVector32<float> zPos(LATENT_DIM);
    mLatentMemory.GetLatentStates(zPos.data(), nullptr, 0);
    
    size_t idx = 0;
    for (size_t i = 0; i < mStateDim; ++i)
    {
        mStateActionBuffer[idx++] = state[i];
    }
    for (size_t i = 0; i < mActionDim; ++i)
    {
        mStateActionBuffer[idx++] = action[i];
    }
    for (size_t i = 0; i < mLatentDim; ++i)
    {
        mStateActionBuffer[idx++] = zPos[i];
    }
    
    float q1[4], q2[4];
    mCritic1.Forward(mStateActionBuffer.data(), q1);
    mCritic2.Forward(mStateActionBuffer.data(), q2);
    
    qValues[0] = q1[0];
    qValues[1] = q1[1];
    qValues[2] = q1[2];
    qValues[3] = q1[3];
}

void SpanActorCritic::ComputeQValuesBatch(const float* states, const float* actions, float* qValues, int batchSize)
{
    for (int b = 0; b < batchSize; ++b)
    {
        ComputeQValues(states + b * mStateDim, actions + b * mActionDim, qValues + b * 4);
    }
}

void SpanActorCritic::ComputeQ1(const float* state, const float* action, float* qValue)
{
    AlignedVector32<float> zPos(LATENT_DIM);
    mLatentMemory.GetLatentStates(zPos.data(), nullptr, 0);
    
    size_t idx = 0;
    for (size_t i = 0; i < mStateDim; ++i)
    {
        mStateActionBuffer[idx++] = state[i];
    }
    for (size_t i = 0; i < mActionDim; ++i)
    {
        mStateActionBuffer[idx++] = action[i];
    }
    for (size_t i = 0; i < mLatentDim; ++i)
    {
        mStateActionBuffer[idx++] = zPos[i];
    }
    
    float q[4];
    mCritic1.Forward(mStateActionBuffer.data(), q);
    *qValue = q[0];
}

void SpanActorCritic::ComputeQ2(const float* state, const float* action, float* qValue)
{
    AlignedVector32<float> zPos(LATENT_DIM);
    mLatentMemory.GetLatentStates(zPos.data(), nullptr, 0);
    
    size_t idx = 0;
    for (size_t i = 0; i < mStateDim; ++i)
    {
        mStateActionBuffer[idx++] = state[i];
    }
    for (size_t i = 0; i < mActionDim; ++i)
    {
        mStateActionBuffer[idx++] = action[i];
    }
    for (size_t i = 0; i < mLatentDim; ++i)
    {
        mStateActionBuffer[idx++] = zPos[i];
    }
    
    float q[4];
    mCritic2.Forward(mStateActionBuffer.data(), q);
    *qValue = q[0];
}

void SpanActorCritic::UpdateTargets(float tau)
{
    mActorTarget.SoftUpdate(mActor, tau);
    mCritic1Target.SoftUpdate(mCritic1, tau);
    mCritic2Target.SoftUpdate(mCritic2, tau);
}
