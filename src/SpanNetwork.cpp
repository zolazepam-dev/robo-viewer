#include "SpanNetwork.h"

#include <algorithm>
#include <immintrin.h>

void TensorProductBSpline::Init(int inputDim, int outputDim, int numKnots, int splineDegree, std::mt19937& rng)
{
    mInputDim = inputDim;
    mOutputDim = outputDim;
    mNumKnots = numKnots;
    mSplineDegree = splineDegree;
    
    ComputeKnotVector();
    
    int numBasis = numKnots + splineDegree + 1;
    int controlPointsPerOutput = numBasis;
    int totalControlPoints = outputDim * controlPointsPerOutput;
    
    mControlPoints.resize(totalControlPoints);
    
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (auto& cp : mControlPoints)
    {
        cp = dist(rng);
    }
    
    mBasisBuffer.resize(numBasis);
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
    int numBasis = mNumKnots + mSplineDegree + 1;
    
    for (int outIdx = 0; outIdx < mOutputDim; ++outIdx)
    {
        float sum = 0.0f;
        
        for (int inIdx = 0; inIdx < mInputDim; ++inIdx)
        {
            float x = input[inIdx];
            x = std::tanhf(x) * 0.5f + 0.5f;
            
            int spanIdx;
            ComputeBasisFunctions(x, mBasisBuffer.data(), spanIdx);
            
            for (int b = 0; b <= mSplineDegree; ++b)
            {
                int basisIdx = spanIdx - mSplineDegree + b;
                if (basisIdx >= 0 && basisIdx < numBasis)
                {
                    int cpIdx = outIdx * numBasis + basisIdx;
                    sum += mBasisBuffer[b] * mControlPoints[cpIdx];
                }
            }
        }
        
        output[outIdx] = sum / static_cast<float>(mInputDim);
    }
}

void TensorProductBSpline::ForwardBatch(const float* input, float* output, int batchSize)
{
    for (int b = 0; b < batchSize; ++b)
    {
        Forward(input + b * mInputDim, output + b * mOutputDim);
    }
}

void TensorProductBSpline::ForwardAVX2(const float* input, float* output)
{
    int numBasis = mNumKnots + mSplineDegree + 1;
    
    for (int outIdx = 0; outIdx < mOutputDim; ++outIdx)
    {
        __m256 sumVec = _mm256_setzero_ps();
        int simdInputDim = mInputDim - (mInputDim % 8);
        
        for (int inIdx = 0; inIdx < simdInputDim; inIdx += 8)
        {
            __m256 x = _mm256_loadu_ps(input + inIdx);
            __m256 tanhX = _mm256_set1_ps(0.0f);
            
            alignas(32) float xArr[8];
            _mm256_store_ps(xArr, x);
            alignas(32) float tanhArr[8];
            for (int i = 0; i < 8; ++i)
            {
                tanhArr[i] = std::tanhf(xArr[i]) * 0.5f + 0.5f;
            }
            tanhX = _mm256_load_ps(tanhArr);
            
            for (int i = 0; i < 8; ++i)
            {
                float xi = tanhArr[i];
                int spanIdx;
                ComputeBasisFunctions(xi, mBasisBuffer.data(), spanIdx);
                
                for (int b = 0; b <= mSplineDegree; ++b)
                {
                    int basisIdx = spanIdx - mSplineDegree + b;
                    if (basisIdx >= 0 && basisIdx < numBasis)
                    {
                        int cpIdx = outIdx * numBasis + basisIdx;
                        sumVec = _mm256_add_ps(sumVec, _mm256_set1_ps(mBasisBuffer[b] * mControlPoints[cpIdx]));
                    }
                }
            }
        }
        
        alignas(32) float sumArr[8];
        _mm256_store_ps(sumArr, sumVec);
        float sum = sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3] + sumArr[4] + sumArr[5] + sumArr[6] + sumArr[7];
        
        for (int inIdx = simdInputDim; inIdx < mInputDim; ++inIdx)
        {
            float x = input[inIdx];
            x = std::tanhf(x) * 0.5f + 0.5f;
            
            int spanIdx;
            ComputeBasisFunctions(x, mBasisBuffer.data(), spanIdx);
            
            for (int b = 0; b <= mSplineDegree; ++b)
            {
                int basisIdx = spanIdx - mSplineDegree + b;
                if (basisIdx >= 0 && basisIdx < numBasis)
                {
                    int cpIdx = outIdx * numBasis + basisIdx;
                    sum += mBasisBuffer[b] * mControlPoints[cpIdx];
                }
            }
        }
        
        output[outIdx] = sum / static_cast<float>(mInputDim);
    }
}

void SpanNetwork::Init(const std::vector<SpanLayerConfig>& layerConfigs, std::mt19937& rng)
{
    mLayers.resize(layerConfigs.size());
    mLayerInputDims.resize(layerConfigs.size());
    mLayerOutputDims.resize(layerConfigs.size());
    
    int maxDim = 0;
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
        
        std::swap(currentInput, currentOutput);
    }
    
    std::copy(currentOutput, currentOutput + mOutputDim, output);
}

void SpanNetwork::ForwardBatch(const float* input, float* output, int batchSize)
{
    for (int b = 0; b < batchSize; ++b)
    {
        Forward(input + b * mInputDim, output + b * mOutputDim);
    }
}

void SpanNetwork::ForwardWithLatent(const float* input, float* output, SecondOrderLatentMemory& latent, int envIdx)
{
    float* zPos = latent.GetPosition(envIdx);
    float* zVel = latent.GetVelocity(envIdx);
    
    int combinedDim = mInputDim + latent.latentDim;
    alignas(32) std::vector<float> combinedInput(combinedDim);
    
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
    
    int actorInputDim = stateDim + latentDim;
    std::vector<SpanLayerConfig> actorConfig = {
        {actorInputDim, hiddenDim, 8, 3},
        {hiddenDim, hiddenDim, 8, 3},
        {hiddenDim, actionDim, 8, 3}
    };
    mActor.Init(actorConfig, rng);
    mActorTarget.Init(actorConfig, rng);
    
    int criticInputDim = stateDim + actionDim + latentDim;
    std::vector<SpanLayerConfig> criticConfig = {
        {criticInputDim, hiddenDim, 8, 3},
        {hiddenDim, hiddenDim, 8, 3},
        {hiddenDim, 4, 8, 3}
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

void SpanActorCritic::SelectAction(const float* state, float* action, float* logProb, bool addNoise)
{
    mLatentMemory.StepLatentDynamics(state, 1);
    
    float zPos[LATENT_DIM];
    float zVel[LATENT_DIM];
    mLatentMemory.GetLatentStates(zPos, zVel, 0);
    
    int combinedDim = mStateDim + mLatentDim;
    alignas(32) std::vector<float> combined(combinedDim);
    std::copy(state, state + mStateDim, combined.begin());
    std::copy(zPos, zPos + mLatentDim, combined.begin() + mStateDim);
    
    mActor.Forward(combined.data(), action);
    
    ForwardMoLU_AVX2(action, mActionDim);
    
    if (addNoise)
    {
        std::normal_distribution<float> noiseDist(0.0f, 0.1f);
        std::mt19937 localRng(0);
        
        float noiseSum = 0.0f;
        for (int i = 0; i < mActionDim; ++i)
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
    float zPos[LATENT_DIM];
    mLatentMemory.GetLatentStates(zPos, nullptr, 0);
    
    int idx = 0;
    for (int i = 0; i < mStateDim; ++i)
    {
        mStateActionBuffer[idx++] = state[i];
    }
    for (int i = 0; i < mActionDim; ++i)
    {
        mStateActionBuffer[idx++] = action[i];
    }
    for (int i = 0; i < mLatentDim; ++i)
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
    float zPos[LATENT_DIM];
    mLatentMemory.GetLatentStates(zPos, nullptr, 0);
    
    int idx = 0;
    for (int i = 0; i < mStateDim; ++i)
    {
        mStateActionBuffer[idx++] = state[i];
    }
    for (int i = 0; i < mActionDim; ++i)
    {
        mStateActionBuffer[idx++] = action[i];
    }
    for (int i = 0; i < mLatentDim; ++i)
    {
        mStateActionBuffer[idx++] = zPos[i];
    }
    
    float q[4];
    mCritic1.Forward(mStateActionBuffer.data(), q);
    *qValue = q[0];
}

void SpanActorCritic::ComputeQ2(const float* state, const float* action, float* qValue)
{
    float zPos[LATENT_DIM];
    mLatentMemory.GetLatentStates(zPos, nullptr, 0);
    
    int idx = 0;
    for (int i = 0; i < mStateDim; ++i)
    {
        mStateActionBuffer[idx++] = state[i];
    }
    for (int i = 0; i < mActionDim; ++i)
    {
        mStateActionBuffer[idx++] = action[i];
    }
    for (int i = 0; i < mLatentDim; ++i)
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
