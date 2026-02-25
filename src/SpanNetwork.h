#pragma once

#include <vector>
#include <random>
#include <cstdint>
#include <cmath>

#include "NeuralMath.h"
#include "LatentMemory.h"

struct SpanLayerConfig
{
    size_t inputDim;
    size_t outputDim;
    int numKnots = 8;
    int splineDegree = 3;
};

class alignas(32) TensorProductBSpline
{
public:
    TensorProductBSpline() = default;
    
    void Init(size_t inputDim, size_t outputDim, int numKnots, int splineDegree, std::mt19937& rng);
    
    void Forward(const float* input, float* output);
    void ForwardBatch(const float* input, float* output, int batchSize);
    void ForwardAVX2(const float* input, float* output);
    
    std::vector<float>& GetControlPoints() { return mControlPoints; }
    const std::vector<float>& GetControlPoints() const { return mControlPoints; }
    
    size_t GetInputDim() const { return mInputDim; }
    size_t GetOutputDim() const { return mOutputDim; }
    int GetNumKnots() const { return mNumKnots; }
    size_t GetNumParams() const { return mControlPoints.size(); }

private:
    void ComputeBasisFunctions(float x, float* basis, int& spanIdx);
    void ComputeKnotVector();
    
    size_t mInputDim = 0;
    size_t mOutputDim = 0;
    int mNumKnots = 8;
    int mSplineDegree = 3;
    
    alignas(32) std::vector<float> mKnots;
    alignas(32) std::vector<float> mControlPoints;
    
    alignas(32) std::vector<float> mBasisBuffer;
    alignas(32) std::vector<float> mTempOutput;
};

class alignas(32) SpanNetwork
{
public:
    SpanNetwork() = default;
    
    void Init(const std::vector<SpanLayerConfig>& layerConfigs, std::mt19937& rng);
    
    void Forward(const float* input, float* output);
    void ForwardBatch(const float* input, float* output, int batchSize);
    void ForwardWithLatent(const float* input, float* output, SecondOrderLatentMemory& latent, int envIdx);
    
    std::vector<float> GetAllWeights() const;
    void SetAllWeights(const std::vector<float>& weights);
    size_t GetNumWeights() const;
    
    TensorProductBSpline& GetLayer(size_t idx) { return mLayers[idx]; }
    const TensorProductBSpline& GetLayer(size_t idx) const { return mLayers[idx]; }
    size_t GetNumLayers() const { return mLayers.size(); }
    
    size_t GetInputDim() const { return mInputDim; }
    size_t GetOutputDim() const { return mOutputDim; }
    
    void SoftUpdate(const SpanNetwork& other, float tau);

private:
    alignas(32) std::vector<TensorProductBSpline> mLayers;
    std::vector<size_t> mLayerInputDims;
    std::vector<size_t> mLayerOutputDims;
    size_t mInputDim = 0;
    size_t mOutputDim = 0;
    
    alignas(32) std::vector<float> mActivationBuffer;
};

class alignas(32) SpanActorCritic
{
public:
    SpanActorCritic() = default;
    
    void Init(size_t stateDim, size_t actionDim, size_t hiddenDim, size_t latentDim, std::mt19937& rng);
    
    void SelectAction(const float* state, float* action, float* logProb, bool addNoise = true);
    void SelectActionBatch(const float* states, float* actions, float* logProbs, int batchSize, bool addNoise = true);
    
    void ComputeQValues(const float* state, const float* action, float* qValues);
    void ComputeQValuesBatch(const float* states, const float* actions, float* qValues, int batchSize);
    
    void ComputeQ1(const float* state, const float* action, float* qValue);
    void ComputeQ2(const float* state, const float* action, float* qValue);
    
    SpanNetwork& GetActor() { return mActor; }
    SpanNetwork& GetCritic1() { return mCritic1; }
    SpanNetwork& GetCritic2() { return mCritic2; }
    SpanNetwork& GetActorTarget() { return mActorTarget; }
    SpanNetwork& GetCritic1Target() { return mCritic1Target; }
    SpanNetwork& GetCritic2Target() { return mCritic2Target; }
    
    const SpanNetwork& GetActor() const { return mActor; }
    const SpanNetwork& GetCritic1() const { return mCritic1; }
    const SpanNetwork& GetCritic2() const { return mCritic2; }
    const SpanNetwork& GetActorTarget() const { return mActorTarget; }
    const SpanNetwork& GetCritic1Target() const { return mCritic1Target; }
    const SpanNetwork& GetCritic2Target() const { return mCritic2Target; }
    
    class LatentMemoryManager& GetLatentMemory() { return mLatentMemory; }
    
    void UpdateTargets(float tau);
    
    size_t GetStateDim() const { return mStateDim; }
    size_t GetActionDim() const { return mActionDim; }
    size_t GetLatentDim() const { return mLatentDim; }

private:
    SpanNetwork mActor;
    SpanNetwork mCritic1;
    SpanNetwork mCritic2;
    SpanNetwork mActorTarget;
    SpanNetwork mCritic1Target;
    SpanNetwork mCritic2Target;
    
    class LatentMemoryManager mLatentMemory;
    
    size_t mStateDim = 0;
    size_t mActionDim = 0;
    size_t mHiddenDim = 0;
    size_t mLatentDim = 0;
    
    alignas(32) std::vector<float> mStateActionBuffer;
    alignas(32) std::vector<float> mLatentBuffer;
    alignas(32) std::vector<float> mNoiseBuffer;
};

struct alignas(32) CriticBatchBuffer
{
    static constexpr size_t BATCH_SIZE = 256;
    static constexpr size_t HIDDEN_ALIGNED = CRITIC_HIDDEN_DIM_ALIGNED;
    
    alignas(32) float preActivation[BATCH_SIZE * HIDDEN_ALIGNED];
    alignas(32) float postActivation[BATCH_SIZE * HIDDEN_ALIGNED];
    alignas(32) float gradients[BATCH_SIZE * HIDDEN_ALIGNED];
    
    alignas(32) float weights[HIDDEN_ALIGNED * HIDDEN_ALIGNED];
    alignas(32) float biases[HIDDEN_ALIGNED];
    
    void Clear()
    {
        std::memset(preActivation, 0, sizeof(preActivation));
        std::memset(postActivation, 0, sizeof(postActivation));
        std::memset(gradients, 0, sizeof(gradients));
    }
};
