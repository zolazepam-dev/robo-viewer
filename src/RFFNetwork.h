#pragma once

#include <vector>
#include <random>
#include <cstdint>
#include <cstring>

#include "RFFLayer.h"
#include "LatentMemory.h"
#include "AlignedAllocator.h"

struct RFFLayerConfig
{
    size_t inputDim;
    size_t outputDim;
    RFFConfig rffConfig;
};

// Cache for storing intermediate values during forward pass (for backprop)
struct SpanCache {
    std::vector<AlignedVector32<float>> layerInputs;
    std::vector<AlignedVector32<float>> layerOutputs;
    std::vector<AlignedVector32<float>> layerFeatures;  // RFF features for each layer
};

/**
 * RFF Network - Container for sequential RFF layers
 * 
 * Supports multiple RFF layers in sequence with MoLU activation between them.
 * Provides batch operations and parameter access for optimization.
 */
class alignas(32) RFFNetwork
{
public:
    RFFNetwork() = default;
    RFFNetwork(const RFFNetwork& other) = default;
    RFFNetwork& operator=(const RFFNetwork& other) = default;

    /**
     * Initialize the RFF network with layer configurations
     * @param layerConfigs Vector of layer configurations
     * @param rng Random number generator
     */
    void Init(const std::vector<RFFLayerConfig>& layerConfigs, std::mt19937& rng);

    /**
     * Forward pass for single sample
     * @param input Input vector [inputDim]
     * @param output Output vector [outputDim]
     */
    void Forward(const float* input, float* output);

    /**
     * Forward pass for batch
     * @param input Input batch [batchSize * inputDim]
     * @param output Output batch [batchSize * outputDim]
     * @param batchSize Number of samples in batch
     */
    void ForwardBatch(const float* input, float* output, int batchSize);

    /**
     * Forward pass for batch using Eigen optimization
     * @param input Input batch [batchSize * inputDim]
     * @param output Output batch [batchSize * outputDim]
     * @param batchSize Number of samples in batch
     */
    void ForwardBatchEigen(const float* input, float* output, int batchSize);

    /**
     * Forward pass with latent memory integration
     * @param input Input vector [inputDim]
     * @param output Output vector [outputDim]
     * @param latent Latent memory to use
     * @param envIdx Environment index
     */
    void ForwardWithLatent(const float* input, float* output, SecondOrderLatentMemory& latent, int envIdx);

    /**
     * Forward pass with caching for backpropagation
     * @param input Input vector [inputDim]
     * @param output Output vector [outputDim]
     * @param cache Cache for storing intermediate values
     */
    void ForwardWithCache(const float* input, float* output, SpanCache& cache);

    /**
     * Forward pass with internal caching
     * @param input Input vector [inputDim]
     * @param output Output vector [outputDim]
     */
    void ForwardWithCache(const float* input, float* output);

    /**
     * Backward pass for gradient computation
     * @param input Original input [inputDim]
     * @param output_grad Gradient of loss w.r.t. output [outputDim]
     * @param input_grad Gradient of loss w.r.t. input [inputDim] (can be nullptr)
     * @param weights_grad Gradient accumulator [numWeights] (can be nullptr)
     * @param cache Cache from forward pass
     */
    void Backward(const float* input, const float* output_grad, float* input_grad, 
                  float* weights_grad, SpanCache& cache);

    /**
     * Backward pass with internal cache
     * @param input Original input [inputDim]
     * @param output_grad Gradient of loss w.r.t. output [outputDim]
     * @param input_grad Gradient of loss w.r.t. input [inputDim] (can be nullptr)
     * @param accumulate_grads Whether to accumulate gradients
     */
    void Backward(const float* input, const float* output_grad, float* input_grad = nullptr, bool accumulate_grads = true);

    /**
     * Get all trainable parameters as a flat vector
     * @return Vector of all trainable weights and biases
     */
    std::vector<float> GetAllWeights() const;

    /**
     * Set all trainable parameters from a flat vector
     * @param weights Vector of all trainable weights and biases
     */
    void SetAllWeights(const std::vector<float>& weights);

    /**
     * Get total number of trainable parameters
     * @return Number of trainable parameters
     */
    size_t GetNumWeights() const;

    /**
     * Get all gradients as a flat vector
     * @return Vector of all gradients
     */
    std::vector<float> GetAllGradients() const;

    /**
     * Set all gradients from a flat vector
     * @param grads Vector of all gradients
     */
    void SetAllGradients(const std::vector<float>& grads);

    /**
     * Zero out all gradients
     */
    void ZeroGradients();

    /**
     * Scale all gradients by a factor
     * @param scale Scale factor
     */
    void ScaleGradients(float scale);

    /**
     * Compute gradients via finite differences (for compatibility)
     * @param input Input batch
     * @param output Target output
     * @param target Target values
     * @param batchSize Batch size
     * @param sampleRate Gradient sampling rate (default: 16)
     */
    void ComputeGradients(const float* input, const float* output, const float* target, 
                          int batchSize, int sampleRate = 16);

    /**
     * Get reference to layer at index
     * @param idx Layer index
     * @return Reference to RFFLayer
     */
    RFFLayer& GetLayer(size_t idx) { return mLayers[idx]; }
    const RFFLayer& GetLayer(size_t idx) const { return mLayers[idx]; }

    /**
     * Get number of layers
     * @return Number of layers in network
     */
    size_t GetNumLayers() const { return mLayers.size(); }

    size_t GetInputDim() const { return mInputDim; }
    size_t GetOutputDim() const { return mOutputDim; }

    /**
     * Soft update from another network (for target networks)
     * @param other Source network
     * @param tau Interpolation factor (0 = keep current, 1 = copy other)
     */
    void SoftUpdate(const RFFNetwork& other, float tau);

private:
    AlignedVector32<RFFLayer> mLayers;
    std::vector<size_t> mLayerInputDims;
    std::vector<size_t> mLayerOutputDims;
    size_t mInputDim = 0;
    size_t mOutputDim = 0;

    AlignedVector32<float> mActivationBuffer;
};

/**
 * RFF-based Actor-Critic network
 * 
 * Replaces SpanActorCritic with RFF networks while maintaining
 * the same interface for compatibility with TD3Trainer.
 */
class alignas(32) RFFActorCritic
{
public:
    RFFActorCritic() = default;
    RFFActorCritic(const RFFActorCritic& other) = default;
    RFFActorCritic& operator=(const RFFActorCritic& other) = default;

    /**
     * Initialize the actor-critic network
     * @param stateDim Dimension of state/observation space
     * @param actionDim Dimension of action space
     * @param hiddenDim Hidden layer dimension
     * @param latentDim Latent memory dimension
     * @param rng Random number generator
     */
    void Init(size_t stateDim, size_t actionDim, size_t hiddenDim, size_t latentDim, std::mt19937& rng);

    /**
     * Select action for single state
     * @param state Input state [stateDim]
     * @param action Output action [actionDim]
     * @param logProb Output log probability (optional)
     * @param addNoise Whether to add exploration noise
     * @param envIdx Environment index
     */
    void SelectAction(const float* state, float* action, float* logProb, bool addNoise = true, int envIdx = 0);

    /**
     * Select actions for batch of states
     * @param states Input states [batchSize * stateDim]
     * @param actions Output actions [batchSize * actionDim]
     * @param logProbs Output log probabilities (optional)
     * @param batchSize Batch size
     * @param addNoise Whether to add exploration noise
     */
    void SelectActionBatch(const float* states, float* actions, float* logProbs, int batchSize, bool addNoise = true);

    /**
     * Select actions with latent memory for batch
     * @param states Input states [batchSize * stateDim]
     * @param actions Output actions [batchSize * actionDim]
     * @param batchSize Batch size
     * @param envIndices Environment indices for each sample
     * @param addNoise Whether to add exploration noise
     */
    void SelectActionBatchWithLatent(const float* states, float* actions, int batchSize, 
                                     const std::vector<int>& envIndices, bool addNoise = true);

    /**
     * Compute Q-values for state-action pairs
     * @param state Input state [stateDim]
     * @param action Input action [actionDim]
     * @param qValues Output Q-values [4]
     */
    void ComputeQValues(const float* state, const float* action, float* qValues);

    /**
     * Compute Q-values for batch of state-action pairs
     * @param states Input states [batchSize * stateDim]
     * @param actions Input actions [batchSize * actionDim]
     * @param qValues Output Q-values [batchSize * 4]
     * @param batchSize Batch size
     */
    void ComputeQValuesBatch(const float* states, const float* actions, float* qValues, int batchSize);

    /**
     * Compute Q1 value only
     * @param state Input state [stateDim]
     * @param action Input action [actionDim]
     * @param qValue Output Q1 value
     */
    void ComputeQ1(const float* state, const float* action, float* qValue);

    /**
     * Compute Q2 value only
     * @param state Input state [stateDim]
     * @param action Input action [actionDim]
     * @param qValue Output Q2 value
     */
    void ComputeQ2(const float* state, const float* action, float* qValue);

    RFFNetwork& GetActor() { return mActor; }
    RFFNetwork& GetCritic1() { return mCritic1; }
    RFFNetwork& GetCritic2() { return mCritic2; }
    RFFNetwork& GetActorTarget() { return mActorTarget; }
    RFFNetwork& GetCritic1Target() { return mCritic1Target; }
    RFFNetwork& GetCritic2Target() { return mCritic2Target; }

    const RFFNetwork& GetActor() const { return mActor; }
    const RFFNetwork& GetCritic1() const { return mCritic1; }
    const RFFNetwork& GetCritic2() const { return mCritic2; }
    const RFFNetwork& GetActorTarget() const { return mActorTarget; }
    const RFFNetwork& GetCritic1Target() const { return mCritic1Target; }
    const RFFNetwork& GetCritic2Target() const { return mCritic2Target; }

    class LatentMemoryManager& GetLatentMemory() { return mLatentMemory; }
    const class LatentMemoryManager& GetLatentMemory() const { return mLatentMemory; }

    /**
     * Update target networks
     * @param tau Interpolation factor
     */
    void UpdateTargets(float tau);

    size_t GetStateDim() const { return mStateDim; }
    size_t GetActionDim() const { return mActionDim; }
    size_t GetLatentDim() const { return mLatentDim; }

private:
    RFFNetwork mActor;
    RFFNetwork mCritic1;
    RFFNetwork mCritic2;
    RFFNetwork mActorTarget;
    RFFNetwork mCritic1Target;
    RFFNetwork mCritic2Target;

    class LatentMemoryManager mLatentMemory;

    size_t mStateDim = 0;
    size_t mActionDim = 0;
    size_t mHiddenDim = 0;
    size_t mLatentDim = 0;

    AlignedVector32<float> mStateActionBuffer;
    AlignedVector32<float> mLatentBuffer;
    AlignedVector32<float> mNoiseBuffer;
};
