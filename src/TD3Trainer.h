#pragma once

#include <memory>
#include <random>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "SpanNetwork.h"
#include "RFFNetwork.h"
#include "RFFLayer.h"
#include "LatentMemory.h"
#include "OpponentPool.h"
#include "NeuralMath.h"
#include "NeuralNetwork.h"
#include "AlignedAllocator.h"
#include "PerformanceProfiler.h"
#include "OptimizedMath.h"
#include "MuonOptimizer.h"

// IO THREAD STATE FOR ASYNC SAVING
struct IOTask {
    enum Type { SAVE_MODEL, SNAPSHOT_OPPONENT };
    Type type;
    std::string path;
    std::string robotPath;
    int numSatellites = 0;
    int obsDim = 0;
    std::vector<float> weights;
};

struct TD3Config
{
    // Model architecture - DO NOT REDUCE (preserved for full capacity)
    int hiddenDim = 128;  // KEEP - full capacity hidden dimension
    int latentDim = 24;   // KEEP - full capacity latent dimension

    // RFF configuration
    int rffNumFeatures = 1024;  // Number of RFF features
    float rffSigma = 1.0f;       // RFF kernel bandwidth

    // Optimizer learning rates
    float actorLR = 3e-4f;
    float criticLR = 3e-4f;

    // TD3 hyperparameters
    float gamma = 0.99f;
    float tau = 0.005f;
    float policyNoise = 0.2f;
    float noiseClip = 0.5f;
    float explNoise = 0.1f;
    int policyDelay = 2;

    // Batch and buffer settings
    int batchSize = 256;  // OPTIMIZED: Increased from 16 to 256 for better GPU utilization
    int bufferSize = 1000000;
    int startSteps = 500;
    int snapshotInterval = 50000;

    // Target network update optimization
    int targetUpdateDelay = 10;  // OPTIMIZED: Update targets every 10 steps (was every step)

    // Gradient accumulation
    bool useGradientAccumulation = true;  // OPTIMIZED: Enable gradient accumulation
    int accumulationSteps = 4;  // OPTIMIZED: Accumulate gradients over 4 steps

    // Muon optimizer settings (OPTIMIZED for speed)
    float muonLR = 0.02f;         // Muon learning rate for matrix params
    float muonBeta = 0.95f;       // Muon momentum beta
    int muonNSSteps = 2;          // Can use 2 steps with analytic gradients (was 1)
    int muonUpdateInterval = 2;   // Update every 2 steps instead of 4 (analytic is faster)
    float muonEpsilon = 1e-6f;    // Small value for numerical stability
    bool useAnalyticGradients = true;  // Use analytic gradients (backpropagation)
    int gradientSampleRate = 1;   // Not used with analytic gradients (kept for compatibility)
};

class TD3Trainer
{
public:
    TD3Trainer(int stateDim, int actionDim, const TD3Config& config = TD3Config());
    TD3Trainer(const TD3Trainer& other) = default;
    TD3Trainer& operator=(const TD3Trainer& other) = default;

    void SelectAction(const float* state, float* action);
    void SelectActionEval(const float* state, float* action);
    void SelectActionWithLatent(const float* state, float* action, int envIdx);
    void SelectActionBatchWithLatent(const float* states, float* actions, int batchSize, const std::vector<int>& envIndices);
    void SelectActionResidual(const float* state, float* residualAction);

    void Train(class ReplayBuffer& buffer);
    void TrainWithVectorRewards(class ReplayBuffer& buffer);

    void Save(const std::string& path) const;
    void Save(const std::string& path, const std::string& robotConfigPath, int numSatellites, int observationDim) const;
    void SaveToDisk(const std::string& path, const std::string& robotConfigPath, int numSatellites, int observationDim) const;
    void Load(const std::string& path);

    int GetStepCount() const { return mStepCount; }
    void IncrementStep() { mStepCount++; }

    SpanActorCritic& GetModel() { return mModel; }
    const SpanActorCritic& GetModel() const { return mModel; }

    void SetPreferenceVector(const std::array<float, VECTOR_REWARD_DIM>& pref) { mPreferenceVector = pref; }
    const std::array<float, VECTOR_REWARD_DIM>& GetPreferenceVector() const { return mPreferenceVector; }
    void SetPreference(float damageDealt, float damageTaken, float airtime, float energy)
    {
        mPreferenceVector[0] = damageDealt;
        mPreferenceVector[1] = damageTaken;
        mPreferenceVector[2] = airtime;
        mPreferenceVector[3] = energy;
    }

    float ComputeScalarReward(const VectorReward& vr) const
    {
        return vr.Dot(mPreferenceVector);
    }

    OpponentPool& GetOpponentPool() { return mOpponentPool; }
    const OpponentPool& GetOpponentPool() const { return mOpponentPool; }

    void SnapshotOpponent();
    bool SampleOpponent();

    // Get trainable parameters from RFF dynamics for optimizer
    std::vector<float> GetDynamicsWeights() const;
    void SetDynamicsWeights(const std::vector<float>& weights);

private:
    void UpdateCritic(class ReplayBuffer& buffer);
    void UpdateActor(class ReplayBuffer& buffer);
    void UpdateTargets();
    void UpdateCriticWithVectorRewards(class ReplayBuffer& buffer);
    float ComputeCriticLoss(RFFNetwork& critic, const float* criticInput, int batchSize);

    // Gradient computation for Muon optimizer
    void ComputeCriticGradients(RFFNetwork& critic, class ReplayBuffer& buffer, bool isCritic1);
    void ComputeActorGradient(class ReplayBuffer& buffer);

    // Checkpoint conversion for expanded observation space
    void ConvertAndLoadWeights(RFFNetwork& network, const std::vector<float>& oldWeights,
                               int oldStateDim, int oldActionDim);

    int mStateDim;
    int mActionDim;
    TD3Config mConfig;
    std::mt19937 mRng;

    std::array<float, VECTOR_REWARD_DIM> mPreferenceVector = {0.5f, 0.3f, 0.1f, 0.05f, 0.05f};

    SpanActorCritic mModel;
    OpponentPool mOpponentPool;
    mutable std::mutex mMutex;

    AlignedVector32<float> mBatchStates;
    AlignedVector32<float> mBatchActions;
    AlignedVector32<float> mBatchRewards;
    AlignedVector32<float> mBatchNextStates;
    AlignedVector32<float> mBatchDones;

    std::vector<VectorReward> mBatchVectorRewards;

    AlignedVector32<float> mNextActions;
    AlignedVector32<float> mQ1Values;
    AlignedVector32<float> mQ2Values;
    AlignedVector32<float> mTargetQ;

    AlignedVector32<float> mGrads;

    AlignedVector32<float> mBatchLogProbs;
    AlignedVector32<float> mTargetLogProbs;
    std::vector<int> mSampledIndices;

    AlignedVector32<float> mCriticInputBuffer;
    AlignedVector32<float> mActorOutputBuffer;
    AlignedVector32<float> mCriticQBuffer;
    AlignedVector32<float> mLatentZPos;
    AlignedVector32<float> mLatentZVel;

    int mStepCount = 0;
    int mUpdateCount = 0;

    // Performance profiling
    float mActionTime = 0.0f;
    float mStepTime = 0.0f;
    float mBufferTime = 0.0f;
    float mTrainTime = 0.0f;
    float mPhysicsTime = 0.0f;
    float mNetworkTime = 0.0f;
    PerformanceMetrics mPerfMetrics;

    // Persistent buffers for parallel training
    std::vector<SpanCache> mCriticCaches; // Caches for critic backprop
    std::vector<SpanCache> mActorCaches;  // Caches for actor backprop
    std::vector<AlignedVector32<float>> mThreadGrads; // For total gradients per thread
    std::vector<AlignedVector32<float>> mThreadQValueBuffers; // Per-thread buffer for Q-values (size 4)
    std::vector<AlignedVector32<float>> mThreadQGradBuffers;  // Per-thread buffer for Q-gradients (size 4)
    std::vector<AlignedVector32<float>> mThreadActorInputBuffers; // Per-thread buffer for actor inputs (state + latent)
    std::vector<AlignedVector32<float>> mThreadActionBuffers; // Per-thread buffer for actor actions (size actionDim)
    std::vector<AlignedVector32<float>> mThreadCriticInputBuffers; // Per-thread buffer for critic inputs (size criticInputDim)
    std::vector<AlignedVector32<float>> mThreadZPosBuffers; // Per-thread buffer for zPos (size latentDim)
    std::vector<AlignedVector32<float>> mThreadCriticInputGradBuffers; // Per-thread buffer for critic input gradients (size criticInputDim)


    // Muon optimizer for gradient-based training
    MuonOptimizer mActorOptimizer;
    MuonOptimizer mCritic1Optimizer;
    MuonOptimizer mCritic2Optimizer;
    MuonOptimizer mDynamicsOptimizer;  // For RFF latent dynamics
};
