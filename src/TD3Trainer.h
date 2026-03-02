#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

#include "SpanNetwork.h"
#include "LatentMemory.h"
#include "OpponentPool.h"
#include "NeuralNetwork.h"
#include "AlignedAllocator.h"
#include "IntrinsicMotivation.h"
#include "AttentionEncoder.h"

struct TD3Config
{
    int hiddenDim = 1024;
    int latentDim = 128;
    float actorLR = 3e-4f;
    float criticLR = 3e-4f;
    float gamma = 0.99f;
    float tau = 0.005f;
    float policyNoise = 0.2f;
    float noiseClip = 0.5f;
    float explNoise = 0.1f;
    int policyDelay = 4;           // Was 8 - more frequent actor updates
    int batchSize = 32;            // Was 16 - better amortization
    int bufferSize = 1000000;
    int startSteps = 2000;         // Was 500 - more exploration initially
    int snapshotInterval = 10000;
    
    // PRIMALPHA: New hyperparameters
    float herRatio = 0.5f;              // HER relabeling ratio
    float intrinsicRewardScale = 0.1f;  // Curiosity reward weight
    bool useAttention = true;           // Enable attention encoding
    int ensembleSubset = 2;             // Number of critics to sample for target
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
    
    void Train(ReplayBuffer& buffer);
    void TrainWithVectorRewards(ReplayBuffer& buffer);
    
    void Save(const std::string& path) const;
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
    
    // PRIMALPHA: New methods
    void SetIntrinsicRewardScale(float scale) { mIntrinsicMotivation.SetCuriosityScale(scale); }
    float GetAverageIntrinsicReward() const { return mIntrinsicMotivation.GetAverageIntrinsicReward(); }
    void SetAttentionEnabled(bool enabled) { mConfig.useAttention = enabled; }

private:
    void UpdateCritic(ReplayBuffer& buffer);
    void UpdateActor(ReplayBuffer& buffer);
    void UpdateTargets();
    void UpdateCriticWithVectorRewards(ReplayBuffer& buffer);
    float ComputeCriticLoss(SpanNetwork& critic, const float* criticInput, int batchSize);
    
    // PRIMALPHA: Ensemble critic methods
    float ComputeEnsembleMinQ(const float* stateActionInput);
    float ComputeEnsembleMeanQ(const float* stateActionInput);
    float ComputeEnsembleStdDev(const float* stateActionInput);
    
    int mStateDim;
    int mActionDim;
    TD3Config mConfig;
    std::mt19937 mRng;
    
    std::array<float, VECTOR_REWARD_DIM> mPreferenceVector = {0.5f, 0.3f, 0.1f, 0.05f, 0.05f};
    
    SpanActorCritic mModel;
    OpponentPool mOpponentPool;
    
    // PRIMALPHA: New components
    IntrinsicMotivation mIntrinsicMotivation;
    AttentionStateEncoder mAttentionEncoder;
    
    AlignedVector32<float> mBatchStates;
    AlignedVector32<float> mBatchActions;
    AlignedVector32<float> mBatchRewards;
    AlignedVector32<float> mBatchNextStates;
    AlignedVector32<float> mBatchDones;

    std::vector<VectorReward> mBatchVectorRewards;
    
    AlignedVector32<float> mNextActions;
    AlignedVector32<float> mQValues[NUM_ENSEMBLE_CRITICS];  // Ensemble Q-values
    AlignedVector32<float> mTargetQ;
    
    AlignedVector32<float> mGrads;

    AlignedVector32<float> mBatchLogProbs;
    AlignedVector32<float> mTargetLogProbs;
    std::vector<int> mSampledIndices;

    AlignedVector32<float> mCriticInputBuffer;
    
    // PRIMALPHA: Additional buffers
    AlignedVector32<float> mAttendedStates;
    AlignedVector32<float> mAttendedNextStates;
    AlignedVector32<float> mIntrinsicRewards;
    AlignedVector32<float> mActorOutputBuffer;
    AlignedVector32<float> mCriticQBuffer;
    AlignedVector32<float> mLatentZPos;
    AlignedVector32<float> mLatentZVel;

    int mStepCount = 0;
    int mUpdateCount = 0;
};
