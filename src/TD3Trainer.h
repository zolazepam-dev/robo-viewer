#pragma once

#include <memory>
#include <random>
#include <vector>
#include <array>
#include <cmath>

#include "SpanNetwork.h"
#include "LatentMemory.h"
#include "OpponentPool.h"
#include "NeuralMath.h"
#include "NeuralNetwork.h"

struct TD3Config
{
    int hiddenDim = 256;
    int latentDim = 64;
    float actorLR = 3e-4f;
    float criticLR = 3e-4f;
    float gamma = 0.99f;
    float tau = 0.005f;
    float policyNoise = 0.2f;
    float noiseClip = 0.5f;
    float explNoise = 0.1f;
    int policyDelay = 2;
    int batchSize = 256;
    int bufferSize = 1000000;
    int startSteps = 10000;
    int snapshotInterval = 10000;
};

class TD3Trainer
{
public:
    TD3Trainer(int stateDim, int actionDim, const TD3Config& config = TD3Config());
    
    void SelectAction(const float* state, float* action);
    void SelectActionEval(const float* state, float* action);
    void SelectActionWithLatent(const float* state, float* action, int envIdx);
    void SelectActionResidual(const float* state, float* residualAction);
    
    void Train(class ReplayBuffer& buffer);
    void TrainWithVectorRewards(class ReplayBuffer& buffer);
    
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

private:
    void UpdateCritic(class ReplayBuffer& buffer);
    void UpdateActor();
    void UpdateTargets();
    void UpdateCriticWithVectorRewards(class ReplayBuffer& buffer);
    
    int mStateDim;
    int mActionDim;
    TD3Config mConfig;

    std::array<float, VECTOR_REWARD_DIM> mPreferenceVector = {0.5f, 0.3f, 0.15f, 0.05f};
    
    SpanActorCritic mModel;
    OpponentPool mOpponentPool;
    
    std::vector<float> mBatchStates;
    std::vector<float> mBatchActions;
    std::vector<float> mBatchRewards;
    std::vector<float> mBatchNextStates;
    std::vector<float> mBatchDones;

    std::vector<VectorReward> mBatchVectorRewards;
    
    std::vector<float> mNextActions;
    std::vector<float> mQ1Values;
    std::vector<float> mQ2Values;
    std::vector<float> mTargetQ;
    
    std::vector<float> mGrads;

    std::vector<float> mBatchLogProbs;
    std::vector<float> mTargetLogProbs;
    std::vector<int> mSampledIndices;
    
    std::mt19937 mRng;
    int mStepCount = 0;
    int mUpdateCount = 0;
};
