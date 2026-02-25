#pragma once

#include <memory>
#include <random>
#include <vector>

#include "NeuralNetwork.h"
#include "VectorizedEnv.h"

// TD3 hyperparameters
struct TD3Config {
    int hiddenDim = 256;
    float actorLR = 3e-4f;
    float criticLR = 3e-4f;
    float gamma = 0.99f;
    float tau = 0.005f;
    float policyNoise = 0.2f;
    float noiseClip = 0.5f;
    float explNoise = 0.1f;
    int policyDelay = 2;  // Delayed policy updates
    int batchSize = 256;
    int bufferSize = 1000000;
    int startSteps = 10000;
};

class TD3Trainer {
public:
    TD3Trainer(int stateDim, int actionDim, const TD3Config& config = TD3Config());
    
    // Select action with exploration noise
    void SelectAction(const float* state, float* action);
    
    // Select action without noise (for evaluation)
    void SelectActionEval(const float* state, float* action);
    
    // Train step
    void Train(ReplayBuffer& buffer);
    
    // Save/load
    void Save(const std::string& path) const;
    void Load(const std::string& path);
    
    // Get step count
    int GetStepCount() const { return mStepCount; }
    void IncrementStep() { mStepCount++; }
    
    NeuralNetwork& GetActor() { return *mActor; }
    const NeuralNetwork& GetActor() const { return *mActor; }
    
private:
    void UpdateCritic(ReplayBuffer& buffer);
    void UpdateActor();
    void UpdateTargets();
    
    int mStateDim;
    int mActionDim;
    TD3Config mConfig;
    
    std::unique_ptr<NeuralNetwork> mActor;
    std::unique_ptr<NeuralNetwork> mActorTarget;
    std::unique_ptr<NeuralNetwork> mCritic1;
    std::unique_ptr<NeuralNetwork> mCritic1Target;
    std::unique_ptr<NeuralNetwork> mCritic2;
    std::unique_ptr<NeuralNetwork> mCritic2Target;
    
    // Workspace for training
    std::vector<float> mBatchStates;
    std::vector<float> mBatchActions;
    std::vector<float> mBatchRewards;
    std::vector<float> mBatchNextStates;
    std::vector<float> mBatchDones;
    
    std::vector<float> mNextActions;
    std::vector<float> mQ1Values;
    std::vector<float> mQ2Values;
    std::vector<float> mTargetQ;
    
    std::vector<float> mGradsW;
    std::vector<float> mGradsB;
    
    std::mt19937 mRng;
    int mStepCount = 0;
    int mUpdateCount = 0;
};
