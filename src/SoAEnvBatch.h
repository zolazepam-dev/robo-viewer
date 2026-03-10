#pragma once

#include "AlignedAllocator.h"
#include "NeuralMath.h"
#include <vector>
#include <cstring>

// ============================================================================
// SOA ENVIRONMENT STATE CONTAINER
// ============================================================================
// Structure of Arrays layout for optimal cache utilization during batch access
// 
// Traditional AoS (Array of Structures):
//   struct EnvState { float obs[256]; float actions[32]; ... };
//   std::vector<EnvState> envs;
//   // Accessing all observations = strided memory access (cache misses)
//
// SoA (Structure of Arrays):
//   struct EnvBatch { float allObs[N*256]; float allActions[N*32]; ... };
//   // Accessing all observations = contiguous memory access (cache friendly)
//
// Expected improvement: 3-5x better cache utilization for batched operations
// ============================================================================

class SoAEnvBatch {
public:
    struct Config {
        int numEnvs;
        int obsDim;
        int actionDim;
        int rewardDim;
        
        Config() : numEnvs(256), obsDim(256), actionDim(32), rewardDim(4) {}
    };
    
    SoAEnvBatch(const Config& config = Config());
    ~SoAEnvBatch() = default;
    
    // Initialize batch for given number of environments
    void Init(int numEnvs, int obsDim, int actionDim, int rewardDim = 4);
    
    // Resize for different batch size
    void Resize(int numEnvs);
    
    // Clear all data
    void Clear();
    
    // =========================================================================
    // GETTERS (contiguous access - cache friendly)
    // =========================================================================
    
    // Get all observations (contiguous: [numEnvs * obsDim * 2] for 2 robots)
    const float* GetAllObservations() const { return mAllObservations.data(); }
    float* GetAllObservations() { return mAllObservations.data(); }
    
    // Get observation for specific environment and robot
    const float* GetObservation(int envIdx, int robotIdx = 0) const {
        return mAllObservations.data() + (envIdx * 2 + robotIdx) * mObsDim;
    }
    float* GetObservation(int envIdx, int robotIdx = 0) {
        return mAllObservations.data() + (envIdx * 2 + robotIdx) * mObsDim;
    }
    
    // Get all rewards (contiguous: [numEnvs * 2])
    const float* GetAllRewards() const { return mAllRewards.data(); }
    float* GetAllRewards() { return mAllRewards.data(); }
    
    // Get reward for specific environment and robot
    float GetReward(int envIdx, int robotIdx = 0) const {
        return mAllRewards[envIdx * 2 + robotIdx];
    }
    void SetReward(int envIdx, int robotIdx, float reward) {
        mAllRewards[envIdx * 2 + robotIdx] = reward;
    }
    
    // Get all vector rewards (contiguous: [numEnvs * 2 * rewardDim])
    const float* GetAllVectorRewards() const { return mAllVectorRewards.data(); }
    float* GetAllVectorRewards() { return mAllVectorRewards.data(); }
    
    // Get all dones (contiguous: [numEnvs * 2])
    const bool* GetAllDones() const { return mAllDones.data(); }
    bool* GetAllDones() { return mAllDones.data(); }
    
    // Get done for specific environment
    bool GetDone(int envIdx, int robotIdx = 0) const {
        return mAllDones[envIdx * 2 + robotIdx];
    }
    void SetDone(int envIdx, int robotIdx, bool done) {
        mAllDones[envIdx * 2 + robotIdx] = done;
    }
    
    // Get all actions (contiguous: [numEnvs * 2 * actionDim])
    const float* GetAllActions() const { return mAllActions.data(); }
    float* GetAllActions() { return mAllActions.data(); }
    
    // Get action for specific environment and robot
    float* GetAction(int envIdx, int robotIdx = 0) {
        return mAllActions.data() + (envIdx * 2 + robotIdx) * mActionDim;
    }
    const float* GetAction(int envIdx, int robotIdx = 0) const {
        return mAllActions.data() + (envIdx * 2 + robotIdx) * mActionDim;
    }
    
    // =========================================================================
    // BATCHED OPERATIONS (optimized for SoA layout)
    // =========================================================================
    
    // Copy observations to contiguous buffer (for neural network input)
    void GatherObservations(float* output, int startEnv = 0, int count = -1) const {
        if (count < 0) count = mNumEnvs - startEnv;
        size_t bytes = count * 2 * mObsDim * sizeof(float);
        std::memcpy(output, GetAllObservations() + startEnv * 2 * mObsDim, bytes);
    }
    
    // Scatter actions from contiguous buffer (from neural network output)
    void ScatterActions(const float* input, int startEnv = 0, int count = -1) {
        if (count < 0) count = mNumEnvs - startEnv;
        size_t bytes = count * 2 * mActionDim * sizeof(float);
        std::memcpy(GetAllActions() + startEnv * 2 * mActionDim, input, bytes);
    }
    
    // Batch normalize observations (in-place)
    void NormalizeObservations(const float* mean, const float* std, int obsDim = -1);
    
    // Batch clip actions to [-1, 1]
    void ClipActions();
    
    // Batch reset dones
    void ResetDones();
    
    // Batch reset rewards
    void ResetRewards();
    
    // =========================================================================
    // STATISTICS
    // =========================================================================
    
    // Get batch statistics (for monitoring)
    float GetMeanReward() const;
    float GetStdReward() const;
    float GetDoneRate() const;
    
    // Get dimensions
    int GetNumEnvs() const { return mNumEnvs; }
    int GetObsDim() const { return mObsDim; }
    int GetActionDim() const { return mActionDim; }
    int GetRewardDim() const { return mRewardDim; }
    
    // Get total sizes
    size_t GetObservationSize() const { return mNumEnvs * 2 * mObsDim; }
    size_t GetActionSize() const { return mNumEnvs * 2 * mActionDim; }
    size_t GetRewardSize() const { return mNumEnvs * 2; }

private:
    // Core data arrays (SoA layout)
    AlignedVector32<float> mAllObservations;  // [numEnvs * 2 * obsDim]
    AlignedVector32<float> mAllActions;       // [numEnvs * 2 * actionDim]
    AlignedVector32<float> mAllRewards;       // [numEnvs * 2]
    AlignedVector32<float> mAllVectorRewards; // [numEnvs * 2 * rewardDim]
    std::vector<bool> mAllDones;              // [numEnvs * 2]
    
    int mNumEnvs;
    int mObsDim;
    int mActionDim;
    int mRewardDim;
    
    // Temporary buffers for operations
    AlignedVector32<float> mTempBuffer;
};

// ============================================================================
// BATCHED ENVIRONMENT STEPPER
// ============================================================================
// Vectorized environment stepping with SoA data layout

class BatchedEnvStepper {
public:
    struct Config {
        int numEnvs;
        int stepsPerEpisode;
        bool useParallelStep;
        int numThreads;
        
        Config() : numEnvs(256), stepsPerEpisode(7200), useParallelStep(true), numThreads(8) {}
    };
    
    BatchedEnvStepper(const Config& config = Config());
    ~BatchedEnvStepper() = default;
    
    // Initialize environments
    void Init(const std::string& robotConfigPath);
    
    // Step all environments in parallel
    void StepAll(const float* actions);
    
    // Step single environment
    void StepEnv(int envIdx, const float* actions);
    
    // Reset all environments
    void ResetAll();
    
    // Reset specific environment
    void ResetEnv(int envIdx);
    
    // Get SoA batch
    SoAEnvBatch& GetBatch() { return mBatch; }
    const SoAEnvBatch& GetBatch() const { return mBatch; }
    
    // Get number of environments
    int GetNumEnvs() const { return mBatch.GetNumEnvs(); }

private:
    SoAEnvBatch mBatch;
    Config mConfig;
    
    // Per-environment state tracking
    std::vector<int> mEpisodeSteps;
    std::vector<int> mEpisodeRewards;
    
    // Physics systems (one per thread for parallel stepping)
    // (Implementation depends on PhysicsCore API)
};

// ============================================================================
// IMPLEMENTATION
// ============================================================================

inline SoAEnvBatch::SoAEnvBatch(const Config& config) 
    : mNumEnvs(config.numEnvs)
    , mObsDim(config.obsDim)
    , mActionDim(config.actionDim)
    , mRewardDim(config.rewardDim) {
    
    // Allocate all arrays with proper alignment
    mAllObservations.resize(mNumEnvs * 2 * mObsDim);
    mAllActions.resize(mNumEnvs * 2 * mActionDim);
    mAllRewards.resize(mNumEnvs * 2);
    mAllVectorRewards.resize(mNumEnvs * 2 * mRewardDim);
    mAllDones.resize(mNumEnvs * 2);
    
    // Clear initial data
    Clear();
}

inline void SoAEnvBatch::Init(int numEnvs, int obsDim, int actionDim, int rewardDim) {
    mNumEnvs = numEnvs;
    mObsDim = obsDim;
    mActionDim = actionDim;
    mRewardDim = rewardDim;
    
    mAllObservations.resize(mNumEnvs * 2 * mObsDim);
    mAllActions.resize(mNumEnvs * 2 * mActionDim);
    mAllRewards.resize(mNumEnvs * 2);
    mAllVectorRewards.resize(mNumEnvs * 2 * mRewardDim);
    mAllDones.resize(mNumEnvs * 2);
    
    Clear();
}

inline void SoAEnvBatch::Resize(int numEnvs) {
    mNumEnvs = numEnvs;
    mAllObservations.resize(mNumEnvs * 2 * mObsDim);
    mAllActions.resize(mNumEnvs * 2 * mActionDim);
    mAllRewards.resize(mNumEnvs * 2);
    mAllVectorRewards.resize(mNumEnvs * 2 * mRewardDim);
    mAllDones.resize(mNumEnvs * 2);
}

inline void SoAEnvBatch::Clear() {
    std::memset(mAllObservations.data(), 0, mAllObservations.size() * sizeof(float));
    std::memset(mAllActions.data(), 0, mAllActions.size() * sizeof(float));
    std::memset(mAllRewards.data(), 0, mAllRewards.size() * sizeof(float));
    std::memset(mAllVectorRewards.data(), 0, mAllVectorRewards.size() * sizeof(float));
    std::fill(mAllDones.begin(), mAllDones.end(), false);
}

inline void SoAEnvBatch::NormalizeObservations(const float* mean, const float* std, int obsDim) {
    if (obsDim < 0) obsDim = mObsDim;
    
    #pragma omp parallel for
    for (int envRobot = 0; envRobot < mNumEnvs * 2; envRobot++) {
        float* obs = mAllObservations.data() + envRobot * obsDim;
        for (int i = 0; i < obsDim; i++) {
            obs[i] = (obs[i] - mean[i]) / (std[i] + 1e-8f);
        }
    }
}

inline void SoAEnvBatch::ClipActions() {
    const int totalActions = mNumEnvs * 2 * mActionDim;
    
    #pragma omp simd aligned(mAllActions: 32)
    for (int i = 0; i < totalActions; i++) {
        mAllActions[i] = std::clamp(mAllActions[i], -1.0f, 1.0f);
    }
}

inline void SoAEnvBatch::ResetDones() {
    std::fill(mAllDones.begin(), mAllDones.end(), false);
}

inline void SoAEnvBatch::ResetRewards() {
    std::memset(mAllRewards.data(), 0, mAllRewards.size() * sizeof(float));
}

inline float SoAEnvBatch::GetMeanReward() const {
    float sum = 0.0f;
    for (float r : mAllRewards) sum += r;
    return sum / mAllRewards.size();
}

inline float SoAEnvBatch::GetStdReward() const {
    float mean = GetMeanReward();
    float var = 0.0f;
    for (float r : mAllRewards) {
        float diff = r - mean;
        var += diff * diff;
    }
    return std::sqrt(var / mAllRewards.size());
}

inline float SoAEnvBatch::GetDoneRate() const {
    int doneCount = 0;
    for (bool d : mAllDones) if (d) doneCount++;
    return static_cast<float>(doneCount) / mAllDones.size();
}

// BatchedEnvStepper implementation
inline BatchedEnvStepper::BatchedEnvStepper(const Config& config) 
    : mConfig(config) {
    
    SoAEnvBatch::Config batchConfig;
    batchConfig.numEnvs = config.numEnvs;
    mBatch.Init(batchConfig);
    
    mEpisodeSteps.resize(config.numEnvs * 2, 0);
    mEpisodeRewards.resize(config.numEnvs * 2, 0);
}

inline void BatchedEnvStepper::Init(const std::string& robotConfigPath) {
    // Initialize physics systems and robots
    // (Implementation depends on PhysicsCore API)
}

inline void BatchedEnvStepper::StepAll(const float* actions) {
    // Scatter actions to batch
    mBatch.ScatterActions(actions);
    
    // Step all environments in parallel
    #pragma omp parallel for
    for (int envIdx = 0; envIdx < mConfig.numEnvs; envIdx++) {
        // Get actions for this environment
        const float* envActions = mBatch.GetAction(envIdx, 0);
        
        // Step physics (implementation depends on PhysicsCore)
        // physicsCore.Step(envIdx, envActions);
        
        // Update episode step counter
        mEpisodeSteps[envIdx * 2]++;
        mEpisodeSteps[envIdx * 2 + 1]++;
        
        // Check for episode end
        if (mEpisodeSteps[envIdx * 2] >= mConfig.stepsPerEpisode) {
            mBatch.SetDone(envIdx, 0, true);
            mEpisodeSteps[envIdx * 2] = 0;
        }
        if (mEpisodeSteps[envIdx * 2 + 1] >= mConfig.stepsPerEpisode) {
            mBatch.SetDone(envIdx, 1, true);
            mEpisodeSteps[envIdx * 2 + 1] = 0;
        }
    }
}

inline void BatchedEnvStepper::StepEnv(int envIdx, const float* actions) {
    // Step single environment
    float* envAction = mBatch.GetAction(envIdx, 0);
    std::memcpy(envAction, actions, mBatch.GetActionDim() * sizeof(float));
    
    // Step physics
    // physicsCore.Step(envIdx, envAction);
    
    mEpisodeSteps[envIdx * 2]++;
    if (mEpisodeSteps[envIdx * 2] >= mConfig.stepsPerEpisode) {
        mBatch.SetDone(envIdx, 0, true);
        mEpisodeSteps[envIdx * 2] = 0;
    }
}

inline void BatchedEnvStepper::ResetAll() {
    mBatch.ResetDones();
    mBatch.ResetRewards();
    std::fill(mEpisodeSteps.begin(), mEpisodeSteps.end(), 0);
    std::fill(mEpisodeRewards.begin(), mEpisodeRewards.end(), 0);
}

inline void BatchedEnvStepper::ResetEnv(int envIdx) {
    mBatch.SetDone(envIdx, 0, false);
    mBatch.SetDone(envIdx, 1, false);
    mEpisodeSteps[envIdx * 2] = 0;
    mEpisodeSteps[envIdx * 2 + 1] = 0;
}
