/**
 * @file VectorEnv.h
 * @brief Batch processor for parallel RL environments
 * 
 * Manages multiple IEnvironment instances and steps them in batch.
 * Provides contiguous observation/reward buffers for efficient RL training.
 */

#pragma once

#include "../common/Types.h"
#include "../environment/IEnvironment.h"
#include "../physics/PhysicsWorld.h"
#include <vector>
#include <memory>
#include <cstdint>

/**
 * @brief Vectorized environment manager
 * 
 * Holds multiple IEnvironment instances and steps them in batch.
 * All environments share a single PhysicsWorld (Dimensional Ghosting).
 */
class VectorEnv {
public:
    /**
     * @brief Constructor
     * @param numEnvs Number of parallel environments
     * @param stepsPerEpisode Maximum steps per episode
     */
    explicit VectorEnv(int numEnvs, int stepsPerEpisode = MAX_EPISODE_STEPS);
    ~VectorEnv();
    
    VectorEnv(const VectorEnv&) = delete;
    VectorEnv& operator=(const VectorEnv&) = delete;
    VectorEnv(VectorEnv&&) = default;
    VectorEnv& operator=(VectorEnv&&) = default;
    
    /**
     * @brief Initialize all environments
     * @param physicsWorld Physics world (non-owning pointer)
     * @param initRobots Whether to create robots (true) or use pre-existing
     */
    void Init(PhysicsWorld* physicsWorld, bool initRobots = true);
    
    /** @brief Shutdown all environments */
    void Shutdown();
    
    /**
     * @brief Step all environments with batched actions
     * @param actions Contiguous action buffer [numEnvs * 2 * actionDim]
     *                Actions are interleaved: [env0_act1, env0_act2, env1_act1, env1_act2, ...]
     */
    void Step(const float* actions);
    
    /**
     * @brief Reset a specific environment
     * @param envIndex Environment index (-1 = reset all)
     */
    void Reset(int envIndex = -1);
    
    /** @brief Reset all done environments */
    void ResetDoneEnvs();
    
    /** @brief Get all observations [numEnvs * 2 * observationDim] */
    const float* GetObservations() const { return mAllObservations.data(); }
    
    /** @brief Get all rewards [numEnvs * 2] */
    const float* GetRewards() const { return mAllRewards.data(); }
    
    /** @brief Get all vector rewards [numEnvs * 2 * VECTOR_REWARD_DIM] */
    const float* GetVectorRewards() const { return mAllVectorRewards.data(); }
    
    /** @brief Get all done flags [numEnvs] */
    const char* GetDones() const { return mAllDones.data(); }
    
    /** @brief Get number of environments */
    int GetNumEnvs() const { return mNumEnvs; }
    
    /** @brief Get observation dimension per robot */
    int GetObservationDim() const { return mObservationDim; }
    
    /** @brief Get action dimension per robot */
    int GetActionDim() const { return mActionDim; }
    
    /** @brief Get total observation size (numEnvs * 2 * observationDim) */
    int GetTotalObservationSize() const { return mNumEnvs * 2 * mObservationDim; }
    
    /** @brief Get total action size (numEnvs * 2 * actionDim) */
    int GetTotalActionSize() const { return mNumEnvs * 2 * mActionDim; }
    
    /** @brief Get specific environment */
    IEnvironment* GetEnv(int index);
    const IEnvironment* GetEnv(int index) const;
    
    /** @brief Get physics world */
    PhysicsWorld* GetPhysicsWorld() { return mPhysicsWorld; }
    const PhysicsWorld* GetPhysicsWorld() const { return mPhysicsWorld; }
    
    /** @brief Check if any environment is done */
    bool AnyDone() const;
    
    /** @brief Get number of done environments */
    int CountDone() const;

private:
    void AllocateBuffers();
    void UpdateBuffers();
    
    PhysicsWorld* mPhysicsWorld = nullptr;
    std::vector<std::unique_ptr<IEnvironment>> mEnvs;
    
    int mNumEnvs;
    int mStepsPerEpisode;
    int mObservationDim = DEFAULT_OBSERVATION_DIM;
    int mActionDim = DEFAULT_ACTION_DIM;
    
    // Contiguous buffers for efficient RL training
    AlignedVector32<float> mAllObservations;    // [numEnvs * 2 * observationDim]
    AlignedVector32<float> mAllRewards;         // [numEnvs * 2]
    AlignedVector32<float> mAllVectorRewards;   // [numEnvs * 2 * VECTOR_REWARD_DIM]
    AlignedVector32<char> mAllDones;            // [numEnvs]
};
