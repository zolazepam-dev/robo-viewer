/**
 * @file IEnvironment.h
 * @brief Abstract interface for RL environments
 * 
 * Defines the contract for all reinforcement learning environments.
 * Implementations include CombatEnv, AircraftEnv, etc.
 */

#pragma once

#include <cstdint>
#include <array>
#include "../common/Types.h"
// VectorReward is defined in Types.h

/**
 * @brief Vector reward structure
 * 
 * Multi-objective reward components:
 * [0] damageDealt - Damage dealt to opponent
 * [1] damageTaken - Damage taken from opponent (negative)
 * [2] airtime - Time spent airborne
 * [3] energy - Energy efficiency
 * [4] survival - Survival bonus
 */

/**
 * @brief Environment state buffers
 */
struct EnvState {
    float* observations;    ///< [2 * observationDim]
    float* rewards;         ///< [2]
    bool done;
    VectorReward vectorReward1;
    VectorReward vectorReward2;
};

/**
 * @brief Abstract environment interface
 * 
 * All RL environments must implement this interface.
 * Thread safety: Implementations are NOT required to be thread-safe.
 * For parallel environments, use VectorEnv which manages synchronization.
 */
class IEnvironment {
public:
    virtual ~IEnvironment() = default;
    
    /**
     * @brief Initialize the environment
     * @param envIndex Environment index (for collision layers)
     * @param physicsWorld Physics world (non-owning pointer)
     */
    virtual void Init(uint32_t envIndex, void* physicsWorld) = 0;
    
    /** @brief Reset environment to initial state */
    virtual void Reset() = 0;
    
    /**
     * @brief Step the environment with actions
     * @param action1 Actions for agent 1 [actionDim]
     * @param action2 Actions for agent 2 [actionDim]
     */
    virtual void Step(const float* action1, const float* action2) = 0;
    
    /**
     * @brief Queue actions for the next physics step
     * @param action1 Actions for agent 1 [actionDim]
     * @param action2 Actions for agent 2 [actionDim]
     */
    virtual void QueueActions(const float* action1, const float* action2) = 0;
    
    /**
     * @brief Get current observations
     * @param obs1 Output buffer for agent 1 [observationDim]
     * @param obs2 Output buffer for agent 2 [observationDim]
     */
    virtual void GetObs(float* obs1, float* obs2) = 0;
    
    /**
     * @brief Get rewards
     * @param robotIdx Robot index (0 or 1)
     * @return Scalar reward
     */
    virtual float GetReward(int robotIdx) const = 0;
    
    /**
     * @brief Get vector rewards
     * @param robotIdx Robot index (0 or 1)
     * @return Vector reward components
     */
    virtual const VectorReward& GetVectorReward(int robotIdx) const = 0;
    
    /** @brief Check if episode is done */
    virtual bool IsDone() const = 0;
    
    /** @brief Get observation dimension */
    virtual int GetObservationDim() const = 0;
    
    /** @brief Get action dimension */
    virtual int GetActionDim() const = 0;
    
    /** @brief Get current step count */
    virtual int GetStepCount() const = 0;
    
    /** @brief Get environment index */
    virtual uint32_t GetEnvIndex() const = 0;
};
