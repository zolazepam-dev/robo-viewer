/**
 * @file CombatEnv.h
 * @brief Combat environment for reinforcement learning
 * 
 * This file contains the CombatEnv class and CombatContactListener class,
 * which define the combat simulation environment for reinforcement learning.
 * The environment manages two robots, their interactions, and the reward system.
 */

#pragma once

#include <Jolt/Jolt.h>
#include <vector>
#include <array>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/ContactListener.h>

#include "PhysicsCore.h"
#include "NeuralMath.h"
#include "NeuralNetwork.h"
#include "CombatRobot.h"
#include "AlignedAllocator.h"

/** Size of the combat arena in meters */
constexpr float ARENA_SIZE = 36.0f;
/** Half-size of the combat arena in meters */
constexpr float ARENA_HALF = ARENA_SIZE * 0.5f;
/** Offset from center for robot spawning in meters */
constexpr float ROBOT_SPAWN_OFFSET = 10.0f;
/** Initial health points for each robot */
constexpr float INITIAL_HP = 100.0f;
/** Multiplier for damage calculations */
constexpr float DAMAGE_MULTIPLIER = 5.0f;
/** Maximum number of steps per episode */
constexpr int MAX_EPISODE_STEPS = 7200; // 2 Minutes at 60Hz

/**
 * @brief Calculate force sensor dimension from number of satellites
 * @param numSatellites Number of satellites
 * @return Force sensor dimension
 */
constexpr int GetForceSensorDim(int numSatellites) {
    return numSatellites * 2;
}

/**
 * @brief Calculate base observation dimension from number of satellites
 * @param numSatellites Number of satellites
 * @return Base observation dimension
 */
constexpr int GetObservationBaseDim(int numSatellites) {
    return 18 + (numSatellites * 6) + (numSatellites * 3) + (numSatellites * 3);
}

/**
 * @class CombatContactListener
 * @brief Contact listener for force sensor data
 * 
 * Captures collision impulses and updates force sensor readings for each robot.
 */
class CombatContactListener : public JPH::ContactListener
{
public:
    CombatContactListener() = default;

    /** @brief Get singleton instance of the contact listener */
    static CombatContactListener& Get();

    /** @brief Called when a new contact is added */
    void OnContactAdded(const JPH::Body& body1, const JPH::Body& body2,
                        const JPH::ContactManifold& manifold, JPH::ContactSettings& settings) override;
    
    /** @brief Called when a contact is persisted */
    void OnContactPersisted(const JPH::Body& body1, const JPH::Body& body2,
                            const JPH::ContactManifold& manifold, JPH::ContactSettings& settings) override;
    
    /** @brief Called when a contact is removed */
    void OnContactRemoved(const JPH::SubShapeIDPair& subShapePair) override;

    /**
     * @brief Get force sensor reading for a specific robot
     * @param envIdx Environment index
     * @param robotIdx Robot index
     * @return Force sensor reading
     */
    ForceSensorReading& GetForceReading(uint32_t envIdx, int robotIdx)
    {
        return mForceReadingsPerEnv[envIdx][robotIdx];
    }

    /**
     * @brief Get const force sensor reading for a specific robot
     * @param envIdx Environment index
     * @param robotIdx Robot index
     * @return Const force sensor reading
     */
    const ForceSensorReading& GetForceReading(uint32_t envIdx, int robotIdx) const
    {
        return mForceReadingsPerEnv[envIdx][robotIdx];
    }

    /**
     * @brief Reset force sensor readings for an environment
     * @param envIdx Environment index
     * @param numSatellites Number of satellites per robot
     */
    void ResetForceReadings(uint32_t envIdx, int numSatellites)
    {
        mForceReadingsPerEnv[envIdx][0].Reset(numSatellites);
        mForceReadingsPerEnv[envIdx][1].Reset(numSatellites);
    }

    /**
     * @brief Extract impulse data from a contact manifold
     * @param body1 First body in contact
     * @param body2 Second body in contact
     * @param manifold Contact manifold
     */
    void ExtractImpulseData(const JPH::Body& body1, const JPH::Body& body2,
                            const JPH::ContactManifold& manifold);

private:
    /** Force sensor readings per environment and robot */
    std::array<std::array<ForceSensorReading, 2>, NUM_PARALLEL_ENVS> mForceReadingsPerEnv;
};

/**
 * @class CombatEnv
 * @brief Combat environment for reinforcement learning
 * 
 * Manages two combat robots, their interactions, and the reward system.
 * Provides methods for initializing, resetting, and stepping the environment.
 */
class CombatEnv
{
public:
    CombatEnv() = default;
    ~CombatEnv() = default;

    CombatEnv(const CombatEnv&) = delete;
    CombatEnv& operator=(const CombatEnv&) = delete;
    CombatEnv(CombatEnv&&) = default;
    CombatEnv& operator=(CombatEnv&&) = default;

     /**
      * @brief Initialize the combat environment
      * @param envIndex Environment index
      * @param globalPhysics Pointer to physics system
      * @param globalLoader Pointer to robot loader
      * @param stepsPerEpisode Maximum number of steps per episode
      */
     void Init(uint32_t envIndex, JPH::PhysicsSystem* globalPhysics, CombatRobotLoader* globalLoader, int stepsPerEpisode = MAX_EPISODE_STEPS);
    
    /** @brief Reset the environment to initial state */
    void Reset();

    /**
     * @brief Queue actions for both robots
     * @param actions1 Actions for robot 1
     * @param actions2 Actions for robot 2
     */
    void QueueActions(const float* actions1, const float* actions2);
    
    /**
     * @brief Harvest state and rewards from the environment
     * @param obs1 Observation vector for robot 1
     * @param obs2 Observation vector for robot 2
     * @param reward1 Reward for robot 1
     * @param reward2 Reward for robot 2
     * @param done Whether the episode is done
     */
    void HarvestState(float* obs1, float* obs2, float* reward1, float* reward2, bool& done);

    /** @brief Get robot 1 data */
    const CombatRobotData& GetRobot1() const { return mRobot1; }
    /** @brief Get robot 2 data */
    const CombatRobotData& GetRobot2() const { return mRobot2; }
    /** @brief Get reference to robot 1 data */
    CombatRobotData& GetRobot1Ref() { return mRobot1; }
    /** @brief Get reference to robot 2 data */
    CombatRobotData& GetRobot2Ref() { return mRobot2; }
    /** @brief Get robot 1 reward structure */
    const VectorReward& GetRobot1Reward() const { return mReward1; }
    /** @brief Get robot 2 reward structure */
    const VectorReward& GetRobot2Reward() const { return mReward2; }
    /** @brief Get current step count */
    int GetStepCount() const { return mStepCount; }
    /** @brief Get done flag */
    bool IsDone() const { return mDone; }
    /** @brief Get observation dimension */
    int GetObservationDim() const { return mObservationDim; }

private:
    /** @brief Check for collisions between robots */
    void CheckCollisions();
    
    /**
     * @brief Calculate rewards for both robots
     * @param r1 Reward for robot 1
     * @param r2 Reward for robot 2
     */
    void CalculateRewards(float& r1, float& r2);
    
    /** @brief Compute airtime for robots */
    float ComputeAirtime() const;
    
    /**
     * @brief Compute energy used from actions
     * @param actions Action vector
     * @param actionDim Action dimension
     * @return Energy used
     */
    float ComputeEnergyUsed(const float* actions, int actionDim) const;
    
    /** @brief Update force sensors */
    void UpdateForceSensors();
    
    /**
     * @brief Build observation vector for a robot
     * @param obs Output observation vector
     * @param robot Robot to observe
     * @param opponent Opponent robot
     * @param forces Force sensor readings
     */
    void BuildObservationVector(float* obs, const CombatRobotData& robot,
                                 const CombatRobotData& opponent, const ForceSensorReading& forces);

    JPH::PhysicsSystem* mPhysicsSystem = nullptr; ///< Pointer to physics system
    CombatRobotLoader* mRobotLoader = nullptr; ///< Pointer to robot loader
    int mStepsPerEpisode = MAX_EPISODE_STEPS; ///< Maximum number of steps per episode

    CombatRobotData mRobot1; ///< Robot 1 data
    CombatRobotData mRobot2; ///< Robot 2 data
    uint32_t mEnvIndex = 0; ///< Environment index
    int mStepCount = 0; ///< Current step count
    bool mDone = false; ///< Done flag
    int mObservationDim = 256; ///< Observation dimension

    JPH::RVec3 mKothPoint{0,0,0}; ///< King of the Hill point
    JPH::BodyID mKothVisualId; ///< King of the Hill visual body ID

    float mPrevHp1 = INITIAL_HP; ///< Previous health of robot 1
    float mPrevHp2 = INITIAL_HP; ///< Previous health of robot 2
    float mPrevEnergy1 = 0.0f; ///< Previous energy of robot 1
    float mPrevEnergy2 = 0.0f; ///< Previous energy of robot 2

    float mAirAccumulator1 = 0.0f; ///< Air time accumulator for robot 1
    float mAirAccumulator2 = 0.0f; ///< Air time accumulator for robot 2
    
    VectorReward mReward1; ///< Reward structure for robot 1
    VectorReward mReward2; ///< Reward structure for robot 2
};
