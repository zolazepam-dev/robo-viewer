/**
 * @file OctopodEnv.h
 * @brief Combat environment for octopod robots
 * 
 * This file contains the OctopodEnv class for reinforcement learning
 * with two octopod robots combatting each other.
 * 
 * Octopod specifications:
 * - 25 bodies (1 central + 3 segments × 8 legs)
 * - 24 hinge constraints (3 per leg: swing, lift, knee)
 * - Action space: 24 motor angles (one per hinge)
 */

#pragma once

#include <Jolt/Jolt.h>
#include <vector>
#include <array>
#include <memory>
#include <map>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>

#include "PhysicsCore.h"
#include "NeuralMath.h"
#include "TD3Trainer.h"
#include "Logging.h"

#ifndef NUM_PARALLEL_ENVS
#define NUM_PARALLEL_ENVS 128
#endif

/** Arena size for octopod combat */
constexpr float OCTOPOD_ARENA_SIZE = 30.0f;
/** Half-size of the arena */
constexpr float OCTOPOD_ARENA_HALF = OCTOPOD_ARENA_SIZE * 0.5f;
/** Initial health points */
constexpr float OCTOPOD_INITIAL_HP = 100.0f;
/** Damage multiplier */
constexpr float OCTOPOD_DAMAGE_MULTIPLIER = 8.0f;
/** Maximum steps per episode */
constexpr int OCTOPOD_MAX_STEPS = 7200;

/** Number of hinge constraints per octopod (3 per leg × 8 legs) */
constexpr int OCTOPOD_ACTION_DIM = 24;

/** Observation dimension for octopod */
constexpr int OCTOPOD_OBS_DIM = 256;

/** Spawn distance between robots */
constexpr float OCTOPOD_SPAWN_DISTANCE = 10.0f;

/** Movement speed multiplier for stable, strong movement */
constexpr float OCTOPOD_SPEED_MULTIPLIER = 3.0f;

/** Reward weights */
constexpr float OCTOPOD_REWARD_DAMAGE = 15.0f;
constexpr float OCTOPOD_REWARD_UPRIGHT = 0.3f;
constexpr float OCTOPOD_REWARD_HEIGHT = 0.2f;
constexpr float OCTOPOD_REWARD_SPEED = 0.3f;
constexpr float OCTOPOD_REWARD_APPROACH = 0.8f;

constexpr float OCTOPOD_REWARD_DAMAGE_DEALT = 5.0f;
constexpr float OCTOPOD_REWARD_DAMAGE_TAKEN = -2.0f;
constexpr float OCTOPOD_REWARD_ENERGY = -0.001f;

/**
 * @brief Contact listener for octopod force sensing
 */
class OctopodContactListener : public JPH::ContactListener
{
public:
    OctopodContactListener() = default;

    /** @brief Get singleton instance */
    static OctopodContactListener& Get();

    void OnContactAdded(const JPH::Body& body1, const JPH::Body& body2,
                        const JPH::ContactManifold& manifold, JPH::ContactSettings& settings) override;
    
    void OnContactPersisted(const JPH::Body& body1, const JPH::Body& body2,
                            const JPH::ContactManifold& manifold, JPH::ContactSettings& settings) override;
    
    void OnContactRemoved(const JPH::SubShapeIDPair& subShapePair) override;

    /** @brief Get force reading for a robot */
    const std::array<float, 25>& GetForceReading(uint32_t envIdx, int robotIdx) const
    {
        return mForceReadingsPerEnv[envIdx][robotIdx];
    }

    /** @brief Reset force readings */
    void ResetForceReadings(uint32_t envIdx)
    {
        if (envIdx >= NUM_PARALLEL_ENVS) return;
        for (int i = 0; i < 25; i++) {
            mForceReadingsPerEnv[envIdx][0][i] = 0.0f;
            mForceReadingsPerEnv[envIdx][1][i] = 0.0f;
        }
    }

private:
    void ExtractImpulseData(const JPH::Body& body1, const JPH::Body& body2,
                            const JPH::ContactManifold& manifold);

    /** Force readings: [env][robot][body_contact_force] */
    std::array<std::array<std::array<float, 25>, 2>, NUM_PARALLEL_ENVS> mForceReadingsPerEnv = {};
};

/**
 * @brief State of a single octopod robot
 */
struct OctopodRobot
{
    std::vector<JPH::BodyID> bodies;          /** All 25 body IDs */
    std::vector<JPH::HingeConstraint*> constraints;  /** All 24 hinge constraints */
    std::map<std::string, JPH::BodyID> bodyMap;      /** Name to body ID mapping */
    JPH::BodyID centralBody;                  /** Central body ID */
    
    float hp = OCTOPOD_INITIAL_HP;
    float totalDamageDealt = 0.0f;
    float totalDamageTaken = 0.0f;
    float totalEnergyUsed = 0.0f;
    
    uint32_t envIndex = 0;
    int robotIndex = 0;
    
    bool IsValid() const { return !centralBody.IsInvalid(); }
};

/**
 * @class OctopodEnv
 * @brief Combat environment for two octopod robots
 * 
 * Action space: 24 hinge motor angles (normalized [-1, 1])
 * Observation space: Central body state, leg states, opponent relative state, forces
 */
class OctopodEnv
{
public:
    OctopodEnv() = default;
    ~OctopodEnv() = default;

    OctopodEnv(const OctopodEnv&) = delete;
    OctopodEnv& operator=(const OctopodEnv&) = delete;
    OctopodEnv(OctopodEnv&&) = default;
    OctopodEnv& operator=(OctopodEnv&&) = default;

    /**
     * @brief Initialize the environment
     * @param envIndex Environment index for collision layers
     * @param core Pointer to physics core
     * @param stepsPerEpisode Maximum steps per episode
     */
    void Init(uint32_t envIndex, PhysicsCore* core, int stepsPerEpisode = OCTOPOD_MAX_STEPS);
    
    /** @brief Reset the environment */
    void Reset();

    /**
     * @brief Apply actions to both robots
     * @param actions1 24 motor angles for robot 1
     * @param actions2 24 motor angles for robot 2
     */
    void QueueActions(const float* actions1, const float* actions2);
    
    /**
     * @brief Step the environment and harvest observations/rewards
     * @param obs1 Output observation for robot 1
     * @param obs2 Output observation for robot 2
     * @param reward1 Output reward for robot 1
     * @param reward2 Output reward for robot 2
     * @param done Whether episode is finished
     */
    void HarvestState(float* obs1, float* obs2, float* reward1, float* reward2, bool& done);

    /** @brief Get robot 1 */
    const OctopodRobot& GetRobot1() const { return mRobot1; }
    /** @brief Get robot 2 */
    const OctopodRobot& GetRobot2() const { return mRobot2; }
    /** @brief Get step count */
    int GetStepCount() const { return mStepCount; }
    /** @brief Get done flag */
    bool IsDone() const { return mDone; }

private:
    /** @brief Check collisions and apply damage */
    void CheckCollisions();
    
    /** @brief Calculate rewards */
    void CalculateRewards(float& r1, float& r2);
    
    /** @brief Update force sensor readings */
    void UpdateForceSensors();
    
    /**
     * @brief Build observation vector
     * @param obs Output observation array
     * @param robot Our robot
     * @param opponent Opponent robot
     * @param forces Force readings
     */
    void BuildObservationVector(float* obs, const OctopodRobot& robot,
                                const OctopodRobot& opponent, const std::array<float, 25>& forces);

    JPH::PhysicsSystem* mPhysicsSystem = nullptr;
    PhysicsCore* mCore = nullptr;
    
    OctopodRobot mRobot1;
    OctopodRobot mRobot2;
    
    uint32_t mEnvIndex = 0;
    int mStepCount = 0;
    int mStepsPerEpisode = OCTOPOD_MAX_STEPS;
    bool mDone = false;

    JPH::RVec3 mKothPoint{0, 0, 0};  /** King of the Hill point */
    JPH::BodyID mKothVisualId;        /** KOTH visual marker */

    float mPrevHp1 = OCTOPOD_INITIAL_HP;
    float mPrevHp2 = OCTOPOD_INITIAL_HP;
    float mPrevEnergy1 = 0.0f;
    float mPrevEnergy2 = 0.0f;

    VectorReward mReward1;
    VectorReward mReward2;

    std::array<JPH::RVec3, 25> mInitialPositions1{};
    std::array<JPH::RVec3, 25> mInitialPositions2{};
};
