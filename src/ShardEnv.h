/**
 * @file ShardEnv.h
 * @brief Environment for orbital shard robot training
 */

#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include "PhysicsCore.h"
#include <vector>

constexpr int SHARD_ACTION_DIM = 4;      // 4 wing motors
constexpr int SHARD_OBS_DIM = 32;        // Observation dimension
constexpr int SHARD_MAX_STEPS = 7200;    // 2 minutes at 60Hz
constexpr float SHARD_INITIAL_HP = 100.0f;

struct ShardRobot {
    std::vector<JPH::BodyID> bodies;
    std::vector<JPH::TwoBodyConstraint*> constraints;
    JPH::BodyID chassis;
    JPH::BodyID wing_fl, wing_fr, wing_bl, wing_br;
    float hp = SHARD_INITIAL_HP;
    uint32_t envIndex;
    uint32_t robotIndex;
    
    bool IsValid() const { return !chassis.IsInvalid(); }
};

class ShardEnv {
public:
    ShardEnv() = default;
    ~ShardEnv() = default;

    void Init(uint32_t envIndex, JPH::PhysicsSystem* physicsSystem, int stepsPerEpisode);
    void Reset();
    void QueueActions(const float* actions1, const float* actions2);
    void HarvestState(float* obs1, float* obs2, float* reward1, float* reward2, bool& done);
    
    const ShardRobot& GetRobot1() const { return mRobot1; }
    const ShardRobot& GetRobot2() const { return mRobot2; }
    int GetStepCount() const { return mStepCount; }
    bool IsDone() const { return mDone; }

private:
    uint32_t mEnvIndex = 0;
    JPH::PhysicsSystem* mPhysicsSystem = nullptr;
    ShardRobot mRobot1;
    ShardRobot mRobot2;
    int mStepCount = 0;
    int mStepsPerEpisode = SHARD_MAX_STEPS;
    bool mDone = false;
    bool mInitialized = false;
    
    void BuildObservationVector(float* obs, const ShardRobot& robot, const ShardRobot& opponent);
    void CalculateRewards(float& r1, float& r2);
};
