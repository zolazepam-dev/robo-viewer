#pragma once

// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Constraints/TwoBodyConstraint.h>

#include <string>
#include <vector>
#include <array>

// Constants for combat system
constexpr int NUM_SATELLITES = 4;
constexpr int ACTIONS_PER_SATELLITE = 4; // yaw, pitch, roll, slide
constexpr int ACTIONS_PER_ROBOT = NUM_SATELLITES * ACTIONS_PER_SATELLITE; // 16

struct SatelliteData
{
    JPH::BodyID coreBodyId;
    JPH::BodyID spikeBodyId;
    JPH::TwoBodyConstraint* yawJoint = nullptr;
    JPH::TwoBodyConstraint* pitchJoint = nullptr;
    JPH::TwoBodyConstraint* rollJoint = nullptr;
    JPH::TwoBodyConstraint* slideJoint = nullptr;
};

struct CombatRobotData
{
    JPH::BodyID mainBodyId;
    std::array<SatelliteData, NUM_SATELLITES> satellites;
    float hp = 100.0f;

    // Identification for Dimensional Ghosting
    uint32_t envIndex = 0;
    int robotIndex = 0;

    // State tracking
    float totalDamageDealt = 0.0f;
    float totalDamageTaken = 0.0f;
};

class CombatRobotLoader
{
public:
    // Loads the robot exactly ONCE at startup and assigns its dimensional layer
    CombatRobotData LoadRobot(
        const std::string& configPath,
        JPH::PhysicsSystem* physicsSystem,
        const JPH::RVec3& position,
        uint32_t envIndex,
        int robotIndex
    );

    // The Necromancer Reset: Teleports the robot and zeroes velocity (Zero allocations)
    void ResetRobot(
        CombatRobotData& robot,
        JPH::PhysicsSystem* physicsSystem,        const JPH::RVec3& spawnPosition
    );

    void ApplyActions(
        CombatRobotData& robot,
        const float* actions,
        JPH::PhysicsSystem* physicsSystem
    );

    // Writes directly to the pre-allocated tensor memory block
    void GetObservations(
        const CombatRobotData& robot,
        const CombatRobotData& opponent,
        float* observations,
        JPH::PhysicsSystem* physicsSystem
    );
};