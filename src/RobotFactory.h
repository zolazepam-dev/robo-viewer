#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>
#include "Robot.h"
#include "RobotConfig.h"

/**
 * @class RobotFactory
 * @brief Handles instantiation of robot bodies and constraints in Jolt
 */
class RobotFactory {
public:
    static JPH::Ref<JPH::GroupFilterTable> mGroupFilter;

    /**
     * @brief Create a robot in the physics system
     */
    static Robot CreateRobot(
        const RobotConfig& config,
        JPH::PhysicsSystem* physicsSystem,
        const JPH::RVec3& position,
        uint32_t envIndex,
        int robotIndex
    );

    /**
     * @brief Reset an existing robot's physical components to spawn position
     */
    static void ResetRobot(
        Robot& robot,
        JPH::PhysicsSystem* physicsSystem,
        const JPH::RVec3& spawnPosition
    );

private:
    static void InitializeGroupFilter();
};
