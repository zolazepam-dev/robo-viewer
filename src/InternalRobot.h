#pragma once

#include "CombatRobot.h"

class InternalRobotLoader {
public:
    static CombatRobotData LoadInternalRobot(
        const std::string& configPath,
        JPH::PhysicsSystem* physicsSystem,
        const JPH::RVec3& position,
        uint32_t envIndex,
        int robotIndex
    );

    static void ApplyInternalActions(
        CombatRobotData& robot,
        const float* actions,
        JPH::PhysicsSystem* physicsSystem
    );
};
