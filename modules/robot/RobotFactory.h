/**
 * @file RobotFactory.h
 * @brief Factory for creating robot bodies in physics world
 * 
 * Uses RobotConfig and PhysicsWorld to create robot bodies and constraints.
 * Returns fully constructed Robot objects.
 */

#pragma once

#include "Robot.h"
#include <memory>

class PhysicsWorld;

/**
 * @brief Robot factory
 * 
 * Creates robot bodies and constraints in the physics world.
 */
class RobotFactory {
public:
    RobotFactory() = default;
    ~RobotFactory() = default;
    
    /**
     * @brief Create a robot in the physics world
     * @param config Robot configuration
     * @param physicsWorld Physics world to create bodies in
     * @param position Spawn position
     * @param envIndex Environment index for collision layers
     * @param robotIndex Robot index (0 or 1) within environment
     * @return Fully constructed Robot
     */
    Robot CreateRobot(
        const RobotConfig& config,
        PhysicsWorld& physicsWorld,
        const JPH::RVec3& position,
        uint32_t envIndex,
        int robotIndex
    );
    
    /**
     * @brief Reset a robot to initial state
     * @param robot Robot to reset
     * @param physicsWorld Physics world
     * @param spawnPosition Spawn position
     */
    void ResetRobot(
        Robot& robot,
        PhysicsWorld& physicsWorld,
        const JPH::RVec3& spawnPosition
    );

private:
    JPH::BodyID CreateCoreBody(
        PhysicsWorld& physicsWorld,
        const RobotConfig& config,
        const JPH::RVec3& position,
        uint16_t objectLayer
    );
    
    SatelliteJoint CreateSatellite(
        PhysicsWorld& physicsWorld,
        const RobotConfig& config,
        const RobotConfig::Satellite& satConfig,
        JPH::BodyID coreBodyId,
        const JPH::RVec3& corePosition,
        uint16_t objectLayer
    );
};
