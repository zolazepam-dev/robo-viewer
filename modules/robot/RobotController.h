/**
 * @file RobotController.h
 * @brief Robot control system
 */

#pragma once

#include "Robot.h"
#include "../physics/PhysicsWorld.h"
#include <vector>

class RobotController {
public:
    RobotController() = default;
    ~RobotController() = default;
    
    void ApplyActions(Robot& robot, const float* actions, PhysicsWorld& physicsWorld, float dt);
    void ApplyResidualActions(Robot& robot, const float* residualActions, PhysicsWorld& physicsWorld, float dt);
    void ComputeBasePIDActions(Robot& robot, PhysicsWorld& physicsWorld, float dt);
    void GetObservations(const Robot& robot, const Robot& opponent, const ForceSensorReading& forces, PhysicsWorld& physicsWorld, float* outObservations);
    void PerformLidarScan(Robot& robot, PhysicsWorld& physicsWorld);

private:
    void BlendResidualWithBase(Robot& robot, const float* residualActions, float* outFinalActions);
    static std::vector<JPH::Vec3> CreateLidarDirections(int numRays);
};
