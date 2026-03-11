/**
 * @file RobotController.cpp
 * @brief Implementation of RobotController class
 */

#include "RobotController.h"
#include "../physics/PhysicsWorld.h"
#include <Jolt/Physics/Body/BodyInterface.h>
#include <cmath>

void RobotController::ApplyActions(
    Robot& robot,
    const float* actions,
    PhysicsWorld& physicsWorld,
    float dt
) {
    auto& system = physicsWorld.GetSystem();
    JPH::BodyInterface& bodyInterface = system.GetBodyInterface();
    
    const auto& config = robot.GetConfig();
    const auto& satellites = robot.GetSatellites();
    
    for (size_t i = 0; i < satellites.size() && i * 4 < static_cast<size_t>(config.actionsPerRobot); ++i) {
        // Actions: [rotX, rotY, rotZ, slide] per satellite
        float rotX = actions[i * 4 + 0] * config.rotationScale;
        float rotY = actions[i * 4 + 1] * config.rotationScale;
        float rotZ = actions[i * 4 + 2] * config.rotationScale;
        float slide = actions[i * 4 + 3] * config.slideScale;
        
        // Apply torques and forces (simplified - would need full implementation)
        // In full implementation, would apply to joints
    }
}

void RobotController::ApplyResidualActions(
    Robot& robot,
    const float* residualActions,
    PhysicsWorld& physicsWorld,
    float dt
) {
    // Compute base PID actions first
    ComputeBasePIDActions(robot, physicsWorld, dt);
    
    // Then blend with residual RL actions
    // (Full implementation would blend here)
    
    ApplyActions(robot, residualActions, physicsWorld, dt);
}

void RobotController::ComputeBasePIDActions(
    Robot& robot,
    PhysicsWorld& physicsWorld,
    float dt
) {
    // Base PID control for stabilization
    // Full implementation would compute target angles and apply PID
}

void RobotController::GetObservations(
    const Robot& robot,
    const Robot& opponent,
    const ForceSensorReading& forces,
    PhysicsWorld& physicsWorld,
    float* outObservations
) {
    auto& system = physicsWorld.GetSystem();
    const JPH::BodyInterface& bodyInterface = system.GetBodyInterface();
    
    int idx = 0;
    
    // Core state (position, velocity, orientation)
    if (!robot.GetMainBodyId().IsInvalid()) {
        JPH::RVec3 pos = bodyInterface.GetPosition(robot.GetMainBodyId());
        JPH::Vec3 linVel = bodyInterface.GetLinearVelocity(robot.GetMainBodyId());
        JPH::Vec3 angVel = bodyInterface.GetAngularVelocity(robot.GetMainBodyId());
        JPH::Quat rot = bodyInterface.GetRotation(robot.GetMainBodyId());
        
        outObservations[idx++] = static_cast<float>(pos.GetX());
        outObservations[idx++] = static_cast<float>(pos.GetY());
        outObservations[idx++] = static_cast<float>(pos.GetZ());
        outObservations[idx++] = linVel.GetX();
        outObservations[idx++] = linVel.GetY();
        outObservations[idx++] = linVel.GetZ();
        outObservations[idx++] = angVel.GetX();
        outObservations[idx++] = angVel.GetY();
        outObservations[idx++] = angVel.GetZ();
        // Add orientation quaternion...
    }
    
    // Opponent relative state
    if (!opponent.GetMainBodyId().IsInvalid()) {
        JPH::RVec3 oppPos = bodyInterface.GetPosition(opponent.GetMainBodyId());
        JPH::RVec3 myPos = bodyInterface.GetPosition(robot.GetMainBodyId());
        JPH::Vec3 relPos = oppPos - myPos;
        
        outObservations[idx++] = static_cast<float>(relPos.GetX());
        outObservations[idx++] = static_cast<float>(relPos.GetY());
        outObservations[idx++] = static_cast<float>(relPos.GetZ());
    }
    
    // Force sensor readings
    for (float impulse : forces.impulseMagnitude) {
        outObservations[idx++] = impulse;
    }
    
    // HP
    outObservations[idx++] = robot.GetHP() / INITIAL_HP;
    outObservations[idx++] = opponent.GetHP() / INITIAL_HP;
    
    // Fill remaining with zeros if needed
    while (idx < robot.GetConfig().observationDim) {
        outObservations[idx++] = 0.0f;
    }
}

void RobotController::PerformLidarScan(
    Robot& robot,
    PhysicsWorld& physicsWorld
) {
    // LIDAR implementation would raycast from robot
    // Full implementation needed
}

void RobotController::BlendResidualWithBase(
    Robot& robot,
    const float* residualActions,
    float* outFinalActions
) {
    // Blend PID base actions with RL residual
    // Full implementation needed
}

std::vector<JPH::Vec3> RobotController::CreateLidarDirections(int numRays) {
    std::vector<JPH::Vec3> directions;
    directions.reserve(numRays);
    
    for (int i = 0; i < numRays; ++i) {
        float angle = 2.0f * 3.14159f * static_cast<float>(i) / static_cast<float>(numRays);
        directions.emplace_back(std::cos(angle), 0.0f, std::sin(angle));
    }
    
    return directions;
}
