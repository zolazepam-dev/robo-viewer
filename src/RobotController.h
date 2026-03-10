#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include "Robot.h"
#include "CombatRobot.h" // For ForceSensorReading and PIDController temporarily

/**
 * @class RobotController
 * @brief Handles control logic, actions, and observations for a Robot entity
 */
class RobotController {
public:
    explicit RobotController(Robot& robot);

    /**
     * @brief Apply raw model actions (residual + PID stabilization)
     */
    void ApplyResidualActions(const float* residualActions, JPH::PhysicsSystem* physicsSystem, float dt);

    /**
     * @brief Build the complete observation vector
     */
    void GetObservations(const Robot& opponent, float* observations, const ForceSensorReading& forces, JPH::PhysicsSystem* physicsSystem);

    /**
     * @brief Internal LIDAR scan
     */
    void PerformLidarScan(JPH::PhysicsSystem* physicsSystem);

    std::vector<float>& GetLidarDistances() { return mLidarDistances; }

private:
    Robot& mRobot;
    
    // Control State
    std::vector<PIDController> mPidX;
    std::vector<PIDController> mPidY;
    std::vector<PIDController> mPidZ;
    
    // Buffers
    std::vector<float> mBaseActions;
    std::vector<float> mFinalActions;
    std::vector<float> mLidarDistances;
    std::vector<float> mResidualActions;

    void ComputeBasePIDActions(JPH::PhysicsSystem* physicsSystem, float dt);
    void ApplyPhysicsActions(const float* actions, JPH::PhysicsSystem* physicsSystem);
    void ApplyAircraftAerodynamics(const float* actions, JPH::PhysicsSystem* physicsSystem, float dt);
    void ApplyShardPhysics(const float* actions, JPH::PhysicsSystem* physicsSystem, float dt);
    static std::vector<JPH::Vec3> CreateLidarDirections(int numRays);
};
