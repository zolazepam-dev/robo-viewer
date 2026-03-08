/**
 * @file CombatRobot.h
 * @brief Legacy definitions and structures for combat robots
 */

#pragma once

#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>
#include "RobotConfig.h"
#include "Robot.h"

#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>

// For JSON parsing
using json = nlohmann::json;

/**
 * @struct ForceSensorReading
 * @brief Contains force and joint stress measurements from the robot
 */
struct ForceSensorReading
{
    std::vector<float> impulseMagnitude; ///< Impulse magnitude per satellite
    std::vector<float> jointStress; ///< Joint stress per satellite

    /** @brief Reset all sensor readings to zero */
    void Reset(int numSatellites)
    {
        int size = (numSatellites > 0) ? numSatellites : 1;
        impulseMagnitude.assign(size, 0.0f);
        jointStress.assign(size, 0.0f);
    }
};

/**
 * @struct PIDController
 * @brief Simple PID controller implementation
 */
struct PIDController
{
    float kp = 50.0f; ///< Proportional gain
    float ki = 0.0f; ///< Integral gain
    float kd = 10.0f; ///< Derivative gain
    float integral = 0.0f; ///< Integral term
    float prevError = 0.0f; ///< Previous error for derivative calculation

    /**
     * @brief Compute PID control output
     */
    float Compute(float setpoint, float current, float dt)
    {
        float error = setpoint - current;
        integral += error * dt;
        float derivative = (dt > 0.0f) ? (error - prevError) / dt : 0.0f;
        prevError = error;
        return kp * error + ki * integral + kd * derivative;
    }

    void Reset()
    {
        integral = 0.0f;
        prevError = 0.0f;
    }
};

/**
 * @struct CombatRobotData
 * @brief Legacy robot data structure (to be phased out)
 */
struct CombatRobotData
{
    RobotType type = RobotType::SATELLITE;
    JPH::BodyID mainBodyId;
    std::vector<SatelliteData> satellites;
    float hp = 100.0f;

    uint32_t envIndex = 0;
    int robotIndex = 0;
    uint32_t collisionGroup = 0;

    float totalDamageDealt = 0.0f;
    float totalDamageTaken = 0.0f;
    float totalEnergyUsed = 0.0f;
    int episodeSteps = 0;

    ResidualActionScale actionScale;

    std::vector<float> baseActions;
    std::vector<float> residualActions;
    std::vector<float> finalActions;
    
    std::vector<float> observationBuffer;
    std::vector<float> lidarDistances;

    RobotConfig config;
};

/**
 * @class CombatRobotLoader
 * @brief Legacy loader (to be phased out)
 */
class CombatRobotLoader
{
public:
    CombatRobotData LoadRobot(const std::string& configPath, JPH::PhysicsSystem* ps, const JPH::RVec3& pos, uint32_t env, int idx);
    void ResetRobot(CombatRobotData& robot, JPH::PhysicsSystem* ps, const JPH::RVec3& pos);
    void ApplyResidualActions(CombatRobotData& robot, const float* actions, JPH::PhysicsSystem* ps);
    void ApplyActions(CombatRobotData& robot, const float* actions, JPH::PhysicsSystem* ps);
    void ComputeBasePIDActions(CombatRobotData& robot, JPH::PhysicsSystem* ps, float dt);
    void PerformLidarScan(CombatRobotData& robot, JPH::PhysicsSystem* ps);
    void GetObservations(CombatRobotData& robot, const CombatRobotData& opp, float* obs, const ForceSensorReading& forces, JPH::PhysicsSystem* ps);
    
private:
    void BlendResidualWithBase(CombatRobotData& robot);
    static std::vector<JPH::Vec3> CreateLidarDirections(int numRays);
    static JPH::Ref<JPH::GroupFilterTable> mGroupFilter;
};
