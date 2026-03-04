/**
 * @file CombatRobot.h
 * @brief Combat robot definition and control system
 * 
 * This file contains the CombatRobotData structure and CombatRobotLoader class,
 * which define the physical structure of the combat robots and provide methods
 * for loading, resetting, controlling, and observing them.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <cmath>

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>

/** Number of satellites per robot */
constexpr int NUM_SATELLITES = 6;
/** Number of actions per satellite */
constexpr int ACTIONS_PER_SATELLITE = 4;
/** Dimension of reaction wheel control */
constexpr int REACTION_WHEEL_DIM = 4;
/** Total number of actions per robot */
constexpr int ACTIONS_PER_ROBOT = 56; // Matching VectorizedEnv: 6*4=24 for satellites, then reaction wheels and bursts at end
/** Number of LIDAR rays */
constexpr int NUM_LIDAR_RAYS = 10;
/** Observation vector dimension */
constexpr int OBSERVATION_DIM = 256; // Expanded for all sensors + padding

/**
 * @struct ForceSensorReading
 * @brief Contains force and joint stress measurements from the robot
 */
struct ForceSensorReading
{
    float impulseMagnitude[NUM_SATELLITES] = {0.0f}; ///< Impulse magnitude per satellite
    float jointStress[NUM_SATELLITES] = {0.0f}; ///< Joint stress per satellite

    /** @brief Reset all sensor readings to zero */
    void Reset()
    {
        for (int i = 0; i < NUM_SATELLITES; ++i)
        {
            impulseMagnitude[i] = 0.0f;
            jointStress[i] = 0.0f;
        }
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
     * @param setpoint Desired value
     * @param current Current value
     * @param dt Time step in seconds
     * @return Control output
     */
    float Compute(float setpoint, float current, float dt)
    {
        float error = setpoint - current;
        integral += error * dt;
        float derivative = (error - prevError) / dt;
        prevError = error;
        return kp * error + ki * integral + kd * derivative;
    }

    /** @brief Reset PID controller state */
    void Reset()
    {
        integral = 0.0f;
        prevError = 0.0f;
    }
};

/**
 * @struct SatelliteData
 * @brief Contains data for a single satellite
 */
struct SatelliteData
{
    JPH::BodyID coreBodyId; ///< Core body ID of the satellite
    JPH::BodyID spikeBodyId; ///< Spike body ID of the satellite
    JPH::SixDOFConstraint* rotationJoint = nullptr; ///< Rotation joint
    JPH::SliderConstraint* slideJoint = nullptr; ///< Slide joint
    PIDController pidX; ///< PID controller for X-axis rotation
    PIDController pidY; ///< PID controller for Y-axis rotation
    PIDController pidZ; ///< PID controller for Z-axis rotation
    
    float currentSlidePosition = 0.0f; ///< Current slide position
    float currentAngularVelX = 0.0f; ///< Current X-axis angular velocity
    float currentAngularVelY = 0.0f; ///< Current Y-axis angular velocity
    float currentAngularVelZ = 0.0f; ///< Current Z-axis angular velocity
};

/**
 * @struct ResidualActionScale
 * @brief Scale factors for residual actions
 */
struct ResidualActionScale
{
    float rotationScale = 25.0f; ///< Rotation action scale factor
    float slideScale = 100.0f; ///< Slide action scale factor
};

/**
 * @enum RobotType
 * @brief Type of combat robot
 */
enum class RobotType {
    SATELLITE, ///< Satellite-based robot design
    INTERNAL_ENGINE ///< Internal engine robot design
};

/**
 * @struct CombatRobotData
 * @brief Contains all data for a single combat robot
 */
struct CombatRobotData
{
    RobotType type = RobotType::SATELLITE; ///< Robot type
    JPH::BodyID mainBodyId; ///< Main body ID
    std::array<SatelliteData, NUM_SATELLITES> satellites; ///< Satellite data
    float hp = 100.0f; ///< Current health points

    uint32_t envIndex = 0; ///< Environment index
    int robotIndex = 0; ///< Robot index within environment
    uint32_t collisionGroup = 0; ///< Collision group

    float totalDamageDealt = 0.0f; ///< Total damage dealt in current episode
    float totalDamageTaken = 0.0f; ///< Total damage taken in current episode
    float totalEnergyUsed = 0.0f; ///< Total energy used in current episode
    int episodeSteps = 0; ///< Number of steps taken in current episode

    ResidualActionScale actionScale; ///< Action scaling factors

    std::array<float, ACTIONS_PER_ROBOT> baseActions{}; ///< Base PID control actions
    std::array<float, ACTIONS_PER_ROBOT> residualActions{}; ///< RL residual actions
    std::array<float, ACTIONS_PER_ROBOT> finalActions{}; ///< Final blended actions
    
    float observationBuffer[OBSERVATION_DIM]; ///< Observation buffer
    float lidarDistances[NUM_LIDAR_RAYS]; ///< LIDAR distance readings
};

/**
 * @class CombatRobotLoader
 * @brief Loads and controls combat robots
 * 
 * Provides methods for loading robots from configuration files, resetting them,
 * applying actions, and getting observations.
 */
class CombatRobotLoader
{
public:
    /**
     * @brief Load a robot from a configuration file
     * @param configPath Path to robot configuration file
     * @param physicsSystem Pointer to physics system
     * @param position Spawn position
     * @param envIndex Environment index
     * @param robotIndex Robot index within environment
     * @return Loaded CombatRobotData
     */
    CombatRobotData LoadRobot(
        const std::string& configPath,
        JPH::PhysicsSystem* physicsSystem,
        const JPH::RVec3& position,
        uint32_t envIndex,
        int robotIndex
    );

    /**
     * @brief Reset a robot to initial state
     * @param robot Robot data to reset
     * @param physicsSystem Pointer to physics system
     * @param spawnPosition Spawn position
     */
    void ResetRobot(
        CombatRobotData& robot,
        JPH::PhysicsSystem* physicsSystem,
        const JPH::RVec3& spawnPosition
    );

    /**
     * @brief Apply actions to a robot
     * @param robot Robot data
     * @param actions Action vector
     * @param physicsSystem Pointer to physics system
     */
    void ApplyActions(
        CombatRobotData& robot,
        const float* actions,
        JPH::PhysicsSystem* physicsSystem
    );

    /**
     * @brief Apply residual RL actions with PID base control
     * @param robot Robot data
     * @param residualActions Residual action vector from RL model
     * @param physicsSystem Pointer to physics system
     */
    void ApplyResidualActions(
        CombatRobotData& robot,
        const float* residualActions,
        JPH::PhysicsSystem* physicsSystem
    );

    /**
     * @brief Compute base PID control actions
     * @param robot Robot data
     * @param physicsSystem Pointer to physics system
     * @param dt Time step in seconds
     */
    void ComputeBasePIDActions(
        CombatRobotData& robot,
        JPH::PhysicsSystem* physicsSystem,
        float dt
    );

    /**
     * @brief Get observations from a robot
     * @param robot Robot data
     * @param opponent Opponent robot data
     * @param observations Output observation buffer
     * @param forces Force sensor readings
     * @param physicsSystem Pointer to physics system
     */
    void GetObservations(
        CombatRobotData& robot,
        const CombatRobotData& opponent,
        float* observations,
        const ForceSensorReading& forces,
        JPH::PhysicsSystem* physicsSystem
    );
    
    /**
     * @brief Perform LIDAR scan
     * @param robot Robot data
     * @param physicsSystem Pointer to physics system
     */
    void PerformLidarScan(
        CombatRobotData& robot,
        JPH::PhysicsSystem* physicsSystem
    );

private:
    /**
     * @brief Blend residual actions with base PID control
     * @param robot Robot data
     */
    void BlendResidualWithBase(CombatRobotData& robot);
    
    static const JPH::Vec3 mLidarDirections[NUM_LIDAR_RAYS]; ///< LIDAR ray directions
    static JPH::Ref<JPH::GroupFilterTable> mGroupFilter; ///< Collision group filter
};
