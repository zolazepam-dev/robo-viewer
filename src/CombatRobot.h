#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>

#include <string>
#include <vector>
#include <array>
#include <cmath>

constexpr int NUM_SATELLITES = 6;
constexpr int ACTIONS_PER_SATELLITE = 4;
constexpr int REACTION_WHEEL_DIM = 4;
constexpr int ACTIONS_PER_ROBOT = 56; // Matching VectorizedEnv: 6*4=24 for satellites, then reaction wheels and bursts at end
constexpr int NUM_LIDAR_RAYS = 10;
constexpr int OBSERVATION_DIM = 256; // Expanded for all sensors + padding

struct ForceSensorReading
{
    float impulseMagnitude[NUM_SATELLITES] = {0.0f};
    float jointStress[NUM_SATELLITES] = {0.0f};

    void Reset()
    {
        for (int i = 0; i < NUM_SATELLITES; ++i)
        {
            impulseMagnitude[i] = 0.0f;
            jointStress[i] = 0.0f;
        }
    }
};

struct PIDController
{
    float kp = 50.0f;
    float ki = 0.0f;
    float kd = 10.0f;
    float integral = 0.0f;
    float prevError = 0.0f;

    float Compute(float setpoint, float current, float dt)
    {
        float error = setpoint - current;
        integral += error * dt;
        float derivative = (error - prevError) / dt;
        prevError = error;
        return kp * error + ki * integral + kd * derivative;
    }

    void Reset()
    {
        integral = 0.0f;
        prevError = 0.0f;
    }
};

struct SatelliteData
{
    JPH::BodyID coreBodyId;
    JPH::BodyID spikeBodyId;
    JPH::SixDOFConstraint* rotationJoint = nullptr;
    JPH::SliderConstraint* slideJoint = nullptr;
    PIDController pidX;
    PIDController pidY;
    PIDController pidZ;
    
    float currentSlidePosition = 0.0f;
    float currentAngularVelX = 0.0f;
    float currentAngularVelY = 0.0f;
    float currentAngularVelZ = 0.0f;
};

struct ResidualActionScale
{
    float rotationScale = 25.0f;
    float slideScale = 100.0f;
};

enum class RobotType {
    SATELLITE,
    INTERNAL_ENGINE
};

struct CombatRobotData
{
    RobotType type = RobotType::SATELLITE;
    JPH::BodyID mainBodyId;
    std::array<SatelliteData, NUM_SATELLITES> satellites;
    float hp = 100.0f;

    uint32_t envIndex = 0;
    int robotIndex = 0;
    uint32_t collisionGroup = 0;

    float totalDamageDealt = 0.0f;
    float totalDamageTaken = 0.0f;
    float totalEnergyUsed = 0.0f;
    int episodeSteps = 0;

    ResidualActionScale actionScale;

    std::array<float, ACTIONS_PER_ROBOT> baseActions{};
    std::array<float, ACTIONS_PER_ROBOT> residualActions{};
    std::array<float, ACTIONS_PER_ROBOT> finalActions{};
    
    float observationBuffer[OBSERVATION_DIM];
    float lidarDistances[NUM_LIDAR_RAYS];
};

class CombatRobotLoader
{
public:
    CombatRobotData LoadRobot(
        const std::string& configPath,
        JPH::PhysicsSystem* physicsSystem,
        const JPH::RVec3& position,
        uint32_t envIndex,
        int robotIndex
    );

    void ResetRobot(
        CombatRobotData& robot,
        JPH::PhysicsSystem* physicsSystem,
        const JPH::RVec3& spawnPosition
    );

    void ApplyActions(
        CombatRobotData& robot,
        const float* actions,
        JPH::PhysicsSystem* physicsSystem
    );

    void ApplyResidualActions(
        CombatRobotData& robot,
        const float* residualActions,
        JPH::PhysicsSystem* physicsSystem
    );

    void ComputeBasePIDActions(
        CombatRobotData& robot,
        JPH::PhysicsSystem* physicsSystem,
        float dt
    );

    void GetObservations(
        CombatRobotData& robot,
        const CombatRobotData& opponent,
        float* observations,
        const ForceSensorReading& forces,
        JPH::PhysicsSystem* physicsSystem
    );
    
    void PerformLidarScan(
        CombatRobotData& robot,
        JPH::PhysicsSystem* physicsSystem
    );

private:
    void BlendResidualWithBase(CombatRobotData& robot);
    
    static const JPH::Vec3 mLidarDirections[NUM_LIDAR_RAYS];
    static JPH::Ref<JPH::GroupFilterTable> mGroupFilter;
};
