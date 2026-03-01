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
constexpr int NUM_REACTION_WHEELS = 3;  // For gyroscopic control
constexpr int ACTIONS_PER_SATELLITE = 4;
constexpr int REACTION_WHEEL_DIM = 4;
constexpr int ACTIONS_PER_ROBOT = 56; // Matching VectorizedEnv: 6*4=24 for satellites, then reaction wheels and bursts at end
constexpr int NUM_LIDAR_RAYS = 10;


// Battery & Power Management System (32-byte aligned for AVX2)
constexpr float MAX_BATTERY_CAPACITY = 1000.0f;
constexpr float BATTERY_CHARGE_RATE = 15.0f;
constexpr float SHIELD_DRAIN_RATE = 25.0f;
constexpr float EMP_COST = 150.0f;
constexpr float SLOWMO_COST = 50.0f;
constexpr float SHIELD_MAX_HEALTH = 50.0f;
constexpr float EMP_DURATION = 2.0f;
constexpr float SLOWMO_DURATION = 3.0f;
constexpr float SLOWMO_FACTOR = 0.5f;

struct alignas(32) BatterySystem {
    // Group 1: State floats (8 floats = 32 bytes)
    float currentCharge = MAX_BATTERY_CAPACITY;
    float shieldHealth = SHIELD_MAX_HEALTH;
    float empTimer = 0.0f;
    float slowmoTimer = 0.0f;
    float totalEnergyGenerated = 0.0f;
    float totalEnergyConsumed = 0.0f;
    float reserved1 = 0.0f;
    float reserved2 = 0.0f;
    
    // Group 2: Boolean flags packed (8 bytes, padded to 32)
    uint8_t shieldActive = 0;
    uint8_t empActive = 0;
    uint8_t slowmoActive = 0;
    uint8_t reserved3 = 0;
    uint32_t reserved4 = 0;
    uint32_t reserved5 = 0;
    uint32_t reserved6 = 0;
    
    inline bool CanUseShield() const {
        return currentCharge >= SHIELD_DRAIN_RATE * 0.1f && shieldHealth > 0.0f;
    }
    
    inline bool CanUseEMP() const {
        return currentCharge >= EMP_COST && empActive == 0;
    }
    
    inline bool CanUseSlowmo() const {
        return currentCharge >= SLOWMO_COST && slowmoActive == 0;
    }
    
    inline void Reset() {
        currentCharge = MAX_BATTERY_CAPACITY;
        shieldHealth = SHIELD_MAX_HEALTH;
        empTimer = 0.0f;
        slowmoTimer = 0.0f;
        totalEnergyGenerated = 0.0f;
        totalEnergyConsumed = 0.0f;
        shieldActive = 0;
        empActive = 0;
        slowmoActive = 0;
    }
};


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


struct ReactionWheelState {
    float rpm = 0.0f;      // Current RPM
    float targetTorque = 0.0f;  // Last applied torque
    
    ReactionWheelState() : rpm(0.0f), targetTorque(0.0f) {}
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
    std::array<ReactionWheelState, NUM_REACTION_WHEELS> reactionWheels;  // Zero-init by default constructor  // Gyroscopic control
    float hp = 100.0f;

    uint32_t envIndex = 0;
    int robotIndex = 0;
    uint32_t collisionGroup = 0;

    float totalDamageDealt = 0.0f;
    float totalDamageTaken = 0.0f;
    float totalEnergyUsed = 0.0f;
    int episodeSteps = 0;

    ResidualActionScale actionScale;
    BatterySystem battery;

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
