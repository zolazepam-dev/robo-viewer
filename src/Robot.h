#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <vector>
#include "RobotConfig.h"

/**
 * @struct SatelliteData
 * @brief Physical handles and state for a single satellite
 */
struct SatelliteData {
    JPH::BodyID coreBodyId;
    JPH::BodyID spikeBodyId;
    JPH::SixDOFConstraint* rotationJoint = nullptr;
    JPH::SliderConstraint* slideJoint = nullptr;
    
    // Legacy state for transition
    float currentSlidePosition = 0.0f;
    float currentAngularVelX = 0.0f;
    float currentAngularVelY = 0.0f;
    float currentAngularVelZ = 0.0f;
    
    // Gyroscopic control state
    JPH::Vec3 gyroAngularVelocity = JPH::Vec3::sZero();  // Internal gyro wheel speed (rad/s)
    float gyroInertia = 0.1f;  // Moment of inertia of gyro wheel (kg·m²)
    
    // Controller state moved to RobotController, but kept here for legacy CombatRobotData compatibility
    struct PIDState {
        float integral = 0.0f;
        float prevError = 0.0f;
        float kp = 50.0f;
        float ki = 0.0f;
        float kd = 10.0f;

        float Compute(float setpoint, float current, float dt) {
            float error = setpoint - current;
            integral += error * dt;
            float derivative = (dt > 0.0f) ? (error - prevError) / dt : 0.0f;
            prevError = error;
            return kp * error + ki * integral + kd * derivative;
        }

        void Reset() {
            integral = 0.0f;
            prevError = 0.0f;
        }
    };
    PIDState pidX, pidY, pidZ, pidSlide;
};

/**
 * @struct ResidualActionScale
 * @brief Scale factors for residual actions
 */
struct ResidualActionScale
{
    float rotationScale = 25.0f;
    float slideScale = 100.0f;
};

/**
 * @class Robot
 * @brief Passive entity representing the robot in the physics world
 */
class Robot {
public:
    RobotType type = RobotType::SATELLITE;
    JPH::BodyID mainBodyId;
    std::vector<SatelliteData> satellites;
    
    // Multi-body support
    std::vector<JPH::BodyID> bodyIds;
    std::vector<JPH::HingeConstraint*> hingeJoints;
    std::vector<JPH::SixDOFConstraint*> sixDofJoints;
    
    ResidualActionScale actionScale;

    // Physical State
    float hp = 100.0f;
    float totalDamageDealt = 0.0f;
    float totalDamageTaken = 0.0f;
    float totalEnergyUsed = 0.0f;
    int episodeSteps = 0;

    // References
    RobotConfig config;
    uint32_t envIndex = 0;
    int robotIndex = 0;
    uint32_t collisionGroup = 0;

    bool IsValid() const { return !mainBodyId.IsInvalid(); }
};
