/**
 * @file Robot.h
 * @brief Robot entity definition
 */

#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <vector>
#include <cstdint>
#include "../common/Types.h"

struct SatelliteJoint {
    JPH::BodyID coreBodyId;
    JPH::BodyID spikeBodyId;
    JPH::SixDOFConstraint* rotationJoint = nullptr;
    JPH::SliderConstraint* slideJoint = nullptr;
    float currentSlidePosition = 0.0f;
};

struct PIDController {
    float kp = 50.0f, ki = 0.0f, kd = 10.0f;
    float integral = 0.0f, prevError = 0.0f;
    float Compute(float setpoint, float current, float dt);
    void Reset();
};

struct RobotConfig {
    float coreRadius = 0.5f, coreMass = 13.0f, coreFriction = 0.5f, coreRestitution = 0.2f;
    
    struct Satellite {
        float offsetAngle = 0.0f, elevation = 0.0f, distance = 1.4f, radius = 0.1f, mass = 3.5f;
    };
    std::vector<Satellite> satellites;
    
    float spikeHalfHeight = 0.2f, spikeRadius = 0.02f, spikeMass = 0.5f;
    float jointDamping = 0.8f, motorTorque = 450.0f, slideMin = 0.0f, slideMax = 0.5f;
    int numLidarRays = 10;
    float lidarMaxDistance = 20.0f;
    int actionsPerSatellite = 4;
    float rotationScale = 25.0f, slideScale = 100.0f;
    
    int numSatellites = 0, actionsPerRobot = 0, observationDim = 256;
    
    static RobotConfig LoadFromJSON(const std::string& path);
    void CalculateDimensions();
};


class Robot {
public:
    Robot() = default;
    ~Robot() = default;
    Robot(const Robot&) = delete;
    Robot& operator=(const Robot&) = delete;
    Robot(Robot&&) = default;
    Robot& operator=(Robot&&) = default;
    
    JPH::BodyID GetMainBodyId() const { return mMainBodyId; }
    const std::vector<SatelliteJoint>& GetSatellites() const { return mSatellites; }
    float GetHP() const { return mHP; }
    uint32_t GetEnvIndex() const { return mEnvIndex; }
    int GetRobotIndex() const { return mRobotIndex; }
    RobotType GetType() const { return mType; }
    const RobotConfig& GetConfig() const { return mConfig; }
    
    void SetMainBodyId(JPH::BodyID id) { mMainBodyId = id; }
    void SetHP(float hp) { mHP = hp; }
    void SetEnvIndex(uint32_t idx) { mEnvIndex = idx; }
    void SetRobotIndex(int idx) { mRobotIndex = idx; }
    void SetType(RobotType type) { mType = type; }
    void SetConfig(const RobotConfig& cfg) { mConfig = cfg; }
    
    float GetTotalDamageDealt() const { return mTotalDamageDealt; }
    float GetTotalDamageTaken() const { return mTotalDamageTaken; }
    float GetTotalEnergyUsed() const { return mTotalEnergyUsed; }
    int GetEpisodeSteps() const { return mEpisodeSteps; }
    
    void AddDamageDealt(float d) { mTotalDamageDealt += d; }
    void AddDamageTaken(float d) { mTotalDamageTaken += d; }
    void AddEnergyUsed(float e) { mTotalEnergyUsed += e; }
    void IncrementEpisodeSteps() { mEpisodeSteps++; }
    void ResetEpisodeStats();
    
    float GetRotationScale() const { return mRotationScale; }
    float GetSlideScale() const { return mSlideScale; }
    void SetRotationScale(float s) { mRotationScale = s; }
    void SetSlideScale(float s) { mSlideScale = s; }
    
    std::vector<float>& GetObservationBuffer() { return mObservationBuffer; }
    std::vector<float>& GetLidarDistances() { return mLidarDistances; }
    void ResizeObservationBuffer(int dim);
    
    void AddSatellite(const SatelliteJoint& joint) { mSatellites.push_back(joint); }

private:
    RobotType mType = RobotType::SATELLITE;
    JPH::BodyID mMainBodyId;
    std::vector<SatelliteJoint> mSatellites;
    float mHP = INITIAL_HP;
    uint32_t mEnvIndex = 0;
    int mRobotIndex = 0;
    float mTotalDamageDealt = 0.0f, mTotalDamageTaken = 0.0f, mTotalEnergyUsed = 0.0f;
    int mEpisodeSteps = 0;
    float mRotationScale = 25.0f, mSlideScale = 100.0f;
    std::vector<float> mObservationBuffer, mLidarDistances;
    RobotConfig mConfig;
};
