#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <vector>
#include <string>

/**
 * @struct Airfoil Section
 * @brief Definition of an aerodynamic surface for force calculation
 */
struct AirfoilSection {
    std::string name;
    JPH::Vec3 relativePos;
    JPH::Quat relativeRot;
    float area;
    float liftCoef;
    float dragCoef;
    int controlType; // 1: Pitch, 2: Roll, 3: Yaw
};

/**
 * @class Aircraft
 * @brief Unified Jolt aircraft model using a single compound body for maximum stability
 */
class Aircraft {
public:
    Aircraft() = default;
    ~Aircraft() = default;

    /**
     * @brief Create the aircraft as a single unified body
     */
    void Create(JPH::PhysicsSystem* physicsSystem, JPH::RVec3 position, JPH::ObjectLayer layer);

    /**
     * @brief Apply stable aerodynamic forces to the unified body
     */
    void ApplyAerodynamics(JPH::PhysicsSystem* physicsSystem, const float* actions, float deltaTime);

    JPH::BodyID GetMainBodyId() const { return mMainBodyId; }

private:
    JPH::BodyID mMainBodyId;
    std::vector<AirfoilSection> mSections;
    float mThrustMax = 350000.0f;
    const float mRho = 1.225f;
};
