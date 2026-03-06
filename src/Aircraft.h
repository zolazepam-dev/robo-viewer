#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <vector>
#include <string>

/**
 * @struct Airfoil
 * @brief Represents an aerodynamic surface (wing, stabilizer, etc.)
 */
struct Airfoil {
    JPH::BodyID bodyId;
    JPH::Vec3 relativePos;
    JPH::Vec3 halfExtents;
    float area;
    float liftCoef;
    float dragCoef;
    int controlType; // 0: none, 1: pitch, 2: roll, 3: yaw
};

/**
 * @class Aircraft
 * @brief Self-contained Jolt aircraft model with aerodynamics
 */
class Aircraft {
public:
    Aircraft() = default;
    ~Aircraft() = default;

    /**
     * @brief Create the aircraft in the physics system
     */
    void Create(JPH::PhysicsSystem* physicsSystem, JPH::RVec3 position, JPH::ObjectLayer layer);

    /**
     * @brief Apply aerodynamic forces based on current state and actions
     * @param actions [thrust (0-1), pitch (-1 to 1), roll (-1 to 1), yaw (-1 to 1)]
     */
    void ApplyAerodynamics(JPH::PhysicsSystem* physicsSystem, const float* actions);

    JPH::BodyID GetMainBodyId() const { return mMainBodyId; }
    float GetHP() const { return mHp; }
    void SetHP(float hp) { mHp = hp; }

private:
    JPH::BodyID mMainBodyId;
    std::vector<Airfoil> mAirfoils;
    float mThrustMax = 150000.0f;
    float mHp = 100.0f;
    
    // Constant for air density at sea level
    const float mRho = 1.225f;
};
