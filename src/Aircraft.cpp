#include <Jolt/Jolt.h>
#include "Aircraft.h"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Constraints/FixedConstraint.h>
#include <algorithm>
#include <cmath>

void Aircraft::Create(JPH::PhysicsSystem* physicsSystem, JPH::RVec3 position, JPH::ObjectLayer layer) {
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();

    // Fuselage (Core)
    JPH::BoxShapeSettings fuselageShape(JPH::Vec3(1.0f, 1.0f, 8.0f));
    JPH::BodyCreationSettings fuselageSettings(fuselageShape.Create().Get(), position, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, layer);
    fuselageSettings.mMassPropertiesOverride.mMass = 10000.0f;
    fuselageSettings.mMassPropertiesOverride.mInertia = JPH::Mat44::sIdentity(); // Simple identity for now
    fuselageSettings.mLinearDamping = 0.01f;
    fuselageSettings.mAngularDamping = 0.05f;

    mMainBodyId = bodyInterface.CreateAndAddBody(fuselageSettings, JPH::EActivation::Activate);

    // Wings and Stabilizers
    struct PartDef {
        JPH::Vec3 relativePos;
        JPH::Vec3 halfExtents;
        float mass;
        float area;
        float liftCoef;
        int controlType;
    };

    std::vector<PartDef> parts = {
        // Left Wing
        { JPH::Vec3(-4.5f, 0.0f, 0.0f), JPH::Vec3(3.5f, 0.1f, 4.0f), 2000.0f, 28.0f, 4.0f, 2 },
        // Right Wing
        { JPH::Vec3(4.5f, 0.0f, 0.0f), JPH::Vec3(3.5f, 0.1f, 4.0f), 2000.0f, 28.0f, 4.0f, 2 },
        // Left Tail (Pitch)
        { JPH::Vec3(-2.5f, 0.0f, -7.0f), JPH::Vec3(2.0f, 0.1f, 2.0f), 1000.0f, 8.0f, 3.0f, 1 },
        // Right Tail (Pitch)
        { JPH::Vec3(2.5f, 0.0f, -7.0f), JPH::Vec3(2.0f, 0.1f, 2.0f), 1000.0f, 8.0f, 3.0f, 1 },
        // Left Vertical Stabilizer (Yaw)
        { JPH::Vec3(-1.5f, 2.0f, -7.0f), JPH::Vec3(0.1f, 2.0f, 2.0f), 500.0f, 8.0f, 2.0f, 3 },
        // Right Vertical Stabilizer (Yaw)
        { JPH::Vec3(1.5f, 2.0f, -7.0f), JPH::Vec3(0.1f, 2.0f, 2.0f), 500.0f, 8.0f, 2.0f, 3 }
    };

    for (const auto& def : parts) {
        JPH::BoxShapeSettings shape(def.halfExtents);
        JPH::RVec3 partPos = position + JPH::RVec3(def.relativePos);
        JPH::BodyCreationSettings settings(shape.Create().Get(), partPos, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, layer);
        settings.mMassPropertiesOverride.mMass = def.mass;
        
        JPH::BodyID partId = bodyInterface.CreateAndAddBody(settings, JPH::EActivation::Activate);
        
        // Fix to fuselage
        JPH::FixedConstraintSettings constraintSettings;
        constraintSettings.mPoint1 = partPos;
        constraintSettings.mPoint2 = partPos;
        physicsSystem->AddConstraint(bodyInterface.CreateConstraint(&constraintSettings, mMainBodyId, partId));

        Airfoil airfoil;
        airfoil.bodyId = partId;
        airfoil.relativePos = def.relativePos;
        airfoil.halfExtents = def.halfExtents;
        airfoil.area = def.area;
        airfoil.liftCoef = def.liftCoef;
        airfoil.dragCoef = 0.05f;
        airfoil.controlType = def.controlType;
        mAirfoils.push_back(airfoil);
    }
}

void Aircraft::ApplyAerodynamics(JPH::PhysicsSystem* physicsSystem, const float* actions) {
    if (mMainBodyId.IsInvalid()) return;
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();

    JPH::Quat rot = bodyInterface.GetRotation(mMainBodyId);
    
    // Thrust: forward is +Z
    JPH::Vec3 forward = rot * JPH::Vec3(0, 0, 1);
    float thrust = std::max(0.0f, actions[0]) * mThrustMax;
    bodyInterface.AddForce(mMainBodyId, forward * thrust);

    for (const auto& airfoil : mAirfoils) {
        if (airfoil.bodyId.IsInvalid()) continue;

        JPH::Vec3 vel = bodyInterface.GetLinearVelocity(airfoil.bodyId);
        float speedSq = vel.LengthSq();
        if (speedSq < 0.1f) continue;

        JPH::Quat partRot = bodyInterface.GetRotation(airfoil.bodyId);
        JPH::Vec3 partUp = partRot * JPH::Vec3(0, 1, 0);
        JPH::Vec3 velDir = vel.Normalized();

        // Angle of attack
        float aoa = std::asin(std::clamp(velDir.Dot(partUp), -1.0f, 1.0f));

        // Control surface deflection
        float deflection = 0.0f;
        if (airfoil.controlType == 1) deflection = actions[1]; // Pitch
        else if (airfoil.controlType == 2) deflection = actions[2] * (airfoil.relativePos.GetX() > 0 ? 1.0f : -1.0f); // Roll
        else if (airfoil.controlType == 3) deflection = actions[3]; // Yaw

        aoa += deflection * 0.3f; // Max ~17 degrees deflection

        float liftMag = 0.5f * mRho * speedSq * airfoil.area * airfoil.liftCoef * std::sin(aoa);
        float dragMag = 0.5f * mRho * speedSq * airfoil.area * (airfoil.dragCoef + std::abs(std::sin(aoa)) * 0.2f);

        JPH::Vec3 liftDir = partUp;
        JPH::Vec3 dragDir = -velDir;

        bodyInterface.AddForce(airfoil.bodyId, liftDir * liftMag + dragDir * dragMag);
    }
}
