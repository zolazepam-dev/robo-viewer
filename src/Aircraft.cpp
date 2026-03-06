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
    JPH::BoxShapeSettings fuselageShapeSettings(JPH::Vec3(1.0f, 1.0f, 8.0f));
    JPH::RefConst<JPH::Shape> fuselageShape = fuselageShapeSettings.Create().Get();
    JPH::BodyCreationSettings fuselageSettings(fuselageShape, position, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, layer);
    
    JPH::MassProperties fuselageMass;
    fuselageMass.mMass = 10000.0f;
    fuselageMass.mInertia = fuselageShape->GetMassProperties().mInertia * (10000.0f / fuselageShape->GetMassProperties().mMass);
    fuselageSettings.mMassPropertiesOverride = fuselageMass;
    fuselageSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    
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
        { JPH::Vec3(-4.5f, 0.0f, 0.0f), JPH::Vec3(3.5f, 0.1f, 4.0f), 2000.0f, 28.0f, 1.0f, 2 },
        // Right Wing
        { JPH::Vec3(4.5f, 0.0f, 0.0f), JPH::Vec3(3.5f, 0.1f, 4.0f), 2000.0f, 28.0f, 1.0f, 2 },
        // Left Tail (Pitch)
        { JPH::Vec3(-2.5f, 0.0f, -7.0f), JPH::Vec3(2.0f, 0.1f, 2.0f), 1000.0f, 8.0f, 0.5f, 1 },
        // Right Tail (Pitch)
        { JPH::Vec3(2.5f, 0.0f, -7.0f), JPH::Vec3(2.0f, 0.1f, 2.0f), 1000.0f, 8.0f, 0.5f, 1 },
        // Left Vertical Stabilizer (Yaw)
        { JPH::Vec3(-1.5f, 2.0f, -7.0f), JPH::Vec3(0.1f, 2.0f, 2.0f), 500.0f, 8.0f, 0.5f, 3 },
        // Right Vertical Stabilizer (Yaw)
        { JPH::Vec3(1.5f, 2.0f, -7.0f), JPH::Vec3(0.1f, 2.0f, 2.0f), 500.0f, 8.0f, 0.5f, 3 }
    };

    for (const auto& def : parts) {
        JPH::BoxShapeSettings shapeSettings(def.halfExtents);
        JPH::RefConst<JPH::Shape> shape = shapeSettings.Create().Get();
        JPH::RVec3 partPos = position + JPH::RVec3(def.relativePos);
        JPH::BodyCreationSettings settings(shape, partPos, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, layer);
        
        JPH::MassProperties partMass;
        partMass.mMass = def.mass;
        partMass.mInertia = shape->GetMassProperties().mInertia * (def.mass / shape->GetMassProperties().mMass);
        settings.mMassPropertiesOverride = partMass;
        settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        
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

        // Simple lift model: CL = CL_alpha * sin(2 * alpha)
        float liftMag = 0.5f * mRho * speedSq * airfoil.area * airfoil.liftCoef * std::sin(2.0f * aoa);
        float dragMag = 0.5f * mRho * speedSq * airfoil.area * (airfoil.dragCoef + 0.01f + std::abs(std::sin(aoa)) * 0.5f);

        JPH::Vec3 liftDir = partUp;
        JPH::Vec3 dragDir = -velDir;

        JPH::Vec3 totalForce = liftDir * liftMag + dragDir * dragMag;
        
        // Clamp force to prevent explosions
        const float maxForce = 1000000.0f; 
        if (totalForce.LengthSq() > maxForce * maxForce) {
            totalForce = totalForce.Normalized() * maxForce;
        }

        bodyInterface.AddForce(airfoil.bodyId, totalForce);
    }
}
