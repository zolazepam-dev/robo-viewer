#include <Jolt/Jolt.h>
#include "Aircraft.h"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace JPH;

void Aircraft::Create(PhysicsSystem* physicsSystem, RVec3 position, ObjectLayer layer) {
    std::cout << "[Aircraft] Creating Basic Jet (Literal Snippet)" << std::endl;
    BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    mSections.clear();

    // --- YOUR SNIPPET LOGIC START ---
    // --- Define Core Dimensions ---
    float fuselageLength = 4.0f;
    float fuselageRadius = 0.6f;
    float wingspan = 3.5f;
    float wingChord = 1.2f;
    float wingThickness = 0.1f;
    float wingSweep = 0.8f; // How far back the wings are placed

    // Create a compound shape settings object. 
    // The shape will automatically recenter itself around the center of mass. [citation:2]
    StaticCompoundShapeSettings compoundSettings;

    // 1. Fuselage (a cylinder, placed at the center)
    // CylinderShape parameters: (half height of the cylinder, radius) [citation:2]
    // We want the cylinder to extend from -fuselageLength/2 to +fuselageLength/2.
    compoundSettings.AddShape(
        Vec3::sZero(), // Position at the center of mass
        Quat::sIdentity(), // No rotation
        new CylinderShapeSettings(fuselageLength / 2.0f, fuselageRadius)
    );

    // 2. Main Wings (two boxes, one on each side)
    float wingYOffset = 0.0f; // Height of wings relative to fuselage center
    float wingZOffset = -fuselageLength * wingSweep; // Position back from center

    // Left Wing
    auto leftWingSettings = new BoxShapeSettings(Vec3(wingChord / 2.0f, wingThickness / 2.0f, wingspan / 4.0f));
    leftWingSettings->mConvexRadius = 0.0f; // Essential Jolt fix for 0.1f thickness
    compoundSettings.AddShape(
        Vec3(-wingspan / 2.0f, wingYOffset, wingZOffset), // Position to the left
        Quat::sIdentity(), // Wings are flat (no dihedral for simplicity)
        leftWingSettings // Box half-extents
    );

    // Right Wing (same but on the opposite side)
    auto rightWingSettings = new BoxShapeSettings(Vec3(wingChord / 2.0f, wingThickness / 2.0f, wingspan / 4.0f));
    rightWingSettings->mConvexRadius = 0.0f; // Essential Jolt fix for 0.1f thickness
    compoundSettings.AddShape(
        Vec3(wingspan / 2.0f, wingYOffset, wingZOffset),
        Quat::sIdentity(),
        rightWingSettings
    );

    // 3. Tail Fin (a small vertical box at the back)
    auto tailSettings = new BoxShapeSettings(Vec3(0.2f, 0.4f, 0.1f));
    tailSettings->mConvexRadius = 0.0f; // Essential Jolt fix
    compoundSettings.AddShape(
        Vec3(0.0f, 0.5f, -fuselageLength / 2.0f + 0.2f), // On top, at the very back
        Quat::sIdentity(),
        tailSettings
    );

    // Create the actual shape from the settings. Error checking omitted for brevity.
    Shape::ShapeResult result = compoundSettings.Create();
    Shape* jetShape = result.Get().GetPtr();
    // --- YOUR SNIPPET LOGIC END ---

    BodyCreationSettings jetSettings(jetShape, position, Quat::sIdentity(), EMotionType::Dynamic, layer);
    jetSettings.mMassPropertiesOverride.mMass = 5000.0f;
    jetSettings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
    jetSettings.mLinearDamping = 0.05f;
    jetSettings.mAngularDamping = 0.1f;

    mMainBodyId = bodyInterface.CreateAndAddBody(jetSettings, EActivation::Activate);

    // Mapping Aero to the User geometry
    mSections = {
        { "LWing", Vec3(-wingspan/2.0f, 0, wingZOffset), Quat::sIdentity(), 10.0f, 2.0f, 0.02f, 2 },
        { "RWing", Vec3(wingspan/2.0f, 0, wingZOffset), Quat::sIdentity(), 10.0f, 2.0f, 0.02f, 2 },
        { "Tail",  Vec3(0, 0.5f, -fuselageLength/2.0f), Quat::sIdentity(), 5.0f, 1.5f, 0.02f, 1 }
    };
}

void Aircraft::ApplyAerodynamics(PhysicsSystem* physicsSystem, const float* actions, float deltaTime) {
    if (mMainBodyId.IsInvalid()) return;
    BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();

    RMat44 worldTransform = bodyInterface.GetWorldTransform(mMainBodyId);
    Quat rot = worldTransform.GetRotation().GetQuaternion();
    Vec3 vel = bodyInterface.GetLinearVelocity(mMainBodyId);
    float speed = vel.Length();

    // Thrust
    float thrust = std::clamp(actions[0], 0.0f, 1.0f) * mThrustMax;
    bodyInterface.AddForce(mMainBodyId, rot * Vec3(0, 0, thrust));

    if (speed < 1.0f) return;
    Vec3 velDir = vel / speed;
    float q = 0.5f * mRho * speed * speed;

    for (const auto& s : mSections) {
        Quat sectionWorldRot = rot * s.relativeRot;
        Vec3 sectionWorldUp = sectionWorldRot * Vec3(0, 1, 0);
        RVec3 sectionWorldPos = worldTransform * s.relativePos;

        float aoa = -std::asin(std::clamp(velDir.Dot(sectionWorldUp), -0.99f, 0.99f));

        float deflection = 0.0f;
        if (s.controlType == 1) deflection = actions[1]; 
        else if (s.controlType == 2) deflection = actions[2] * (s.relativePos.GetX() > 0 ? -1.0f : 1.0f); 
        else if (s.controlType == 3) deflection = actions[3]; 

        aoa += deflection * DegreesToRadians(20.0f); 

        float Cl = s.liftCoef * std::sin(2.0f * aoa);
        float Cd = s.dragCoef + (Cl * Cl * 0.15f); 

        Vec3 cross1 = velDir.Cross(sectionWorldUp);
        if (cross1.LengthSq() < 1e-6f) continue;
        
        Vec3 liftDir = cross1.Cross(velDir).Normalized();
        if (liftDir.Dot(sectionWorldUp) < 0) liftDir = -liftDir;

        Vec3 force = (liftDir * Cl + (-velDir * Cd)) * (q * s.area);
        bodyInterface.AddForce(mMainBodyId, force, sectionWorldPos);
    }
}
