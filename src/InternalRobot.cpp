#include "InternalRobot.h"
#include "PhysicsCore.h"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <cmath>

using json = nlohmann::json;

// We use a specific filter that ALLOWS internal collisions but prevents "self-collision" 
// between the overlapping shell panels if needed. 
// However, Jolt's CompoundShape sub-shapes don't collide with each other anyway.
// The real issue was disabling all collisions in the previous version.

CombatRobotData InternalRobotLoader::LoadInternalRobot(
    const std::string& configPath,
    JPH::PhysicsSystem* physicsSystem,
    const JPH::RVec3& position,
    uint32_t envIndex,
    int robotIndex)
{
    CombatRobotData robotData;
    robotData.envIndex = envIndex;
    robotData.robotIndex = robotIndex;
    robotData.collisionGroup = envIndex * 2 + robotIndex;
    robotData.mainBodyId = JPH::BodyID(); 
    robotData.type = RobotType::INTERNAL_ENGINE;
    
    for(int i=0; i<NUM_SATELLITES; ++i) {
        robotData.satellites[i].coreBodyId = JPH::BodyID();
        robotData.satellites[i].spikeBodyId = JPH::BodyID();
    }

    std::ifstream file(configPath);
    if (!file.is_open())
    {
        std::cerr << "[InternalRobot] FATAL: Failed to open " << configPath << std::endl;
        std::cerr << "[InternalRobot] Current CWD: " << std::filesystem::current_path() << std::endl;
        std::cerr << "[InternalRobot] Absolute Path tried: " << std::filesystem::absolute(configPath) << std::endl;
        return robotData;
    }
    json config;
    file >> config;

    float shellRadius = config["shell"].value("radius", 2.0f);
    float thickness = config["shell"].value("thickness", 0.3f); // Use config or default to thicker
    float shellMass = config["shell"].value("mass", 25.0f);

    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    JPH::ObjectLayer ghostLayer = Layers::MOVING_BASE + envIndex;

    // --- BUILD ICOSAHEDRON SHELL ---
    // 12 vertices using golden ratio method
    const float phi = (1.0f + std::sqrt(5.0f)) / 2.0f;

    // Normalize vertices to unit sphere, then scale to desired radius
    auto normalize = [](const JPH::Vec3& v, float scale) -> JPH::Vec3 {
        return v.Normalized() * scale;
    };

    // 12 vertices of icosahedron
    JPH::Vec3 verts[12];
    verts[0]  = normalize(JPH::Vec3(0,  1,  phi), shellRadius);
    verts[1]  = normalize(JPH::Vec3(0,  1, -phi), shellRadius);
    verts[2]  = normalize(JPH::Vec3(0, -1,  phi), shellRadius);
    verts[3]  = normalize(JPH::Vec3(0, -1, -phi), shellRadius);
    verts[4]  = normalize(JPH::Vec3( 1,  phi, 0), shellRadius);
    verts[5]  = normalize(JPH::Vec3( 1, -phi, 0), shellRadius);
    verts[6]  = normalize(JPH::Vec3(-1,  phi, 0), shellRadius);
    verts[7]  = normalize(JPH::Vec3(-1, -phi, 0), shellRadius);
    verts[8]  = normalize(JPH::Vec3( phi, 0,  1), shellRadius);
    verts[9]  = normalize(JPH::Vec3( phi, 0, -1), shellRadius);
    verts[10] = normalize(JPH::Vec3(-phi, 0,  1), shellRadius);
    verts[11] = normalize(JPH::Vec3(-phi, 0, -1), shellRadius);

    // 20 triangular faces (indices into verts array)
    // Each face winds counter-clockwise when viewed from outside
    int faces[20][3] = {
        {0, 2, 8},    // Top cap faces
        {0, 8, 4},
        {0, 4, 6},
        {0, 6, 10},
        {0, 10, 2},
        {3, 1, 9},    // Bottom cap faces
        {3, 9, 5},
        {3, 5, 7},
        {3, 7, 11},
        {3, 11, 1},
        {1, 4, 9},    // Middle ring faces (upper)
        {1, 6, 4},
        {1, 11, 6},
        {2, 5, 8},    // Middle ring faces (lower)
        {2, 7, 5},
        {2, 10, 7},
        {4, 8, 9},    // Connecting faces
        {6, 11, 10},
        {5, 9, 8},
        {7, 10, 11}
    };

    // Create static compound shape for the icosahedron shell
    JPH::StaticCompoundShapeSettings compoundSettings;

    // Create an inner shell radius for perfect mitering
    float innerRadius = shellRadius - thickness;

    for (int f = 0; f < 20; ++f) {
        // Get vertices for this face
        JPH::Vec3 v0_out = verts[faces[f][0]];
        JPH::Vec3 v1_out = verts[faces[f][1]];
        JPH::Vec3 v2_out = verts[faces[f][2]];

        // Calculate corresponding inner vertices (perfectly aligned with the center)
        JPH::Vec3 v0_in = v0_out * (innerRadius / shellRadius);
        JPH::Vec3 v1_in = v1_out * (innerRadius / shellRadius);
        JPH::Vec3 v2_in = v2_out * (innerRadius / shellRadius);

        // Define the 6 points for the triangular wedge (prism)
        std::vector<JPH::Vec3> wedgePoints = {
            v0_out, v1_out, v2_out,
            v0_in, v1_in, v2_in
        };

        // Create a ConvexHullShape for this wedge
        JPH::ConvexHullShapeSettings wedgeSettings(wedgePoints.data(), wedgePoints.size());
        JPH::ShapeSettings::ShapeResult wedgeResult = wedgeSettings.Create();
        if (!wedgeResult.IsValid()) {
            std::cerr << "[InternalRobot] Failed to create wedge shape for face " << f << std::endl;
            continue;
        }
        JPH::RefConst<JPH::Shape> wedgeShape = wedgeResult.Get();

        // Add to compound shape (pos and rot are 0 because points are in global/local model space)
        compoundSettings.AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), wedgeShape);
    }

    JPH::ShapeSettings::ShapeResult shellResult = compoundSettings.Create();
    if (!shellResult.IsValid()) {
        std::cerr << "[InternalRobot] FATAL: Failed to create icosahedron shell shape" << std::endl;
        return robotData;
    }
    JPH::RefConst<JPH::Shape> shellShape = shellResult.Get();

    // Create shell body at robot position
    JPH::BodyCreationSettings shellSettings(shellShape, position, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, ghostLayer);
    shellSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    shellSettings.mMassPropertiesOverride.mMass = shellMass;
    shellSettings.mRestitution = 0.1f;
    shellSettings.mFriction = 0.2f;
    shellSettings.mMotionQuality = JPH::EMotionQuality::LinearCast; // CCD Enabled

    JPH::Body* shellBody = bodyInterface.CreateBody(shellSettings);
    if (shellBody == nullptr) {
        std::cerr << "[InternalRobot] FATAL: Failed to create shell body" << std::endl;
        return robotData;
    }
    robotData.mainBodyId = shellBody->GetID();
    bodyInterface.AddBody(robotData.mainBodyId, JPH::EActivation::Activate);

    // --- INTERNAL ENGINES ---
    for (int i = 0; i < 3; ++i) {
        float engineRadius = config["engines"][i].value("radius", 0.25f);
        float engineMass = config["engines"][i].value("mass", 1.5f); // Heavier engines for better collision stability

        JPH::SphereShapeSettings engShapeSettings(engineRadius);
        JPH::RefConst<JPH::Shape> engShape = engShapeSettings.Create().Get();
        
        // Position engines slightly spread out from center
        JPH::RVec3 engPos = position + JPH::RVec3((i - 1) * 0.2f, 0.0f, 0.0f);
        
        JPH::BodyCreationSettings engSettings(engShape, engPos, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, ghostLayer);
        engSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        engSettings.mMassPropertiesOverride.mMass = engineMass;
        
        // CRITICAL: CCD for engines
        engSettings.mMotionQuality = JPH::EMotionQuality::LinearCast;
        
        engSettings.mRestitution = 0.2f;
        engSettings.mFriction = 0.1f;
        
        JPH::Body* engBody = bodyInterface.CreateBody(engSettings);
        robotData.satellites[i].coreBodyId = engBody->GetID();
        bodyInterface.AddBody(robotData.satellites[i].coreBodyId, JPH::EActivation::Activate);
    }

    return robotData;
}

void InternalRobotLoader::ApplyInternalActions(
    CombatRobotData& robot,
    const float* actions,
    JPH::PhysicsSystem* physicsSystem)
{
    if (robot.mainBodyId.IsInvalid()) return;
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    
    // WIRE UP GUI: Use the dynamic scale from the robot configuration
    const float maxForce = robot.actionScale.slideScale; 
    float energySum = 0.0f;

    JPH::RVec3 shellPos = bodyInterface.GetPosition(robot.mainBodyId);
    float containmentRadius = 1.8f; // Soft boundary slightly inside the 2.0m shell

    for (int i = 0; i < 3; ++i) {
        if (robot.satellites[i].coreBodyId.IsInvalid()) continue;
        
        JPH::RVec3 engPos = bodyInterface.GetPosition(robot.satellites[i].coreBodyId);
        JPH::Vec3 relPos = JPH::Vec3(engPos - shellPos);
        float dist = relPos.Length();

        // --- PREVENTATIVE CONTAINMENT (Magnetic Well) ---
        // Apply a spring force pulling back to center if they move too far
        if (dist > 0.1f) {
            float springK = 500.0f;
            if (dist > containmentRadius) springK = 5000.0f; // Stiff recall
            
            JPH::Vec3 containmentForce = -relPos.Normalized() * (dist * springK);
            bodyInterface.AddForce(robot.satellites[i].coreBodyId, containmentForce);
        }

        // --- ACTION FORCE ---
        JPH::Vec3 force(actions[i * 3 + 0] * maxForce, actions[i * 3 + 1] * maxForce, actions[i * 3 + 2] * maxForce);
        bodyInterface.AddForce(robot.satellites[i].coreBodyId, force);
        energySum += force.Length();
    }

    // Reaction Wheels (Main Body Rotation)
    const float reactionTorqueScale = robot.actionScale.rotationScale;
    JPH::Vec3 reactionTorque(actions[52] * reactionTorqueScale, actions[53] * reactionTorqueScale, actions[54] * reactionTorqueScale);
    bodyInterface.AddTorque(robot.mainBodyId, reactionTorque);

    robot.totalEnergyUsed += energySum * 0.0001f;
}
