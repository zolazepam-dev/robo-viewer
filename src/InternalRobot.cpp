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
    float thickness = 0.15f;
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

    for (int f = 0; f < 20; ++f) {
        // Get the three vertices of this face
        JPH::Vec3 v0 = verts[faces[f][0]];
        JPH::Vec3 v1 = verts[faces[f][1]];
        JPH::Vec3 v2 = verts[faces[f][2]];

        // Calculate face center
        JPH::Vec3 faceCenter = (v0 + v1 + v2) / 3.0f;

        // Calculate face normal (pointing outward)
        JPH::Vec3 edge1 = v1 - v0;
        JPH::Vec3 edge2 = v2 - v0;
        JPH::Vec3 normal = edge1.Cross(edge2).Normalized();

        // Calculate basis vectors for the triangle plane
        JPH::Vec3 basisX = edge1.Normalized();
        JPH::Vec3 basisZ = normal;
        JPH::Vec3 basisY = basisZ.Cross(basisX).Normalized();

        // Re-calculate basisX to ensure orthogonality
        basisX = basisY.Cross(basisZ).Normalized();

        // Build rotation matrix from basis vectors
        JPH::Mat44 rotationMatrix(
            JPH::Vec4(basisX, 0),
            JPH::Vec4(basisY, 0),
            JPH::Vec4(basisZ, 0),
            JPH::Vec4(0, 0, 0, 1)
        );
        JPH::Quat rotation = JPH::Quat::sFromTo(JPH::Vec3::sAxisZ(), normal);

        // Calculate triangle dimensions for panel sizing
        // The panel needs to cover the entire triangle with slight overlap
        float edgeLength = (v1 - v0).Length();

        // Triangle height (distance from base to opposite vertex)
        float triangleHeight = std::sqrt(3.0f) / 2.0f * edgeLength;

        // Create triangular panel using a box shape
        // Scale up slightly (1.08x) to ensure overlap at edges
        float panelWidth = edgeLength * 1.08f;
        float panelHeight = triangleHeight * 1.08f;
        float panelThickness = thickness;

        // Create box shape for the panel
        JPH::BoxShapeSettings panelShapeSettings(JPH::Vec3(panelWidth * 0.5f, panelHeight * 0.5f, panelThickness * 0.5f));
        JPH::RefConst<JPH::Shape> panelShape = panelShapeSettings.Create().Get();

        // Position panel slightly outward from face center
        JPH::Vec3 panelPos = faceCenter + normal * (panelThickness * 0.5f);

        // Add to compound shape
        compoundSettings.AddShape(panelPos, rotation, panelShape);
    }

    JPH::RefConst<JPH::Shape> shellShape = compoundSettings.Create().Get();

    // Create shell body at robot position
    JPH::BodyCreationSettings shellSettings(shellShape, position, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, ghostLayer);
    shellSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    shellSettings.mMassPropertiesOverride.mMass = shellMass;
    shellSettings.mRestitution = 0.05f;
    shellSettings.mFriction = 0.1f;
    shellSettings.mMotionQuality = JPH::EMotionQuality::LinearCast;

    JPH::Body* shellBody = bodyInterface.CreateBody(shellSettings);
    robotData.mainBodyId = shellBody->GetID();
    bodyInterface.AddBody(robotData.mainBodyId, JPH::EActivation::Activate);

    // --- INTERNAL ENGINES ---
    for (int i = 0; i < 3; ++i) {
        float engineRadius = config["engines"][i].value("radius", 0.25f);
         float engineMass = config["engines"][i].value("mass", 0.8f); // lighter engines for less "kick"

        JPH::SphereShapeSettings engShapeSettings(engineRadius);
        JPH::RefConst<JPH::Shape> engShape = engShapeSettings.Create().Get();
        
        // Position engines very close to the center to minimize escape risk
        JPH::RVec3 engPos = position + JPH::RVec3((i - 1) * 0.1f, 0.0f, 0.0f);
        
        JPH::BodyCreationSettings engSettings(engShape, engPos, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, ghostLayer);
        engSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        engSettings.mMassPropertiesOverride.mMass = engineMass;
        
        // CRITICAL: Continuous Collision Detection (CCD)
        // This prevents the fast-moving internal spheres from tunneling through the shell walls
        engSettings.mMotionQuality = JPH::EMotionQuality::LinearCast;
        
        engSettings.mRestitution = 0.3f; // Reduced restitution to prevent bouncing out
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
    
      const float maxForce = 20.0f; // Drastically reduced to prevent engine escape - was 50.0f
    float energySum = 0.0f;

        JPH::RVec3 shellPos = bodyInterface.GetPosition(robot.mainBodyId);
        // Calculate containment radius from spherical shell dimensions
        float shellRadius = 2.0f; // Fixed shell radius from config defaults
        float thickness = 0.8f; // Fixed shell thickness
        float containmentRadius = shellRadius; // Inner radius of the spherical shell

    for (int i = 0; i < 3; ++i) {
        if (robot.satellites[i].coreBodyId.IsInvalid()) continue;
        
        JPH::RVec3 engPos = bodyInterface.GetPosition(robot.satellites[i].coreBodyId);
        JPH::Vec3 relPos = JPH::Vec3(engPos - shellPos);
        float dist = relPos.Length();

        // --- DEBUG MONITORING ---
        if (dist > containmentRadius) {
            std::cout << "[InternalRobot] ALERT: Engine " << i << " escaped! Dist: " << dist << ". Applying recall force." << std::endl;
            // Physical Recall Force (Magnetic Containment) - Significantly increased
            JPH::Vec3 recall = -relPos.Normalized() * 10000.0f;
            bodyInterface.AddForce(robot.satellites[i].coreBodyId, recall);
        }

        JPH::Vec3 force(actions[i * 3 + 0] * maxForce, actions[i * 3 + 1] * maxForce, actions[i * 3 + 2] * maxForce);
        bodyInterface.AddForce(robot.satellites[i].coreBodyId, force);
        energySum += (std::abs(force.GetX()) + std::abs(force.GetY()) + std::abs(force.GetZ()));
    }

    // Reaction Wheels
    const float reactionTorqueScale = 6000.0f;
    JPH::Vec3 reactionTorque(actions[52] * reactionTorqueScale, actions[53] * reactionTorqueScale, actions[54] * reactionTorqueScale);
    bodyInterface.AddTorque(robot.mainBodyId, reactionTorque);

    robot.totalEnergyUsed += energySum * 0.0001f;
}
