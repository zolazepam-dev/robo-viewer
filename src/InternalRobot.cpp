#include "InternalRobot.h"
#include "PhysicsCore.h"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
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
    float thickness = 0.4f; // Even thicker shell for absolute containment
    float shellMass = config["shell"].value("mass", 15.0f);

    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    JPH::ObjectLayer ghostLayer = Layers::MOVING_BASE + envIndex;

    // --- BUILD HOLLOW DODECAHEDRON ---
    JPH::Ref<JPH::StaticCompoundShapeSettings> compoundSettings = new JPH::StaticCompoundShapeSettings();
    
    const float phi = (1.0f + std::sqrt(5.0f)) / 2.0f;
    std::vector<JPH::Vec3> normals = {
        {0, 1, phi}, {0, 1, -phi}, {0, -1, phi}, {0, -1, -phi},
        {1, phi, 0}, {1, -phi, 0}, {-1, phi, 0}, {-1, -phi, 0},
        {phi, 0, 1}, {phi, 0, -1}, {-phi, 0, 1}, {-phi, 0, -1}
    };

    // Panels are now significantly oversized to ensure no vertex gaps
    float panelHalfSize = shellRadius * 1.8f; 
    JPH::BoxShapeSettings panelSettings(JPH::Vec3(panelHalfSize, thickness * 0.5f, panelHalfSize));
    JPH::RefConst<JPH::Shape> panelShape = panelSettings.Create().Get();

    for (auto& n : normals) {
        JPH::Vec3 unitN = n.Normalized();
        JPH::Vec3 pos = unitN * (shellRadius + thickness * 0.25f);
        JPH::Quat rot = JPH::Quat::sFromTo(JPH::Vec3::sAxisY(), unitN);
        compoundSettings->AddShape(pos, rot, panelShape);
    }

    auto shellResult = compoundSettings->Create();
    JPH::RefConst<JPH::Shape> shellShape = shellResult.Get();

    JPH::BodyCreationSettings shellSettings(shellShape, position, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, ghostLayer);
    shellSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    shellSettings.mMassPropertiesOverride.mMass = shellMass;
    shellSettings.mRestitution = 0.2f; // Low restitution to prevent engine-shell jitter
    shellSettings.mFriction = 0.5f;
    
    // We REMOVE the GroupFilter for the internal bot to allow shell-engine collisions
    // Jolt default behavior: bodies in the same Layer collide unless filtered.
    
    JPH::Body* shellBody = bodyInterface.CreateBody(shellSettings);
    robotData.mainBodyId = shellBody->GetID();
    bodyInterface.AddBody(robotData.mainBodyId, JPH::EActivation::Activate);

    // --- INTERNAL ENGINES ---
    for (int i = 0; i < 3; ++i) {
        float engineRadius = config["engines"][i].value("radius", 0.25f);
        float engineMass = config["engines"][i].value("mass", 3.0f); // Heavier engines for more "kick"

        JPH::SphereShapeSettings engShapeSettings(engineRadius);
        JPH::RefConst<JPH::Shape> engShape = engShapeSettings.Create().Get();
        
        JPH::RVec3 engPos = position + JPH::RVec3((i - 1) * 0.3f, 0.0f, 0.0f);
        
        JPH::BodyCreationSettings engSettings(engShape, engPos, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, ghostLayer);
        engSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        engSettings.mMassPropertiesOverride.mMass = engineMass;
        
        // CRITICAL: Continuous Collision Detection (CCD)
        // This prevents the fast-moving internal spheres from tunneling through the shell walls
        engSettings.mMotionQuality = JPH::EMotionQuality::LinearCast;
        
        engSettings.mRestitution = 0.7f;
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
    
    const float maxForce = 750.0f; // Reduced from 1500.0f
    float energySum = 0.0f;

    JPH::RVec3 shellPos = bodyInterface.GetPosition(robot.mainBodyId);
    float containmentRadius = 2.5f; // Slightly larger than shellRadius

    for (int i = 0; i < 3; ++i) {
        if (robot.satellites[i].coreBodyId.IsInvalid()) continue;
        
        JPH::RVec3 engPos = bodyInterface.GetPosition(robot.satellites[i].coreBodyId);
        JPH::Vec3 relPos = JPH::Vec3(engPos - shellPos);
        float dist = relPos.Length();

        // --- DEBUG MONITORING ---
        if (dist > containmentRadius) {
            std::cout << "[InternalRobot] ALERT: Engine " << i << " escaped! Dist: " << dist << ". Applying recall force." << std::endl;
            // Physical Recall Force (Magnetic Containment)
            JPH::Vec3 recall = -relPos.Normalized() * 5000.0f;
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
