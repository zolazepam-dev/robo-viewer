#include "InternalRobot.h"
#include "PhysicsCore.h"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

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
    for(int i=0; i<NUM_SATELLITES; ++i) {
        robotData.satellites[i].coreBodyId = JPH::BodyID();
        robotData.satellites[i].spikeBodyId = JPH::BodyID();
    }

    std::ifstream file(configPath);
    if (!file.is_open())
    {
        std::cerr << "[InternalRobot] FATAL: Failed to open " << configPath << " at " << std::filesystem::current_path() << std::endl;
        return robotData;
    }
    json config;
    file >> config;

    float shellRadius = config["shell"].value("radius", 2.0f);
    float thickness = config["shell"].value("thickness", 0.1f);
    float shellMass = config["shell"].value("mass", 15.0f);

    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    JPH::ObjectLayer ghostLayer = Layers::MOVING_BASE + envIndex;

    // --- BUILD SOPHISTICATED HOLLOW SPHERE (26 PANELS) ---
    // We use a combination of face panels, edge panels, and corner panels.
    JPH::Ref<JPH::StaticCompoundShapeSettings> compoundSettings = new JPH::StaticCompoundShapeSettings();
    
    // 1. Core faces (6) - These are the large flat parts
    float faceSize = shellRadius * 0.8f;
    JPH::BoxShapeSettings faceSettings(JPH::Vec3(faceSize, thickness, faceSize));
    JPH::RefConst<JPH::Shape> faceShape = faceSettings.Create().Get();

    // Orientations for the 6 faces
    compoundSettings->AddShape(JPH::Vec3(0, shellRadius, 0), JPH::Quat::sIdentity(), faceShape);
    compoundSettings->AddShape(JPH::Vec3(0, -shellRadius, 0), JPH::Quat::sIdentity(), faceShape);
    compoundSettings->AddShape(JPH::Vec3(shellRadius, 0, 0), JPH::Quat::sRotation(JPH::Vec3::sAxisZ(), JPH::JPH_PI / 2.0f), faceShape);
    compoundSettings->AddShape(JPH::Vec3(-shellRadius, 0, 0), JPH::Quat::sRotation(JPH::Vec3::sAxisZ(), JPH::JPH_PI / 2.0f), faceShape);
    compoundSettings->AddShape(JPH::Vec3(0, 0, shellRadius), JPH::Quat::sRotation(JPH::Vec3::sAxisX(), JPH::JPH_PI / 2.0f), faceShape);
    compoundSettings->AddShape(JPH::Vec3(0, 0, -shellRadius), JPH::Quat::sRotation(JPH::Vec3::sAxisX(), JPH::JPH_PI / 2.0f), faceShape);

    // 2. Edge panels (12) - These bridge the faces at 45 degrees
    float edgeLen = faceSize;
    float edgeWidth = shellRadius * 0.5f;
    JPH::BoxShapeSettings edgeSettings(JPH::Vec3(edgeLen, thickness, edgeWidth));
    JPH::RefConst<JPH::Shape> edgeShape = edgeSettings.Create().Get();

    float offset = shellRadius * 0.707f; // cos(45) * R
    
    // Y-Z plane edges
    compoundSettings->AddShape(JPH::Vec3(0, offset, offset), JPH::Quat::sRotation(JPH::Vec3::sAxisX(), JPH::JPH_PI / 4.0f), edgeShape);
    compoundSettings->AddShape(JPH::Vec3(0, -offset, offset), JPH::Quat::sRotation(JPH::Vec3::sAxisX(), -JPH::JPH_PI / 4.0f), edgeShape);
    compoundSettings->AddShape(JPH::Vec3(0, offset, -offset), JPH::Quat::sRotation(JPH::Vec3::sAxisX(), -JPH::JPH_PI / 4.0f), edgeShape);
    compoundSettings->AddShape(JPH::Vec3(0, -offset, -offset), JPH::Quat::sRotation(JPH::Vec3::sAxisX(), JPH::JPH_PI / 4.0f), edgeShape);

    // X-Y plane edges
    JPH::Quat zRot45 = JPH::Quat::sRotation(JPH::Vec3::sAxisZ(), JPH::JPH_PI / 4.0f);
    JPH::Quat yRot90 = JPH::Quat::sRotation(JPH::Vec3::sAxisY(), JPH::JPH_PI / 2.0f);
    compoundSettings->AddShape(JPH::Vec3(offset, offset, 0), zRot45 * yRot90, edgeShape);
    compoundSettings->AddShape(JPH::Vec3(-offset, offset, 0), JPH::Quat::sRotation(JPH::Vec3::sAxisZ(), -JPH::JPH_PI / 4.0f) * yRot90, edgeShape);
    compoundSettings->AddShape(JPH::Vec3(offset, -offset, 0), JPH::Quat::sRotation(JPH::Vec3::sAxisZ(), -JPH::JPH_PI / 4.0f) * yRot90, edgeShape);
    compoundSettings->AddShape(JPH::Vec3(-offset, -offset, 0), zRot45 * yRot90, edgeShape);

    // X-Z plane edges
    JPH::Quat xRot90 = JPH::Quat::sRotation(JPH::Vec3::sAxisX(), JPH::JPH_PI / 2.0f);
    compoundSettings->AddShape(JPH::Vec3(offset, 0, offset), JPH::Quat::sRotation(JPH::Vec3::sAxisY(), JPH::JPH_PI / 4.0f) * xRot90, edgeShape);
    compoundSettings->AddShape(JPH::Vec3(-offset, 0, offset), JPH::Quat::sRotation(JPH::Vec3::sAxisY(), -JPH::JPH_PI / 4.0f) * xRot90, edgeShape);
    compoundSettings->AddShape(JPH::Vec3(offset, 0, -offset), JPH::Quat::sRotation(JPH::Vec3::sAxisY(), -JPH::JPH_PI / 4.0f) * xRot90, edgeShape);
    compoundSettings->AddShape(JPH::Vec3(-offset, 0, -offset), JPH::Quat::sRotation(JPH::Vec3::sAxisY(), JPH::JPH_PI / 4.0f) * xRot90, edgeShape);

    auto shellResult = compoundSettings->Create();
    JPH::RefConst<JPH::Shape> shellShape = shellResult.Get();

    JPH::BodyCreationSettings shellSettings(shellShape, position, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, ghostLayer);
    shellSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    shellSettings.mMassPropertiesOverride.mMass = shellMass;
    shellSettings.mFriction = 0.4f;
    shellSettings.mRestitution = 0.6f; // Bouncy shell
    
    JPH::Body* shellBody = bodyInterface.CreateBody(shellSettings);
    robotData.mainBodyId = shellBody->GetID();
    bodyInterface.AddBody(robotData.mainBodyId, JPH::EActivation::Activate);

    // --- SPAWN INTERNAL ENGINES ---
    for (int i = 0; i < 3; ++i) {
        float engineRadius = config["engines"][i].value("radius", 0.2f);
        float engineMass = config["engines"][i].value("mass", 2.0f);

        JPH::SphereShapeSettings engShapeSettings(engineRadius);
        JPH::RefConst<JPH::Shape> engShape = engShapeSettings.Create().Get();

        // Place them safely inside
        JPH::RVec3 engPos = position + JPH::RVec3((i - 1) * 0.4f, 0.5f, 0);
        
        JPH::BodyCreationSettings engSettings(engShape, engPos, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, ghostLayer);
        engSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        engSettings.mMassPropertiesOverride.mMass = engineMass;
        engSettings.mFriction = 0.1f; // Slippery internal engines
        engSettings.mRestitution = 0.8f; // Very bouncy
        
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
    const float maxForce = 800.0f; // Cranked up force
    float energySum = 0.0f;

    // 1. Thruster Actions (0-8)
    for (int i = 0; i < 3; ++i) {
        if (robot.satellites[i].coreBodyId.IsInvalid()) continue;

        JPH::Vec3 force(
            actions[i * 3 + 0] * maxForce,
            actions[i * 3 + 1] * maxForce,
            actions[i * 3 + 2] * maxForce
        );

        bodyInterface.AddForce(robot.satellites[i].coreBodyId, force);
        energySum += (std::abs(force.GetX()) + std::abs(force.GetY()) + std::abs(force.GetZ()));
    }

    // 2. Individual Engine Mass Shifting (9, 10, 11)
    for (int i = 0; i < 3; ++i) {
        if (robot.satellites[i].coreBodyId.IsInvalid()) continue;
        float massMult = std::lerp(0.5f, 8.0f, (actions[9 + i] + 1.0f) * 0.5f);
        
        JPH::MassProperties props;
        props.mMass = 2.0f * massMult;
        
        JPH::BodyLockWrite lock(physicsSystem->GetBodyLockInterface(), robot.satellites[i].coreBodyId);
        if (lock.Succeeded()) {
            lock.GetBody().GetMotionProperties()->SetMassProperties(JPH::EAllowedDOFs::All, props);
        }
    }

    // 3. Global Mass Scaling (Action 51)
    float globalMult = std::lerp(0.5f, 3.0f, (actions[51] + 1.0f) * 0.5f);
    JPH::MassProperties shellProps;
    shellProps.mMass = 15.0f * globalMult;
    
    JPH::BodyLockWrite shellLock(physicsSystem->GetBodyLockInterface(), robot.mainBodyId);
    if (shellLock.Succeeded()) {
        shellLock.GetBody().GetMotionProperties()->SetMassProperties(JPH::EAllowedDOFs::All, shellProps);
    }

    // 4. Reaction Wheels (52-54) for the shell
    const float reactionTorqueScale = 4000.0f;
    JPH::Vec3 reactionTorque(
        actions[52] * reactionTorqueScale,
        actions[53] * reactionTorqueScale,
        actions[54] * reactionTorqueScale
    );
    bodyInterface.AddTorque(robot.mainBodyId, reactionTorque);

    robot.totalEnergyUsed += energySum * 0.0001f;
}
