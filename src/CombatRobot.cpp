#include <Jolt/Jolt.h>
#include "CombatRobot.h"

#include <cmath>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>

#include "PhysicsCore.h"

using json = nlohmann::json;

CombatRobotData CombatRobotLoader::LoadRobot(
    const std::string& configPath,
    JPH::PhysicsSystem* physicsSystem,
    const JPH::RVec3& position,
    uint32_t envIndex,
    int robotIndex)
{
    CombatRobotData robotData;
    robotData.envIndex = envIndex;
    robotData.robotIndex = robotIndex;
    robotData.hp = 100.0f;

    std::ifstream file(configPath);
    if (!file.is_open())
    {
        std::cerr << "[JOLTrl] FATAL: Failed to open " << configPath << std::endl;
        return robotData;
    }

    json config;
    file >> config;

    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();

    // Dimensional Ghosting Layer Mapping
    JPH::ObjectLayer ghostLayer = Layers::MOVING_BASE + envIndex;

    const float coreRadius = config["core"].value("radius", 0.5f);
    JPH::SphereShapeSettings coreShapeSettings(coreRadius);
    JPH::Ref<JPH::Shape> coreShape = coreShapeSettings.Create().Get();

    JPH::BodyCreationSettings coreSettings(
        coreShape,
        position,
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        ghostLayer
    );

    coreSettings.mFriction = 0.0f;
    coreSettings.mRestitution = 0.8f;
    coreSettings.mLinearDamping = 0.05f;
    coreSettings.mAngularDamping = 0.05f;

    JPH::Body* coreBody = bodyInterface.CreateBody(coreSettings);
    robotData.mainBodyId = coreBody->GetID();
    bodyInterface.AddBody(robotData.mainBodyId, JPH::EActivation::Activate);

    const auto& satellitesConfig = config["satellites"];

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        const float angle = satellitesConfig[i].value("offset_angle", static_cast<float>(i * 90));
        const float distance = satellitesConfig[i].value("distance", 1.4f);
        const float satRadius = 0.3f;

        const float angleRad = JPH::DegreesToRadians(angle);
        JPH::RVec3 satPosition = position + JPH::RVec3(
            std::cos(angleRad) * distance, 0.0, std::sin(angleRad) * distance
        );

        JPH::SphereShapeSettings sphereSettings(satRadius);
        JPH::Ref<JPH::Shape> satShape = sphereSettings.Create().Get();

        JPH::BodyCreationSettings satSettings(
            satShape,
            satPosition,
            JPH::Quat::sIdentity(),
            JPH::EMotionType::Dynamic,
            ghostLayer
        );

        satSettings.mFriction = 0.0f;
        satSettings.mRestitution = 0.8f;
        satSettings.mLinearDamping = 0.05f;
        satSettings.mAngularDamping = 0.05f;

        JPH::Body* satBody = bodyInterface.CreateBody(satSettings);
        robotData.satellites[i].coreBodyId = satBody->GetID();
        bodyInterface.AddBody(robotData.satellites[i].coreBodyId, JPH::EActivation::Activate);

        robotData.satellites[i].yawJoint = nullptr;
        robotData.satellites[i].spikeBodyId = robotData.satellites[i].coreBodyId;
        robotData.satellites[i].slideJoint = nullptr;
    }

    return robotData;
}

void CombatRobotLoader::ResetRobot(
    CombatRobotData& robot,
    JPH::PhysicsSystem* physicsSystem,
    const JPH::RVec3& spawnPosition)
{
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();

    robot.hp = 100.0f;
    robot.totalDamageDealt = 0.0f;
    robot.totalDamageTaken = 0.0f;

    bodyInterface.SetPositionAndRotation(robot.mainBodyId, spawnPosition, JPH::Quat::sIdentity(),
                                         JPH::EActivation::Activate);
    bodyInterface.SetLinearAndAngularVelocity(robot.mainBodyId, JPH::Vec3::sZero(), JPH::Vec3::sZero());

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        const float angleRad = JPH::DegreesToRadians(static_cast<float>(i * 90));
        const float distance = 1.4f;

        JPH::RVec3 satPos = spawnPosition + JPH::RVec3(
            std::cos(angleRad) * distance, 0.0, std::sin(angleRad) * distance
        );

        bodyInterface.SetPositionAndRotation(robot.satellites[i].coreBodyId, satPos, JPH::Quat::sIdentity(),
                                             JPH::EActivation::Activate);
        bodyInterface.SetLinearAndAngularVelocity(robot.satellites[i].coreBodyId, JPH::Vec3::sZero(),
                                                  JPH::Vec3::sZero());
    }
}

void CombatRobotLoader::ApplyActions(
    CombatRobotData& robot,
    const float* actions,
    JPH::PhysicsSystem* physicsSystem)
{
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        const float tx = actions[i * ACTIONS_PER_SATELLITE + 0] * 50.0f;
        const float ty = actions[i * ACTIONS_PER_SATELLITE + 1] * 50.0f;
        const float tz = actions[i * ACTIONS_PER_SATELLITE + 2] * 50.0f;

        bodyInterface.AddTorque(robot.satellites[i].coreBodyId, JPH::Vec3(tx, ty, tz));
    }
}

void CombatRobotLoader::GetObservations(
    const CombatRobotData& robot,
    const CombatRobotData& opponent,
    float* observations,
    JPH::PhysicsSystem* physicsSystem)
{
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    int idx = 0;

    JPH::RVec3 myPos = bodyInterface.GetPosition(robot.mainBodyId);
    JPH::Vec3 myVel = bodyInterface.GetLinearVelocity(robot.mainBodyId);
    JPH::Vec3 myAngVel = bodyInterface.GetAngularVelocity(robot.mainBodyId);

    observations[idx++] = static_cast<float>(myPos.GetX());
    observations[idx++] = static_cast<float>(myPos.GetY());
    observations[idx++] = static_cast<float>(myPos.GetZ());
    observations[idx++] = myVel.GetX();
    observations[idx++] = myVel.GetY();
    observations[idx++] = myVel.GetZ();
    observations[idx++] = myAngVel.GetX();
    observations[idx++] = myAngVel.GetY();
    observations[idx++] = myAngVel.GetZ();

    JPH::RVec3 oppPos = bodyInterface.GetPosition(opponent.mainBodyId);
    JPH::Vec3 oppVel = bodyInterface.GetLinearVelocity(opponent.mainBodyId);
    JPH::RVec3 relPos = oppPos - myPos;

    observations[idx++] = static_cast<float>(relPos.GetX());
    observations[idx++] = static_cast<float>(relPos.GetY());
    observations[idx++] = static_cast<float>(relPos.GetZ());
    observations[idx++] = oppVel.GetX();
    observations[idx++] = oppVel.GetY();
    observations[idx++] = oppVel.GetZ();

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        JPH::RVec3 pos = bodyInterface.GetPosition(robot.satellites[i].coreBodyId);
        JPH::Vec3 vel = bodyInterface.GetLinearVelocity(robot.satellites[i].coreBodyId);
        observations[idx++] = static_cast<float>(pos.GetX());
        observations[idx++] = static_cast<float>(pos.GetY());
        observations[idx++] = static_cast<float>(pos.GetZ());
        observations[idx++] = vel.GetX();
        observations[idx++] = vel.GetY();
        observations[idx++] = vel.GetZ();
    }

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        JPH::RVec3 pos = bodyInterface.GetPosition(robot.satellites[i].spikeBodyId);
        observations[idx++] = static_cast<float>(pos.GetX());
        observations[idx++] = static_cast<float>(pos.GetY());
        observations[idx++] = static_cast<float>(pos.GetZ());
    }

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        JPH::RVec3 pos = bodyInterface.GetPosition(opponent.satellites[i].spikeBodyId);
        observations[idx++] = static_cast<float>(pos.GetX());
        observations[idx++] = static_cast<float>(pos.GetY());
        observations[idx++] = static_cast<float>(pos.GetZ());
    }

    observations[idx++] = robot.hp;
    observations[idx++] = opponent.hp;
    observations[idx++] = static_cast<float>((oppPos - myPos).Length());
}