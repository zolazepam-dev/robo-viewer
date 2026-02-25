#include <stdexcept>
#include <Jolt/Jolt.h>
#include "CombatRobot.h"

#include <cmath>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/PhysicsMaterial.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>

#include "PhysicsCore.h"

using json = nlohmann::json;

const JPH::Vec3 CombatRobotLoader::mLidarDirections[NUM_LIDAR_RAYS] = {
    JPH::Vec3(1.0f, 0.0f, 0.0f),
    JPH::Vec3(0.707f, 0.0f, 0.707f),
    JPH::Vec3(0.707f, 0.0f, -0.707f),
    JPH::Vec3(0.5f, 0.0f, 0.866f),
    JPH::Vec3(0.5f, 0.0f, -0.866f),
    JPH::Vec3(0.0f, 0.0f, 1.0f),
    JPH::Vec3(0.0f, 0.0f, -1.0f),
    JPH::Vec3(-1.0f, 0.0f, 0.0f),
    JPH::Vec3(0.0f, 1.0f, 0.0f),
    JPH::Vec3(0.0f, -1.0f, 0.0f)
};

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
    robotData.totalEnergyUsed = 0.0f;
    robotData.collisionGroup = envIndex * 2 + robotIndex;

    std::ifstream file(configPath);
    if (!file.is_open())
    {
        std::cerr << "[JOLTrl] FATAL: Failed to open " << configPath << std::endl;
        return robotData;
    }

    json config;
    file >> config;

    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();

    JPH::ObjectLayer ghostLayer = Layers::MOVING_BASE + envIndex;

    if (mGroupFilter == nullptr)
    {
        mGroupFilter = new JPH::GroupFilterTable(256);
    }

    const float coreRadius = config["core"].value("radius", 0.5f);
    const float coreMass = config["core"].value("mass", 13.0f);
    
    JPH::SphereShapeSettings coreShapeSettings(coreRadius);
    coreShapeSettings.SetDensity(coreMass / (4.0f / 3.0f * 3.14159f * coreRadius * coreRadius * coreRadius));
    JPH::RefConst<JPH::Shape> coreShape = coreShapeSettings.Create().Get();

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
    coreSettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
    coreSettings.mCollisionGroup.SetGroupID(robotData.collisionGroup);
    coreSettings.mCollisionGroup.SetSubGroupID(0);

    JPH::Body* coreBody = bodyInterface.CreateBody(coreSettings);
    if (!coreBody) throw std::runtime_error("FATAL: Failed to create body!");
    robotData.mainBodyId = coreBody->GetID();
    bodyInterface.AddBody(robotData.mainBodyId, JPH::EActivation::Activate);

    const auto& satellitesConfig = config["satellites"];
    
    float jointDamping = 0.8f;
    float jointArmature = 0.5f;
    float motorTorque = 450.0f;
    float slideMin = 0.0f;
    float slideMax = 0.5f;
    
    if (config.contains("joints"))
    {
        jointDamping = config["joints"].value("hinge_damping", 0.8f);
        jointArmature = config["joints"].value("hinge_armature", 0.5f);
        motorTorque = config["joints"].value("motor_torque", 450.0f);
        if (config["joints"]["slide_range"].is_array() && config["joints"]["slide_range"].size() >= 2)
        {
            slideMin = config["joints"]["slide_range"][0].get<float>();
            slideMax = config["joints"]["slide_range"][1].get<float>();
        }
    }

    std::cout << "[LoadRobot" << robotIndex << "] Step 5: Entering satellite loop" << std::endl;
    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        std::cout << "[LoadRobot" << robotIndex << "] Step 5." << i << ".1: Processing satellite " << i << std::endl;
        const float azimuth = JPH::DegreesToRadians(satellitesConfig[i].value("offset_angle", 0.0f));
        const float elevation = JPH::DegreesToRadians(satellitesConfig[i].value("elevation", 0.0f));
        const float dist = satellitesConfig[i].value("distance", 1.4f);
        
        JPH::RVec3 satPos = position + JPH::RVec3(
            dist * std::cos(elevation) * std::cos(azimuth),
            dist * std::sin(elevation),
            dist * std::cos(elevation) * std::sin(azimuth)
        );

        const float satRadius = 0.1f;
        const float satMass = 3.5f;
        
        JPH::SphereShapeSettings sphereSettings(satRadius);
        sphereSettings.SetDensity(satMass / (4.0f / 3.0f * 3.14159f * satRadius * satRadius * satRadius));
        JPH::RefConst<JPH::Shape> satShape = sphereSettings.Create().Get();

        JPH::BodyCreationSettings satSettings(
            satShape,
            satPos,
            JPH::Quat::sIdentity(),
            JPH::EMotionType::Dynamic,
            ghostLayer
        );

        satSettings.mFriction = 0.0f;
        satSettings.mRestitution = 0.8f;
        satSettings.mLinearDamping = 0.05f;
        satSettings.mAngularDamping = 0.05f;
        satSettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
        satSettings.mCollisionGroup.SetGroupID(robotData.collisionGroup);
        satSettings.mCollisionGroup.SetSubGroupID(i + 1);

        JPH::Body* satBody = bodyInterface.CreateBody(satSettings);
        if (!satBody) throw std::runtime_error("FATAL: Failed to create body!");
        robotData.satellites[i].coreBodyId = satBody->GetID();
        bodyInterface.AddBody(robotData.satellites[i].coreBodyId, JPH::EActivation::Activate);

        JPH::SixDOFConstraintSettings rotSettings;
        rotSettings.mSpace = JPH::EConstraintSpace::WorldSpace;
        rotSettings.mPosition1 = position;
        rotSettings.mPosition2 = position;
        
        rotSettings.mLimitMin[JPH::SixDOFConstraintSettings::EAxis::TranslationX] = 0.0f;
        rotSettings.mLimitMax[JPH::SixDOFConstraintSettings::EAxis::TranslationX] = 0.0f;
        rotSettings.mLimitMin[JPH::SixDOFConstraintSettings::EAxis::TranslationY] = 0.0f;
        rotSettings.mLimitMax[JPH::SixDOFConstraintSettings::EAxis::TranslationY] = 0.0f;
        rotSettings.mLimitMin[JPH::SixDOFConstraintSettings::EAxis::TranslationZ] = 0.0f;
        rotSettings.mLimitMax[JPH::SixDOFConstraintSettings::EAxis::TranslationZ] = 0.0f;
        
        rotSettings.mMotorSettings[JPH::SixDOFConstraintSettings::EAxis::RotationX] = 
            JPH::MotorSettings(motorTorque, jointDamping);
        rotSettings.mMotorSettings[JPH::SixDOFConstraintSettings::EAxis::RotationY] = 
            JPH::MotorSettings(motorTorque, jointDamping);
        rotSettings.mMotorSettings[JPH::SixDOFConstraintSettings::EAxis::RotationZ] = 
            JPH::MotorSettings(motorTorque, jointDamping);

        robotData.satellites[i].rotationJoint = static_cast<JPH::SixDOFConstraint*>(
            bodyInterface.CreateConstraint(&rotSettings, coreBody->GetID(), satBody->GetID()));
        if (!robotData.satellites[i].rotationJoint) throw std::runtime_error("FATAL: Constraint creation returned nullptr!");
        physicsSystem->AddConstraint(robotData.satellites[i].rotationJoint);
        
        robotData.satellites[i].rotationJoint->SetMotorState(
            JPH::SixDOFConstraintSettings::EAxis::RotationX, JPH::EMotorState::Velocity);
        robotData.satellites[i].rotationJoint->SetMotorState(
            JPH::SixDOFConstraintSettings::EAxis::RotationY, JPH::EMotorState::Velocity);
        robotData.satellites[i].rotationJoint->SetMotorState(
            JPH::SixDOFConstraintSettings::EAxis::RotationZ, JPH::EMotorState::Velocity);

        const float spikeHalfHeight = 0.2f;
        const float spikeRadius = 0.02f;
        const float spikeMass = 0.5f;
        
        JPH::CylinderShapeSettings spikeShapeSettings(spikeHalfHeight, spikeRadius);
        spikeShapeSettings.SetDensity(spikeMass / (3.14159f * spikeRadius * spikeRadius * 2.0f * spikeHalfHeight));
        JPH::RefConst<JPH::Shape> spikeShape = spikeShapeSettings.Create().Get();

        JPH::Vec3 direction = JPH::Vec3(
            std::cos(elevation) * std::cos(azimuth),
            std::sin(elevation),
            std::cos(elevation) * std::sin(azimuth)
        );
        
        JPH::Quat spikeRotation = JPH::Quat::sFromTo(JPH::Vec3::sAxisY(), direction);

        JPH::RVec3 spikePos = satPos + JPH::RVec3(direction * (satRadius + spikeHalfHeight));

        JPH::BodyCreationSettings spikeSettings(
            spikeShape,
            spikePos,
            spikeRotation,
            JPH::EMotionType::Dynamic,
            ghostLayer
        );

        spikeSettings.mFriction = 0.0f;
        spikeSettings.mRestitution = 0.3f;
        spikeSettings.mMotionQuality = JPH::EMotionQuality::LinearCast;
        spikeSettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
        spikeSettings.mCollisionGroup.SetGroupID(robotData.collisionGroup);
        spikeSettings.mCollisionGroup.SetSubGroupID(NUM_SATELLITES + i + 1);

        JPH::Body* spikeBody = bodyInterface.CreateBody(spikeSettings);
        if (!spikeBody) throw std::runtime_error("FATAL: Failed to create body!");
        robotData.satellites[i].spikeBodyId = spikeBody->GetID();
        bodyInterface.AddBody(robotData.satellites[i].spikeBodyId, JPH::EActivation::Activate);

        JPH::SliderConstraintSettings slideSettings;
        slideSettings.mSpace = JPH::EConstraintSpace::WorldSpace;
        slideSettings.mPoint1 = spikePos;
        slideSettings.mPoint2 = spikePos;
        slideSettings.SetSliderAxis(direction);
        slideSettings.mLimitsMin = slideMin;
        slideSettings.mLimitsMax = slideMax;
        slideSettings.mMotorSettings = JPH::MotorSettings(200.0f, 1.0f);

        robotData.satellites[i].slideJoint = static_cast<JPH::SliderConstraint*>(
            bodyInterface.CreateConstraint(&slideSettings, satBody->GetID(), spikeBody->GetID()));
        if (!robotData.satellites[i].slideJoint) throw std::runtime_error("FATAL: Constraint creation returned nullptr!");
        physicsSystem->AddConstraint(robotData.satellites[i].slideJoint);
        
        robotData.satellites[i].slideJoint->SetMotorState(JPH::EMotorState::Velocity);

        robotData.satellites[i].pidX = PIDController{50.0f, 0.0f, 10.0f, 0.0f, 0.0f};
        robotData.satellites[i].pidY = PIDController{50.0f, 0.0f, 10.0f, 0.0f, 0.0f};
        robotData.satellites[i].pidZ = PIDController{50.0f, 0.0f, 10.0f, 0.0f, 0.0f};
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
    robot.totalEnergyUsed = 0.0f;

    bodyInterface.SetPositionAndRotation(robot.mainBodyId, spawnPosition, JPH::Quat::sIdentity(),
                                         JPH::EActivation::Activate);
    bodyInterface.SetLinearAndAngularVelocity(robot.mainBodyId, JPH::Vec3::sZero(), JPH::Vec3::sZero());

    const float distance = 1.4f;
    
    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        float azimuth, elevation;
        
        switch (i)
        {
            case 0: azimuth = 0.0f; elevation = 0.0f; break;
            case 1: azimuth = 72.0f; elevation = 0.0f; break;
            case 2: azimuth = 144.0f; elevation = 0.0f; break;
            case 3: azimuth = 216.0f; elevation = 0.0f; break;
            case 4: azimuth = 288.0f; elevation = 0.0f; break;
            case 5: azimuth = 0.0f; elevation = 45.0f; break;
            case 6: azimuth = 90.0f; elevation = 45.0f; break;
            case 7: azimuth = 180.0f; elevation = 45.0f; break;
            case 8: azimuth = 270.0f; elevation = 45.0f; break;
            case 9: azimuth = 0.0f; elevation = -45.0f; break;
            case 10: azimuth = 90.0f; elevation = -45.0f; break;
            case 11: azimuth = 180.0f; elevation = -45.0f; break;
            case 12: azimuth = 270.0f; elevation = -45.0f; break;
            default: azimuth = 0.0f; elevation = 0.0f;
        }
        
        azimuth = JPH::DegreesToRadians(azimuth);
        elevation = JPH::DegreesToRadians(elevation);
        
        JPH::RVec3 satPos = spawnPosition + JPH::RVec3(
            distance * std::cos(elevation) * std::cos(azimuth),
            distance * std::sin(elevation),
            distance * std::cos(elevation) * std::sin(azimuth)
        );

        bodyInterface.SetPositionAndRotation(robot.satellites[i].coreBodyId, satPos, JPH::Quat::sIdentity(),
                                             JPH::EActivation::Activate);
        bodyInterface.SetLinearAndAngularVelocity(robot.satellites[i].coreBodyId, JPH::Vec3::sZero(),
                                                  JPH::Vec3::sZero());

        const float satRadius = 0.1f;
        const float spikeHalfHeight = 0.2f;
        
        JPH::Vec3 direction = JPH::Vec3(
            std::cos(elevation) * std::cos(azimuth),
            std::sin(elevation),
            std::cos(elevation) * std::sin(azimuth)
        );
        
        JPH::RVec3 spikePos = satPos + JPH::RVec3(direction * (satRadius + spikeHalfHeight));
        JPH::Quat spikeRotation = JPH::Quat::sFromTo(JPH::Vec3::sAxisY(), direction);

        bodyInterface.SetPositionAndRotation(robot.satellites[i].spikeBodyId, spikePos, spikeRotation,
                                             JPH::EActivation::Activate);
        bodyInterface.SetLinearAndAngularVelocity(robot.satellites[i].spikeBodyId, JPH::Vec3::sZero(),
                                                  JPH::Vec3::sZero());

        robot.satellites[i].pidX.Reset();
        robot.satellites[i].pidY.Reset();
        robot.satellites[i].pidZ.Reset();
        robot.satellites[i].currentSlidePosition = 0.0f;
        robot.satellites[i].currentAngularVelX = 0.0f;
        robot.satellites[i].currentAngularVelY = 0.0f;
        robot.satellites[i].currentAngularVelZ = 0.0f;
    }
}

void CombatRobotLoader::ComputeBasePIDActions(
    CombatRobotData& robot,
    JPH::PhysicsSystem* physicsSystem,
    float dt)
{
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        JPH::Vec3 angVel = bodyInterface.GetAngularVelocity(robot.satellites[i].coreBodyId);
        
        float torqueX = robot.satellites[i].pidX.Compute(0.0f, angVel.GetX(), dt);
        float torqueY = robot.satellites[i].pidY.Compute(0.0f, angVel.GetY(), dt);
        float torqueZ = robot.satellites[i].pidZ.Compute(0.0f, angVel.GetZ(), dt);

        robot.baseActions[i * ACTIONS_PER_SATELLITE + 0] = torqueX;
        robot.baseActions[i * ACTIONS_PER_SATELLITE + 1] = torqueY;
        robot.baseActions[i * ACTIONS_PER_SATELLITE + 2] = torqueZ;
        robot.baseActions[i * ACTIONS_PER_SATELLITE + 3] = 0.0f;
    }
}

void CombatRobotLoader::BlendResidualWithBase(CombatRobotData& robot)
{
    for (int i = 0; i < ACTIONS_PER_ROBOT; ++i)
    {
        robot.finalActions[i] = robot.baseActions[i] + 
            robot.residualActions[i] * robot.actionScale.rotationScale;
    }
}

void CombatRobotLoader::ApplyActions(
    CombatRobotData& robot,
    const float* actions,
    JPH::PhysicsSystem* physicsSystem)
{
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    float energySum = 0.0f;

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        const float vx = actions[i * ACTIONS_PER_SATELLITE + 0] * robot.actionScale.rotationScale;
        const float vy = actions[i * ACTIONS_PER_SATELLITE + 1] * robot.actionScale.rotationScale;
        const float vz = actions[i * ACTIONS_PER_SATELLITE + 2] * robot.actionScale.rotationScale;
        const float slideVel = actions[i * ACTIONS_PER_SATELLITE + 3] * robot.actionScale.slideScale;

        if (robot.satellites[i].rotationJoint != nullptr)
        {
            robot.satellites[i].rotationJoint->SetTargetVelocityCS(
                JPH::Vec3(vx, vy, vz));
        }

        if (robot.satellites[i].slideJoint != nullptr)
        {
            robot.satellites[i].slideJoint->SetTargetVelocity(slideVel);
        }

        energySum += std::abs(vx) + std::abs(vy) + std::abs(vz) + std::abs(slideVel);
    }

    const float reactionTorqueScale = 450.0f;
    JPH::Vec3 reactionTorque(
        actions[52] * reactionTorqueScale,
        actions[53] * reactionTorqueScale,
        actions[54] * reactionTorqueScale
    );
    bodyInterface.AddTorque(robot.mainBodyId, reactionTorque);

    const float omniSpikeBurst = actions[55] * robot.actionScale.slideScale;
    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        if (robot.satellites[i].slideJoint != nullptr)
        {
            float currentVel = robot.satellites[i].slideJoint->GetTargetVelocity();
            robot.satellites[i].slideJoint->SetTargetVelocity(currentVel + omniSpikeBurst);
        }
    }

    robot.totalEnergyUsed += energySum * 0.001f;
}

void CombatRobotLoader::ApplyResidualActions(
    CombatRobotData& robot,
    const float* residualActions,
    JPH::PhysicsSystem* physicsSystem)
{
    for (int i = 0; i < ACTIONS_PER_ROBOT; ++i)
    {
        robot.residualActions[i] = residualActions[i];
    }

    ComputeBasePIDActions(robot, physicsSystem, 1.0f / 60.0f);
    BlendResidualWithBase(robot);
    ApplyActions(robot, robot.finalActions.data(), physicsSystem);
}

void CombatRobotLoader::PerformLidarScan(
    CombatRobotData& robot,
    JPH::PhysicsSystem* physicsSystem)
{
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    
    JPH::RVec3 rootPos = bodyInterface.GetPosition(robot.mainBodyId);
    JPH::Quat rootRot = bodyInterface.GetRotation(robot.mainBodyId);
    
    const float maxDistance = 20.0f;
    
    const JPH::NarrowPhaseQuery& narrowPhaseQuery = physicsSystem->GetNarrowPhaseQuery();
    
    for (int i = 0; i < NUM_LIDAR_RAYS; ++i)
    {
        JPH::Vec3 worldDir = rootRot * mLidarDirections[i];
        
        JPH::RRayCast ray;
        ray.mOrigin = rootPos;
        ray.mDirection = JPH::RVec3(worldDir * maxDistance);
        
        JPH::RayCastResult result;
        
        JPH::IgnoreSingleBodyFilter bodyFilter(robot.mainBodyId);
        
        bool hit = narrowPhaseQuery.CastRay(ray, result, JPH::BroadPhaseLayerFilter(), JPH::ObjectLayerFilter(), bodyFilter);
        
        if (hit)
        {
            robot.lidarDistances[i] = static_cast<float>(result.mFraction * maxDistance);
        }
        else
        {
            robot.lidarDistances[i] = maxDistance;
        }
    }
}

void CombatRobotLoader::GetObservations(
    CombatRobotData& robot,
    const CombatRobotData& opponent,
    float* observations,
    const ForceSensorReading& forces,
    JPH::PhysicsSystem* physicsSystem)
{
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    int idx = 0;

    JPH::RVec3 myPos = bodyInterface.GetPosition(robot.mainBodyId);
    JPH::Vec3 myVel = bodyInterface.GetLinearVelocity(robot.mainBodyId);
    JPH::Vec3 myAngVel = bodyInterface.GetAngularVelocity(robot.mainBodyId);
    JPH::Quat myRot = bodyInterface.GetRotation(robot.mainBodyId);

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
        JPH::Vec3 angVel = bodyInterface.GetAngularVelocity(robot.satellites[i].coreBodyId);
        
        observations[idx++] = static_cast<float>(pos.GetX());
        observations[idx++] = static_cast<float>(pos.GetY());
        observations[idx++] = static_cast<float>(pos.GetZ());
        observations[idx++] = vel.GetX();
        observations[idx++] = vel.GetY();
        observations[idx++] = vel.GetZ();
        observations[idx++] = angVel.GetX();
        observations[idx++] = angVel.GetY();
        observations[idx++] = angVel.GetZ();
    }

    PerformLidarScan(robot, physicsSystem);
    for (int i = 0; i < NUM_LIDAR_RAYS; ++i)
    {
        observations[idx++] = robot.lidarDistances[i] / 20.0f;
    }

    observations[idx++] = robot.hp / 100.0f;
    observations[idx++] = opponent.hp / 100.0f;
    observations[idx++] = static_cast<float>((oppPos - myPos).Length()) / 20.0f;
    
    JPH::Vec3 myForward = myRot.RotateAxisY();
    JPH::Vec3 toOpponent = (oppPos - myPos).Normalized();
    float facingDot = myForward.Dot(toOpponent);
    observations[idx++] = facingDot;
    
    float healthDiff = (robot.hp - opponent.hp) / 100.0f;
    observations[idx++] = healthDiff;
    
    float mySpeed = myVel.Length();
    float oppSpeed = oppVel.Length();
    observations[idx++] = mySpeed / 10.0f;
    observations[idx++] = oppSpeed / 10.0f;
    
    float speedRatio = (oppSpeed > 0.01f) ? (mySpeed / oppSpeed) : 1.0f;
    observations[idx++] = std::clamp(speedRatio, 0.0f, 5.0f) / 5.0f;
    
    JPH::Vec3 relVel = oppVel - myVel;
    observations[idx++] = relVel.GetX() / 10.0f;
    observations[idx++] = relVel.GetY() / 10.0f;
    observations[idx++] = relVel.GetZ() / 10.0f;
    
    float closingSpeed = -relVel.Dot(toOpponent);
    observations[idx++] = closingSpeed / 10.0f;
    
    JPH::Vec3 crossProduct = myVel.Cross(oppVel);
    observations[idx++] = crossProduct.GetX() / 10.0f;
    observations[idx++] = crossProduct.GetY() / 10.0f;
    observations[idx++] = crossProduct.GetZ() / 10.0f;
    
    observations[idx++] = robot.totalDamageDealt / 100.0f;
    observations[idx++] = robot.totalDamageTaken / 100.0f;
    observations[idx++] = robot.episodeSteps / 1000.0f;
    
    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        observations[idx++] = forces.impulseMagnitude[i];
    }
    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        observations[idx++] = forces.jointStress[i];
    }
    
    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        JPH::RVec3 satPos = bodyInterface.GetPosition(robot.satellites[i].coreBodyId);
        observations[idx++] = static_cast<float>(satPos.GetY()) / 10.0f;
    }
    
    JPH::Vec3 worldGravity(0.0f, -1.0f, 0.0f);
    JPH::Vec3 localGravity = myRot.Conjugated() * worldGravity;
    observations[idx++] = localGravity.GetX();
    observations[idx++] = localGravity.GetY();
    observations[idx++] = localGravity.GetZ();
    
    constexpr float coreMass = 13.0f;
    observations[idx++] = myAngVel.GetX() * coreMass;
    observations[idx++] = myAngVel.GetY() * coreMass;
    observations[idx++] = myAngVel.GetZ() * coreMass;
    
    observations[idx++] = static_cast<float>(myPos.GetX()) / 100.0f;
    observations[idx++] = static_cast<float>(myPos.GetZ()) / 100.0f;
    
    float dist = static_cast<float>((oppPos - myPos).Length());
    float timeToCollision = dist / std::max(std::abs(closingSpeed), 0.1f);
    observations[idx++] = timeToCollision / 20.0f;
}