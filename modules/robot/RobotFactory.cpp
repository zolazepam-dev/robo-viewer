/**
 * @file RobotFactory.cpp
 * @brief Implementation of RobotFactory class
 */

#include "RobotFactory.h"
#include "../physics/PhysicsWorld.h"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>

Robot RobotFactory::CreateRobot(
    const RobotConfig& config,
    PhysicsWorld& physicsWorld,
    const JPH::RVec3& position,
    uint32_t envIndex,
    int robotIndex
) {
    Robot robot;
    robot.SetConfig(config);
    robot.SetEnvIndex(envIndex);
    robot.SetRobotIndex(robotIndex);
    robot.SetType(RobotType::SATELLITE);
    robot.ResizeObservationBuffer(config.observationDim);
    
    uint16_t objectLayer = PhysicsLayers::GetEnvLayer(envIndex);
    
    // Create core body
    JPH::BodyID coreId = CreateCoreBody(physicsWorld, config, position, objectLayer);
    robot.SetMainBodyId(coreId);
    
    // Create satellites
    auto& system = physicsWorld.GetSystem();
    const JPH::BodyInterface& bodyInterface = system.GetBodyInterface();
    JPH::RVec3 corePos = bodyInterface.GetPosition(coreId);
    
    for (const auto& satConfig : config.satellites) {
        SatelliteJoint joint = CreateSatellite(
            physicsWorld, config, satConfig, coreId, corePos, objectLayer
        );
        // robot.AddSatellite(joint); // Would need method to add
    }
    
    return robot;
}

void RobotFactory::ResetRobot(
    Robot& robot,
    PhysicsWorld& physicsWorld,
    const JPH::RVec3& spawnPosition
) {
    auto& system = physicsWorld.GetSystem();
    JPH::BodyInterface& bodyInterface = system.GetBodyInterface();
    
    // Reset main body
    if (!robot.GetMainBodyId().IsInvalid()) {
        bodyInterface.SetPositionAndRotation(
            robot.GetMainBodyId(),
            spawnPosition,
            JPH::Quat::sIdentity(),
            JPH::EActivation::Activate
        );
        bodyInterface.SetLinearVelocity(robot.GetMainBodyId(), JPH::Vec3::sZero());
        bodyInterface.SetAngularVelocity(robot.GetMainBodyId(), JPH::Vec3::sZero());
    }
    
    robot.SetHP(INITIAL_HP);
    robot.ResetEpisodeStats();
}

JPH::BodyID RobotFactory::CreateCoreBody(
    PhysicsWorld& physicsWorld,
    const RobotConfig& config,
    const JPH::RVec3& position,
    uint16_t objectLayer
) {
    JPH::SphereShapeSettings shapeSettings(config.coreRadius);
    JPH::BodyCreationSettings bodySettings(
        shapeSettings.Create().Get(),
        position,
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        objectLayer
    );
    
    bodySettings.mMassPropertiesOverride.mMass = config.coreMass;
    bodySettings.mFriction = config.coreFriction;
    bodySettings.mRestitution = config.coreRestitution;
    
    auto& system = physicsWorld.GetSystem();
    return system.GetBodyInterface().CreateAndAddBody(bodySettings, JPH::EActivation::Activate);
}

SatelliteJoint RobotFactory::CreateSatellite(
    PhysicsWorld& physicsWorld,
    const RobotConfig& config,
    const RobotConfig::Satellite& satConfig,
    JPH::BodyID coreBodyId,
    const JPH::RVec3& corePosition,
    uint16_t objectLayer
) {
    SatelliteJoint joint;
    
    // Calculate satellite position based on offset angle and elevation
    float angle = satConfig.offsetAngle * 3.14159f / 180.0f;
    float elevation = satConfig.elevation * 3.14159f / 180.0f;
    
    JPH::Vec3 offset(
        std::cos(angle) * std::cos(elevation) * satConfig.distance,
        std::sin(elevation) * satConfig.distance,
        std::sin(angle) * std::cos(elevation) * satConfig.distance
    );
    
    JPH::RVec3 satPosition = corePosition + offset;
    
    // Create satellite core body
    JPH::SphereShapeSettings coreShape(satConfig.radius);
    JPH::BodyCreationSettings coreSettings(
        coreShape.Create().Get(),
        satPosition,
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        objectLayer
    );
    coreSettings.mMassPropertiesOverride.mMass = satConfig.mass;
    
    auto& system = physicsWorld.GetSystem();
    joint.coreBodyId = system.GetBodyInterface().CreateAndAddBody(coreSettings, JPH::EActivation::Activate);
    
    // Create spike body (simplified - would need full implementation)
    joint.spikeBodyId = joint.coreBodyId; // Placeholder
    
    // Create joints (simplified - would need full implementation)
    joint.rotationJoint = nullptr;
    joint.slideJoint = nullptr;
    
    return joint;
}
