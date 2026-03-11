/**
 * @file CombatRobot.cpp
 * @brief Implementation of the CombatRobotLoader class
 * 
 * This file implements all the functionality for loading, controlling, and
 * observing combat robots in the JOLTrl reinforcement learning environment.
 */

#include <stdexcept>
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include "CombatRobot.h"
#include "RobotConfig.h"
#include "Robot.h"
#include "RobotFactory.h"
#include "RobotController.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>

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
#ifndef NUM_LIDAR_RAYS
constexpr int NUM_LIDAR_RAYS = 10;
#endif
#ifndef NUM_SATELLITES
constexpr int NUM_SATELLITES = 4;
#endif

// Define the static member
JPH::Ref<JPH::GroupFilterTable> CombatRobotLoader::mGroupFilter = nullptr;

using json = nlohmann::json;

// JPH::Ref counter corruption from concurrent thread execution.

/**
 * @brief Create LIDAR directions for 360-degree scanning
 * 
 * Generates evenly distributed LIDAR rays based on the configured number of rays.
 * 
 * @param numRays Number of LIDAR rays to generate
 * @return Vector of LIDAR ray directions
 */
std::vector<JPH::Vec3> CombatRobotLoader::CreateLidarDirections(int numRays)
{
    std::vector<JPH::Vec3> directions;
    directions.reserve(numRays);

    // If numRays is 10, use the legacy configuration for compatibility
    if (numRays == 10)
    {
        directions = {
            JPH::Vec3(1.0f, 0.0f, 0.0f),        // Right
            JPH::Vec3(0.707f, 0.0f, 0.707f),   // Right-Forward
            JPH::Vec3(0.707f, 0.0f, -0.707f),  // Right-Back
            JPH::Vec3(0.5f, 0.0f, 0.866f),     // Forward-Right
            JPH::Vec3(0.5f, 0.0f, -0.866f),    // Back-Right
            JPH::Vec3(0.0f, 0.0f, 1.0f),       // Forward
            JPH::Vec3(0.0f, 0.0f, -1.0f),      // Back
            JPH::Vec3(-1.0f, 0.0f, 0.0f),      // Left
            JPH::Vec3(0.0f, 1.0f, 0.0f),       // Up
            JPH::Vec3(0.0f, -1.0f, 0.0f)       // Down
        };
    }
    else
    {
        // Generate evenly distributed rays around the robot
        for (int i = 0; i < numRays; ++i)
        {
            float angle = 2.0f * 3.14159f * i / numRays;
            float x = std::cos(angle);
            float z = std::sin(angle);
            directions.emplace_back(x, 0.0f, z);
        }
    }

    return directions;
}

/**
 * @brief Load a combat robot from a configuration file
 * 
 * This function loads a combat robot from a JSON configuration file, creates all
 * its physical components, and adds them to the physics system.
 * 
 * @param configPath Path to the JSON configuration file
 * @param physicsSystem Pointer to the Jolt Physics system
 * @param position Spawn position of the robot
 * @param envIndex Environment index for collision filtering
 * @param robotIndex Robot index within the environment
 * @return CombatRobotData structure containing the loaded robot
 * 
 * @throws std::runtime_error If there's an error loading the robot
 */
CombatRobotData CombatRobotLoader::LoadRobot(
    const std::string& configPath,
    JPH::PhysicsSystem* ps,
    const JPH::RVec3& pos,
    uint32_t env,
    int idx)
{
    auto loadStart = std::chrono::high_resolution_clock::now();
    std::cout << "[LoadRobot" << idx << "] Start loading" << std::endl;
    
    // Force sequential loading to prevent Jolt memory allocator collisions 
    // and JPH::Ref counter corruption from concurrent thread execution.
    static std::mutex sLoadMutex;
    std::lock_guard<std::mutex> lock(sLoadMutex);
    
    CombatRobotData robotData;
    robotData.envIndex = env;
    robotData.robotIndex = idx;
    robotData.hp = 100.0f;
    robotData.totalEnergyUsed = 0.0f;
    robotData.collisionGroup = env * 2 + idx;

    std::ifstream file(configPath);
    if (!file.is_open())
    {
        std::cerr << "[JOLTrl] FATAL: Failed to open " << configPath << std::endl;
        return robotData;
    }

    json config;
    file >> config;

    // Load robot configuration FIRST
    robotData.config = RobotConfig::LoadFromJSON(config);

    // Initialize vector fields AFTER config is loaded
    robotData.baseActions.resize(robotData.config.actionsPerRobot);
    robotData.residualActions.resize(robotData.config.actionsPerRobot);
    robotData.finalActions.resize(robotData.config.actionsPerRobot);
    robotData.observationBuffer.resize(robotData.config.observationDim);
    robotData.lidarDistances.resize(robotData.config.numLidarRays);

    JPH::BodyInterface& bodyInterface = ps->GetBodyInterface();

    JPH::ObjectLayer ghostLayer = Layers::MOVING_BASE + env;

    if (mGroupFilter == nullptr)
    {
        mGroupFilter = new JPH::GroupFilterTable(256);
        
        // --- PREVENT INTERNAL EXPLOSIONS ---
        // Disable all self-collisions between parts of the same robot
        for (int i = 0; i < 256; ++i) {
            for (int j = 0; j < 256; ++j) {
                mGroupFilter->DisableCollision(i, j);
            }
        }
        // -----------------------------------
    }

    // Create core body using configuration
    JPH::SphereShapeSettings coreShapeSettings(robotData.config.coreRadius);
    coreShapeSettings.SetDensity(robotData.config.coreMass / (4.0f / 3.0f * 3.14159f * 
        pow(robotData.config.coreRadius, 3)));
    
    auto coreResult = coreShapeSettings.Create();
    if (coreResult.HasError()) throw std::runtime_error("Core Shape Error: " + std::string(coreResult.GetError().c_str()));
    JPH::RefConst<JPH::Shape> coreShape = coreResult.Get();

    JPH::BodyCreationSettings coreSettings(
        coreShape,
        pos,
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        ghostLayer
    );

    coreSettings.mFriction = robotData.config.coreFriction;
    coreSettings.mRestitution = robotData.config.coreRestitution;
    coreSettings.mLinearDamping = robotData.config.coreLinearDamping;
    coreSettings.mAngularDamping = robotData.config.coreAngularDamping;
    coreSettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
    coreSettings.mCollisionGroup.SetGroupID(robotData.collisionGroup);
    coreSettings.mCollisionGroup.SetSubGroupID(0);

    JPH::Body* coreBody = bodyInterface.CreateBody(coreSettings);
    if (!coreBody) throw std::runtime_error("FATAL: Failed to create body!");
    robotData.mainBodyId = coreBody->GetID();
    bodyInterface.AddBody(robotData.mainBodyId, JPH::EActivation::Activate);

    std::cout << "[LoadRobot" << idx << "] Config: bodies=" << robotData.config.bodies.size() 
              << ", joints=" << robotData.config.joints.size() 
              << ", satellites=" << robotData.config.numSatellites << std::endl;

    // Handle multi-body configs (bodies + constraints/joints)
    if (!robotData.config.bodies.empty()) {
        std::cout << "[LoadRobot" << idx << "] Loading multi-body config with " 
                  << robotData.config.bodies.size() << " bodies and "
                  << robotData.config.joints.size() << " joints" << std::endl;
        
        robotData.bodies.resize(robotData.config.bodies.size());
        
        // Create all bodies first
        for (size_t i = 0; i < robotData.config.bodies.size(); ++i) {
            const auto& bodyConfig = robotData.config.bodies[i];
            JPH::Ref<JPH::Shape> bodyShape;
            
            if (bodyConfig.shapeType == "sphere") {
                float radius = bodyConfig.shapeParams.empty() ? 1.0f : bodyConfig.shapeParams[0];
                JPH::SphereShapeSettings shapeSettings(radius);
                shapeSettings.SetDensity(bodyConfig.mass / (4.0f / 3.0f * 3.14159f * pow(radius, 3)));
                auto result = shapeSettings.Create();
                if (result.HasError()) throw std::runtime_error("Body Shape Error: " + std::string(result.GetError().c_str()));
                bodyShape = result.Get();
            } else if (bodyConfig.shapeType == "box") {
                if (bodyConfig.shapeParams.size() >= 3) {
                    JPH::BoxShapeSettings shapeSettings(JPH::Vec3(bodyConfig.shapeParams[0], bodyConfig.shapeParams[1], bodyConfig.shapeParams[2]));
                    shapeSettings.SetDensity(bodyConfig.mass / (8.0f * bodyConfig.shapeParams[0] * bodyConfig.shapeParams[1] * bodyConfig.shapeParams[2]));
                    auto result = shapeSettings.Create();
                    if (result.HasError()) throw std::runtime_error("Box Shape Error: " + std::string(result.GetError().c_str()));
                    bodyShape = result.Get();
                }
            }
            
            if (!bodyShape) continue;
            
            JPH::Vec3 position(bodyConfig.position[0], bodyConfig.position[1], bodyConfig.position[2]);
            JPH::Quat rotation = JPH::Quat::sIdentity();
            if (bodyConfig.rotation.size() >= 3) {
                rotation = JPH::Quat::sEulerAngles(JPH::Vec3(bodyConfig.rotation[0], bodyConfig.rotation[1], bodyConfig.rotation[2]));
            }
            
            JPH::BodyCreationSettings bodySettings(bodyShape, pos + position, rotation, JPH::EMotionType::Dynamic, ghostLayer);
            bodySettings.mFriction = bodyConfig.friction;
            bodySettings.mRestitution = bodyConfig.restitution;
            bodySettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
            bodySettings.mCollisionGroup.SetGroupID(robotData.collisionGroup);
            bodySettings.mCollisionGroup.SetSubGroupID(static_cast<uint32_t>(i));
            
            JPH::Body* body = bodyInterface.CreateBody(bodySettings);
            if (body) {
                robotData.bodies[i] = body->GetID();
                bodyInterface.AddBody(robotData.bodies[i], JPH::EActivation::Activate);
                
                // Set mainBodyId to first body for IsValid() check
                if (i == 0) robotData.mainBodyId = body->GetID();
            }
        }
        
        // Create joints/constraints
        for (const auto& jointConfig : robotData.config.joints) {
            if (jointConfig.body1.empty() || jointConfig.body2.empty()) continue;
            
            // Find body indices by name
            int body1Idx = -1, body2Idx = -1;
            for (size_t i = 0; i < robotData.config.bodies.size(); ++i) {
                if (robotData.config.bodies[i].name == jointConfig.body1) body1Idx = i;
                if (robotData.config.bodies[i].name == jointConfig.body2) body2Idx = i;
            }
            
            if (body1Idx < 0 || body2Idx < 0) continue;
            if (body1Idx >= (int)robotData.bodies.size() || body2Idx >= (int)robotData.bodies.size()) continue;
            
            JPH::BodyID body1 = robotData.bodies[body1Idx];
            JPH::BodyID body2 = robotData.bodies[body2Idx];
            
            if (jointConfig.type == "sixdof" || jointConfig.type == "SixDOF") {
                JPH::SixDOFConstraintSettings sixDofSettings;
                sixDofSettings.mSpace = JPH::EConstraintSpace::WorldSpace;
                
                // Use joint position from config (in local body space)
                JPH::Vec3 jointPos(0.0f, 0.0f, 0.0f);
                if (jointConfig.position.size() >= 3) {
                    jointPos = JPH::Vec3(jointConfig.position[0], jointConfig.position[1], jointConfig.position[2]);
                }
                sixDofSettings.mPosition1 = pos + jointPos;
                sixDofSettings.mPosition2 = pos + jointPos;
                
                // Lock translations
                for (int axis = 0; axis < 3; ++axis) {
                    sixDofSettings.mLimitMin[axis] = 0.0f;
                    sixDofSettings.mLimitMax[axis] = 0.0f;
                }
                
                // Setup motors for rotations
                if (jointConfig.hasMotor) {
                    for (int axis = 3; axis < 6; ++axis) {
                        sixDofSettings.mMotorSettings[axis].mSpringSettings.mFrequency = 0.0f;
                        sixDofSettings.mMotorSettings[axis].mMinTorqueLimit = jointConfig.motorMaxTorque * -1.0f;
                        sixDofSettings.mMotorSettings[axis].mMaxTorqueLimit = jointConfig.motorMaxTorque;
                    }
                }
                
                JPH::SixDOFConstraint* constraint = static_cast<JPH::SixDOFConstraint*>(
                    bodyInterface.CreateConstraint(&sixDofSettings, body1, body2));
                if (constraint) {
                    robotData.sixDofJoints.push_back(constraint);
                    ps->AddConstraint(constraint);
                    
                    if (jointConfig.hasMotor) {
                        for (int axis = 3; axis < 6; ++axis) {
                            constraint->SetMotorState(static_cast<JPH::SixDOFConstraintSettings::EAxis>(axis), JPH::EMotorState::Velocity);
                        }
                    }
                }
            } else if (jointConfig.type == "hinge" || jointConfig.type == "Hinge") {
                JPH::HingeConstraintSettings hingeSettings;
                hingeSettings.mSpace = JPH::EConstraintSpace::WorldSpace;
                hingeSettings.mPoint1 = pos;
                hingeSettings.mPoint2 = pos;
                hingeSettings.mHingeAxis1 = JPH::Vec3(0, 1, 0);
                hingeSettings.mHingeAxis2 = JPH::Vec3(0, 1, 0);
                hingeSettings.mNormalAxis1 = JPH::Vec3(1, 0, 0);
                hingeSettings.mNormalAxis2 = JPH::Vec3(1, 0, 0);
                
                if (jointConfig.hasMotor) {
                    hingeSettings.mMotorSettings.mSpringSettings.mFrequency = 0.0f;
                    hingeSettings.mMotorSettings.mMinTorqueLimit = jointConfig.motorMaxTorque * -1.0f;
                    hingeSettings.mMotorSettings.mMaxTorqueLimit = jointConfig.motorMaxTorque;
                }
                
                JPH::HingeConstraint* constraint = static_cast<JPH::HingeConstraint*>(
                    bodyInterface.CreateConstraint(&hingeSettings, body1, body2));
                if (constraint) {
                    robotData.hingeJoints.push_back(constraint);
                    ps->AddConstraint(constraint);
                    
                    if (jointConfig.hasMotor) {
                        constraint->SetMotorState(JPH::EMotorState::Velocity);
                    }
                }
            }
        }
        
        std::cout << "[LoadRobot" << idx << "] Multi-body config loaded: " 
                  << robotData.bodies.size() << " bodies, "
                  << robotData.hingeJoints.size() << " hinge joints, "
                  << robotData.sixDofJoints.size() << " 6DOF joints" << std::endl;
    } else {
        // Original satellite-based loading
        std::cout << "[LoadRobot" << idx << "] Step 5: Entering satellite loop" << std::endl;
        robotData.satellites.resize(robotData.config.numSatellites);
    }
    
    for (int i = 0; i < robotData.config.numSatellites; ++i)
    {
        std::cout << "[LoadRobot" << idx << "] Step 5." << i << ".1: Processing satellite " << i << std::endl;
        const SatelliteConfig& satConfig = robotData.config.satellites[i];
        const float azimuth = JPH::DegreesToRadians(satConfig.offsetAngle);
        const float elevation = JPH::DegreesToRadians(satConfig.elevation);
        const float dist = satConfig.distance;
        
        JPH::RVec3 satPos = pos + JPH::RVec3(
            dist * std::cos(elevation) * std::cos(azimuth),
            dist * std::sin(elevation),
            dist * std::cos(elevation) * std::sin(azimuth)
        );

        JPH::SphereShapeSettings sphereSettings(satConfig.radius);
        sphereSettings.SetDensity(satConfig.mass / (4.0f / 3.0f * 3.14159f * 
            pow(satConfig.radius, 3)));
        
        auto satResult = sphereSettings.Create();
        if (satResult.HasError()) throw std::runtime_error("Sat Shape Error: " + std::string(satResult.GetError().c_str()));
        JPH::RefConst<JPH::Shape> satShape = satResult.Get();

        JPH::BodyCreationSettings satSettings(
            satShape,
            satPos,
            JPH::Quat::sIdentity(),
            JPH::EMotionType::Dynamic,
            ghostLayer
        );

        satSettings.mFriction = satConfig.friction;
        satSettings.mRestitution = satConfig.restitution;
        satSettings.mLinearDamping = satConfig.linearDamping;
        satSettings.mAngularDamping = satConfig.angularDamping;
        satSettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
        satSettings.mCollisionGroup.SetGroupID(robotData.collisionGroup);
        satSettings.mCollisionGroup.SetSubGroupID(i + 1);

        JPH::Body* satBody = bodyInterface.CreateBody(satSettings);
        if (!satBody) throw std::runtime_error("FATAL: Failed to create body!");
        robotData.satellites[i].coreBodyId = satBody->GetID();
        bodyInterface.AddBody(robotData.satellites[i].coreBodyId, JPH::EActivation::Activate);

        JPH::SixDOFConstraintSettings rotSettings;
        rotSettings.mSpace = JPH::EConstraintSpace::WorldSpace;
        rotSettings.mPosition1 = pos;
        rotSettings.mPosition2 = pos;
        
        rotSettings.mLimitMin[JPH::SixDOFConstraintSettings::EAxis::TranslationX] = 0.0f;
        rotSettings.mLimitMax[JPH::SixDOFConstraintSettings::EAxis::TranslationX] = 0.0f;
        rotSettings.mLimitMin[JPH::SixDOFConstraintSettings::EAxis::TranslationY] = 0.0f;
        rotSettings.mLimitMax[JPH::SixDOFConstraintSettings::EAxis::TranslationY] = 0.0f;
        rotSettings.mLimitMin[JPH::SixDOFConstraintSettings::EAxis::TranslationZ] = 0.0f;
        rotSettings.mLimitMax[JPH::SixDOFConstraintSettings::EAxis::TranslationZ] = 0.0f;
        
        for (int axis = (int)JPH::SixDOFConstraintSettings::EAxis::RotationX; axis <= (int)JPH::SixDOFConstraintSettings::EAxis::RotationZ; ++axis) {
            rotSettings.mMotorSettings[axis].mSpringSettings.mFrequency = 0.0f; // Pure velocity motor
            rotSettings.mMotorSettings[axis].mMinTorqueLimit = -500.0f;
            rotSettings.mMotorSettings[axis].mMaxTorqueLimit = 500.0f;
        }

        robotData.satellites[i].rotationJoint = static_cast<JPH::SixDOFConstraint*>(
            bodyInterface.CreateConstraint(&rotSettings, coreBody->GetID(), satBody->GetID()));
        if (!robotData.satellites[i].rotationJoint) throw std::runtime_error("FATAL: Constraint creation returned nullptr!");
        ps->AddConstraint(robotData.satellites[i].rotationJoint);
        
        robotData.satellites[i].rotationJoint->SetMotorState(
            JPH::SixDOFConstraintSettings::EAxis::RotationX, JPH::EMotorState::Velocity);
        robotData.satellites[i].rotationJoint->SetMotorState(
            JPH::SixDOFConstraintSettings::EAxis::RotationY, JPH::EMotorState::Velocity);
        robotData.satellites[i].rotationJoint->SetMotorState(
            JPH::SixDOFConstraintSettings::EAxis::RotationZ, JPH::EMotorState::Velocity);

        // Create spike body using configuration
        JPH::CylinderShapeSettings spikeShapeSettings(
            robotData.config.spike.halfHeight,
            robotData.config.spike.radius,
            robotData.config.spike.convexRadius
        );
        spikeShapeSettings.SetDensity(robotData.config.spike.mass / (3.14159f * 
            pow(robotData.config.spike.radius, 2) * 2.0f * robotData.config.spike.halfHeight));
        
        auto spikeResult = spikeShapeSettings.Create();
        if (spikeResult.HasError()) throw std::runtime_error("Spike Shape Error: " + std::string(spikeResult.GetError().c_str()));
        JPH::RefConst<JPH::Shape> spikeShape = spikeResult.Get();

        JPH::Vec3 direction = JPH::Vec3(
            std::cos(elevation) * std::cos(azimuth),
            std::sin(elevation),
            std::cos(elevation) * std::sin(azimuth)
        );
        
        JPH::Quat spikeRotation = JPH::Quat::sFromTo(JPH::Vec3::sAxisY(), direction);

        JPH::RVec3 spikePos = satPos + JPH::RVec3(direction * (satConfig.radius + robotData.config.spike.halfHeight));

        JPH::BodyCreationSettings spikeSettings(
            spikeShape,
            spikePos,
            spikeRotation,
            JPH::EMotionType::Dynamic,
            ghostLayer
        );

        spikeSettings.mFriction = robotData.config.spike.friction;
        spikeSettings.mRestitution = robotData.config.spike.restitution;
        spikeSettings.mMotionQuality = JPH::EMotionQuality::LinearCast;
        spikeSettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
        spikeSettings.mCollisionGroup.SetGroupID(robotData.collisionGroup);
        spikeSettings.mCollisionGroup.SetSubGroupID(robotData.config.numSatellites + i + 1);

        JPH::Body* spikeBody = bodyInterface.CreateBody(spikeSettings);
        if (!spikeBody) throw std::runtime_error("FATAL: Failed to create body!");
        robotData.satellites[i].spikeBodyId = spikeBody->GetID();
        bodyInterface.AddBody(robotData.satellites[i].spikeBodyId, JPH::EActivation::Activate);

        JPH::SliderConstraintSettings slideSettings;
        slideSettings.mSpace = JPH::EConstraintSpace::WorldSpace;
        slideSettings.mPoint1 = spikePos;
        slideSettings.mPoint2 = spikePos;
        slideSettings.SetSliderAxis(direction);
        slideSettings.mLimitsMin = robotData.config.slideMin;
        slideSettings.mLimitsMax = robotData.config.slideMax;
        slideSettings.mMotorSettings.mSpringSettings.mFrequency = 0.0f; // Pure velocity
        slideSettings.mMotorSettings.mMinForceLimit = robotData.config.motorMinTorqueLimit;
        slideSettings.mMotorSettings.mMaxForceLimit = robotData.config.motorMaxTorqueLimit;

        robotData.satellites[i].slideJoint = static_cast<JPH::SliderConstraint*>(
            bodyInterface.CreateConstraint(&slideSettings, satBody->GetID(), spikeBody->GetID()));
        if (!robotData.satellites[i].slideJoint) throw std::runtime_error("FATAL: Constraint creation returned nullptr!");
        ps->AddConstraint(robotData.satellites[i].slideJoint);
        
        robotData.satellites[i].slideJoint->SetMotorState(JPH::EMotorState::Velocity);

         robotData.satellites[i].pidX = {0.0f, 0.0f, 200.0f, 5.0f, 50.0f};
         robotData.satellites[i].pidY = {0.0f, 0.0f, 200.0f, 5.0f, 50.0f};
         robotData.satellites[i].pidZ = {0.0f, 0.0f, 200.0f, 5.0f, 50.0f};
    }

    auto loadEnd = std::chrono::high_resolution_clock::now();
    auto loadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - loadStart).count();
    std::cout << "[LoadRobot" << idx << "] Loaded in " << loadDuration << "ms" << std::endl;
    
    return robotData;
}

/**
 * @brief Reset a combat robot to its initial state
 * 
 * This function resets a robot to its initial state by moving all its parts
 * to the spawn position and resetting all internal state.
 * 
 * @param robot Reference to the CombatRobotData structure
 * @param physicsSystem Pointer to the Jolt Physics system
 * @param spawnPosition Spawn position for the robot
 */
void CombatRobotLoader::ResetRobot(
    CombatRobotData& robot,
    JPH::PhysicsSystem* ps,
    const JPH::RVec3& pos)
{
    JPH::BodyInterface& bodyInterface = ps->GetBodyInterface();

    robot.hp = 100.0f;
    robot.totalDamageDealt = 0.0f;
    robot.totalDamageTaken = 0.0f;
    robot.totalEnergyUsed = 0.0f;

    bodyInterface.SetPositionAndRotation(robot.mainBodyId, pos, JPH::Quat::sIdentity(),
                                         JPH::EActivation::Activate);
    bodyInterface.SetLinearAndAngularVelocity(robot.mainBodyId, JPH::Vec3::sZero(), JPH::Vec3::sZero());

    for (int i = 0; i < robot.config.numSatellites; ++i)
    {
        const SatelliteConfig& satConfig = robot.config.satellites[i];
        const float azimuth = JPH::DegreesToRadians(satConfig.offsetAngle);
        const float elevation = JPH::DegreesToRadians(satConfig.elevation);
        const float distance = satConfig.distance;
        
        JPH::RVec3 satPos = pos + JPH::RVec3(
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

/**
 * @brief Compute base PID control actions for the robot
 * 
 * This function calculates the base control actions using PID controllers
 * to stabilize the satellite rotations.
 * 
 * @param robot Reference to the CombatRobotData structure
 * @param physicsSystem Pointer to the Jolt Physics system
 * @param dt Time step in seconds
 */
void CombatRobotLoader::ComputeBasePIDActions(
    CombatRobotData& robot,
    JPH::PhysicsSystem* ps,
    float dt)
{
    JPH::BodyInterface& bodyInterface = ps->GetBodyInterface();

    for (int i = 0; i < robot.config.numSatellites; ++i)
    {
        JPH::Vec3 angVel = bodyInterface.GetAngularVelocity(robot.satellites[i].coreBodyId);
        
        float torqueX = robot.satellites[i].pidX.Compute(0.0f, angVel.GetX(), dt);
        float torqueY = robot.satellites[i].pidY.Compute(0.0f, angVel.GetY(), dt);
        float torqueZ = robot.satellites[i].pidZ.Compute(0.0f, angVel.GetZ(), dt);

        robot.baseActions[i * robot.config.actionsPerSatellite + 0] = torqueX;
        robot.baseActions[i * robot.config.actionsPerSatellite + 1] = torqueY;
        robot.baseActions[i * robot.config.actionsPerSatellite + 2] = torqueZ;
        robot.baseActions[i * robot.config.actionsPerSatellite + 3] = 0.0f;
    }
}

/**
 * @brief Blend residual actions with base PID control
 * 
 * This function blends the residual actions from the RL model with the base
 * PID control actions to produce the final control signals.
 * 
 * @param robot Reference to the CombatRobotData structure
 */
void CombatRobotLoader::BlendResidualWithBase(CombatRobotData& robot)
{
    for (int i = 0; i < robot.config.numSatellites; ++i)
    {
        int base = i * robot.config.actionsPerSatellite;
        // PID output (baseActions) is already in target velocity units
        // Model output (residualActions) is -1 to 1, scaled by actionScale
        robot.finalActions[base + 0] = robot.baseActions[base + 0] + robot.residualActions[base + 0] * robot.actionScale.rotationScale;
        robot.finalActions[base + 1] = robot.baseActions[base + 1] + robot.residualActions[base + 1] * robot.actionScale.rotationScale;
        robot.finalActions[base + 2] = robot.baseActions[base + 2] + robot.residualActions[base + 2] * robot.actionScale.rotationScale;
        robot.finalActions[base + 3] = robot.baseActions[base + 3] + robot.residualActions[base + 3] * robot.actionScale.slideScale;
    }
    
    // For reaction wheels and burst, we just pass them through to ApplyActions
    // (ApplyActions will handle their specific scales)
    int satelliteActions = robot.config.numSatellites * robot.config.actionsPerSatellite;
    for (int i = satelliteActions; i < robot.config.actionsPerRobot; ++i)
    {
        robot.finalActions[i] = robot.residualActions[i];
    }
}

/**
 * @brief Apply control actions to the robot
 * 
 * This function applies the control actions to the robot's physical components,
 * including satellite rotations, spike slides, reaction wheels, and spike bursts.
 * 
 * @param robot Reference to the CombatRobotData structure
 * @param actions Action vector containing control signals
 * @param physicsSystem Pointer to the Jolt Physics system
 */
void CombatRobotLoader::ApplyActions(
    CombatRobotData& robot,
    const float* actions,
    JPH::PhysicsSystem* ps)
{
    JPH::BodyInterface& bodyInterface = ps->GetBodyInterface();
    float energySum = 0.0f;

    for (int i = 0; i < robot.config.numSatellites; ++i)
    {
        // actions here are already blended and scaled if coming from ApplyResidualActions
        const float vx = actions[i * robot.config.actionsPerSatellite + 0];
        const float vy = actions[i * robot.config.actionsPerSatellite + 1];
        const float vz = actions[i * robot.config.actionsPerSatellite + 2];
        const float slideVel = actions[i * robot.config.actionsPerSatellite + 3];

        if (robot.config.useDirectTorque) {
            // Apply direct torque to orbiters instead of using motors
            const float torqueScale = robot.config.orbiterTorqueScale;
            JPH::Vec3 orbiterTorque(vx * torqueScale, vy * torqueScale, vz * torqueScale);
            bodyInterface.AddTorque(robot.satellites[i].coreBodyId, orbiterTorque);
        } else {
            // Traditional motor-based control
            if (robot.satellites[i].rotationJoint != nullptr)
            {
                robot.satellites[i].rotationJoint->SetTargetVelocityCS(
                    JPH::Vec3(vx, vy, vz));
            }
        }

        if (robot.satellites[i].slideJoint != nullptr)
        {
            robot.satellites[i].slideJoint->SetTargetVelocity(slideVel);
        }

        energySum += std::abs(vx) + std::abs(vy) + std::abs(vz) + std::abs(slideVel);
    }

    const float reactionTorqueScale = robot.config.reactionTorqueScale;
    // Indices after satellite actions for reaction torque, burst is last
    int satelliteActions = robot.config.numSatellites * robot.config.actionsPerSatellite;
    JPH::Vec3 reactionTorque(
        actions[satelliteActions] * reactionTorqueScale,
        actions[satelliteActions + 1] * reactionTorqueScale,
        actions[satelliteActions + 2] * reactionTorqueScale
    );
    bodyInterface.AddTorque(robot.mainBodyId, reactionTorque);

    // // Omni spike burst functionality disabled
    // const float omniSpikeBurst = actions[satelliteActions + 3] * robot.actionScale.slideScale;
    // for (int i = 0; i < robot.config.numSatellites; ++i)
    // {
    //     if (robot.satellites[i].slideJoint != nullptr)
    //     {
    //         float currentVel = robot.satellites[i].slideJoint->GetTargetVelocity();
    //         robot.satellites[i].slideJoint->SetTargetVelocity(currentVel + omniSpikeBurst);
    //     }
    // }

    robot.totalEnergyUsed += energySum * 0.001f;
}

/**
 * @brief Apply residual actions with PID base control
 * 
 * This function applies residual actions from the RL model, combining them
 * with PID base control for stable operation.
 * 
 * @param robot Reference to the CombatRobotData structure
 * @param residualActions Residual action vector from the RL model
 * @param physicsSystem Pointer to the Jolt Physics system
 */
void CombatRobotLoader::ApplyResidualActions(
    CombatRobotData& robot,
    const float* residualActions,
    JPH::PhysicsSystem* ps)
{
    // 1. Store the residual actions from the model
    for (int i = 0; i < robot.config.actionsPerRobot; ++i)
    {
        robot.residualActions[i] = residualActions[i];
    }

    // 2. Compute the base stability actions (PID)
    // Using 120Hz control frequency to match viewer physicshz
    ComputeBasePIDActions(robot, ps, 1.0f / 120.0f);

    // 3. Blend and Scale
    BlendResidualWithBase(robot);

    // 4. Apply to Jolt
    ApplyActions(robot, robot.finalActions.data(), ps);
}

/**
 * @brief Perform a LIDAR scan for the robot
 * 
 * This function performs a 360-degree LIDAR scan around the robot's core,
 * measuring distances to obstacles in 10 different directions.
 * 
 * @param robot Reference to the CombatRobotData structure
 * @param physicsSystem Pointer to the Jolt Physics system
 */
void CombatRobotLoader::PerformLidarScan(
    CombatRobotData& robot,
    JPH::PhysicsSystem* ps)
{
    JPH::BodyInterface& bodyInterface = ps->GetBodyInterface();
    
    JPH::RVec3 rootPos = bodyInterface.GetPosition(robot.mainBodyId);
    JPH::Quat rootRot = bodyInterface.GetRotation(robot.mainBodyId);
    
    const float maxDistance = robot.config.lidarMaxDistance;
    
    const JPH::NarrowPhaseQuery& narrowPhaseQuery = ps->GetNarrowPhaseQuery();
    
    JPH::IgnoreMultipleBodiesFilter bodyFilter;
    bodyFilter.IgnoreBody(robot.mainBodyId);
    for (int i = 0; i < robot.config.numSatellites; ++i) {
        if (!robot.satellites[i].coreBodyId.IsInvalid()) bodyFilter.IgnoreBody(robot.satellites[i].coreBodyId);
        if (!robot.satellites[i].spikeBodyId.IsInvalid()) bodyFilter.IgnoreBody(robot.satellites[i].spikeBodyId);
    }
    
    std::vector<JPH::Vec3> lidarDirections = CreateLidarDirections(robot.config.numLidarRays);
    
    for (int i = 0; i < robot.config.numLidarRays; ++i)
    {
        JPH::Vec3 worldDir = rootRot * lidarDirections[i];
        
        JPH::RRayCast ray;
        ray.mOrigin = rootPos;
        ray.mDirection = JPH::RVec3(worldDir); // Fixed: direction should be unit vector, not scaled
        
        JPH::RayCastResult result;
        
        bool hit = narrowPhaseQuery.CastRay(ray, result, JPH::BroadPhaseLayerFilter(), JPH::ObjectLayerFilter(), bodyFilter);
        
        if (hit && result.mFraction <= maxDistance)
        {
            robot.lidarDistances[i] = static_cast<float>(result.mFraction);
        }
        else
        {
            robot.lidarDistances[i] = maxDistance;
        }
    }
}

/**
 * @brief Get observations for the robot
 * 
 * This function collects all sensory observations from the robot's environment
 * and opponent, formatting them into a single observation vector for the
 * reinforcement learning algorithm.
 * 
 * @param robot Reference to the CombatRobotData structure
 * @param opponent Reference to the opponent's CombatRobotData structure
 * @param observations Output buffer for the observation vector
 * @param forces Force sensor readings
 * @param physicsSystem Pointer to the Jolt Physics system
 */
void CombatRobotLoader::GetObservations(
    CombatRobotData& robot,
    const CombatRobotData& opponent,
    float* observations,
    const ForceSensorReading& forces,
    JPH::PhysicsSystem* ps)
{
    JPH::BodyInterface& bodyInterface = ps->GetBodyInterface();
    int idx = 0;

    // ========== MY CORE BODY (9) ==========
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
    
    // ========== MY ORIENTATION (4 + 6) ==========
    observations[idx++] = myRot.GetX();
    observations[idx++] = myRot.GetY();
    observations[idx++] = myRot.GetZ();
    observations[idx++] = myRot.GetW();
    
    // Direction vectors
    
    // ========== OPPONENT INFO (18) ==========
    JPH::RVec3 oppPos = bodyInterface.GetPosition(opponent.mainBodyId);
    JPH::Vec3 oppVel = bodyInterface.GetLinearVelocity(opponent.mainBodyId);
    JPH::Vec3 oppAngVel = bodyInterface.GetAngularVelocity(opponent.mainBodyId);
    JPH::Quat oppRot = bodyInterface.GetRotation(opponent.mainBodyId);
    JPH::RVec3 relPos = oppPos - myPos;
    
    observations[idx++] = static_cast<float>(relPos.GetX());
    observations[idx++] = static_cast<float>(relPos.GetY());
    observations[idx++] = static_cast<float>(relPos.GetZ());
    observations[idx++] = oppVel.GetX();
    observations[idx++] = oppVel.GetY();
    observations[idx++] = oppVel.GetZ();
    observations[idx++] = oppAngVel.GetX();
    observations[idx++] = oppAngVel.GetY();
    observations[idx++] = oppAngVel.GetZ();
    observations[idx++] = oppRot.GetX();
    observations[idx++] = oppRot.GetY();
    observations[idx++] = oppRot.GetZ();
    observations[idx++] = oppRot.GetW();
    
    // Relative velocity (9)
    JPH::Vec3 relVel = oppVel - myVel;
    observations[idx++] = relVel.GetX();
    observations[idx++] = relVel.GetY();
    observations[idx++] = relVel.GetZ();
    
    // Relative angular velocity (3)
    JPH::Vec3 relAngVel = oppAngVel - myAngVel;
    observations[idx++] = relAngVel.GetX();
    observations[idx++] = relAngVel.GetY();
    observations[idx++] = relAngVel.GetZ();
    
    // ========== MY SATELLITES (numSat * 9 = 27 for 3 sats) ==========
    for (int i = 0; i < robot.config.numSatellites; ++i)
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
    
    // ========== OPPONENT SATELLITES (numSat * 9 = 27 for 3 sats) ==========
    for (int i = 0; i < opponent.config.numSatellites; ++i)
    {
        JPH::RVec3 pos = bodyInterface.GetPosition(opponent.satellites[i].coreBodyId);
        JPH::Vec3 vel = bodyInterface.GetLinearVelocity(opponent.satellites[i].coreBodyId);
        JPH::Vec3 angVel = bodyInterface.GetAngularVelocity(opponent.satellites[i].coreBodyId);
        
        // Relative position to each opponent sat
        JPH::RVec3 satRelPos = pos - myPos;
        observations[idx++] = static_cast<float>(satRelPos.GetX());
        observations[idx++] = static_cast<float>(satRelPos.GetY());
        observations[idx++] = static_cast<float>(satRelPos.GetZ());
        observations[idx++] = vel.GetX();
        observations[idx++] = vel.GetY();
        observations[idx++] = vel.GetZ();
        observations[idx++] = angVel.GetX();
        observations[idx++] = angVel.GetY();
        observations[idx++] = angVel.GetZ();
    }
    
    // ========== LIDAR (16) ==========
    PerformLidarScan(robot, ps);
    for (int i = 0; i < robot.config.numLidarRays; ++i)
    {
        observations[idx++] = robot.lidarDistances[i] / robot.config.lidarMaxDistance;
    }
    
    // ========== STATUS (30) ==========
    // Health (4)
    observations[idx++] = robot.hp / 100.0f;
    observations[idx++] = opponent.hp / 100.0f;
    float healthDiff = (robot.hp - opponent.hp) / 100.0f;
    observations[idx++] = healthDiff;
    float healthRatio = (opponent.hp > 0.01f) ? (robot.hp / opponent.hp) : 1.0f;
    observations[idx++] = std::clamp(healthRatio, 0.0f, 3.0f) / 3.0f;
    
    // Distance & geometry (8)
    float dist = static_cast<float>((oppPos - myPos).Length());
    observations[idx++] = dist / 20.0f;  // Normalized distance
    observations[idx++] = dist * dist / 400.0f;  // Squared distance
    
    JPH::Vec3 myForward = myRot.RotateAxisY();
    JPH::Vec3 toOpponent = (oppPos - myPos).Normalized();
    float facingDot = myForward.Dot(toOpponent);
    observations[idx++] = facingDot;  // -1 to 1
    
    // Lateral component (reuse myRight declared earlier)
    JPH::Vec3 myRight = myRot.RotateAxisX();
    JPH::Vec3 myUp = myRot.RotateAxisZ();
    observations[idx++] = myRight.Dot(toOpponent);
    observations[idx++] = myUp.Dot(toOpponent);
    
    // Relative orientation (4)
    JPH::Quat relRot = myRot.Conjugated() * oppRot;
    observations[idx++] = relRot.GetX();
    observations[idx++] = relRot.GetY();
    observations[idx++] = relRot.GetZ();
    observations[idx++] = relRot.GetW();
    
    // Speed metrics (8)
    float mySpeed = myVel.Length();
    float oppSpeed = oppVel.Length();
    observations[idx++] = mySpeed / 10.0f;
    observations[idx++] = oppSpeed / 10.0f;
    float speedRatio = (oppSpeed > 0.01f) ? (mySpeed / oppSpeed) : 1.0f;
    observations[idx++] = std::clamp(speedRatio, 0.0f, 5.0f) / 5.0f;
    
    float closingSpeed = -relVel.Dot(toOpponent);
    observations[idx++] = closingSpeed / 10.0f;
    observations[idx++] = std::abs(closingSpeed) / 10.0f;
    
    // Angular speed (4)
    float myAngSpeed = myAngVel.Length();
    float oppAngSpeed = oppAngVel.Length();
    observations[idx++] = myAngSpeed / 20.0f;
    observations[idx++] = oppAngSpeed / 20.0f;
    
    // Cross products (9)
    JPH::Vec3 crossProduct = myVel.Cross(oppVel);
    observations[idx++] = crossProduct.GetX() / 10.0f;
    observations[idx++] = crossProduct.GetY() / 10.0f;
    observations[idx++] = crossProduct.GetZ() / 10.0f;
    
    // Momentum (6)
    float myMomentum = mySpeed * 30.0f;  // Assuming mass ~30
    float oppMomentum = oppSpeed * 30.0f;
    observations[idx++] = myMomentum / 100.0f;
    observations[idx++] = oppMomentum / 100.0f;
    observations[idx++] = (myMomentum - oppMomentum) / 100.0f;
    
    // Angular momentum (6)
    JPH::Vec3 myAngMom = myAngVel * 10.0f;  // Approximate I
    JPH::Vec3 oppAngMom = oppAngVel * 10.0f;
    observations[idx++] = myAngMom.GetX() / 20.0f;
    observations[idx++] = myAngMom.GetY() / 20.0f;
    observations[idx++] = myAngMom.GetZ() / 20.0f;
    
    // ========== DAMAGE & ENERGY (9) ==========
    observations[idx++] = robot.totalDamageDealt / 100.0f;
    observations[idx++] = robot.totalDamageTaken / 100.0f;
    observations[idx++] = robot.totalEnergyUsed / 10000.0f;
    observations[idx++] = robot.totalDamageDealt / std::max(robot.totalDamageTaken, 0.01f);  // Efficiency
    observations[idx++] = robot.episodeSteps / 1000.0f;
    observations[idx++] = robot.episodeSteps / 10000.0f;  // Longer term
    
    // Time to collision estimate (2)
    float timeToCollision = dist / std::max(std::abs(closingSpeed), 0.1f);
    observations[idx++] = timeToCollision / 20.0f;
    observations[idx++] = std::clamp(timeToCollision, 0.0f, 20.0f) / 20.0f;
    
    // ========== SATELLITE FORCES (numSat * 4 = 12 for 3 sats) ==========
    for (int i = 0; i < robot.config.numSatellites; ++i)
    {
        if (i < forces.impulseMagnitude.size())
            observations[idx++] = forces.impulseMagnitude[i];
        else
            observations[idx++] = 0.0f;
    }
    for (int i = 0; i < robot.config.numSatellites; ++i)
    {
        if (i < forces.jointStress.size())
            observations[idx++] = forces.jointStress[i];
        else
            observations[idx++] = 0.0f;
    }
    
    // ========== SATELLITE ALTITUDES (3) ==========
    for (int i = 0; i < robot.config.numSatellites; ++i)
    {
        JPH::RVec3 satPos = bodyInterface.GetPosition(robot.satellites[i].coreBodyId);
        observations[idx++] = static_cast<float>(satPos.GetY()) / 10.0f;
    }
    
    // ========== GRAVITY & ORIENTATION (6) ==========
    JPH::Vec3 worldGravity(0.0f, -1.0f, 0.0f);
    JPH::Vec3 localGravity = myRot.Conjugated() * worldGravity;
    observations[idx++] = localGravity.GetX();
    observations[idx++] = localGravity.GetY();
    observations[idx++] = localGravity.GetZ();
    
    // Gravity magnitude indicator
    observations[idx++] = localGravity.Length();
    
    // Up vector alignment (reuse myUp)
    observations[idx++] = myUp.GetY();  // 1 = upright, -1 = upside down
    
    // ========== POSITION BOUNDS (4) ==========
    observations[idx++] = static_cast<float>(myPos.GetX()) / 100.0f;
    observations[idx++] = static_cast<float>(myPos.GetZ()) / 100.0f;
    observations[idx++] = static_cast<float>(oppPos.GetX()) / 100.0f;
    observations[idx++] = static_cast<float>(oppPos.GetZ()) / 100.0f;
    
    // ========== WORLD INFO (4) ==========
    // Arena bounds check
    float arenaRadius = 15.0f;
    float distFromCenter = std::sqrt(myPos.GetX()*myPos.GetX() + myPos.GetZ()*myPos.GetZ());
    observations[idx++] = distFromCenter / arenaRadius;
    observations[idx++] = (distFromCenter > arenaRadius * 0.8f) ? 1.0f : 0.0f;  // Near edge warning
    
    // Opponent near edge
    float oppDistFromCenter = std::sqrt(oppPos.GetX()*oppPos.GetX() + oppPos.GetZ()*oppPos.GetZ());
    observations[idx++] = oppDistFromCenter / arenaRadius;
    observations[idx++] = (oppDistFromCenter > arenaRadius * 0.8f) ? 1.0f : 0.0f;
    
    // ========== COMPOUND OBSERVATIONS (combinations) ==========
    
    // Energy-based (6)
    float kineticEnergy = 0.5f * 30.0f * mySpeed * mySpeed;
    float potentialEnergy = 30.0f * 9.81f * myPos.GetY();
    float energyRatio = (potentialEnergy > 0.01f) ? kineticEnergy / potentialEnergy : 0.0f;
    observations[idx++] = kineticEnergy / 10000.0f;
    observations[idx++] = potentialEnergy / 10000.0f;
    observations[idx++] = (kineticEnergy + potentialEnergy) / 10000.0f;
    observations[idx++] = energyRatio;
    
    // Momentum × position interaction (3)
    observations[idx++] = myMomentum * dist / 1000.0f;
    observations[idx++] = oppMomentum * dist / 1000.0f;
    observations[idx++] = (myMomentum - oppMomentum) * dist / 1000.0f;
    
    // Threat assessment (6)
    float threatLevel = 0.0f;
    for (int i = 0; i < opponent.config.numSatellites; ++i) {
        JPH::RVec3 oppSatPos = bodyInterface.GetPosition(opponent.satellites[i].coreBodyId);
        float satDist = static_cast<float>((oppSatPos - myPos).Length());
        if (satDist < 3.0f) threatLevel += (3.0f - satDist) / 3.0f;
    }
    observations[idx++] = threatLevel / 3.0f;  // Normalized threat
    
    // Attack opportunity (3)
    float attackAngle = facingDot;  // Already computed
    float attackSpeed = closingSpeed;
    observations[idx++] = attackAngle * attackSpeed;  // Combined attack metric
    observations[idx++] = (1.0f - std::abs(attackAngle)) * oppSpeed;  // Flanking opportunity
    observations[idx++] = facingDot * oppSpeed;  // Head-on intensity
    
    // Defense metrics (6)
    observations[idx++] = (mySpeed > oppSpeed) ? 1.0f : 0.0f;  // Can outrun
    observations[idx++] = (dist < 5.0f) ? 1.0f : 0.0f;  // In danger zone
    observations[idx++] = (healthRatio < 1.0f) ? 1.0f : 0.0f;  // Health disadvantage
    
    // Relative orientation advantages (4)
    float myHeading = std::atan2(myForward.GetX(), myForward.GetZ());
    float oppHeading = std::atan2(oppRot.GetX(), oppRot.GetZ());
    float headingDiff = myHeading - oppHeading;
    observations[idx++] = std::sin(headingDiff);
    observations[idx++] = std::cos(headingDiff);
    observations[idx++] = std::sin(headingDiff) * mySpeed;
    observations[idx++] = std::cos(headingDiff) * oppSpeed;
    
    // Spin detection - are we rotating towards each other? (3)
    float spinToFace = myAngVel.Dot(toOpponent);
    observations[idx++] = spinToFace / 20.0f;
    observations[idx++] = (spinToFace > 0) ? 1.0f : 0.0f;  // Turning to face
    observations[idx++] = (spinToFace > 0 && dist < 10.0f) ? 1.0f : 0.0f;  // Aggressive spin
    
    // Historical/momentum features (6) - declare statics first
    static JPH::Vec3 prevVel(0,0,0);
    static JPH::Vec3 prevAngVel(0,0,0);
    static JPH::Vec3 prevAccel(0,0,0);
    static float prevDist = 0.0f;
    static float prevClosingSpeed = 0.0f;
    
    // Acceleration direction relative to opponent (3)
    JPH::Vec3 linAccelLocal = (myVel - prevVel) / 0.00833f;
    JPH::Vec3 accelDir = linAccelLocal.Normalized();
    observations[idx++] = accelDir.Dot(toOpponent);  // Moving toward/away
    observations[idx++] = accelDir.Dot(myRight);  // Strafing
    observations[idx++] = accelDir.Dot(myUp);  // Diving/climbing
    
    // Historical/momentum features (6)
    float distRate = (dist - prevDist) / 0.00833f;
    float closingAccel = (closingSpeed - prevClosingSpeed) / 0.00833f;
    prevDist = dist;
    prevClosingSpeed = closingSpeed;
    
    observations[idx++] = distRate / 10.0f;
    observations[idx++] = closingAccel / 50.0f;
    observations[idx++] = std::abs(distRate) / 10.0f;
    observations[idx++] = std::abs(closingAccel) / 50.0f;
    observations[idx++] = (distRate > 0) ? 1.0f : 0.0f;  // Moving away
    observations[idx++] = (closingAccel > 0) ? 1.0f : 0.0f;  // Accelerating toward
    
    // Combined combat assessment (4)
    observations[idx++] = healthRatio * (mySpeed / std::max(oppSpeed, 0.1f));  // Advantage metric
    observations[idx++] = (facingDot + 1.0f) / 2.0f * (dist / 20.0f);  // Position quality
    observations[idx++] = healthRatio * (1.0f - std::abs(facingDot));  // Can attack from flank
    observations[idx++] = std::clamp(robot.totalDamageDealt - robot.totalDamageTaken, -100.0f, 100.0f) / 100.0f;
    
    // Satellite coverage - are my sats covering opponent? (3)
    float satCoverage = 0.0f;
    for (int i = 0; i < robot.config.numSatellites; ++i) {
        JPH::RVec3 mySatPos = bodyInterface.GetPosition(robot.satellites[i].coreBodyId);
        JPH::Vec3 toSatFromCore = mySatPos - myPos;
        float satAngle = toSatFromCore.Normalized().Dot(toOpponent);
        if (satAngle > 0.5f) satCoverage += satAngle;
    }
    observations[idx++] = satCoverage / 3.0f;
    observations[idx++] = satCoverage * (1.0f / std::max(dist, 0.1f));  // Coverage × proximity
    observations[idx++] = (satCoverage > 1.5f && dist < 5.0f) ? 1.0f : 0.0f;  // Locked on
    
    // ========== PAD TO MULTIPLE OF 8 ==========
    // Final padding
    int finalIdx = idx;
    int paddedSize = ((finalIdx + 7) / 8) * 8;
    while (idx < paddedSize) {
        observations[idx++] = 0.0f;
    }
    
    // Debug output for first call
    static bool printed = false;
    if (!printed) {
        printf("[CombatRobot] Observation dimensions: %d (padded to %d)\n", finalIdx, paddedSize);
        printed = true;
    }
}