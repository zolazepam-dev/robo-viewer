/**
 * @file CombatRobot.cpp
 * @brief Implementation of the CombatRobotLoader class
 * 
 * This file implements all the functionality for loading, controlling, and
 * observing combat robots in the JOLTrl reinforcement learning environment.
 */

#include <stdexcept>
#include <Jolt/Jolt.h>
#include "CombatRobot.h"

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

/**
 * @brief Load RobotConfig from JSON configuration
 * @param config JSON object containing robot configuration
 * @return Loaded RobotConfig
 */
RobotConfig RobotConfig::LoadFromJSON(const json& config)
{
    RobotConfig robotConfig;

    // Load core configuration
    if (config.contains("core"))
    {
        const auto& core = config["core"];
        robotConfig.coreRadius = core.value("radius", 0.5f);
        robotConfig.coreMass = core.value("mass", 13.0f);
        robotConfig.coreFriction = core.value("friction", 0.5f);
        robotConfig.coreRestitution = core.value("restitution", 0.2f);
        robotConfig.coreLinearDamping = core.value("linear_damping", 0.1f);
        robotConfig.coreAngularDamping = core.value("angular_damping", 0.1f);
    }

    // Load satellite configuration
    if (config.contains("satellites") && config["satellites"].is_array())
    {
        const auto& satellites = config["satellites"];
        for (const auto& satConfig : satellites)
        {
            RobotConfig::Satellite satellite;
            satellite.offsetAngle = satConfig.value("offset_angle", 0.0f);
            satellite.elevation = satConfig.value("elevation", 0.0f);
            satellite.distance = satConfig.value("distance", 1.4f);
            satellite.radius = satConfig.value("radius", 0.1f);
            satellite.mass = satConfig.value("mass", 3.5f);
            satellite.friction = satConfig.value("friction", 0.5f);
            satellite.restitution = satConfig.value("restitution", 0.2f);
            satellite.linearDamping = satConfig.value("linear_damping", 0.1f);
            satellite.angularDamping = satConfig.value("angular_damping", 0.1f);
            robotConfig.satellites.push_back(satellite);
        }
    }
    else
    {
        // Default satellite configuration (6 satellites)
        std::vector<float> azimuths = {0.0f, 72.0f, 144.0f, 216.0f, 288.0f, 0.0f};
        std::vector<float> elevations = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 45.0f};
        for (size_t i = 0; i < 6; ++i)
        {
            RobotConfig::Satellite satellite;
            satellite.offsetAngle = azimuths[i];
            satellite.elevation = elevations[i];
            robotConfig.satellites.push_back(satellite);
        }
    }

    // Load spike configuration
    if (config.contains("spike"))
    {
        const auto& spike = config["spike"];
        robotConfig.spikeHalfHeight = spike.value("half_height", 0.2f);
        robotConfig.spikeRadius = spike.value("radius", 0.02f);
        robotConfig.spikeMass = spike.value("mass", 0.5f);
        robotConfig.spikeFriction = spike.value("friction", 0.0f);
        robotConfig.spikeRestitution = spike.value("restitution", 0.3f);
        robotConfig.spikeConvexRadius = spike.value("convex_radius", 0.01f);
    }

    // Load joint configuration
    if (config.contains("joints"))
    {
        const auto& joints = config["joints"];
        robotConfig.jointDamping = joints.value("hinge_damping", 0.8f);
        robotConfig.jointArmature = joints.value("hinge_armature", 0.5f);
        robotConfig.motorTorque = joints.value("motor_torque", 450.0f);
        
        if (joints["slide_range"].is_array() && joints["slide_range"].size() >= 2)
        {
            robotConfig.slideMin = joints["slide_range"][0].get<float>();
            robotConfig.slideMax = joints["slide_range"][1].get<float>();
        }
        
        robotConfig.motorMinTorqueLimit = joints.value("motor_min_torque", -500.0f);
        robotConfig.motorMaxTorqueLimit = joints.value("motor_max_torque", 500.0f);
    }

    // Load sensor configuration
    if (config.contains("sensors"))
    {
        const auto& sensors = config["sensors"];
        robotConfig.numLidarRays = sensors.value("lidar_rays", 10);
        robotConfig.lidarMaxDistance = sensors.value("lidar_max_distance", 20.0f);
    }

    // Load action configuration
    if (config.contains("actions"))
    {
        const auto& actions = config["actions"];
        robotConfig.actionsPerSatellite = actions.value("per_satellite", 4);
        robotConfig.reactionWheelDim = actions.value("reaction_wheel_dim", 4);
        robotConfig.rotationScale = actions.value("rotation_scale", 25.0f);
        robotConfig.slideScale = actions.value("slide_scale", 100.0f);
        robotConfig.reactionTorqueScale = actions.value("reaction_torque_scale", 5000.0f);
    }

    // Calculate dynamic dimensions
    robotConfig.CalculateDimensions();

    return robotConfig;
}

/**
 * @brief Calculate dynamic dimensions based on configuration
 */
void RobotConfig::CalculateDimensions()
{
    numSatellites = static_cast<int>(satellites.size());
    actionsPerRobot = numSatellites * actionsPerSatellite + reactionWheelDim;
    
    // Calculate actual required observation dimension based on robot configuration
    int actualObservationDim = 38 + (numSatellites * 12) + numLidarRays;
    
    // Pad to next multiple of 8 for SIMD alignment
    observationDim = ((actualObservationDim + 7) / 8) * 8;
}

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
    JPH::PhysicsSystem* physicsSystem,
    const JPH::RVec3& position,
    uint32_t envIndex,
    int robotIndex)
{
    auto loadStart = std::chrono::high_resolution_clock::now();
    std::cout << "[LoadRobot" << robotIndex << "] Start loading" << std::endl;
    
    // Force sequential loading to prevent Jolt memory allocator collisions 
    // and JPH::Ref counter corruption from concurrent thread execution.
    static std::mutex sLoadMutex;
    std::lock_guard<std::mutex> lock(sLoadMutex);
    
    CombatRobotData robotData;
    robotData.envIndex = envIndex;
    robotData.robotIndex = robotIndex;
    robotData.hp = 100.0f;
    robotData.totalEnergyUsed = 0.0f;
    robotData.collisionGroup = envIndex * 2 + robotIndex;
    
    // Initialize vector fields
    robotData.baseActions.resize(robotData.config.actionsPerRobot);
    robotData.residualActions.resize(robotData.config.actionsPerRobot);
    robotData.finalActions.resize(robotData.config.actionsPerRobot);
    robotData.observationBuffer.resize(robotData.config.observationDim);
    robotData.lidarDistances.resize(robotData.config.numLidarRays);

    std::ifstream file(configPath);
    if (!file.is_open())
    {
        std::cerr << "[JOLTrl] FATAL: Failed to open " << configPath << std::endl;
        return robotData;
    }

    json config;
    file >> config;

    // Load robot configuration
    robotData.config = RobotConfig::LoadFromJSON(config);

    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();

    JPH::ObjectLayer ghostLayer = Layers::MOVING_BASE + envIndex;

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
        position,
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

    std::cout << "[LoadRobot" << robotIndex << "] Step 5: Entering satellite loop" << std::endl;
    robotData.satellites.resize(robotData.config.numSatellites);
    
    for (int i = 0; i < robotData.config.numSatellites; ++i)
    {
        std::cout << "[LoadRobot" << robotIndex << "] Step 5." << i << ".1: Processing satellite " << i << std::endl;
        const RobotConfig::Satellite& satConfig = robotData.config.satellites[i];
        const float azimuth = JPH::DegreesToRadians(satConfig.offsetAngle);
        const float elevation = JPH::DegreesToRadians(satConfig.elevation);
        const float dist = satConfig.distance;
        
        JPH::RVec3 satPos = position + JPH::RVec3(
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
        rotSettings.mPosition1 = position;
        rotSettings.mPosition2 = position;
        
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
        physicsSystem->AddConstraint(robotData.satellites[i].rotationJoint);
        
        robotData.satellites[i].rotationJoint->SetMotorState(
            JPH::SixDOFConstraintSettings::EAxis::RotationX, JPH::EMotorState::Velocity);
        robotData.satellites[i].rotationJoint->SetMotorState(
            JPH::SixDOFConstraintSettings::EAxis::RotationY, JPH::EMotorState::Velocity);
        robotData.satellites[i].rotationJoint->SetMotorState(
            JPH::SixDOFConstraintSettings::EAxis::RotationZ, JPH::EMotorState::Velocity);

        // Create spike body using configuration
        JPH::CylinderShapeSettings spikeShapeSettings(
            robotData.config.spikeHalfHeight,
            robotData.config.spikeRadius,
            robotData.config.spikeConvexRadius
        );
        spikeShapeSettings.SetDensity(robotData.config.spikeMass / (3.14159f * 
            pow(robotData.config.spikeRadius, 2) * 2.0f * robotData.config.spikeHalfHeight));
        
        auto spikeResult = spikeShapeSettings.Create();
        if (spikeResult.HasError()) throw std::runtime_error("Spike Shape Error: " + std::string(spikeResult.GetError().c_str()));
        JPH::RefConst<JPH::Shape> spikeShape = spikeResult.Get();

        JPH::Vec3 direction = JPH::Vec3(
            std::cos(elevation) * std::cos(azimuth),
            std::sin(elevation),
            std::cos(elevation) * std::sin(azimuth)
        );
        
        JPH::Quat spikeRotation = JPH::Quat::sFromTo(JPH::Vec3::sAxisY(), direction);

        JPH::RVec3 spikePos = satPos + JPH::RVec3(direction * (satConfig.radius + robotData.config.spikeHalfHeight));

        JPH::BodyCreationSettings spikeSettings(
            spikeShape,
            spikePos,
            spikeRotation,
            JPH::EMotionType::Dynamic,
            ghostLayer
        );

        spikeSettings.mFriction = robotData.config.spikeFriction;
        spikeSettings.mRestitution = robotData.config.spikeRestitution;
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
        physicsSystem->AddConstraint(robotData.satellites[i].slideJoint);
        
        robotData.satellites[i].slideJoint->SetMotorState(JPH::EMotorState::Velocity);

         robotData.satellites[i].pidX = PIDController{200.0f, 5.0f, 50.0f, 0.0f, 0.0f};
        robotData.satellites[i].pidY = PIDController{200.0f, 5.0f, 50.0f, 0.0f, 0.0f};
        robotData.satellites[i].pidZ = PIDController{200.0f, 5.0f, 50.0f, 0.0f, 0.0f};
    }

    auto loadEnd = std::chrono::high_resolution_clock::now();
    auto loadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - loadStart).count();
    std::cout << "[LoadRobot" << robotIndex << "] Loaded in " << loadDuration << "ms" << std::endl;
    
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

    for (int i = 0; i < robot.config.numSatellites; ++i)
    {
        const RobotConfig::Satellite& satConfig = robot.config.satellites[i];
        const float azimuth = JPH::DegreesToRadians(satConfig.offsetAngle);
        const float elevation = JPH::DegreesToRadians(satConfig.elevation);
        const float distance = satConfig.distance;
        
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
    JPH::PhysicsSystem* physicsSystem,
    float dt)
{
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();

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
    JPH::PhysicsSystem* physicsSystem)
{
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    float energySum = 0.0f;

    for (int i = 0; i < robot.config.numSatellites; ++i)
    {
        // actions here are already blended and scaled if coming from ApplyResidualActions
        const float vx = actions[i * robot.config.actionsPerSatellite + 0];
        const float vy = actions[i * robot.config.actionsPerSatellite + 1];
        const float vz = actions[i * robot.config.actionsPerSatellite + 2];
        const float slideVel = actions[i * robot.config.actionsPerSatellite + 3];

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
    JPH::PhysicsSystem* physicsSystem)
{
    // 1. Store the residual actions from the model
    for (int i = 0; i < robot.config.actionsPerRobot; ++i)
    {
        robot.residualActions[i] = residualActions[i];
    }

    // 2. Compute the base stability actions (PID)
    // Using 120Hz control frequency to match viewer physicshz
    ComputeBasePIDActions(robot, physicsSystem, 1.0f / 120.0f);

    // 3. Blend and Scale
    BlendResidualWithBase(robot);

    // 4. Apply to Jolt
    ApplyActions(robot, robot.finalActions.data(), physicsSystem);
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
    JPH::PhysicsSystem* physicsSystem)
{
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    
    JPH::RVec3 rootPos = bodyInterface.GetPosition(robot.mainBodyId);
    JPH::Quat rootRot = bodyInterface.GetRotation(robot.mainBodyId);
    
    const float maxDistance = robot.config.lidarMaxDistance;
    
    const JPH::NarrowPhaseQuery& narrowPhaseQuery = physicsSystem->GetNarrowPhaseQuery();
    
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

    PerformLidarScan(robot, physicsSystem);
    for (int i = 0; i < robot.config.numLidarRays; ++i)
    {
        observations[idx++] = robot.lidarDistances[i] / robot.config.lidarMaxDistance;
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
    
    for (int i = 0; i < robot.config.numSatellites; ++i)
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