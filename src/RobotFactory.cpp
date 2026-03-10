#include "RobotFactory.h"
#include <Jolt/RegisterTypes.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/PointConstraint.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include "PhysicsCore.h"
#include <iostream>
#include <map>
#include <string>

JPH::Ref<JPH::GroupFilterTable> RobotFactory::mGroupFilter = nullptr;

namespace {
    JPH::RefConst<JPH::Shape> CreateShape(const RobotConfig::PartShape& p) {
        if (p.type == "sphere" && p.dimensions.size() >= 1) {
            return new JPH::SphereShape(p.dimensions[0]);
        } else if (p.type == "box" && p.dimensions.size() >= 3) {
            return new JPH::BoxShape(JPH::Vec3(p.dimensions[0], p.dimensions[1], p.dimensions[2]));
        } else if (p.type == "cylinder" && p.dimensions.size() >= 2) {
            return new JPH::CylinderShape(p.dimensions[0], p.dimensions[1]);
        } else if (p.type == "hull" && p.dimensions.size() >= 3 && (p.dimensions.size() % 3 == 0)) {
            std::vector<JPH::Vec3> points;
            for (size_t i = 0; i < p.dimensions.size(); i += 3) {
                points.emplace_back(p.dimensions[i], p.dimensions[i+1], p.dimensions[i+2]);
            }
            JPH::ConvexHullShapeSettings settings(points.data(), static_cast<int>(points.size()), 0.01f);
            auto res = settings.Create();
            if (res.HasError()) {
                std::cerr << "[RobotFactory] Hull creation failed: " << res.GetError() << " - Using box fallback." << std::endl;
                // Use the bounding box of the points to create a box that roughly matches
                return new JPH::BoxShape(JPH::Vec3(1.5f, 0.1f, 1.5f));
            }
            return res.Get();
        }
        return new JPH::SphereShape(0.5f);
    }
}

void RobotFactory::InitializeGroupFilter() {
    if (JPH::Factory::sInstance == nullptr) {
        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    }
    if (mGroupFilter == nullptr) {
        mGroupFilter = new JPH::GroupFilterTable(256);
        for (int i = 0; i < 256; ++i) {
            for (int j = 0; j < 256; ++j) {
                mGroupFilter->DisableCollision(i, j);
            }
        }
    }
}

Robot RobotFactory::CreateRobot(const RobotConfig& config, JPH::PhysicsSystem* physicsSystem, const JPH::RVec3& position, uint32_t envIndex, int robotIndex) {
    InitializeGroupFilter();
    Robot robot;
    robot.config = config;
    robot.envIndex = envIndex;
    robot.robotIndex = robotIndex;
    robot.collisionGroup = envIndex * 2 + robotIndex;
    robot.type = config.type;

    // OPTIMIZATION: Removed verbose logging
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    JPH::ObjectLayer movingLayer = Layers::MOVING_BASE + envIndex;

    // --- NEW: General Multi-Body Loader (Octopod-style) ---
    if (!config.bodies.empty()) {
        std::map<std::string, JPH::BodyID> bodyMap;
        
        for (const auto& bConfig : config.bodies) {
            JPH::RefConst<JPH::Shape> shape;
            if (bConfig.shapeType == "box" && bConfig.shapeParams.size() >= 3) {
                shape = new JPH::BoxShape(JPH::Vec3(bConfig.shapeParams[0], bConfig.shapeParams[1], bConfig.shapeParams[2]));
            } else if (bConfig.shapeType == "sphere" && !bConfig.shapeParams.empty()) {
                shape = new JPH::SphereShape(bConfig.shapeParams[0]);
            } else {
                shape = new JPH::BoxShape(JPH::Vec3(0.5f, 0.5f, 0.5f));
            }

            JPH::RVec3 bodyPos = position;
            if (bConfig.position.size() >= 3) {
                bodyPos = position + JPH::RVec3(bConfig.position[0], bConfig.position[1], bConfig.position[2]);
            }

            JPH::Quat bodyRot = JPH::Quat::sIdentity();
            if (bConfig.rotation.size() >= 4) {
                bodyRot = JPH::Quat(bConfig.rotation[0], bConfig.rotation[1], bConfig.rotation[2], bConfig.rotation[3]);
            }

            JPH::BodyCreationSettings bodySettings(shape, bodyPos, bodyRot, JPH::EMotionType::Dynamic, movingLayer);
            bodySettings.mMassPropertiesOverride.mMass = bConfig.mass;
            bodySettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            bodySettings.mFriction = bConfig.friction;
            bodySettings.mRestitution = bConfig.restitution;
            
            if (mGroupFilter) {
                bodySettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
                bodySettings.mCollisionGroup.SetGroupID(robot.collisionGroup);
            }

            JPH::BodyID bid = bodyInterface.CreateAndAddBody(bodySettings, JPH::EActivation::Activate);
            if (bid.IsInvalid()) {
                std::cerr << "[RobotFactory] ERROR: Failed to create body " << bConfig.name << std::endl;
                continue;
            }
            robot.bodyIds.push_back(bid);
            bodyMap[bConfig.name] = bid;

            if (robot.mainBodyId.IsInvalid() || bConfig.name.find("central") != std::string::npos || bConfig.name.find("body") != std::string::npos) {
                robot.mainBodyId = bid;
            }
        }

        for (const auto& jConfig : config.joints) {
            auto b1it = bodyMap.find(jConfig.body1);
            auto b2it = bodyMap.find(jConfig.body2);
            if (b1it == bodyMap.end() || b2it == bodyMap.end()) continue;

            if (jConfig.type == "hinge") {
                JPH::HingeConstraintSettings settings;
                settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                if (jConfig.position.size() >= 3) {
                    settings.mPoint1 = settings.mPoint2 = position + JPH::RVec3(jConfig.position[0], jConfig.position[1], jConfig.position[2]);
                }
                if (jConfig.axis.size() >= 3) {
                    settings.mHingeAxis1 = settings.mHingeAxis2 = JPH::Vec3(jConfig.axis[0], jConfig.axis[1], jConfig.axis[2]).Normalized();
                }
                settings.mLimitsMin = jConfig.minLimit;
                settings.mLimitsMax = jConfig.maxLimit;
                
                if (jConfig.hasMotor) {
                    settings.mMotorSettings.mMinTorqueLimit = -jConfig.motorMaxTorque;
                    settings.mMotorSettings.mMaxTorqueLimit = jConfig.motorMaxTorque;
                }

                auto* constraint = static_cast<JPH::HingeConstraint*>(bodyInterface.CreateConstraint(&settings, b1it->second, b2it->second));
                if (constraint) {
                    physicsSystem->AddConstraint(constraint);
                    robot.hingeJoints.push_back(constraint);
                    if (jConfig.hasMotor) constraint->SetMotorState(JPH::EMotorState::Velocity);
                } else {
                    std::cerr << "[RobotFactory] WARNING: Failed to create joint " << jConfig.name << std::endl;
                }
            } else if (jConfig.type == "point") {
                JPH::PointConstraintSettings settings;
                settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                if (jConfig.position.size() >= 3) {
                    settings.mPoint1 = settings.mPoint2 = position + JPH::RVec3(jConfig.position[0], jConfig.position[1], jConfig.position[2]);
                }
                auto* constraint = static_cast<JPH::PointConstraint*>(bodyInterface.CreateConstraint(&settings, b1it->second, b2it->second));
                if (constraint) physicsSystem->AddConstraint(constraint);
            } else if (jConfig.type == "sixdof") {
                JPH::SixDOFConstraintSettings settings;
                settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                if (jConfig.position.size() >= 3) {
                    settings.mPosition1 = settings.mPosition2 = position + JPH::RVec3(jConfig.position[0], jConfig.position[1], jConfig.position[2]);
                }
                // Fix all translations
                for (int i = 0; i < 3; ++i) settings.MakeFixedAxis((JPH::SixDOFConstraintSettings::EAxis)i);
                
                if (jConfig.hasMotor) {
                    for (int i = 3; i < 6; ++i) {
                        settings.mMotorSettings[i].mMinTorqueLimit = -jConfig.motorMaxTorque;
                        settings.mMotorSettings[i].mMaxTorqueLimit = jConfig.motorMaxTorque;
                    }
                }
                auto* constraint = static_cast<JPH::SixDOFConstraint*>(bodyInterface.CreateConstraint(&settings, b1it->second, b2it->second));
                if (constraint) {
                    physicsSystem->AddConstraint(constraint);
                    robot.sixDofJoints.push_back(constraint);
                    if (jConfig.hasMotor) {
                        for (int i = 3; i < 6; ++i) constraint->SetMotorState((JPH::SixDOFConstraintSettings::EAxis)i, JPH::EMotorState::Velocity);
                    }
                }
            }
        }
        return robot;
    }

    // --- OLD: Special Cases and Satellite Logic ---
    if (config.type == RobotType::INTERNAL_ENGINE) {
        std::cout << "[RobotFactory] Building Hardcoded Shard Robot (Internal Engine)..." << std::endl;
        
        // 1. Chassis (The flat square)
        std::cout << "[RobotFactory] Creating chassis shape..." << std::endl;
        JPH::BoxShapeSettings chassisShapeSettings(JPH::Vec3(1.0f, 0.1f, 1.0f), 0.01f);
        auto chassisShapeRes = chassisShapeSettings.Create();
        if (chassisShapeRes.HasError()) {
            std::cerr << "[RobotFactory] FATAL: Chassis shape creation failed: " << chassisShapeRes.GetError() << std::endl;
            return robot;
        }
        JPH::RefConst<JPH::Shape> chassisShape = chassisShapeRes.Get();
        
        JPH::BodyCreationSettings chassisSettings(chassisShape, position, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, movingLayer);
        if (mGroupFilter) {
            chassisSettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
            chassisSettings.mCollisionGroup.SetGroupID(robot.collisionGroup);
            chassisSettings.mCollisionGroup.SetSubGroupID(0);
        }
        chassisSettings.mMassPropertiesOverride.mMass = 1500.0f;
        chassisSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        std::cout << "[RobotFactory] Creating chassis body..." << std::endl;
        robot.mainBodyId = bodyInterface.CreateAndAddBody(chassisSettings, JPH::EActivation::Activate);
        if (robot.mainBodyId.IsInvalid()) {
            std::cerr << "[RobotFactory] FATAL: Chassis body creation failed!" << std::endl;
            return robot;
        }
        std::cout << "[RobotFactory] Chassis body created: " << robot.mainBodyId.GetIndex() << std::endl;

        robot.satellites.clear();
        
        // 2. Wings (Improved: 4 Scythe Wings)
        std::cout << "[RobotFactory] Creating wing shape (Box)..." << std::endl;
        JPH::BoxShapeSettings wsSettings(JPH::Vec3(0.5f, 0.05f, 1.2f), 0.01f);
        auto wingShapeRes = wsSettings.Create();
        if (wingShapeRes.HasError()) {
            std::cerr << "[RobotFactory] FATAL: Wing shape creation failed: " << wingShapeRes.GetError() << std::endl;
            return robot;
        }
        JPH::RefConst<JPH::Shape> wingShape = wingShapeRes.Get();

        for (int i = 0; i < 4; ++i) {
            std::cout << "[RobotFactory] Creating wing body " << i << "..." << std::endl;
            // Position wings at corners: FL, FR, BL, BR
            float sideX = (i % 2 == 0) ? -1.0f : 1.0f;
            float sideZ = (i < 2) ? 1.0f : -1.0f;
            JPH::RVec3 wingPos = position + JPH::RVec3(sideX * 1.5f, 0.0f, sideZ * 0.8f);
            
            JPH::BodyCreationSettings wingSettings(wingShape, wingPos, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, movingLayer);
            if (mGroupFilter) {
                wingSettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
                wingSettings.mCollisionGroup.SetGroupID(robot.collisionGroup);
                wingSettings.mCollisionGroup.SetSubGroupID(i + 1);
            }
            wingSettings.mMassPropertiesOverride.mMass = 500.0f;
            wingSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            
            SatelliteData sat;
            sat.coreBodyId = bodyInterface.CreateAndAddBody(wingSettings, JPH::EActivation::Activate);
            
            if (sat.coreBodyId.IsInvalid()) {
                std::cerr << "[RobotFactory] Wing body creation FAILED!" << std::endl;
                continue;
            }
            std::cout << "[RobotFactory] Wing body " << i << " added with ID: " << sat.coreBodyId.GetIndex() << std::endl;

            // 3. Hinges (Using SixDOF)
            std::cout << "[RobotFactory] Creating hinge " << i << "..." << std::endl;
            JPH::SixDOFConstraintSettings hinge;
            hinge.mSpace = JPH::EConstraintSpace::WorldSpace;
            hinge.mPosition1 = position + JPH::RVec3(sideX * 1.0f, 0.0f, sideZ * 0.8f);
            hinge.mPosition2 = hinge.mPosition1;
            
            for (int a = 0; a < 3; ++a) hinge.MakeFixedAxis((JPH::SixDOFConstraintSettings::EAxis)a);
            hinge.MakeFixedAxis(JPH::SixDOFConstraintSettings::EAxis::RotationX);
            hinge.MakeFixedAxis(JPH::SixDOFConstraintSettings::EAxis::RotationY);
            hinge.SetLimitedAxis(JPH::SixDOFConstraintSettings::EAxis::RotationZ, -0.8f, 0.8f);
            
            hinge.mMotorSettings[JPH::SixDOFConstraintSettings::EAxis::RotationZ].mMaxTorqueLimit = 50000.0f;
            hinge.mMotorSettings[JPH::SixDOFConstraintSettings::EAxis::RotationZ].mMinTorqueLimit = -50000.0f;
            
            sat.rotationJoint = static_cast<JPH::SixDOFConstraint*>(bodyInterface.CreateConstraint(&hinge, robot.mainBodyId, sat.coreBodyId));
            physicsSystem->AddConstraint(sat.rotationJoint);
            sat.rotationJoint->SetMotorState(JPH::SixDOFConstraintSettings::EAxis::RotationZ, JPH::EMotorState::Velocity);
            
            robot.satellites.push_back(sat);
            std::cout << "[RobotFactory] Hinge " << i << " created and added." << std::endl;
        }
        
        std::cout << "[RobotFactory] Shard robot creation complete." << std::endl;
        return robot;
    }

    if (config.isUnifiedBody) {
        JPH::StaticCompoundShapeSettings compoundSettings;
        if (!config.partShapes.empty()) {
            for (const auto& p : config.partShapes) {
                JPH::Vec3 pos = p.relativePos.size() >= 3 ? JPH::Vec3(p.relativePos[0], p.relativePos[1], p.relativePos[2]) : JPH::Vec3::sZero();
                JPH::Quat rot = p.relativeRot.size() >= 4 ? JPH::Quat(p.relativeRot[0], p.relativeRot[1], p.relativeRot[2], p.relativeRot[3]) : JPH::Quat::sIdentity();
                compoundSettings.AddShape(pos, rot, CreateShape(p));
            }
        } else {
            compoundSettings.AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), new JPH::CylinderShape(2.0f, 0.6f));
        }

                auto res = compoundSettings.Create();
        if (res.HasError()) {
            std::cerr << "[RobotFactory] FATAL: Compound Shape creation failed: " << res.GetError() << std::endl;
            for (const auto& p : config.partShapes) {
                std::cerr << "  - Part Type: " << p.type << " at pos [" << p.relativePos[0] << ", " << p.relativePos[1] << ", " << p.relativePos[2] << "]" << std::endl;
            }
            return robot; // Returns invalid robot
        }
        if (res.HasError()) {
            std::cerr << "[RobotFactory] Compound Shape Error: " << res.GetError() << std::endl;
            return robot;
        }

        JPH::BodyCreationSettings jetSettings(res.Get(), position, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, movingLayer);
        jetSettings.mMassPropertiesOverride.mMass = config.coreMass;
        jetSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        jetSettings.mFriction = config.coreFriction;
        jetSettings.mRestitution = config.coreRestitution;
        jetSettings.mLinearDamping = config.coreLinearDamping;
        jetSettings.mAngularDamping = config.coreAngularDamping;
        
        robot.mainBodyId = bodyInterface.CreateAndAddBody(jetSettings, JPH::EActivation::Activate);
    } else {
        JPH::SphereShapeSettings coreShapeSettings(config.coreRadius);
        coreShapeSettings.SetDensity(config.coreMass / (4.0f / 3.0f * 3.14159f * pow(config.coreRadius, 3)));
        JPH::BodyCreationSettings coreSettings(coreShapeSettings.Create().Get(), position, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, movingLayer);
        coreSettings.mFriction = config.coreFriction;
        coreSettings.mRestitution = config.coreRestitution;
        coreSettings.mLinearDamping = config.coreLinearDamping;
        coreSettings.mAngularDamping = config.coreAngularDamping;
        coreSettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
        coreSettings.mCollisionGroup.SetGroupID(robot.collisionGroup);
        coreSettings.mCollisionGroup.SetSubGroupID(0);
        robot.mainBodyId = bodyInterface.CreateAndAddBody(coreSettings, JPH::EActivation::Activate);

        robot.satellites.resize(config.numSatellites);
        for (int i = 0; i < config.numSatellites; ++i) {
            const auto& satConfig = config.satellites[i];
            float azimuth = JPH::DegreesToRadians(satConfig.offsetAngle), elevation = JPH::DegreesToRadians(satConfig.elevation), dist = satConfig.distance;
            JPH::RVec3 satPos = position + JPH::RVec3(dist * std::cos(elevation) * std::cos(azimuth), dist * std::sin(elevation), dist * std::cos(elevation) * std::sin(azimuth));
            JPH::SphereShapeSettings satShapeSettings(satConfig.radius);
            satShapeSettings.SetDensity(satConfig.mass / (4.0f / 3.0f * 3.14159f * pow(satConfig.radius, 3)));
            JPH::BodyCreationSettings satSettings(satShapeSettings.Create().Get(), satPos, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, movingLayer);
            satSettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
            satSettings.mCollisionGroup.SetGroupID(robot.collisionGroup);
            satSettings.mCollisionGroup.SetSubGroupID(i + 1);
            robot.satellites[i].coreBodyId = bodyInterface.CreateAndAddBody(satSettings, JPH::EActivation::Activate);

            JPH::SixDOFConstraintSettings rotSettings;
            rotSettings.mSpace = JPH::EConstraintSpace::WorldSpace;
            rotSettings.mPosition1 = position; rotSettings.mPosition2 = position;
            for (int j = 0; j < 3; ++j) rotSettings.MakeFixedAxis((JPH::SixDOFConstraintSettings::EAxis)j);
            for (int j = 3; j < 6; ++j) {
                rotSettings.mMotorSettings[j].mMinTorqueLimit = config.motorMinTorqueLimit;
                rotSettings.mMotorSettings[j].mMaxTorqueLimit = config.motorMaxTorqueLimit;
            }
            robot.satellites[i].rotationJoint = static_cast<JPH::SixDOFConstraint*>(bodyInterface.CreateConstraint(&rotSettings, robot.mainBodyId, robot.satellites[i].coreBodyId));
            physicsSystem->AddConstraint(robot.satellites[i].rotationJoint);
            for (int axis = 0; axis < 3; ++axis) robot.satellites[i].rotationJoint->SetMotorState((JPH::SixDOFConstraintSettings::EAxis)(axis+3), JPH::EMotorState::Velocity);

            JPH::CylinderShapeSettings spikeShapeSettings(config.spike.halfHeight, config.spike.radius, config.spike.convexRadius);
            spikeShapeSettings.SetDensity(config.spike.mass / (3.14159f * pow(config.spike.radius, 2) * 2.0f * config.spike.halfHeight));
            JPH::Vec3 dir(std::cos(elevation) * std::cos(azimuth), std::sin(elevation), std::cos(elevation) * std::sin(azimuth));
            JPH::RVec3 spikePos = satPos + JPH::RVec3(dir * (satConfig.radius + config.spike.halfHeight));
            JPH::BodyCreationSettings spikeSettings(spikeShapeSettings.Create().Get(), spikePos, JPH::Quat::sFromTo(JPH::Vec3::sAxisY(), dir), JPH::EMotionType::Dynamic, movingLayer);
            spikeSettings.mCollisionGroup.SetGroupFilter(mGroupFilter);
            spikeSettings.mCollisionGroup.SetGroupID(robot.collisionGroup);
            spikeSettings.mCollisionGroup.SetSubGroupID(config.numSatellites + i + 1);
            robot.satellites[i].spikeBodyId = bodyInterface.CreateAndAddBody(spikeSettings, JPH::EActivation::Activate);

            JPH::SliderConstraintSettings slideSettings;
            slideSettings.SetSliderAxis(dir);
            slideSettings.mLimitsMin = config.slideMin; slideSettings.mLimitsMax = config.slideMax;
            slideSettings.mMotorSettings.mMinForceLimit = -10000.0f; slideSettings.mMotorSettings.mMaxForceLimit = 10000.0f;
            robot.satellites[i].slideJoint = static_cast<JPH::SliderConstraint*>(bodyInterface.CreateConstraint(&slideSettings, robot.satellites[i].coreBodyId, robot.satellites[i].spikeBodyId));
            physicsSystem->AddConstraint(robot.satellites[i].slideJoint);
            robot.satellites[i].slideJoint->SetMotorState(JPH::EMotorState::Velocity);
        }
    }
    return robot;
}

void RobotFactory::ResetRobot(Robot& robot, JPH::PhysicsSystem* physicsSystem, const JPH::RVec3& spawnPosition) {
    if (robot.mainBodyId.IsInvalid()) {
        std::cerr << "[ResetRobot] Robot has invalid main body ID – skipping reset." << std::endl;
        return;
    }
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface(); 
    robot.hp = 100.0f;
    robot.totalDamageDealt = 0.0f;
    robot.totalDamageTaken = 0.0f;
    robot.totalEnergyUsed = 0.0f;
    robot.episodeSteps = 0;

    if (!robot.bodyIds.empty()) {
        for (size_t i = 0; i < robot.bodyIds.size(); ++i) {
            JPH::RVec3 relPos = JPH::RVec3::sZero();
            if (i < robot.config.bodies.size() && robot.config.bodies[i].position.size() >= 3) {
                relPos = JPH::RVec3(robot.config.bodies[i].position[0], robot.config.bodies[i].position[1], robot.config.bodies[i].position[2]);
            }
            JPH::Quat relRot = JPH::Quat::sIdentity();
            if (i < robot.config.bodies.size() && robot.config.bodies[i].rotation.size() >= 4) {
                relRot = JPH::Quat(robot.config.bodies[i].rotation[0], robot.config.bodies[i].rotation[1], robot.config.bodies[i].rotation[2], robot.config.bodies[i].rotation[3]);
            }
            bodyInterface.SetPositionAndRotation(robot.bodyIds[i], spawnPosition + relPos, relRot, JPH::EActivation::Activate);
            bodyInterface.SetLinearAndAngularVelocity(robot.bodyIds[i], JPH::Vec3::sZero(), JPH::Vec3::sZero());
        }
    } else {
        bodyInterface.SetPositionAndRotation(robot.mainBodyId, spawnPosition, JPH::Quat::sIdentity(), JPH::EActivation::Activate);
        bodyInterface.SetLinearAndAngularVelocity(robot.mainBodyId, JPH::Vec3::sZero(), JPH::Vec3::sZero());
        
        for (int i = 0; i < (int)robot.satellites.size(); ++i) {
            if (!robot.satellites[i].coreBodyId.IsInvalid()) {
                if (robot.type == RobotType::INTERNAL_ENGINE) {
                     float sideX = (i % 2 == 0) ? -1.0f : 1.0f;
                     float sideZ = (i < 2) ? 1.0f : -1.0f;
                     bodyInterface.SetPositionAndRotation(robot.satellites[i].coreBodyId, spawnPosition + JPH::RVec3(sideX * 1.5f, 0.0f, sideZ * 0.8f), JPH::Quat::sIdentity(), JPH::EActivation::Activate);
                }
                bodyInterface.SetLinearAndAngularVelocity(robot.satellites[i].coreBodyId, JPH::Vec3::sZero(), JPH::Vec3::sZero());
            }
            if (!robot.satellites[i].spikeBodyId.IsInvalid()) {
                bodyInterface.SetLinearAndAngularVelocity(robot.satellites[i].spikeBodyId, JPH::Vec3::sZero(), JPH::Vec3::sZero());
            }
        }
    }
}
