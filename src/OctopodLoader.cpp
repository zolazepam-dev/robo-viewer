/**
 * OctopodLoader - Loads octopod.json with native Jolt format
 * 
 * Direct mapping from JSON to Jolt Physics API.
 * Supports the octopod robot with 25 bodies and 24 hinge constraints.
 */

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "PhysicsCore.h"

using json = nlohmann::json;

namespace OctopodLoader {

// Parse JPH::Vec3 from JSON array
JPH::Vec3 ParseVec3(const json& j) {
    if (!j.is_array() || j.size() < 3) return JPH::Vec3::sZero();
    return JPH::Vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
}

// Parse JPH::Quat from JSON array [x, y, z, w]
JPH::Quat ParseQuat(const json& j) {
    if (!j.is_array() || j.size() < 4) return JPH::Quat::sIdentity();
    return JPH::Quat(j[0].get<float>(), j[1].get<float>(), j[2].get<float>(), j[3].get<float>());
}

// Create shape from native JSON shape definition
JPH::Ref<JPH::Shape> CreateShape(const json& shapeJson) {
    std::string shapeType = shapeJson.value("type", "box");
    
    if (shapeType == "box") {
        auto halfExtents = ParseVec3(shapeJson.at("halfExtents"));
        float convexRadius = shapeJson.value("convexRadius", 0.0f);
        JPH::BoxShapeSettings settings(halfExtents, convexRadius);
        auto result = settings.Create();
        if (result.HasError()) {
            std::cerr << "OctopodLoader: Box shape creation failed" << std::endl;
            return nullptr;
        }
        return result.Get();
    }
    else if (shapeType == "sphere") {
        float radius = shapeJson.at("radius").get<float>();
        JPH::SphereShapeSettings settings(radius);
        auto result = settings.Create();
        return result.Get();
    }
    else if (shapeType == "capsule") {
        float halfHeight = shapeJson.at("halfHeight").get<float>();
        float radius = shapeJson.at("radius").get<float>();
        JPH::CapsuleShapeSettings settings(halfHeight, radius);
        auto result = settings.Create();
        return result.Get();
    }
    
    std::cerr << "OctopodLoader: Unsupported shape type: " << shapeType << std::endl;
    return nullptr;
}

struct LoadedOctopod {
    std::vector<JPH::BodyID> bodies;
    std::vector<JPH::HingeConstraint*> constraints;
    std::map<std::string, JPH::BodyID> bodyMap;
    JPH::BodyID centralBody;
};

LoadedOctopod LoadOctopod(
    const std::string& configPath,
    JPH::PhysicsSystem* physicsSystem,
    const JPH::RVec3& spawnPosition,
    uint32_t envIndex,
    JPH::GroupFilter* groupFilter)
{
    LoadedOctopod octopod;
    
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "OctopodLoader: Failed to open " << configPath << std::endl;
        return octopod;
    }
    
    json data;
    try {
        file >> data;
    } catch (const std::exception& e) {
        std::cerr << "OctopodLoader: JSON parse error: " << e.what() << std::endl;
        return octopod;
    }
    
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    
    std::cout << "[OctopodLoader] Loading " << data.at("bodies").size() << " bodies..." << std::endl;
    
    // Load bodies
    for (const auto& bodyJson : data.at("bodies")) {
        std::string name = bodyJson.value("name", "");
        if (name.empty()) continue;
        
        // Parse position with spawn offset
        JPH::RVec3 position = spawnPosition;
        if (bodyJson.contains("position")) {
            auto p = ParseVec3(bodyJson.at("position"));
            position = spawnPosition + JPH::RVec3(p.GetX(), p.GetY(), p.GetZ());
        }
        
        // Parse rotation (quaternion)
        JPH::Quat rotation = JPH::Quat::sIdentity();
        if (bodyJson.contains("rotation")) {
            rotation = ParseQuat(bodyJson.at("rotation"));
        }
        
        // Create shape
        if (!bodyJson.contains("shape")) continue;
        JPH::Ref<JPH::Shape> shape = CreateShape(bodyJson.at("shape"));
        if (!shape) continue;
        
        // Parse motion type and layer
        std::string typeStr = bodyJson.value("type", "dynamic");
        JPH::EMotionType motionType = (typeStr == "static") ? JPH::EMotionType::Static : JPH::EMotionType::Dynamic;
        JPH::ObjectLayer layer = (motionType == JPH::EMotionType::Static) ? Layers::STATIC : (Layers::MOVING_BASE + envIndex);
        
        // Create body settings - DIRECT JOLT API MAPPING
        if (!shape) {
            std::cerr << "OctopodLoader: Failed to create shape for body " << name << std::endl;
            continue;
        }
        JPH::BodyCreationSettings bodySettings(shape, position, rotation, motionType, layer);
        
        // Apply mass for dynamic bodies
        if (motionType == JPH::EMotionType::Dynamic && bodyJson.contains("mass")) {
            float mass = bodyJson.at("mass").get<float>();
            bodySettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            bodySettings.mMassPropertiesOverride.mMass = mass;
        }
        
        // Apply material properties - DIRECT JOLT API MAPPING
        if (bodyJson.contains("material")) {
            const auto& mat = bodyJson.at("material");
            bodySettings.mFriction = mat.value("friction", 0.5f);
            bodySettings.mRestitution = mat.value("restitution", 0.0f);
            bodySettings.mLinearDamping = mat.value("linearDamping", 0.0f);
            bodySettings.mAngularDamping = mat.value("angularDamping", 0.0f);
        }
        
        // Apply motion quality
        if (bodyJson.contains("motionQuality")) {
            std::string quality = bodyJson.at("motionQuality");
            if (quality == "linear_cast") {
                bodySettings.mMotionQuality = JPH::EMotionQuality::LinearCast;
            }
        }
        
        // Apply allowSleeping
        if (bodyJson.contains("allowSleeping")) {
            bodySettings.mAllowSleeping = bodyJson.at("allowSleeping").get<bool>();
        }
        
    // Set collision group to prevent self-collision
    bodySettings.mCollisionGroup.SetGroupID(envIndex * 100 + 1);  // Robot group
    
    // Use the global group filter from PhysicsCore if available
    if (groupFilter) {
        bodySettings.mCollisionGroup.SetGroupFilter(groupFilter);
    }

    // Disable collision between directly connected parts
    if (name == "body_central" || name.find("coxa") != std::string::npos) {
        bodySettings.mCollisionGroup.SetSubGroupID(0); // Body and Coxas don't collide
    } else {
        bodySettings.mCollisionGroup.SetSubGroupID(1); // Femur and Tibia CAN collide with body
    }
        
        // Create and add body
        JPH::Body* body = bodyInterface.CreateBody(bodySettings);
        if (!body) continue;
        
        bodyInterface.AddBody(body->GetID(), 
                              motionType != JPH::EMotionType::Static ? 
                                  JPH::EActivation::Activate : JPH::EActivation::DontActivate);
        
        octopod.bodies.push_back(body->GetID());
        octopod.bodyMap[name] = body->GetID();
        
        // Track central body
        if (name.find("central") != std::string::npos || name.find("base") != std::string::npos) {
            octopod.centralBody = body->GetID();
        }
        
        if (octopod.bodies.size() <= 3) {
            std::cout << "  Body: " << name << " at (" 
                      << position.GetX() << ", " << position.GetY() << ", " << position.GetZ() << ")" << std::endl;
        }
    }
    
    std::cout << "[OctopodLoader] Loaded " << octopod.bodies.size() << " bodies" << std::endl;
    
    // Load constraints (hinge joints)
    if (data.contains("constraints")) {
        std::cout << "[OctopodLoader] Loading " << data.at("constraints").size() << " constraints..." << std::endl;
        
        for (const auto& cJson : data.at("constraints")) {
            std::string type = cJson.value("type", "hinge");
            std::string name = cJson.value("name", "");
            std::string body1Name = cJson.value("body1", "");
            std::string body2Name = cJson.value("body2", "");
            
            auto b1it = octopod.bodyMap.find(body1Name);
            auto b2it = octopod.bodyMap.find(body2Name);
            
            if (b1it == octopod.bodyMap.end() || b2it == octopod.bodyMap.end()) {
                std::cerr << "  Constraint '" << name << "': bodies not found" << std::endl;
                continue;
            }
            
            if (type == "hinge") {
                JPH::HingeConstraintSettings settings;
                settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                
                // Constraint position
                if (cJson.contains("position")) {
                    auto p = ParseVec3(cJson.at("position"));
                    settings.mPoint1 = settings.mPoint2 = spawnPosition + JPH::RVec3(p.GetX(), p.GetY(), p.GetZ());
                }
                
                // Hinge axis
                if (cJson.contains("hingeAxis")) {
                    JPH::Vec3 axis = ParseVec3(cJson.at("hingeAxis")).Normalized();
                    settings.mHingeAxis1 = axis;
                    settings.mHingeAxis2 = axis;
                    settings.mNormalAxis1 = axis.GetNormalizedPerpendicular();
                    settings.mNormalAxis2 = settings.mNormalAxis1;
                }
                
                // Limits
                if (cJson.contains("limits")) {
                    settings.mLimitsMin = cJson.at("limits").value("min", -JPH::JPH_PI);
                    settings.mLimitsMax = cJson.at("limits").value("max", JPH::JPH_PI);
                }
                
                // Motor settings for control
                if (cJson.contains("motor")) {
                    const auto& motor = cJson.at("motor");
                    settings.mMotorSettings.mSpringSettings.mFrequency = motor.value("frequency", 10.0f);
                    settings.mMotorSettings.mSpringSettings.mDamping = motor.value("damping", 1.0f);
                    settings.mMotorSettings.mMinTorqueLimit = motor.value("minForce", -100.0f);
                    settings.mMotorSettings.mMaxTorqueLimit = motor.value("maxForce", 100.0f);
                } else {
                    // Default motor settings for RL control
                    settings.mMotorSettings.mSpringSettings.mFrequency = 5.0f;
                    settings.mMotorSettings.mSpringSettings.mDamping = 2.0f;
                    settings.mMotorSettings.mMinTorqueLimit = -20.0f;
                    settings.mMotorSettings.mMaxTorqueLimit = 20.0f;
                }
                
                JPH::TwoBodyConstraint* constraint = bodyInterface.CreateConstraint(
                    &settings, b1it->second, b2it->second);
                
                if (constraint) {
                    physicsSystem->AddConstraint(constraint);
                    octopod.constraints.push_back(static_cast<JPH::HingeConstraint*>(constraint));
                }
            }
        }
        
        std::cout << "[OctopodLoader] Loaded " << octopod.constraints.size() << " constraints" << std::endl;
    }
    
    // Apply initial state if specified
    if (data.contains("initialState")) {
        const auto& state = data.at("initialState");
        
        if (!octopod.centralBody.IsInvalid() && state.contains("position")) {
            auto p = ParseVec3(state.at("position"));
            bodyInterface.SetPosition(octopod.centralBody, spawnPosition + JPH::RVec3(p.GetX(), p.GetY(), p.GetZ()), JPH::EActivation::Activate);
        }
        
        if (!octopod.centralBody.IsInvalid() && state.contains("rotation")) {
            bodyInterface.SetRotation(octopod.centralBody, ParseQuat(state.at("rotation")), JPH::EActivation::Activate);
        }
    }
    
    return octopod;
}

// Apply motor actions to hinge constraints (for RL)
void ApplyMotorActions(
    std::vector<JPH::HingeConstraint*>& constraints,
    const std::vector<float>& actions,
    float actionScale = 1.0f)
{
    for (size_t i = 0; i < constraints.size() && i < actions.size(); i++) {
        if (!constraints[i]) continue;
        
        float targetAngle = actions[i] * actionScale;
        
        constraints[i]->SetMotorState(JPH::EMotorState::Position);
        constraints[i]->SetTargetAngle(targetAngle);
    }
}

// Get observations from octopod state
void GetObservations(
    const LoadedOctopod& octopod,
    JPH::PhysicsSystem* physicsSystem,
    float* observations,
    int obsDim)
{
    if (observations == nullptr || obsDim <= 0) return;
    if (octopod.centralBody.IsInvalid()) {
        std::memset(observations, 0, obsDim * sizeof(float));
        return;
    }
    
    JPH::BodyInterface& bi = physicsSystem->GetBodyInterface();
    
    // Central body position and velocity
    JPH::RVec3 pos = bi.GetPosition(octopod.centralBody);
    JPH::Vec3 vel = bi.GetLinearVelocity(octopod.centralBody);
    JPH::Vec3 angVel = bi.GetAngularVelocity(octopod.centralBody);
    JPH::Quat rot = bi.GetRotation(octopod.centralBody);
    
    int idx = 0;
    
    // Position (3)
    observations[idx++] = pos.GetX();
    observations[idx++] = pos.GetY();
    observations[idx++] = pos.GetZ();
    
    // Linear velocity (3)
    observations[idx++] = vel.GetX();
    observations[idx++] = vel.GetY();
    observations[idx++] = vel.GetZ();
    
    // Angular velocity (3)
    observations[idx++] = angVel.GetX();
    observations[idx++] = angVel.GetY();
    observations[idx++] = angVel.GetZ();
    
    // Rotation as quaternion (4)
    observations[idx++] = rot.GetX();
    observations[idx++] = rot.GetY();
    observations[idx++] = rot.GetZ();
    observations[idx++] = rot.GetW();
    
    // Fill remaining with zeros
    while (idx < obsDim) {
        observations[idx++] = 0.0f;
    }
}

} // namespace OctopodLoader
