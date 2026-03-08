/**
 * RobotLoader - Universal Robot JSON Loader
 * 
 * Supports two JSON formats:
 * 1. LEGACY: Simple format (test_bot.json) - flat shape strings, Euler angles
 * 2. NATIVE: Jolt-native format (octopod.json) - nested shapes, quaternions, direct API mapping
 */

#include <Jolt/Jolt.h>
#include "RobotLoader.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <sstream>

#include <nlohmann/json.hpp>

#include <Jolt/Math/Math.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/PointConstraint.h>
#include <Jolt/Physics/Constraints/FixedConstraint.h>

#include "PhysicsCore.h"

namespace NativeLoader {

// Parse JPH::Vec3 from JSON array
JPH::Vec3 ParseVec3(const nlohmann::json& j) {
    if (!j.is_array() || j.size() < 3) return JPH::Vec3::sZero();
    return JPH::Vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
}

// Parse JPH::Quat from JSON array [x, y, z, w]
JPH::Quat ParseQuat(const nlohmann::json& j) {
    if (!j.is_array() || j.size() < 4) return JPH::Quat::sIdentity();
    return JPH::Quat(j[0].get<float>(), j[1].get<float>(), j[2].get<float>(), j[3].get<float>());
}

// Parse JPH::EMotionType from string
JPH::EMotionType ParseMotionType(const std::string& type) {
    if (type == "static") return JPH::EMotionType::Static;
    if (type == "kinematic") return JPH::EMotionType::Kinematic;
    return JPH::EMotionType::Dynamic;
}

// Parse JPH::EMotionQuality from string
JPH::EMotionQuality ParseMotionQuality(const std::string& quality) {
    if (quality == "linear_cast") return JPH::EMotionQuality::LinearCast;
    return JPH::EMotionQuality::Discrete;
}

// Create shape from native JSON shape definition
JPH::Ref<JPH::Shape> CreateShape(const nlohmann::json& shapeJson) {
    std::string shapeType = shapeJson.value("type", "box");
    
    if (shapeType == "box") {
        auto halfExtents = ParseVec3(shapeJson.at("halfExtents"));
        float convexRadius = shapeJson.value("convexRadius", 0.0f);
        JPH::BoxShapeSettings settings(halfExtents, convexRadius);
        auto result = settings.Create();
        if (result.HasError()) {
            std::cerr << "NativeLoader: Box shape creation failed" << std::endl;
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
    else if (shapeType == "cylinder") {
        float halfHeight = shapeJson.at("halfHeight").get<float>();
        float radius = shapeJson.at("radius").get<float>();
        JPH::CylinderShapeSettings settings(halfHeight, radius);
        auto result = settings.Create();
        return result.Get();
    }
    
    std::cerr << "NativeLoader: Unsupported shape type: " << shapeType << std::endl;
    return nullptr;
}

// Load bodies from native JSON format
void LoadBodies(const nlohmann::json& data, 
                JPH::BodyInterface& bodyInterface,
                std::map<std::string, JPH::BodyID>& bodyMap,
                RobotData& outData) {
    
    const auto& bodies = data.at("bodies");
    
    for (const auto& bodyJson : bodies) {
        std::string name = bodyJson.value("name", "");
        if (name.empty()) continue;
        
        JPH::RVec3 position;
        if (bodyJson.contains("position")) {
            JPH::Vec3 pos = ParseVec3(bodyJson.at("position"));
            position = JPH::RVec3(pos.GetX(), pos.GetY(), pos.GetZ());
        }
        
        JPH::Quat rotation = JPH::Quat::sIdentity();
        if (bodyJson.contains("rotation")) {
            rotation = ParseQuat(bodyJson.at("rotation"));
        }
        
        JPH::Ref<JPH::Shape> shape = CreateShape(bodyJson.at("shape"));
        if (!shape) continue;
        
        std::string typeStr = bodyJson.value("type", "dynamic");
        JPH::EMotionType motionType = ParseMotionType(typeStr);
        JPH::ObjectLayer layer = (motionType == JPH::EMotionType::Static) ? 
                                  Layers::STATIC : Layers::MOVING_BASE;
        
        JPH::BodyCreationSettings bodySettings(shape, position, rotation, motionType, layer);
        
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
        
        if (bodyJson.contains("motionQuality")) {
            bodySettings.mMotionQuality = ParseMotionQuality(bodyJson.at("motionQuality"));
        }
        
        if (bodyJson.contains("allowSleeping")) {
            bodySettings.mAllowSleeping = bodyJson.at("allowSleeping").get<bool>();
        }
        
        JPH::Body* body = bodyInterface.CreateBody(bodySettings);
        if (!body) continue;
        
        bodyInterface.AddBody(body->GetID(), 
                              motionType != JPH::EMotionType::Static ? 
                                  JPH::EActivation::Activate : JPH::EActivation::DontActivate);
        
        bodyMap[name] = body->GetID();
        outData.bodies.push_back(body->GetID());
        
        std::cout << "NativeLoader: Created body '" << name << "'" << std::endl;
    }
}

// Load constraints from native JSON format
void LoadConstraints(const nlohmann::json& data,
                     JPH::BodyInterface& bodyInterface,
                     JPH::PhysicsSystem& physicsSystem,
                     const std::map<std::string, JPH::BodyID>& bodyMap,
                     RobotData& outData) {
    
    nlohmann::json::const_iterator it;
    if (data.contains("constraints")) it = data.find("constraints");
    else if (data.contains("joints")) it = data.find("joints");
    else return;
    
    for (const auto& cJson : *it) {
        std::string type = cJson.value("type", "hinge");
        std::string name = cJson.value("name", "");
        std::string body1Name = cJson.value("body1", "");
        std::string body2Name = cJson.value("body2", "");
        
        auto b1it = bodyMap.find(body1Name);
        auto b2it = bodyMap.find(body2Name);
        if (b1it == bodyMap.end() || b2it == bodyMap.end()) continue;
        
        JPH::RVec3 position;
        if (cJson.contains("position")) {
            JPH::Vec3 pos = ParseVec3(cJson.at("position"));
            position = JPH::RVec3(pos.GetX(), pos.GetY(), pos.GetZ());
        }
        
        JPH::TwoBodyConstraint* constraint = nullptr;
        
        if (type == "hinge") {
            JPH::HingeConstraintSettings settings;
            settings.mSpace = JPH::EConstraintSpace::WorldSpace;
            settings.mPoint1 = position;
            settings.mPoint2 = position;
            
            if (cJson.contains("hingeAxis")) {
                JPH::Vec3 axis = ParseVec3(cJson.at("hingeAxis")).Normalized();
                settings.mHingeAxis1 = axis;
                settings.mHingeAxis2 = axis;
                settings.mNormalAxis1 = axis.GetNormalizedPerpendicular();
                settings.mNormalAxis2 = settings.mNormalAxis1;
            }
            
            if (cJson.contains("limits")) {
                const auto& limits = cJson.at("limits");
                settings.mLimitsMin = limits.value("min", -JPH::JPH_PI);
                settings.mLimitsMax = limits.value("max", JPH::JPH_PI);
            }
            
            // Motor settings - DIRECT JOLT API MAPPING
            if (cJson.contains("motor")) {
                const auto& motor = cJson.at("motor");
                settings.mMotorSettings.mSpringSettings.mFrequency = motor.value("frequency", 10.0f);
                settings.mMotorSettings.mSpringSettings.mDamping = motor.value("damping", 1.0f);
                settings.mMotorSettings.mMinTorqueLimit = motor.value("minForce", -100.0f);
                settings.mMotorSettings.mMaxTorqueLimit = motor.value("maxForce", 100.0f);
            }
            
            constraint = bodyInterface.CreateConstraint(&settings, b1it->second, b2it->second);
            
            // Apply motor state
            if (constraint && cJson.contains("motor")) {
                JPH::HingeConstraint* hinge = static_cast<JPH::HingeConstraint*>(constraint);
                const auto& motor = cJson.at("motor");
                std::string motorType = motor.value("type", "velocity");
                
                if (motorType == "position") {
                    hinge->SetMotorState(JPH::EMotorState::Position);
                    hinge->SetTargetAngle(motor.value("targetPosition", 0.0f));
                } else {
                    hinge->SetMotorState(JPH::EMotorState::Velocity);
                    hinge->SetTargetAngularVelocity(motor.value("targetVelocity", 0.0f));
                }
            }
        }
        else if (type == "slider") {
            JPH::SliderConstraintSettings settings;
            settings.mSpace = JPH::EConstraintSpace::WorldSpace;
            settings.mPoint1 = position;
            settings.mPoint2 = position;
            if (cJson.contains("sliderAxis")) {
                settings.mSliderAxis1 = ParseVec3(cJson.at("sliderAxis")).Normalized();
                settings.mSliderAxis2 = settings.mSliderAxis1;
            }
            constraint = bodyInterface.CreateConstraint(&settings, b1it->second, b2it->second);
        }
        else if (type == "point") {
            JPH::PointConstraintSettings settings;
            settings.mSpace = JPH::EConstraintSpace::WorldSpace;
            settings.mPoint1 = position;
            settings.mPoint2 = position;
            constraint = bodyInterface.CreateConstraint(&settings, b1it->second, b2it->second);
        }
        else if (type == "fixed") {
            JPH::FixedConstraintSettings settings;
            settings.mSpace = JPH::EConstraintSpace::WorldSpace;
            settings.mPoint1 = position;
            settings.mPoint2 = position;
            constraint = bodyInterface.CreateConstraint(&settings, b1it->second, b2it->second);
        }
        
        if (constraint) {
            physicsSystem.AddConstraint(constraint);
            outData.constraints.push_back(constraint);
            std::cout << "NativeLoader: Created constraint '" << name << "' (" << type << ")" << std::endl;
        }
    }
}

// Apply initial state
void ApplyInitialState(const nlohmann::json& data,
                       JPH::BodyInterface& bodyInterface,
                       const std::map<std::string, JPH::BodyID>& bodyMap) {
    if (!data.contains("initialState") || bodyMap.empty()) return;
    
    const auto& state = data.at("initialState");
    JPH::BodyID rootBody = bodyMap.begin()->second;
    
    // Find root/central body
    for (const auto& [name, id] : bodyMap) {
        if (name.find("central") != std::string::npos || name.find("base") != std::string::npos) {
            rootBody = id;
            break;
        }
    }
    
    if (state.contains("position")) {
        JPH::Vec3 pos = ParseVec3(state.at("position"));
        bodyInterface.SetPosition(rootBody, JPH::RVec3(pos.GetX(), pos.GetY(), pos.GetZ()), JPH::EActivation::Activate);
    }
    if (state.contains("rotation")) {
        bodyInterface.SetRotation(rootBody, ParseQuat(state.at("rotation")), JPH::EActivation::Activate);
    }
    if (state.contains("linearVelocity")) {
        bodyInterface.SetLinearVelocity(rootBody, ParseVec3(state.at("linearVelocity")));
    }
    if (state.contains("angularVelocity")) {
        bodyInterface.SetAngularVelocity(rootBody, ParseVec3(state.at("angularVelocity")));
    }
}

} // namespace NativeLoader

RobotData RobotLoader::LoadRobot(const std::string& filepath, JPH::PhysicsSystem* physicsSystem) {
    RobotData robotData;
    
    if (!physicsSystem) {
        std::cerr << "RobotLoader: PhysicsSystem is null" << std::endl;
        return robotData;
    }
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "RobotLoader: Failed to open " << filepath << std::endl;
        return robotData;
    }
    
    nlohmann::json data;
    try {
        file >> data;
    } catch (const std::exception& e) {
        std::cerr << "RobotLoader: JSON parse error: " << e.what() << std::endl;
        return robotData;
    }
    
    // Detect format: native has nested shape object with "type" field
    bool isNativeFormat = false;
    if (data.contains("bodies") && data.at("bodies").is_array() && !data.at("bodies").empty()) {
        const auto& firstBody = data.at("bodies")[0];
        if (firstBody.contains("shape")) {
            const auto& shape = firstBody.at("shape");
            isNativeFormat = shape.is_object() && shape.contains("type");
        }
    }
    
    std::cout << "RobotLoader: " << (isNativeFormat ? "NATIVE" : "LEGACY") 
              << " format detected for " << filepath << std::endl;
    
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    std::map<std::string, JPH::BodyID> bodyMap;
    
    if (isNativeFormat) {
        // NATIVE MODE: Direct Jolt API mapping
        NativeLoader::LoadBodies(data, bodyInterface, bodyMap, robotData);
        NativeLoader::LoadConstraints(data, bodyInterface, *physicsSystem, bodyMap, robotData);
        NativeLoader::ApplyInitialState(data, bodyInterface, bodyMap);
    } else {
        // LEGACY MODE: Simple format
        const auto& bodies = data.at("bodies");
        for (const auto& bodyJson : bodies) {
            std::string name = bodyJson.value("name", "");
            std::string shapeType = bodyJson.value("shape", "box");
            std::string type = bodyJson.value("type", "dynamic");
            
            if (name.empty()) continue;
            
            const auto& pos = bodyJson.at("position");
            JPH::RVec3 position(pos[0].get<double>(), pos[1].get<double>(), pos[2].get<double>());
            
            JPH::Quat rotation = JPH::Quat::sIdentity();
            if (bodyJson.contains("rotation")) {
                const auto& rot = bodyJson.at("rotation");
                float rx = JPH::DegreesToRadians(rot[0].get<float>());
                float ry = JPH::DegreesToRadians(rot[1].get<float>());
                float rz = JPH::DegreesToRadians(rot[2].get<float>());
                rotation = JPH::Quat::sEulerAngles(JPH::Vec3(rx, ry, rz));
            }
            
            JPH::EMotionType motionType = (type == "static") ? JPH::EMotionType::Static : JPH::EMotionType::Dynamic;
            JPH::ObjectLayer layer = (motionType == JPH::EMotionType::Static) ? Layers::STATIC : Layers::MOVING_BASE;
            
            JPH::Ref<JPH::Shape> shape;
            if (shapeType == "box") {
                const auto& he = bodyJson.at("half_extents");
                shape = JPH::BoxShapeSettings(JPH::Vec3(he[0].get<float>(), he[1].get<float>(), he[2].get<float>())).Create().Get();
            } else if (shapeType == "sphere") {
                shape = JPH::SphereShapeSettings(bodyJson.at("radius").get<float>()).Create().Get();
            }
            
            if (!shape) continue;
            
            JPH::BodyCreationSettings settings(shape, position, rotation, motionType, layer);
            if (motionType == JPH::EMotionType::Dynamic && bodyJson.contains("mass")) {
                settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
                settings.mMassPropertiesOverride.mMass = bodyJson.at("mass").get<float>();
            }
            
            JPH::Body* body = bodyInterface.CreateBody(settings);
            if (body) {
                bodyInterface.AddBody(body->GetID(), motionType == JPH::EMotionType::Dynamic ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
                bodyMap[name] = body->GetID();
                robotData.bodies.push_back(body->GetID());
            }
        }
        
        // Legacy joints
        if (data.contains("joints")) {
            for (const auto& jJson : data.at("joints")) {
                std::string type = jJson.value("type", "hinge");
                std::string pName = jJson.value("parent", "");
                std::string cName = jJson.value("child", "");
                
                auto pit = bodyMap.find(pName);
                auto cit = bodyMap.find(cName);
                if (pit == bodyMap.end() || cit == bodyMap.end()) continue;
                
                if (type == "hinge") {
                    const auto& pt = jJson.at("point");
                    const auto& ax = jJson.at("axis");
                    
                    JPH::HingeConstraintSettings settings;
                    settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                    settings.mPoint1 = JPH::RVec3(pt[0].get<double>(), pt[1].get<double>(), pt[2].get<double>());
                    settings.mPoint2 = settings.mPoint1;
                    settings.mHingeAxis1 = JPH::Vec3(ax[0].get<float>(), ax[1].get<float>(), ax[2].get<float>()).Normalized();
                    settings.mHingeAxis2 = settings.mHingeAxis1;
                    settings.mNormalAxis1 = settings.mHingeAxis1.GetNormalizedPerpendicular();
                    settings.mNormalAxis2 = settings.mNormalAxis1;
                    settings.mLimitsMin = -JPH::JPH_PI;
                    settings.mLimitsMax = JPH::JPH_PI;
                    
                    JPH::TwoBodyConstraint* c = bodyInterface.CreateConstraint(&settings, pit->second, cit->second);
                    if (c) {
                        physicsSystem->AddConstraint(c);
                        robotData.constraints.push_back(c);
                    }
                }
            }
        }
    }
    
    std::cout << "RobotLoader: Loaded " << robotData.bodies.size() << " bodies, " 
              << robotData.constraints.size() << " constraints" << std::endl;
    
    return robotData;
}
