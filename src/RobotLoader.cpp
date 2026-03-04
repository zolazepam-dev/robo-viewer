#include <Jolt/Jolt.h>
#include "RobotLoader.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <algorithm>

#include <nlohmann/json.hpp>
#include <tinyxml2.h>

#include <Jolt/Math/Math.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>

#include "PhysicsCore.h"

namespace {

struct MJCFBodyInstance
{
    std::string name;
    JPH::Ref<JPH::Shape> shape;
    float mass = 1.0f;
    JPH::RVec3 worldPos {0, 0, 0};
    JPH::Quat worldRot = JPH::Quat::sIdentity();
    bool isStatic = false;
};

JPH::Vec3 ParseVec3(const char* str, const JPH::Vec3& def = JPH::Vec3::sZero())
{
    if (!str) return def;
    float x=0,y=0,z=0;
    std::stringstream ss(str);
    ss >> x >> y >> z;
    if (ss.fail()) return def;
    return JPH::Vec3(x, y, z);
}

JPH::Quat ParseEuler(const char* str)
{
    if (!str) return JPH::Quat::sIdentity();
    float rx=0, ry=0, rz=0;
    std::stringstream ss(str);
    ss >> rx >> ry >> rz;
    if (ss.fail()) return JPH::Quat::sIdentity();
    return JPH::Quat::sEulerAngles(JPH::Vec3(JPH::DegreesToRadians(rx), JPH::DegreesToRadians(ry), JPH::DegreesToRadians(rz)));
}

JPH::Ref<JPH::Shape> ParseGeomShape(const tinyxml2::XMLElement* geom)
{
    const char* type = geom->Attribute("type");
    std::string t = type ? type : "box";
    if (t == "sphere") {
        float radius = geom->FloatAttribute("size", 0.5f);
        JPH::SphereShapeSettings settings(radius);
        auto res = settings.Create();
        return res.HasError() ? nullptr : res.Get();
    }
    if (t == "box") {
        JPH::Vec3 half = JPH::Vec3::sReplicate(geom->FloatAttribute("size", 0.5f));
        const char* sizeStr = geom->Attribute("size");
        if (sizeStr) {
            std::stringstream ss(sizeStr);
            float x, y, z;
            ss >> x >> y >> z;
            if (!ss.fail()) {
                half.SetX(x);
                half.SetY(y);
                half.SetZ(z);
            }
        }
        JPH::BoxShapeSettings settings(half);
        auto res = settings.Create();
        return res.HasError() ? nullptr : res.Get();
    }
    if (t == "cylinder") {
        // MuJoCo cylinder: size[0] = radius, size[1] = half-height (z)
        JPH::Vec3 size = ParseVec3(geom->Attribute("size"), JPH::Vec3(0.1f, 0.1f, 0.1f));
        float radius = size.GetX();
        float halfHeight = size.GetY();
        JPH::CylinderShapeSettings settings(halfHeight, radius);
        auto res = settings.Create();
        return res.HasError() ? nullptr : res.Get();
    }
    std::cerr << "RobotLoader: Unsupported MJCF geom type: " << t << std::endl;
    return nullptr;
}

void BuildMJCFRecursive(const tinyxml2::XMLElement* bodyElem,
                        const JPH::RVec3& parentPos,
                        const JPH::Quat& parentRot,
                        JPH::BodyInterface& bodyInterface,
                        JPH::PhysicsSystem& physicsSystem,
                        const JPH::ObjectLayer dynamicLayer,
                        std::map<std::string, JPH::BodyID>& bodyMap,
                        RobotData& outData)
{
    JPH::Vec3 localPos = ParseVec3(bodyElem->Attribute("pos"));
    JPH::Quat localRot = ParseEuler(bodyElem->Attribute("euler"));
    JPH::RVec3 worldPos = parentPos + parentRot * JPH::RVec3(localPos);
    JPH::Quat worldRot = parentRot * localRot;

    const char* bodyName = bodyElem->Attribute("name");
    std::string name = bodyName ? bodyName : "";

    // Geom (take first)
    const tinyxml2::XMLElement* geom = bodyElem->FirstChildElement("geom");
    JPH::Ref<JPH::Shape> shape;
    float mass = 1.0f;
    bool isStatic = false;
    if (geom) {
        shape = ParseGeomShape(geom);
        if (geom->Attribute("mass")) mass = geom->FloatAttribute("mass", 1.0f);
        if (geom->Attribute("density")) {
            // ignore density for now
        }
    }

    // Freejoint or no joints => root is dynamic unless mass==0
    const bool hasFreeJoint = bodyElem->FirstChildElement("freejoint") != nullptr;
    if (!shape) {
        // default tiny sphere to keep hierarchy valid
        JPH::SphereShapeSettings s(0.1f);
        auto res = s.Create();
        shape = res.HasError() ? nullptr : res.Get();
    }
    if (shape) {
        JPH::BodyCreationSettings settings(shape,
                                           worldPos,
                                           worldRot,
                                           mass > 0.0f ? JPH::EMotionType::Dynamic : JPH::EMotionType::Static,
                                           mass > 0.0f ? dynamicLayer : Layers::STATIC);
        if (mass > 0.0f) {
            settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            settings.mMassPropertiesOverride.mMass = mass;
        }
        JPH::Body* body = bodyInterface.CreateBody(settings);
        if (body) {
            bodyInterface.AddBody(body->GetID(), mass > 0.0f ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
            if (!name.empty()) bodyMap[name] = body->GetID();
            outData.bodies.push_back(body->GetID());
        }
    }

    // Constraints for joints declared in children referencing this as parent
    const tinyxml2::XMLElement* child = bodyElem->FirstChildElement("body");
    for (; child; child = child->NextSiblingElement("body")) {
        // Pre-compute child's world transform for joint anchors/axes
        JPH::Vec3 childLocalPos = ParseVec3(child->Attribute("pos"));
        JPH::Quat childLocalRot = ParseEuler(child->Attribute("euler"));
        JPH::RVec3 childWorldPos = worldPos + worldRot * JPH::RVec3(childLocalPos);
        JPH::Quat childWorldRot = worldRot * childLocalRot;

        BuildMJCFRecursive(child, worldPos, worldRot, bodyInterface, physicsSystem, dynamicLayer, bodyMap, outData);

        // Find joint inside child (hinge/slide)
        const tinyxml2::XMLElement* joint = child->FirstChildElement("joint");
        if (joint) {
            const char* jtype = joint->Attribute("type");
            std::string jt = jtype ? jtype : "hinge";
            JPH::Vec3 axis = ParseVec3(joint->Attribute("axis"), JPH::Vec3(0, 0, 1));
            if (axis.IsNearZero()) axis = JPH::Vec3::sAxisZ();
            axis = (childWorldRot * axis).Normalized();

            JPH::RVec3 anchor = childWorldPos; // joint at child origin in world
            auto parentIt = bodyMap.find(name);
            auto childIt = bodyMap.find(child->Attribute("name") ? child->Attribute("name") : "");
            if (parentIt != bodyMap.end() && childIt != bodyMap.end()) {
                if (jt == "hinge") {
                    JPH::HingeConstraintSettings settings;
                    settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                    settings.mPoint1 = anchor;
                    settings.mPoint2 = anchor;
                    settings.mHingeAxis1 = axis;
                    settings.mHingeAxis2 = axis;
                    settings.mNormalAxis1 = axis.GetNormalizedPerpendicular();
                    settings.mNormalAxis2 = settings.mNormalAxis1;
                    if (joint->Attribute("range")) {
                        float lo=0, hi=0; std::stringstream ss(joint->Attribute("range")); ss >> lo >> hi;
                        if (!ss.fail()) { settings.mLimitsMin = JPH::DegreesToRadians(lo); settings.mLimitsMax = JPH::DegreesToRadians(hi); }
                    }
                    JPH::TwoBodyConstraint* c = bodyInterface.CreateConstraint(&settings, parentIt->second, childIt->second);
                    if (c) { physicsSystem.AddConstraint(c); outData.constraints.push_back(c); }
                } else if (jt == "slide") {
                    JPH::SliderConstraintSettings settings;
                    settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                    settings.mPoint1 = anchor;
                    settings.mPoint2 = anchor;
                    settings.mSliderAxis1 = axis;
                    settings.mSliderAxis2 = axis;
                    if (joint->Attribute("range")) {
                        float lo=0, hi=0; std::stringstream ss(joint->Attribute("range")); ss >> lo >> hi;
                        if (!ss.fail()) { settings.mLimitsMin = lo; settings.mLimitsMax = hi; }
                    }
                    JPH::TwoBodyConstraint* c = bodyInterface.CreateConstraint(&settings, parentIt->second, childIt->second);
                    if (c) { physicsSystem.AddConstraint(c); outData.constraints.push_back(c); }
                }
            }
        }
    }
}

bool HasXmlExtension(const std::string& path)
{
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return false;
    std::string ext = path.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".xml";
}

}

RobotData RobotLoader::LoadRobot(const std::string& filepath, JPH::PhysicsSystem* physicsSystem)
{
    RobotData robot_data;

    if (physicsSystem == nullptr) {
        std::cerr << "RobotLoader: PhysicsSystem is null." << std::endl;
        return robot_data;
    }

    // --- MJCF path ---
    if (HasXmlExtension(filepath)) {
        tinyxml2::XMLDocument doc;
        if (doc.LoadFile(filepath.c_str()) != tinyxml2::XML_SUCCESS) {
            std::cerr << "RobotLoader: Failed to load MJCF " << filepath << std::endl;
            return robot_data;
        }
        const tinyxml2::XMLElement* mj = doc.FirstChildElement("mujoco");
        if (!mj) {
            std::cerr << "RobotLoader: Not an MJCF file (missing <mujoco>)" << std::endl;
            return robot_data;
        }
        JPH::BodyInterface& body_interface = physicsSystem->GetBodyInterface();
        const JPH::ObjectLayer dynamicLayer = Layers::MOVING_BASE;

        const tinyxml2::XMLElement* world = mj->FirstChildElement("worldbody");
        if (!world) {
            std::cerr << "RobotLoader: MJCF missing <worldbody>" << std::endl;
            return robot_data;
        }

        std::map<std::string, JPH::BodyID> body_map;
        for (const tinyxml2::XMLElement* body = world->FirstChildElement("body"); body; body = body->NextSiblingElement("body")) {
            BuildMJCFRecursive(body, JPH::RVec3::sZero(), JPH::Quat::sIdentity(), body_interface, *physicsSystem, dynamicLayer, body_map, robot_data);
        }
        return robot_data;
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "RobotLoader: Failed to open " << filepath << std::endl;
        return robot_data;
    }

    nlohmann::json data;
    file >> data;

    std::map<std::string, JPH::BodyID> body_map;
    JPH::BodyInterface& body_interface = physicsSystem->GetBodyInterface();

    const auto& bodies = data.at("bodies");
    for (const auto& body_data : bodies) {
        const std::string name = body_data.value("name", "");
        const std::string shape = body_data.value("shape", "box");
        const std::string type = body_data.value("type", "dynamic");

        if (name.empty()) {
            std::cerr << "RobotLoader: Body missing name." << std::endl;
            continue;
        }

        const auto& position = body_data.at("position");
        const double px = position.at(0).get<double>();
        const double py = position.at(1).get<double>();
        const double pz = position.at(2).get<double>();

        JPH::Quat rotation = JPH::Quat::sIdentity();
        auto rotation_it = body_data.find("rotation");
        if (rotation_it != body_data.end()) {
            const auto& rot = *rotation_it;
            const float rx = JPH::DegreesToRadians(rot.at(0).get<float>());
            const float ry = JPH::DegreesToRadians(rot.at(1).get<float>());
            const float rz = JPH::DegreesToRadians(rot.at(2).get<float>());
            rotation = JPH::Quat::sEulerAngles(JPH::Vec3(rx, ry, rz));
        }

        const JPH::EMotionType motion_type = (type == "static") ? JPH::EMotionType::Static : JPH::EMotionType::Dynamic;
        const JPH::ObjectLayer layer = (motion_type == JPH::EMotionType::Static) ? Layers::STATIC : Layers::MOVING_BASE;

        JPH::ShapeSettings::ShapeResult shape_result;
        if (shape == "box") {
            const auto& half_extents = body_data.at("half_extents");
            const float hx = half_extents.at(0).get<float>();
            const float hy = half_extents.at(1).get<float>();
            const float hz = half_extents.at(2).get<float>();
            JPH::BoxShapeSettings shape_settings(JPH::Vec3(hx, hy, hz));
            shape_result = shape_settings.Create();
        } else if (shape == "sphere" || shape == "circle") {
            const float radius = body_data.at("radius").get<float>();
            JPH::SphereShapeSettings shape_settings(radius);
            shape_result = shape_settings.Create();
        } else {
            std::cerr << "RobotLoader: Unsupported shape for body '" << name << "': " << shape << std::endl;
            continue;
        }

        if (shape_result.HasError()) {
            std::cerr << "RobotLoader: Shape creation failed for body '" << name << "': " << shape_result.GetError() << std::endl;
            continue;
        }

        JPH::Ref<JPH::Shape> shape_ref = shape_result.Get();
        JPH::BodyCreationSettings body_settings(
            shape_ref,
            JPH::RVec3(px, py, pz),
            rotation,
            motion_type,
            layer
        );

        if (motion_type == JPH::EMotionType::Dynamic) {
            const float mass = body_data.value("mass", 1.0f);
            body_settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            body_settings.mMassPropertiesOverride.mMass = mass;

            body_settings.mFriction = 0.0f;
            body_settings.mRestitution = 1.0f;
            body_settings.mLinearDamping = 0.0f;
            body_settings.mAngularDamping = 0.0f;
        }

        JPH::Body* body = body_interface.CreateBody(body_settings);
        if (body == nullptr) {
            std::cerr << "RobotLoader: Failed to create body '" << name << "'." << std::endl;
            continue;
        }

        const JPH::BodyID body_id = body->GetID();
        body_interface.AddBody(body_id, motion_type == JPH::EMotionType::Dynamic ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);

        body_map[name] = body_id;
        robot_data.bodies.push_back(body_id);
    }

    auto joints_it = data.find("joints");
    if (joints_it != data.end()) {
        for (const auto& joint_data : *joints_it) {
            const std::string type = joint_data.value("type", "");
            if (type != "hinge") {
                std::cerr << "RobotLoader: Unsupported joint type: " << type << std::endl;
                continue;
            }

            const std::string parent_name = joint_data.value("parent", "");
            const std::string child_name = joint_data.value("child", "");

            auto parent_it = body_map.find(parent_name);
            auto child_it = body_map.find(child_name);
            if (parent_it == body_map.end() || child_it == body_map.end()) {
                std::cerr << "RobotLoader: Joint references unknown bodies: " << parent_name << " -> " << child_name << std::endl;
                continue;
            }

            const auto& point = joint_data.at("point");
            const auto& axis = joint_data.at("axis");

            const double px = point.at(0).get<double>();
            const double py = point.at(1).get<double>();
            const double pz = point.at(2).get<double>();

            JPH::Vec3 hinge_axis(
                axis.at(0).get<float>(),
                axis.at(1).get<float>(),
                axis.at(2).get<float>()
            );

            if (hinge_axis.IsNearZero()) {
                hinge_axis = JPH::Vec3::sAxisY();
            } else {
                hinge_axis = hinge_axis.Normalized();
            }

            const float axis_dot_up = std::fabs(hinge_axis.GetY());
            JPH::Vec3 reference = axis_dot_up > 0.9f ? JPH::Vec3::sAxisX() : JPH::Vec3::sAxisY();
            JPH::Vec3 normal_axis = hinge_axis.Cross(reference);
            if (normal_axis.IsNearZero()) {
                normal_axis = hinge_axis.GetNormalizedPerpendicular();
            } else {
                normal_axis = normal_axis.Normalized();
            }

            JPH::HingeConstraintSettings settings;
            settings.mSpace = JPH::EConstraintSpace::WorldSpace;
            settings.mPoint1 = JPH::RVec3(px, py, pz);
            settings.mPoint2 = JPH::RVec3(px, py, pz);
            settings.mHingeAxis1 = hinge_axis;
            settings.mHingeAxis2 = hinge_axis;
            settings.mNormalAxis1 = normal_axis;
            settings.mNormalAxis2 = normal_axis;
            settings.mLimitsMin = -JPH::JPH_PI;
            settings.mLimitsMax = JPH::JPH_PI;
            settings.mLimitsSpringSettings.mFrequency = 0.0f;
            settings.mLimitsSpringSettings.mDamping = 0.0f;
            settings.mMaxFrictionTorque = 0.0f;

            const float motor_speed = joint_data.value("motor_speed", 0.0f);
            const float motor_torque = joint_data.value("motor_torque", 0.0f);
            if (motor_torque > 0.0f) {
                settings.mMotorSettings.mSpringSettings.mFrequency = 10.0f;
                settings.mMotorSettings.mSpringSettings.mDamping = 1.0f;
                settings.mMotorSettings.mMinTorqueLimit = -motor_torque;
                settings.mMotorSettings.mMaxTorqueLimit = motor_torque;
            }

            JPH::TwoBodyConstraint* constraint = body_interface.CreateConstraint(&settings, parent_it->second, child_it->second);
            if (constraint == nullptr) {
                std::cerr << "RobotLoader: Failed to create hinge constraint for joint." << std::endl;
                continue;
            }

            if (motor_torque > 0.0f) {
                JPH::HingeConstraint* hinge = static_cast<JPH::HingeConstraint*>(constraint);
                hinge->SetMotorState(JPH::EMotorState::Velocity);
                hinge->SetTargetAngularVelocity(0.0f);
            }

            physicsSystem->AddConstraint(constraint);
            robot_data.constraints.push_back(constraint);
        }
    }

    return robot_data;
}
