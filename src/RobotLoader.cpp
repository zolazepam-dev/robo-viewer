#include <Jolt/Jolt.h>
#include "RobotLoader.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

#include <nlohmann/json.hpp>

#include <Jolt/Math/Math.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>

#include "PhysicsCore.h"

RobotData RobotLoader::LoadRobot(const std::string& filepath, JPH::PhysicsSystem* physicsSystem)
{
    RobotData robot_data;

    if (physicsSystem == nullptr) {
        std::cerr << "RobotLoader: PhysicsSystem is null." << std::endl;
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
        const JPH::ObjectLayer layer = (motion_type == JPH::EMotionType::Static) ? Layers::JOINT_ANCHOR : Layers::MOVING;

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
