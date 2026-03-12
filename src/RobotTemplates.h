#pragma once

#include <Jolt/Jolt.h>
#include <vector>
#include <string>
#include "CombatRobot.h"

/**
 * @brief Fluent builder for constructing robots programmatically
 */
class RobotBuilder {
public:
    RobotBuilder& SetName(const std::string& name);
    RobotBuilder& SetSpawnPos(const JPH::RVec3& pos);
    RobotBuilder& SetEnvIndex(uint32_t env);
    RobotBuilder& SetRobotIndex(int idx);
    
    // Body creation
    RobotBuilder& AddSphereBody(const std::string& name, float radius, float mass, const JPH::Vec3& offset = JPH::Vec3::sZero());
    RobotBuilder& AddBoxBody(const std::string& name, const JPH::Vec3& halfExtents, float mass, const JPH::Vec3& offset = JPH::Vec3::sZero());
    RobotBuilder& AddCapsuleBody(const std::string& name, float radius, float halfHeight, float mass, const JPH::Vec3& offset = JPH::Vec3::sZero());
    RobotBuilder& AddCylinderBody(const std::string& name, float radius, float halfHeight, float mass, const JPH::Vec3& offset = JPH::Vec3::sZero());
    RobotBuilder& AddStaticBoxBody(const std::string& name, const JPH::Vec3& halfExtents, const JPH::Vec3& offset = JPH::Vec3::sZero());
    
    // Joint creation
    RobotBuilder& AddHingeJoint(const std::string& parent, const std::string& child, const JPH::Vec3& axis, float motorTorque = 100.0f, float motorSpeed = 10.0f);
    RobotBuilder& AddSixDOFJoint(const std::string& parent, const std::string& child, const JPH::Vec3& position);
    RobotBuilder& AddSliderJoint(const std::string& parent, const std::string& child, const JPH::Vec3& axis, float minSlide = -0.5f, float maxSlide = 0.5f);
    
    // Motor configuration
    RobotBuilder& SetMotorTorque(const std::string& jointName, float torque);
    RobotBuilder& SetMotorSpeed(const std::string& jointName, float speed);
    
    // Build the robot
    CombatRobotData Build(JPH::PhysicsSystem* physicsSystem);
    
private:
    std::string m_name = "robot";
    JPH::RVec3 m_spawnPos = JPH::RVec3::sZero();
    uint32_t m_envIndex = 0;
    int m_robotIndex = 0;
    
    struct BodyDef {
        std::string name;
        std::string type;  // "sphere", "box", "capsule", "cylinder"
        std::vector<float> params;  // radius, height, etc.
        float mass;
        JPH::Vec3 offset;
    };
    
    struct JointDef {
        std::string name;
        std::string type;  // "hinge", "sixdof", "slider"
        std::string parent;
        std::string child;
        JPH::Vec3 axis;
        JPH::Vec3 position;
        float motorTorque = 100.0f;
        float motorSpeed = 10.0f;
        float minSlide = -0.5f;
        float maxSlide = 0.5f;
    };
    
    std::vector<BodyDef> m_bodies;
    std::vector<JointDef> m_joints;
    
    JPH::GroupFilterTable* m_groupFilter = nullptr;
};

/**
 * @brief Pre-built robot templates
 */
class RobotTemplates {
public:
    // Arm manipulator robots
    static CombatRobotData CreateArm3DOF(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env = 0);
    static CombatRobotData CreateArm5DOF(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env = 0);
    static CombatRobotData CreateArm7DOF(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env = 0);
    
    // Snake robots
    static CombatRobotData CreateSnake4Segment(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env = 0);
    static CombatRobotData CreateSnake8Segment(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env = 0);
    static CombatRobotData CreateSnake12Segment(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env = 0);
    
    // Hybrid robots
    static CombatRobotData CreateArmWithGripper(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env = 0);
    static CombatRobotData CreateSnakeWithHead(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env = 0);
};
