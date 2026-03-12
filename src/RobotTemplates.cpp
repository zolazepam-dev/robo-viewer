#include "RobotTemplates.h"
#include "PhysicsCore.h"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <iostream>
#include <map>

// ============================================================================
// ROBOT BUILDER IMPLEMENTATION
// ============================================================================

RobotBuilder& RobotBuilder::SetName(const std::string& name) {
    m_name = name;
    return *this;
}

RobotBuilder& RobotBuilder::SetSpawnPos(const JPH::RVec3& pos) {
    m_spawnPos = pos;
    return *this;
}

RobotBuilder& RobotBuilder::SetEnvIndex(uint32_t env) {
    m_envIndex = env;
    return *this;
}

RobotBuilder& RobotBuilder::SetRobotIndex(int idx) {
    m_robotIndex = idx;
    return *this;
}

RobotBuilder& RobotBuilder::AddSphereBody(const std::string& name, float radius, float mass, const JPH::Vec3& offset) {
    BodyDef def;
    def.name = name;
    def.type = "sphere";
    def.params = {radius};
    def.mass = mass;
    def.offset = offset;
    m_bodies.push_back(def);
    return *this;
}

RobotBuilder& RobotBuilder::AddBoxBody(const std::string& name, const JPH::Vec3& halfExtents, float mass, const JPH::Vec3& offset) {
    BodyDef def;
    def.name = name;
    def.type = "box";
    def.params = {halfExtents.GetX(), halfExtents.GetY(), halfExtents.GetZ()};
    def.mass = mass;
    def.offset = offset;
    m_bodies.push_back(def);
    return *this;
}

RobotBuilder& RobotBuilder::AddCapsuleBody(const std::string& name, float radius, float halfHeight, float mass, const JPH::Vec3& offset) {
    BodyDef def;
    def.name = name;
    def.type = "capsule";
    def.params = {radius, halfHeight};
    def.mass = mass;
    def.offset = offset;
    m_bodies.push_back(def);
    return *this;
}

RobotBuilder& RobotBuilder::AddCylinderBody(const std::string& name, float radius, float halfHeight, float mass, const JPH::Vec3& offset) {
    BodyDef def;
    def.name = name;
    def.type = "cylinder";
    def.params = {radius, halfHeight};
    def.mass = mass;
    def.offset = offset;
    m_bodies.push_back(def);
    return *this;
}

RobotBuilder& RobotBuilder::AddStaticBoxBody(const std::string& name, const JPH::Vec3& halfExtents, const JPH::Vec3& offset) {
    BodyDef def;
    def.name = name;
    def.type = "static_box";
    def.params = {halfExtents.GetX(), halfExtents.GetY(), halfExtents.GetZ()};
    def.mass = 0.0f;  // Static bodies have no mass
    def.offset = offset;
    m_bodies.push_back(def);
    return *this;
}

RobotBuilder& RobotBuilder::AddHingeJoint(const std::string& parent, const std::string& child, const JPH::Vec3& axis, float motorTorque, float motorSpeed) {
    JointDef def;
    def.name = parent + "_" + child + "_hinge";
    def.type = "hinge";
    def.parent = parent;
    def.child = child;
    def.axis = axis;
    def.motorTorque = motorTorque;
    def.motorSpeed = motorSpeed;
    m_joints.push_back(def);
    return *this;
}

RobotBuilder& RobotBuilder::AddSixDOFJoint(const std::string& parent, const std::string& child, const JPH::Vec3& position) {
    JointDef def;
    def.name = parent + "_" + child + "_6dof";
    def.type = "sixdof";
    def.parent = parent;
    def.child = child;
    def.position = position;
    m_joints.push_back(def);
    return *this;
}

RobotBuilder& RobotBuilder::AddSliderJoint(const std::string& parent, const std::string& child, const JPH::Vec3& axis, float minSlide, float maxSlide) {
    JointDef def;
    def.name = parent + "_" + child + "_slider";
    def.type = "slider";
    def.parent = parent;
    def.child = child;
    def.axis = axis;
    def.minSlide = minSlide;
    def.maxSlide = maxSlide;
    m_joints.push_back(def);
    return *this;
}

RobotBuilder& RobotBuilder::SetMotorTorque(const std::string& jointName, float torque) {
    for (auto& joint : m_joints) {
        if (joint.name == jointName) {
            joint.motorTorque = torque;
            break;
        }
    }
    return *this;
}

RobotBuilder& RobotBuilder::SetMotorSpeed(const std::string& jointName, float speed) {
    for (auto& joint : m_joints) {
        if (joint.name == jointName) {
            joint.motorSpeed = speed;
            break;
        }
    }
    return *this;
}

CombatRobotData RobotBuilder::Build(JPH::PhysicsSystem* physicsSystem) {
    CombatRobotData robotData;
    robotData.envIndex = m_envIndex;
    robotData.robotIndex = m_robotIndex;
    robotData.hp = 100.0f;
    robotData.totalEnergyUsed = 0.0f;
    robotData.collisionGroup = m_envIndex * 2 + m_robotIndex;
    
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    
    // Create group filter to prevent self-collision
    if (m_groupFilter == nullptr) {
        m_groupFilter = new JPH::GroupFilterTable(256);
        for (int i = 0; i < 256; ++i) {
            for (int j = 0; j < 256; ++j) {
                m_groupFilter->DisableCollision(i, j);
            }
        }
    }
    
    JPH::ObjectLayer layer = Layers::MOVING_BASE + m_envIndex;
    
    // Create all bodies
    std::map<std::string, JPH::BodyID> bodyMap;
    
    for (const auto& bodyDef : m_bodies) {
        JPH::RefConst<JPH::Shape> shape;
        float volume = 1.0f;
        
        if (bodyDef.type == "sphere") {
            float radius = bodyDef.params[0];
            JPH::SphereShapeSettings shapeSettings(radius);
            volume = 4.0f / 3.0f * 3.14159f * radius * radius * radius;
            shapeSettings.SetDensity(bodyDef.mass / volume);
            auto result = shapeSettings.Create();
            if (result.HasError()) {
                std::cerr << "Shape error for " << bodyDef.name << ": " << result.GetError().c_str() << std::endl;
                continue;
            }
            shape = result.Get();
        } else if (bodyDef.type == "box") {
            JPH::Vec3 halfExtents(bodyDef.params[0], bodyDef.params[1], bodyDef.params[2]);
            JPH::BoxShapeSettings shapeSettings(halfExtents);
            volume = 8.0f * halfExtents.GetX() * halfExtents.GetY() * halfExtents.GetZ();
            shapeSettings.SetDensity(bodyDef.mass / volume);
            auto result = shapeSettings.Create();
            if (result.HasError()) {
                std::cerr << "Shape error for " << bodyDef.name << ": " << result.GetError().c_str() << std::endl;
                continue;
            }
            shape = result.Get();
        } else if (bodyDef.type == "capsule") {
            float radius = bodyDef.params[0];
            float halfHeight = bodyDef.params[1];
            JPH::CapsuleShapeSettings shapeSettings(halfHeight, radius);
            volume = 3.14159f * radius * radius * (4.0f / 3.0f * radius + 2.0f * halfHeight);
            shapeSettings.SetDensity(bodyDef.mass / volume);
            auto result = shapeSettings.Create();
            if (result.HasError()) {
                std::cerr << "Shape error for " << bodyDef.name << ": " << result.GetError().c_str() << std::endl;
                continue;
            }
            shape = result.Get();
        } else if (bodyDef.type == "cylinder") {
            float radius = bodyDef.params[0];
            float halfHeight = bodyDef.params[1];
            JPH::CylinderShapeSettings shapeSettings(halfHeight, radius);
            volume = 3.14159f * radius * radius * 2.0f * halfHeight;
            shapeSettings.SetDensity(bodyDef.mass / volume);
            auto result = shapeSettings.Create();
            if (result.HasError()) {
                std::cerr << "Shape error for " << bodyDef.name << ": " << result.GetError().c_str() << std::endl;
                continue;
            }
            shape = result.Get();
        } else if (bodyDef.type == "static_box") {
            JPH::Vec3 halfExtents(bodyDef.params[0], bodyDef.params[1], bodyDef.params[2]);
            JPH::BoxShapeSettings shapeSettings(halfExtents);
            auto result = shapeSettings.Create();
            if (result.HasError()) {
                std::cerr << "Shape error for " << bodyDef.name << ": " << result.GetError().c_str() << std::endl;
                continue;
            }
            shape = result.Get();
        }
        
        if (!shape) {
            std::cerr << "Failed to create shape for body: " << bodyDef.name << std::endl;
            continue;
        }
        
        // Static bodies (like base platforms) don't move
        bool isStatic = (bodyDef.type == "static_box");
        
        JPH::BodyCreationSettings bodySettings(
            shape,
            m_spawnPos + bodyDef.offset,
            JPH::Quat::sIdentity(),
            isStatic ? JPH::EMotionType::Static : JPH::EMotionType::Dynamic,
            layer
        );
        
        bodySettings.mCollisionGroup.SetGroupFilter(m_groupFilter);
        bodySettings.mCollisionGroup.SetGroupID(robotData.collisionGroup);
        bodySettings.mFriction = 0.8f;  // Higher friction for stability
        bodySettings.mRestitution = 0.1f;
        
        std::cout << "  Creating " << (isStatic ? "STATIC " : "") << "body '" << bodyDef.name << "' at Y=" << (m_spawnPos + bodyDef.offset).GetY() << std::endl;
        
        JPH::Body* body = bodyInterface.CreateBody(bodySettings);
        if (body) {
            std::cout << "    Body created with ID index=" << body->GetID().GetIndex() << std::endl;
            bodyInterface.AddBody(body->GetID(), JPH::EActivation::Activate);
            bodyMap[bodyDef.name] = body->GetID();
            robotData.bodies.push_back(body->GetID());
            
            // First body is main body
            if (robotData.mainBodyId.IsInvalid()) {
                robotData.mainBodyId = body->GetID();
                std::cout << "    Set as main body" << std::endl;
            }
        } else {
            std::cerr << "    FAILED to create body!" << std::endl;
        }
    }
    
    // Create all joints
    for (const auto& jointDef : m_joints) {
        auto it1 = bodyMap.find(jointDef.parent);
        auto it2 = bodyMap.find(jointDef.child);
        
        if (it1 == bodyMap.end() || it2 == bodyMap.end()) {
            std::cerr << "Joint '" << jointDef.name << "' failed: parent or child not found" << std::endl;
            continue;
        }
        
        JPH::BodyID body1 = it1->second;
        JPH::BodyID body2 = it2->second;
        
        // Get actual body positions for joint placement
        JPH::RVec3 pos1 = bodyInterface.GetPosition(body1);
        JPH::RVec3 pos2 = bodyInterface.GetPosition(body2);
        
        // Joint position is midpoint between connected bodies
        JPH::RVec3 jointPos = (pos1 + pos2) * 0.5;
        
        std::cout << "  Creating joint '" << jointDef.name << "' between '" << jointDef.parent 
                  << "' and '" << jointDef.child << "' at Y=" << jointPos.GetY() << std::endl;
        
        if (jointDef.type == "hinge") {
            JPH::HingeConstraintSettings hingeSettings;
            hingeSettings.mSpace = JPH::EConstraintSpace::WorldSpace;
            hingeSettings.mPoint1 = jointPos;
            hingeSettings.mPoint2 = jointPos;
            hingeSettings.mHingeAxis1 = jointDef.axis;
            hingeSettings.mHingeAxis2 = jointDef.axis;
            hingeSettings.mNormalAxis1 = JPH::Vec3::sAxisX();
            hingeSettings.mNormalAxis2 = JPH::Vec3::sAxisX();
            
            if (jointDef.motorTorque > 0) {
                hingeSettings.mMotorSettings.mSpringSettings.mFrequency = 0.0f;
                hingeSettings.mMotorSettings.mMinTorqueLimit = -jointDef.motorTorque;
                hingeSettings.mMotorSettings.mMaxTorqueLimit = jointDef.motorTorque;
            }
            
            JPH::HingeConstraint* hinge = static_cast<JPH::HingeConstraint*>(
                bodyInterface.CreateConstraint(&hingeSettings, body1, body2)
            );
            
            if (hinge) {
                if (jointDef.motorSpeed > 0) {
                    hinge->SetMotorState(JPH::EMotorState::Velocity);
                    hinge->SetTargetAngularVelocity(jointDef.motorSpeed);
                }
                physicsSystem->AddConstraint(hinge);
                robotData.hingeJoints.push_back(hinge);
            }
        } else if (jointDef.type == "sixdof") {
            JPH::SixDOFConstraintSettings sixDofSettings;
            sixDofSettings.mSpace = JPH::EConstraintSpace::WorldSpace;
            sixDofSettings.mPosition1 = m_spawnPos + jointDef.position;
            sixDofSettings.mPosition2 = m_spawnPos + jointDef.position;
            
            JPH::SixDOFConstraint* sixDof = static_cast<JPH::SixDOFConstraint*>(
                bodyInterface.CreateConstraint(&sixDofSettings, body1, body2)
            );
            
            if (sixDof) {
                physicsSystem->AddConstraint(sixDof);
                robotData.sixDofJoints.push_back(sixDof);
            }
        } else if (jointDef.type == "slider") {
            JPH::SliderConstraintSettings sliderSettings;
            sliderSettings.mSpace = JPH::EConstraintSpace::WorldSpace;
            sliderSettings.mPoint1 = m_spawnPos + jointDef.position;
            sliderSettings.mPoint2 = m_spawnPos + jointDef.position;
            sliderSettings.mSliderAxis1 = jointDef.axis;
            sliderSettings.mSliderAxis2 = jointDef.axis;
            sliderSettings.mLimitsMin = jointDef.minSlide;
            sliderSettings.mLimitsMax = jointDef.maxSlide;
            
            JPH::SliderConstraint* slider = static_cast<JPH::SliderConstraint*>(
                bodyInterface.CreateConstraint(&sliderSettings, body1, body2)
            );
            
            if (slider) {
                physicsSystem->AddConstraint(slider);
            }
        }
    }
    
    return robotData;
}

// ============================================================================
// ROBOT TEMPLATES IMPLEMENTATION
// ============================================================================

// ARM ROBOTS
CombatRobotData RobotTemplates::CreateArm3DOF(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env) {
    RobotBuilder builder;
    builder.SetName("arm_3dof")
           .SetSpawnPos(spawnPos)
           .SetEnvIndex(env)
           .SetRobotIndex(0);
    
    // Static base platform attached to ground
    builder.AddStaticBoxBody("base_platform", JPH::Vec3(0.5f, 0.1f, 0.5f), JPH::Vec3::sZero());
    
    // Rotating base joint
    builder.AddBoxBody("base", JPH::Vec3(0.25f, 0.15f, 0.25f), 4.0f, JPH::Vec3(0, 0.25f, 0));
    
    // Lower arm - stacked on base
    builder.AddCapsuleBody("lower_arm", 0.06f, 0.35f, 1.5f, JPH::Vec3(0, 0.55f, 0));
    
    // Upper arm - stacked on lower
    builder.AddCapsuleBody("upper_arm", 0.05f, 0.3f, 1.2f, JPH::Vec3(0, 0.95f, 0));
    
    // Wrist
    builder.AddBoxBody("wrist", JPH::Vec3(0.08f, 0.08f, 0.12f), 0.4f, JPH::Vec3(0, 1.25f, 0));
    
    // Gripper base
    builder.AddBoxBody("gripper_base", JPH::Vec3(0.12f, 0.06f, 0.1f), 0.3f, JPH::Vec3(0, 1.38f, 0));
    
    // Gripper fingers (left and right)
    builder.AddBoxBody("finger_left", JPH::Vec3(0.03f, 0.08f, 0.04f), 0.15f, JPH::Vec3(-0.07f, 1.45f, 0));
    builder.AddBoxBody("finger_right", JPH::Vec3(0.03f, 0.08f, 0.04f), 0.15f, JPH::Vec3(0.07f, 1.45f, 0));
    
    // Joints
    builder.AddHingeJoint("base_platform", "base", JPH::Vec3::sAxisY(), 100.0f, 4.0f);  // Base rotation
    builder.AddHingeJoint("base", "lower_arm", JPH::Vec3::sAxisZ(), 120.0f, 6.0f);     // Shoulder
    builder.AddHingeJoint("lower_arm", "upper_arm", JPH::Vec3::sAxisZ(), 80.0f, 8.0f);  // Elbow
    builder.AddHingeJoint("upper_arm", "wrist", JPH::Vec3::sAxisZ(), 50.0f, 10.0f);    // Wrist tilt
    builder.AddHingeJoint("wrist", "gripper_base", JPH::Vec3::sAxisX(), 30.0f, 12.0f); // Wrist rotate
    builder.AddHingeJoint("gripper_base", "finger_left", JPH::Vec3::sAxisX(), 15.0f, 8.0f);   // Left finger
    builder.AddHingeJoint("gripper_base", "finger_right", JPH::Vec3::sAxisX(), 15.0f, 8.0f);  // Right finger
    
    return builder.Build(physics);
}

CombatRobotData RobotTemplates::CreateArm5DOF(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env) {
    RobotBuilder builder;
    builder.SetName("arm_5dof_gripper")
           .SetSpawnPos(spawnPos)
           .SetEnvIndex(env)
           .SetRobotIndex(0);
    
    // Static base platform
    builder.AddStaticBoxBody("base_platform", JPH::Vec3(0.6f, 0.15f, 0.6f), JPH::Vec3::sZero());
    
    // Rotating base
    builder.AddCylinderBody("base", 0.2f, 0.15f, 5.0f, JPH::Vec3(0, 0.25f, 0));
    
    // Shoulder
    builder.AddBoxBody("shoulder", JPH::Vec3(0.15f, 0.12f, 0.15f), 3.5f, JPH::Vec3(0, 0.45f, 0));
    
    // Lower arm
    builder.AddCapsuleBody("lower_arm", 0.07f, 0.4f, 2.5f, JPH::Vec3(0, 0.85f, 0));
    
    // Upper arm
    builder.AddCapsuleBody("upper_arm", 0.06f, 0.35f, 2.0f, JPH::Vec3(0, 1.3f, 0));
    
    // Forearm
    builder.AddBoxBody("forearm", JPH::Vec3(0.08f, 0.08f, 0.15f), 1.2f, JPH::Vec3(0, 1.65f, 0));
    
    // Wrist
    builder.AddBoxBody("wrist", JPH::Vec3(0.1f, 0.08f, 0.12f), 0.6f, JPH::Vec3(0, 1.8f, 0));
    
    // Gripper base
    builder.AddBoxBody("gripper_base", JPH::Vec3(0.15f, 0.08f, 0.12f), 0.4f, JPH::Vec3(0, 1.92f, 0));
    
    // Gripper fingers
    builder.AddBoxBody("finger_left", JPH::Vec3(0.04f, 0.1f, 0.05f), 0.2f, JPH::Vec3(-0.09f, 2.0f, 0));
    builder.AddBoxBody("finger_right", JPH::Vec3(0.04f, 0.1f, 0.05f), 0.2f, JPH::Vec3(0.09f, 2.0f, 0));
    
    // Joints (5 DOF + gripper)
    builder.AddHingeJoint("base_platform", "base", JPH::Vec3::sAxisY(), 150.0f, 3.0f);        // Base rotation
    builder.AddHingeJoint("base", "shoulder", JPH::Vec3::sAxisZ(), 180.0f, 5.0f);             // Shoulder tilt
    builder.AddHingeJoint("shoulder", "lower_arm", JPH::Vec3::sAxisZ(), 150.0f, 7.0f);        // Shoulder
    builder.AddHingeJoint("lower_arm", "upper_arm", JPH::Vec3::sAxisZ(), 120.0f, 9.0f);       // Elbow
    builder.AddHingeJoint("upper_arm", "forearm", JPH::Vec3::sAxisZ(), 80.0f, 11.0f);         // Wrist tilt
    builder.AddHingeJoint("forearm", "wrist", JPH::Vec3::sAxisX(), 60.0f, 13.0f);             // Wrist rotate
    builder.AddHingeJoint("wrist", "gripper_base", JPH::Vec3::sAxisZ(), 40.0f, 10.0f);        // Gripper tilt
    builder.AddHingeJoint("gripper_base", "finger_left", JPH::Vec3::sAxisX(), 20.0f, 8.0f);   // Left finger close/open
    builder.AddHingeJoint("gripper_base", "finger_right", JPH::Vec3::sAxisX(), 20.0f, 8.0f);  // Right finger close/open
    
    return builder.Build(physics);
}

CombatRobotData RobotTemplates::CreateArm7DOF(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env) {
    RobotBuilder builder;
    builder.SetName("arm_7dof")
           .SetSpawnPos(spawnPos)
           .SetEnvIndex(env)
           .SetRobotIndex(0);
    
    // Base
    builder.AddBoxBody("base", JPH::Vec3(0.5f, 0.2f, 0.5f), 10.0f, JPH::Vec3::sZero());
    
    // Segments
    builder.AddCylinderBody("shoulder", 0.18f, 0.12f, 4.0f, JPH::Vec3(0, 0.2f, 0));
    builder.AddCapsuleBody("upper_arm", 0.09f, 0.4f, 3.0f, JPH::Vec3(0, 0.6f, 0));
    builder.AddCapsuleBody("elbow", 0.08f, 0.15f, 2.5f, JPH::Vec3(0, 1.0f, 0));
    builder.AddCapsuleBody("forearm", 0.07f, 0.35f, 2.0f, JPH::Vec3(0, 1.35f, 0));
    builder.AddCylinderBody("wrist", 0.06f, 0.1f, 1.5f, JPH::Vec3(0, 1.7f, 0));
    builder.AddBoxBody("hand", JPH::Vec3(0.15f, 0.08f, 0.2f), 1.0f, JPH::Vec3(0, 1.9f, 0));
    
    // 7 DOF: shoulder pan, shoulder tilt, upper arm twist, elbow, forearm twist, wrist tilt, wrist rotate
    builder.AddHingeJoint("base", "shoulder", JPH::Vec3::sAxisY(), 250.0f, 3.0f);
    builder.AddHingeJoint("shoulder", "upper_arm", JPH::Vec3::sAxisZ(), 180.0f, 5.0f);
    builder.AddHingeJoint("upper_arm", "elbow", JPH::Vec3::sAxisX(), 150.0f, 6.0f);
    builder.AddHingeJoint("elbow", "forearm", JPH::Vec3::sAxisZ(), 120.0f, 8.0f);
    builder.AddHingeJoint("forearm", "wrist", JPH::Vec3::sAxisX(), 80.0f, 10.0f);
    builder.AddHingeJoint("wrist", "hand", JPH::Vec3::sAxisZ(), 60.0f, 12.0f);
    builder.AddHingeJoint("hand", "hand", JPH::Vec3::sAxisY(), 40.0f, 15.0f);
    
    return builder.Build(physics);
}

// SNAKE ROBOTS
CombatRobotData RobotTemplates::CreateSnake4Segment(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env) {
    RobotBuilder builder;
    builder.SetName("snake_4seg")
           .SetSpawnPos(spawnPos)
           .SetEnvIndex(env)
           .SetRobotIndex(0);
    
    float segmentLength = 0.4f;
    float segmentRadius = 0.1f;
    float segmentMass = 1.5f;
    
    // Head
    builder.AddCapsuleBody("head", segmentRadius, segmentLength * 0.6f, segmentMass * 1.2f, JPH::Vec3(0, segmentRadius + 0.1f, 0));
    
    // Body segments
    builder.AddCapsuleBody("seg1", segmentRadius, segmentLength, segmentMass, JPH::Vec3(0, segmentRadius + 0.1f, -segmentLength));
    builder.AddCapsuleBody("seg2", segmentRadius, segmentLength, segmentMass, JPH::Vec3(0, segmentRadius + 0.1f, -segmentLength * 2));
    builder.AddCapsuleBody("seg3", segmentRadius, segmentLength, segmentMass, JPH::Vec3(0, segmentRadius + 0.1f, -segmentLength * 3));
    
    // Tail
    builder.AddCapsuleBody("tail", segmentRadius * 0.8f, segmentLength * 0.5f, segmentMass * 0.5f, JPH::Vec3(0, segmentRadius + 0.1f, -segmentLength * 4));
    
    // Joints between segments (yaw joints for undulation)
    builder.AddHingeJoint("head", "seg1", JPH::Vec3::sAxisY(), 30.0f, 5.0f);
    builder.AddHingeJoint("seg1", "seg2", JPH::Vec3::sAxisY(), 30.0f, 5.0f);
    builder.AddHingeJoint("seg2", "seg3", JPH::Vec3::sAxisY(), 30.0f, 5.0f);
    builder.AddHingeJoint("seg3", "tail", JPH::Vec3::sAxisY(), 20.0f, 5.0f);
    
    return builder.Build(physics);
}

CombatRobotData RobotTemplates::CreateSnake8Segment(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env) {
    RobotBuilder builder;
    builder.SetName("snake_8seg")
           .SetSpawnPos(spawnPos)
           .SetEnvIndex(env)
           .SetRobotIndex(0);
    
    float segmentLength = 0.35f;
    float segmentRadius = 0.08f;
    float segmentMass = 1.2f;
    
    // Head
    builder.AddCapsuleBody("head", segmentRadius, segmentLength * 0.7f, segmentMass * 1.3f, JPH::Vec3(0, segmentRadius + 0.1f, 0));
    
    // Body segments
    for (int i = 0; i < 7; ++i) {
        std::string name = "seg" + std::to_string(i);
        builder.AddCapsuleBody(name, segmentRadius, segmentLength, segmentMass, 
                               JPH::Vec3(0, segmentRadius + 0.1f, -segmentLength * (i + 1)));
    }
    
    // Tail
    builder.AddCapsuleBody("tail", segmentRadius * 0.7f, segmentLength * 0.4f, segmentMass * 0.4f, 
                           JPH::Vec3(0, segmentRadius + 0.1f, -segmentLength * 8));
    
    // Joints (yaw for undulation)
    builder.AddHingeJoint("head", "seg0", JPH::Vec3::sAxisY(), 25.0f, 6.0f);
    for (int i = 0; i < 6; ++i) {
        std::string parent = "seg" + std::to_string(i);
        std::string child = "seg" + std::to_string(i + 1);
        builder.AddHingeJoint(parent.c_str(), child.c_str(), JPH::Vec3::sAxisY(), 25.0f, 6.0f);
    }
    builder.AddHingeJoint("seg6", "tail", JPH::Vec3::sAxisY(), 15.0f, 6.0f);
    
    return builder.Build(physics);
}

CombatRobotData RobotTemplates::CreateSnake12Segment(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env) {
    RobotBuilder builder;
    builder.SetName("snake_12seg")
           .SetSpawnPos(spawnPos)
           .SetEnvIndex(env)
           .SetRobotIndex(0);
    
    float segmentLength = 0.3f;
    float segmentRadius = 0.07f;
    float segmentMass = 1.0f;
    
    // Head
    builder.AddCapsuleBody("head", segmentRadius, segmentLength * 0.8f, segmentMass * 1.4f, JPH::Vec3(0, segmentRadius + 0.1f, 0));
    
    // Body segments
    for (int i = 0; i < 11; ++i) {
        std::string name = "seg" + std::to_string(i);
        builder.AddCapsuleBody(name, segmentRadius, segmentLength, segmentMass, 
                               JPH::Vec3(0, segmentRadius + 0.1f, -segmentLength * (i + 1)));
    }
    
    // Tail
    builder.AddCapsuleBody("tail", segmentRadius * 0.6f, segmentLength * 0.3f, segmentMass * 0.3f, 
                           JPH::Vec3(0, segmentRadius + 0.1f, -segmentLength * 12));
    
    // Joints (yaw for undulation + pitch for vertical movement)
    builder.AddHingeJoint("head", "seg0", JPH::Vec3::sAxisY(), 20.0f, 8.0f);
    for (int i = 0; i < 10; ++i) {
        std::string parent = "seg" + std::to_string(i);
        std::string child = "seg" + std::to_string(i + 1);
        builder.AddHingeJoint(parent.c_str(), child.c_str(), JPH::Vec3::sAxisY(), 20.0f, 8.0f);
    }
    builder.AddHingeJoint("seg10", "tail", JPH::Vec3::sAxisY(), 10.0f, 8.0f);
    
    return builder.Build(physics);
}

// HYBRID ROBOTS
CombatRobotData RobotTemplates::CreateArmWithGripper(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env) {
    RobotBuilder builder;
    builder.SetName("arm_gripper")
           .SetSpawnPos(spawnPos)
           .SetEnvIndex(env)
           .SetRobotIndex(0);
    
    // Base
    builder.AddBoxBody("base", JPH::Vec3(0.4f, 0.15f, 0.4f), 8.0f, JPH::Vec3::sZero());
    
    // Arm segments
    builder.AddCylinderBody("shoulder", 0.15f, 0.1f, 3.0f, JPH::Vec3(0, 0.15f, 0));
    builder.AddCapsuleBody("lower_arm", 0.08f, 0.35f, 2.5f, JPH::Vec3(0, 0.5f, 0));
    builder.AddCapsuleBody("upper_arm", 0.07f, 0.3f, 2.0f, JPH::Vec3(0, 0.9f, 0));
    builder.AddCapsuleBody("forearm", 0.06f, 0.2f, 1.5f, JPH::Vec3(0, 1.2f, 0));
    
    // Gripper base
    builder.AddBoxBody("gripper_base", JPH::Vec3(0.1f, 0.05f, 0.1f), 0.5f, JPH::Vec3(0, 1.4f, 0));
    
    // Gripper fingers
    builder.AddBoxBody("finger_left", JPH::Vec3(0.03f, 0.08f, 0.05f), 0.2f, JPH::Vec3(-0.06f, 1.45f, 0));
    builder.AddBoxBody("finger_right", JPH::Vec3(0.03f, 0.08f, 0.05f), 0.2f, JPH::Vec3(0.06f, 1.45f, 0));
    
    // Arm joints
    builder.AddHingeJoint("base", "shoulder", JPH::Vec3::sAxisY(), 200.0f, 4.0f);
    builder.AddHingeJoint("shoulder", "lower_arm", JPH::Vec3::sAxisZ(), 150.0f, 6.0f);
    builder.AddHingeJoint("lower_arm", "upper_arm", JPH::Vec3::sAxisZ(), 120.0f, 8.0f);
    builder.AddHingeJoint("upper_arm", "forearm", JPH::Vec3::sAxisZ(), 80.0f, 10.0f);
    builder.AddHingeJoint("forearm", "gripper_base", JPH::Vec3::sAxisX(), 50.0f, 12.0f);
    
    // Gripper finger joints (slider for open/close)
    builder.AddSliderJoint("gripper_base", "finger_left", JPH::Vec3::sAxisX(), -0.03f, 0.03f);
    builder.AddSliderJoint("gripper_base", "finger_right", JPH::Vec3::sAxisX(), -0.03f, 0.03f);
    
    return builder.Build(physics);
}

CombatRobotData RobotTemplates::CreateSnakeWithHead(JPH::PhysicsSystem* physics, JPH::RVec3 spawnPos, uint32_t env) {
    RobotBuilder builder;
    builder.SetName("snake_head")
           .SetSpawnPos(spawnPos)
           .SetEnvIndex(env)
           .SetRobotIndex(0);
    
    float segmentLength = 0.35f;
    float segmentRadius = 0.08f;
    float segmentMass = 1.2f;
    
    // Head with sensors (larger)
    builder.AddCapsuleBody("head", segmentRadius * 1.3f, segmentLength * 0.8f, segmentMass * 1.5f, JPH::Vec3(0, segmentRadius + 0.1f, 0));
    
    // Neck (more flexible)
    builder.AddCapsuleBody("neck", segmentRadius * 0.9f, segmentLength * 0.5f, segmentMass * 0.8f, JPH::Vec3(0, segmentRadius + 0.1f, -segmentLength * 0.7f));
    
    // Body segments
    for (int i = 0; i < 6; ++i) {
        std::string name = "seg" + std::to_string(i);
        builder.AddCapsuleBody(name, segmentRadius, segmentLength, segmentMass, 
                               JPH::Vec3(0, segmentRadius + 0.1f, -segmentLength * (0.7f + i + 1)));
    }
    
    // Tail
    builder.AddCapsuleBody("tail", segmentRadius * 0.6f, segmentLength * 0.4f, segmentMass * 0.4f, 
                           JPH::Vec3(0, segmentRadius + 0.1f, -segmentLength * 7.5f));
    
    // Head/neck joints (yaw + pitch capability via angled axes)
    builder.AddHingeJoint("head", "neck", JPH::Vec3::sAxisY(), 30.0f, 8.0f);
    builder.AddHingeJoint("neck", "seg0", JPH::Vec3::sAxisY(), 25.0f, 7.0f);
    
    // Body joints
    for (int i = 0; i < 5; ++i) {
        std::string parent = "seg" + std::to_string(i);
        std::string child = "seg" + std::to_string(i + 1);
        builder.AddHingeJoint(parent.c_str(), child.c_str(), JPH::Vec3::sAxisY(), 25.0f, 7.0f);
    }
    builder.AddHingeJoint("seg5", "tail", JPH::Vec3::sAxisY(), 15.0f, 7.0f);
    
    return builder.Build(physics);
}
