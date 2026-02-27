
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <iostream>
#include <chrono>

#include "PhysicsCore.h"

int main() {
    // Initialize physics core
    PhysicsCore physicsCore;
    physicsCore.Init(1);
    
    JPH::PhysicsSystem& physicsSystem = physicsCore.GetPhysicsSystem();
    JPH::BodyInterface& bodyInterface = physicsSystem.GetBodyInterface();
    
    // Create static floor
    JPH::BoxShapeSettings floorShapeSettings(JPH::Vec3(10.0f, 0.5f, 10.0f));
    JPH::RefConst<JPH::Shape> floorShape = floorShapeSettings.Create().Get();
    JPH::BodyCreationSettings floorSettings(
        floorShape,
        JPH::RVec3(0, 0, 0),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Static,
        Layers::STATIC
    );
    bodyInterface.CreateAndAddBody(floorSettings, JPH::EActivation::DontActivate);
    
    // Create dynamic sphere
    JPH::SphereShapeSettings sphereShapeSettings(0.5f);
    JPH::RefConst<JPH::Shape> sphereShape = sphereShapeSettings.Create().Get();
    JPH::BodyCreationSettings sphereSettings(
        sphereShape,
        JPH::RVec3(0, 5.0f, 0),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING_BASE
    );
    sphereSettings.mFriction = 0.5f;
    sphereSettings.mRestitution = 0.3f;
    JPH::Body* sphereBody = bodyInterface.CreateBody(sphereSettings);
    JPH::BodyID sphereBodyId = sphereBody->GetID();
    bodyInterface.AddBody(sphereBodyId, JPH::EActivation::Activate);
    
    std::cout << "Initial position: (" 
              << bodyInterface.GetPosition(sphereBodyId).GetX() << ", " 
              << bodyInterface.GetPosition(sphereBodyId).GetY() << ", " 
              << bodyInterface.GetPosition(sphereBodyId).GetZ() << ")" << std::endl;
    
    // Step physics
    for (int i = 0; i < 60; ++i) {
        physicsCore.Step(1.0f / 60.0f);
        
        JPH::RVec3 pos = bodyInterface.GetPosition(sphereBodyId);
        JPH::Vec3 vel = bodyInterface.GetLinearVelocity(sphereBodyId);
        
        std::cout << "Step " << i+1 << ": Position = (" << pos.GetX() << ", " << pos.GetY() << ", " << pos.GetZ() 
                  << "), Velocity = (" << vel.GetX() << ", " << vel.GetY() << ", " << vel.GetZ() << ")" << std::endl;
        
        if (pos.GetY() <= 0.5f) {
            std::cout << "Sphere hit floor!" << std::endl;
            break;
        }
    }
    
    return 0;
}
