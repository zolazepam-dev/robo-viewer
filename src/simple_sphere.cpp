#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <iostream>
#include <random>
#include "PhysicsCore.h"

int main() {
    std::cout << "[JOLTrl] Starting Simple Sphere Simulation..." << std::endl;

    PhysicsCore physicsCore;
    if (!physicsCore.Init(1)) {
        std::cerr << "Failed to initialize PhysicsCore" << std::endl;
        return 1;
    }

    auto& bodyInterface = physicsCore.GetPhysicsSystem().GetBodyInterface();

    // 1. Create Ground
    JPH::BoxShapeSettings groundShapeSettings(JPH::Vec3(50.0f, 1.0f, 50.0f));
    JPH::ShapeSettings::ShapeResult groundShapeResult = groundShapeSettings.Create();
    JPH::ShapeRefC groundShape = groundShapeResult.Get();

    JPH::BodyCreationSettings groundSettings(groundShape, JPH::RVec3(0, -1, 0), JPH::Quat::Identity(), JPH::EMotionType::Static, Layers::STATIC);
    JPH::BodyID groundID = bodyInterface.CreateAndAddBody(groundSettings, JPH::EActivation::DontActivate);

    // 2. Create Sphere
    JPH::SphereShapeSettings sphereShapeSettings(1.0f);
    JPH::ShapeSettings::ShapeResult sphereShapeResult = sphereShapeSettings.Create();
    JPH::ShapeRefC sphereShape = sphereShapeResult.Get();

    // Use MOVING_BASE layer for dynamic objects
    JPH::BodyCreationSettings sphereSettings(sphereShape, JPH::RVec3(0, 5, 0), JPH::Quat::Identity(), JPH::EMotionType::Dynamic, Layers::MOVING_BASE);
    JPH::BodyID sphereID = bodyInterface.CreateAndAddBody(sphereSettings, JPH::EActivation::Activate);

    // 3. Simulation Loop with Random Actions
    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::cout << "Starting simulation loop (100 steps)..." << std::endl;
    for (int i = 0; i < 100; ++i) {
        // Random "Action": Apply an impulse
        JPH::Vec3 randomImpulse(dist(gen), 0, dist(gen));
        bodyInterface.AddImpulse(sphereID, randomImpulse);

        // Step physics
        physicsCore.Step(1.0f / 60.0f);

        // Output position
        JPH::RVec3 pos = bodyInterface.GetCenterOfMassPosition(sphereID);
        if (i % 10 == 0) {
            std::cout << "Step " << i << ": Sphere Position = (" << pos.GetX() << ", " << pos.GetY() << ", " << pos.GetZ() << ")" << std::endl;
        }
    }

    std::cout << "Simulation complete." << std::endl;
    physicsCore.Shutdown();
    return 0;
}
