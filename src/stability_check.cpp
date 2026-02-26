#include <iostream>
#include <vector>
#include <cmath>

// Jolt must be first
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

#include "src/PhysicsCore.h"
#include "src/CombatRobot.h"

int main() {
    PhysicsCore physicsCore;
    if (!physicsCore.Init(1)) {
        std::cerr << "Failed to init physics" << std::endl;
        return 1;
    }

    auto& bodyInterface = physicsCore.GetPhysicsSystem().GetBodyInterface();

    // Create Floor
    JPH::BoxShapeSettings floorShapeSettings(JPH::Vec3(100.0f, 1.0f, 100.0f));
    auto floorShapeResult = floorShapeSettings.Create();
    JPH::BodyCreationSettings floorSettings(floorShapeResult.Get(), JPH::RVec3(0, -1, 0), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC);
    bodyInterface.CreateAndAddBody(floorSettings, JPH::EActivation::DontActivate);

    // Load Robot
    CombatRobotLoader loader;
    std::string configPath = "robots/combat_bot.json";
    JPH::RVec3 spawnPos(0, 5, 0);
    CombatRobotData robot = loader.LoadRobot(configPath, &physicsCore.GetPhysicsSystem(), spawnPos, 0, 0);

    std::cout << "Starting stability check (5 seconds simulation)..." << std::endl;

    JPH::RVec3 lastPos = bodyInterface.GetCenterOfMassPosition(robot.mainBodyId);

    for (int step = 0; step < 300; ++step) {
        physicsCore.Step(1.0f / 60.0f);

        if (step % 60 == 0) {
            JPH::RVec3 currentPos = bodyInterface.GetCenterOfMassPosition(robot.mainBodyId);
            float drift = (float)(currentPos - lastPos).Length();
            std::cout << "Step " << step << " | PosY: " << (float)currentPos.GetY() << " | Drift: " << drift << std::endl;
            lastPos = currentPos;
        }
    }

    JPH::RVec3 finalPos = bodyInterface.GetCenterOfMassPosition(robot.mainBodyId);
    std::cout << "Final Position: " << (float)finalPos.GetX() << ", " << (float)finalPos.GetY() << ", " << (float)finalPos.GetZ() << std::endl;

    // Check drift in the last second
    JPH::RVec3 startFinalCheck = finalPos;
    for (int step = 0; step < 60; ++step) {
        physicsCore.Step(1.0f / 60.0f);
    }
    JPH::RVec3 endFinalCheck = bodyInterface.GetCenterOfMassPosition(robot.mainBodyId);
    float finalDrift = (float)(endFinalCheck - startFinalCheck).Length();

    std::cout << "Final drift over 1 second: " << finalDrift << std::endl;

    if (finalDrift < 0.001f) {
        std::cout << "SUCCESS: Robot is stable at rest." << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE: Robot is drifting!" << std::endl;
        return 1;
    }
}
