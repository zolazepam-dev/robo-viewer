#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>

#include <iostream>

#include "PhysicsCore.h"
#include "CombatRobot.h"

int main() {
    std::cout << "[SEQUENTIAL TEST] Starting sequential robot loading test..." << std::endl;
    
    // Create our physics core
    PhysicsCore physicsCore;
    if (!physicsCore.Init(1)) {  // Just 1 environment for testing
        std::cerr << "[SEQUENTIAL TEST] Failed to initialize PhysicsCore" << std::endl;
        return 1;
    }
    
    std::cout << "[SEQUENTIAL TEST] PhysicsCore initialized successfully" << std::endl;
    
    // Try to load robots sequentially
    CombatRobotLoader robotLoader;
    JPH::RVec3 spawnPosition1(0, 5, 0);
    JPH::RVec3 spawnPosition2(3, 5, 0);
    
    try {
        std::cout << "[SEQUENTIAL TEST] Loading first robot..." << std::endl;
        CombatRobotData robot1 = robotLoader.LoadRobot(
            "robots/combat_bot.json",
            &physicsCore.GetPhysicsSystem(),
            spawnPosition1,
            0,  // envIndex
            0   // robotIndex
        );
        std::cout << "[SEQUENTIAL TEST] First robot loaded successfully!" << std::endl;
        
        std::cout << "[SEQUENTIAL TEST] Loading second robot..." << std::endl;
        CombatRobotData robot2 = robotLoader.LoadRobot(
            "robots/combat_bot.json",
            &physicsCore.GetPhysicsSystem(),
            spawnPosition2,
            0,  // envIndex
            1   // robotIndex
        );
        std::cout << "[SEQUENTIAL TEST] Second robot loaded successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[SEQUENTIAL TEST] Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[SEQUENTIAL TEST] Unknown exception caught" << std::endl;
        return 1;
    }
    
    std::cout << "[SEQUENTIAL TEST] Done." << std::endl;
    return 0;
}