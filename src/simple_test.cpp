#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>

#include <iostream>

#include "PhysicsCore.h"
#include "CombatRobot.h"

int main() {
    std::cout << "[TEST] Starting simple test..." << std::endl;
    
    // Create our physics core
    PhysicsCore physicsCore;
    if (!physicsCore.Init(1)) {  // Just 1 environment for testing
        std::cerr << "[TEST] Failed to initialize PhysicsCore" << std::endl;
        return 1;
    }
    
    std::cout << "[TEST] PhysicsCore initialized successfully" << std::endl;
    
    // Try to load a single robot
    std::cout << "[TEST] Attempting to load robot..." << std::endl;
    
    CombatRobotLoader robotLoader;
    JPH::RVec3 spawnPosition(0, 5, 0);
    
    try {
        CombatRobotData robot = robotLoader.LoadRobot(
            "robots/combat_bot.json",
            &physicsCore.GetPhysicsSystem(),
            spawnPosition,
            0,  // envIndex
            0   // robotIndex
        );
        
        std::cout << "[TEST] Robot loaded successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[TEST] Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[TEST] Unknown exception caught" << std::endl;
        return 1;
    }
    
    std::cout << "[TEST] Done." << std::endl;
    return 0;
}