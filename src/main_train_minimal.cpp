// Minimal test to isolate the issue
#include <iostream>
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>

#include "VectorizedEnv.h"

#include "Config.h"

int main() {
    std::cout << "Main function started" << std::endl;
    
    // Load runtime configuration (defaults used if file missing)
    LoadConfig("config/game_config.json");
    
    try {
        std::cout << "Creating VectorizedEnv(1)" << std::endl;
        VectorizedEnv vecEnv(1);
        
        std::cout << "Calling Init()" << std::endl;
        vecEnv.Init();
        std::cout << "Init() completed" << std::endl;
        
        std::cout << "Calling GetNumEnvs(): " << vecEnv.GetNumEnvs() << std::endl;
        
        // Test if we can get action dimension directly
        std::cout << "Calling GetActionDim(): " << vecEnv.GetActionDim() << std::endl;
        
        // Test GetObservationDim() specifically
        std::cout << "Calling GetObservationDim(): " << vecEnv.GetObservationDim() << std::endl;
        
        // This should work if VectorizedEnv is initialized correctly
        std::cout << "Test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        return 2;
    }
    
    std::cout << "Main function exiting normally" << std::endl;
    return 0;
}
