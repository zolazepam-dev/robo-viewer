#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    std::cout << "[JSON TEST] Starting JSON load test..." << std::endl;
    
    std::ifstream file("robots/combat_bot.json");
    if (!file.is_open()) {
        std::cerr << "[JSON TEST] FATAL: Failed to open robots/combat_bot.json" << std::endl;
        return 1;
    }
    
    std::cout << "[JSON TEST] File opened successfully" << std::endl;
    
    try {
        json config;
        file >> config;
        
        std::cout << "[JSON TEST] JSON parsed successfully" << std::endl;
        std::cout << "[JSON TEST] Core radius: " << config["core"].value("radius", 0.0f) << std::endl;
        std::cout << "[JSON TEST] Core mass: " << config["core"].value("mass", 0.0f) << std::endl;
        
        const auto& satellitesConfig = config["satellites"];
        std::cout << "[JSON TEST] Number of satellites: " << satellitesConfig.size() << std::endl;
        
        for (size_t i = 0; i < satellitesConfig.size(); ++i) {
            std::cout << "[JSON TEST] Satellite " << i << ": id=" << satellitesConfig[i].value("id", 0) 
                      << ", angle=" << satellitesConfig[i].value("offset_angle", 0.0f) 
                      << ", elevation=" << satellitesConfig[i].value("elevation", 0.0f) 
                      << ", distance=" << satellitesConfig[i].value("distance", 0.0f) << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[JSON TEST] Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[JSON TEST] Unknown exception caught" << std::endl;
        return 1;
    }
    
    std::cout << "[JSON TEST] Done." << std::endl;
    return 0;
}