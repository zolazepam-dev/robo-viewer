#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    std::cout << "Testing JSON file loading..." << std::endl;
    
    // Open the JSON file
    std::ifstream file("robots/combat_bot.json");
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }
    
    std::cout << "File opened successfully" << std::endl;
    
    // Parse the JSON
    json config;
    file >> config;
    
    std::cout << "JSON parsed successfully" << std::endl;
    
    // Access core configuration
    if (config.contains("core")) {
        std::cout << "Core configuration found" << std::endl;
        std::cout << "  Radius: " << config["core"].value("radius", 0.0f) << std::endl;
        std::cout << "  Mass: " << config["core"].value("mass", 0.0f) << std::endl;
    } else {
        std::cout << "Core configuration NOT found" << std::endl;
    }
    
    // Access satellites configuration
    if (config.contains("satellites")) {
        std::cout << "Satellites configuration found" << std::endl;
        std::cout << "  Number of satellites: " << config["satellites"].size() << std::endl;
        
        // Try to access each satellite
        for (size_t i = 0; i < config["satellites"].size(); ++i) {
            std::cout << "  Satellite " << i << ": ";
            try {
                auto& satellite = config["satellites"][i];
                std::cout << "ID=" << satellite.value("id", 0) 
                          << ", Angle=" << satellite.value("offset_angle", 0.0f);
            } catch (const std::exception& e) {
                std::cout << "ERROR: " << e.what();
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "Satellites configuration NOT found" << std::endl;
    }
    
    return 0;
}
