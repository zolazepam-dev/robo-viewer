#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    std::cout << "Testing minimal JSON loading..." << std::endl;
    
    std::ifstream file("robots/combat_bot.json");
    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        return 1;
    }
    
    std::cout << "File opened successfully" << std::endl;
    
    json config;
    file >> config;
    file.close();
    
    std::cout << "JSON parsing completed" << std::endl;
    
    const auto& satellitesConfig = config["satellites"];
    std::cout << "satellitesConfig type: " << satellitesConfig.type_name() << std::endl;
    
    if (satellitesConfig.is_array()) {
        std::cout << "satellitesConfig is array, size: " << satellitesConfig.size() << std::endl;
        
        for (int i = 0; i < 13; ++i) {
            std::cout << "Index " << i << ": ";
            try {
                if (i < satellitesConfig.size()) {
                    auto sat = satellitesConfig[i];
                    std::cout << "exists. Type: " << sat.type_name() << ", contains 'offset_angle': " << sat.contains("offset_angle");
                    if (sat.contains("offset_angle")) {
                        std::cout << ", value type: " << sat["offset_angle"].type_name();
                        if (sat["offset_angle"].is_number()) {
                            float value = sat["offset_angle"].get<float>();
                            std::cout << ", value: " << value;
                        }
                    }
                } else {
                    std::cout << "out of bounds";
                }
            } catch (const std::exception& e) {
                std::cout << "EXCEPTION: " << e.what();
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "satellitesConfig is NOT an array!" << std::endl;
    }
    
    return 0;
}
