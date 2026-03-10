// Include standard libraries first to avoid Jolt conflicts
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>

// Include Jolt and other libraries
#include <Jolt/Jolt.h>
#include <nlohmann/json.hpp>

// Include our header after the dependencies are resolved
#include "ConfigManager.h"

using json = nlohmann::json;

void CentralConfig::LoadFromJSON(const json& j) {
    // Graphics settings
    if (j.contains("graphics")) {
        const auto& g = j["graphics"];
        if (g.contains("cameraAzimuth")) graphics.cameraAzimuth = g["cameraAzimuth"];
        if (g.contains("cameraDistance")) graphics.cameraDistance = g["cameraDistance"];
        if (g.contains("cameraElevation")) graphics.cameraElevation = g["cameraElevation"];
        if (g.contains("showAABBs")) graphics.showAABBs = g["showAABBs"];
        if (g.contains("showCollisionShapes")) graphics.showCollisionShapes = g["showCollisionShapes"];
        if (g.contains("showContactPoints")) graphics.showContactPoints = g["showContactPoints"];
        if (g.contains("showInternalEngines")) graphics.showInternalEngines = g["showInternalEngines"];
        if (g.contains("showRobot1")) graphics.showRobot1 = g["showRobot1"];
        if (g.contains("showRobot2")) graphics.showRobot2 = g["showRobot2"];
    }

    // Physics settings
    if (j.contains("physics")) {
        const auto& p = j["physics"];
        if (p.contains("Baumgarte")) physics.Baumgarte = p["Baumgarte"];
        if (p.contains("allowSleep")) physics.allowSleep = p["allowSleep"];
        if (p.contains("gravityY")) physics.gravityY = p["gravityY"];
        if (p.contains("penetrationSlop")) physics.penetrationSlop = p["penetrationSlop"];
        if (p.contains("positionSteps")) physics.positionSteps = p["positionSteps"];
        if (p.contains("stepsPerEpisode")) physics.stepsPerEpisode = p["stepsPerEpisode"];
        if (p.contains("timeScale")) physics.timeScale = p["timeScale"];
        if (p.contains("timestep")) physics.timestep = p["timestep"];
        if (p.contains("velocitySteps")) physics.velocitySteps = p["velocitySteps"];
        if (p.contains("speculativeContactDistance")) physics.speculativeContactDistance = p["speculativeContactDistance"];
        
        // NEW EXHAUSTIVE PARAMS
        if (p.contains("friction")) physics.friction = p["friction"];
        if (p.contains("restitution")) physics.restitution = p["restitution"];
        if (p.contains("linearDamping")) physics.linearDamping = p["linearDamping"];
        if (p.contains("angularDamping")) physics.angularDamping = p["angularDamping"];
        if (p.contains("maxPenetrationVelocity")) physics.maxPenetrationVelocity = p["maxPenetrationVelocity"];
        if (p.contains("numSubSteps")) physics.numSubSteps = p["numSubSteps"];
        if (p.contains("warmStarting")) physics.warmStarting = p["warmStarting"];
    }

    // Training settings
    if (j.contains("training")) {
        const auto& t = j["training"];
        if (t.contains("robotConfigPath")) training.robotConfigPath = t["robotConfigPath"];
        if (t.contains("checkpointDir")) training.checkpointDir = t["checkpointDir"];
        if (t.contains("checkpointInterval")) training.checkpointInterval = t["checkpointInterval"];
        if (t.contains("numEnvs")) training.numEnvs = t["numEnvs"];
    }

    // Available robot definitions
    if (j.contains("robotDefinitions") && j["robotDefinitions"].is_array()) {
        robotDefinitions.clear();
        for (const auto& robDef : j["robotDefinitions"]) {
            RobotDefinition def;
            if (robDef.contains("name")) def.name = robDef["name"];
            if (robDef.contains("configPath")) def.configPath = robDef["configPath"];
            if (robDef.contains("description")) def.description = robDef["description"];
            robotDefinitions.push_back(def);
        }
    }

    // Current selected robot
    if (j.contains("currentRobotName")) {
        currentRobotName = j["currentRobotName"];
    }
}

json CentralConfig::ToJSON() const {
    json j;
    
    // Graphics settings
    j["graphics"]["cameraAzimuth"] = graphics.cameraAzimuth;
    j["graphics"]["cameraDistance"] = graphics.cameraDistance;
    j["graphics"]["cameraElevation"] = graphics.cameraElevation;
    j["graphics"]["showAABBs"] = graphics.showAABBs;
    j["graphics"]["showCollisionShapes"] = graphics.showCollisionShapes;
    j["graphics"]["showContactPoints"] = graphics.showContactPoints;
    j["graphics"]["showInternalEngines"] = graphics.showInternalEngines;
    j["graphics"]["showRobot1"] = graphics.showRobot1;
    j["graphics"]["showRobot2"] = graphics.showRobot2;
    
    // Physics settings
    j["physics"]["Baumgarte"] = physics.Baumgarte;
    j["physics"]["allowSleep"] = physics.allowSleep;
    j["physics"]["gravityY"] = physics.gravityY;
    j["physics"]["penetrationSlop"] = physics.penetrationSlop;
    j["physics"]["positionSteps"] = physics.positionSteps;
    j["physics"]["stepsPerEpisode"] = physics.stepsPerEpisode;
    j["physics"]["timeScale"] = physics.timeScale;
    j["physics"]["timestep"] = physics.timestep;
    j["physics"]["velocitySteps"] = physics.velocitySteps;
    j["physics"]["speculativeContactDistance"] = physics.speculativeContactDistance;
    
    j["physics"]["friction"] = physics.friction;
    j["physics"]["restitution"] = physics.restitution;
    j["physics"]["linearDamping"] = physics.linearDamping;
    j["physics"]["angularDamping"] = physics.angularDamping;
    j["physics"]["maxPenetrationVelocity"] = physics.maxPenetrationVelocity;
    j["physics"]["numSubSteps"] = physics.numSubSteps;
    j["physics"]["warmStarting"] = physics.warmStarting;
    
    // Training settings
    j["training"]["robotConfigPath"] = training.robotConfigPath;
    j["training"]["checkpointDir"] = training.checkpointDir;
    j["training"]["checkpointInterval"] = training.checkpointInterval;
    j["training"]["numEnvs"] = training.numEnvs;

    // Available robot definitions
    json robotDefsArray = json::array();
    for (const auto& def : robotDefinitions) {
        json robotDef;
        robotDef["name"] = def.name;
        robotDef["configPath"] = def.configPath;
        robotDef["description"] = def.description;
        robotDefsArray.push_back(robotDef);
    }
    j["robotDefinitions"] = robotDefsArray;

    // Current selected robot
    j["currentRobotName"] = currentRobotName;

    return j;
}

bool ConfigManager::LoadConfig(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        // Create default configuration if file doesn't exist
        std::cout << "[ConfigManager] Config file not found: " << filepath << ", creating defaults" << std::endl;
        
        // Add default robot definitions
        m_config.robotDefinitions = {
            {"BouncyOrbiter", "robots/bouncy_orbiter.json", "Large bouncy sphere with 3 double-sized bouncy orbiters"},
            {"InternalEngineBot", "robots/internal_engine.json", "Standard internal engine robot"},
            {"ShellBot", "robots/shell_bot.json", "Shell-based combat robot"},
            {"WedgeBot", "robots/wedge_bot.json", "Wedge-shaped robot"}
        };
        
        // Save the defaults so the file exists
        SaveConfig(filepath);
        
        return true; // We now have a valid (default) config
    }

    try {
        json j;
        file >> j;
        m_config.LoadFromJSON(j);
        std::cout << "[ConfigManager] Loaded configuration from: " << filepath << std::endl;
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ConfigManager] Error loading config: " << e.what() << std::endl;
        file.close();
        return false;
    }
}

bool ConfigManager::SaveConfig(const std::string& filepath) const {
    try {
        json j = m_config.ToJSON();
        std::ofstream file(filepath);
        if (file.is_open()) {
            file << j.dump(2);
            file.close();
            std::cout << "[ConfigManager] Saved configuration to: " << filepath << std::endl;
            return true;
        }
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[ConfigManager] Error saving config: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::string> ConfigManager::GetAvailableRobotNames() const {
    std::vector<std::string> names;
    for (const auto& def : m_config.robotDefinitions) {
        names.push_back(def.name);
    }
    return names;
}

const RobotDefinition* ConfigManager::GetRobotDefinition(const std::string& name) const {
    for (const auto& def : m_config.robotDefinitions) {
        if (def.name == name) {
            return &def;
        }
    }
    return nullptr;
}

bool ConfigManager::SetCurrentRobot(const std::string& name) {
    for (const auto& def : m_config.robotDefinitions) {
        if (def.name == name) {
            m_config.currentRobotName = name;
            m_config.training.robotConfigPath = def.configPath;
            return true;
        }
    }
    return false;
}
