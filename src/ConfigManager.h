#pragma once

#include <Jolt/Jolt.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <iostream>

using json = nlohmann::json;

struct RobotDefinition {
    std::string name;
    std::string configPath;
    std::string description;
};

struct CentralConfig {
    struct Graphics {
        float cameraAzimuth = 45.0f;
        float cameraDistance = 20.0f;
        float cameraElevation = 30.0f;
        bool showAABBs = false;
        bool showCollisionShapes = false;
        bool showContactPoints = false;
        bool showInternalEngines = true;
        bool showRobot1 = true;
        bool showRobot2 = true;
    } graphics;

    struct Physics {
        float Baumgarte = 0.3f;
        bool allowSleep = false;
        float gravityY = -9.81f;
        float penetrationSlop = 0.005f;
        int positionSteps = 3;
        int stepsPerEpisode = 1000;
        float timeScale = 1.0f;
        float timestep = 1.0f / 120.0f;
        int velocitySteps = 8;
        float speculativeContactDistance = 0.01f;
        
        // NEW ADVANCED SOLVER PARAMS
        float friction = 0.5f;
        float restitution = 0.0f;
        float linearDamping = 0.05f;
        float angularDamping = 0.05f;
        float maxPenetrationVelocity = 1.0f;
        int numSubSteps = 1;
        bool warmStarting = true;
    } physics;

    struct Training {
        std::string robotConfigPath = "robots/bouncy_orbiter.json";
        std::string checkpointDir = "checkpoints";
        int checkpointInterval = 50000;
        int numEnvs = 1;
    } training;

    std::vector<RobotDefinition> robotDefinitions;
    std::string currentRobotName = "BouncyOrbiter";

    void LoadFromJSON(const json& j);
    json ToJSON() const;
};

class ConfigManager {
public:
    static ConfigManager& GetInstance() {
        static ConfigManager instance;
        return instance;
    }

    bool LoadConfig(const std::string& filepath = "viewer_config.json");
    bool SaveConfig(const std::string& filepath = "viewer_config.json") const;

    CentralConfig& GetConfig() { return m_config; }
    const CentralConfig& GetConfig() const { return m_config; }

    std::vector<std::string> GetAvailableRobotNames() const;
    const RobotDefinition* GetRobotDefinition(const std::string& name) const;
    bool SetCurrentRobot(const std::string& name);

private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    CentralConfig m_config;
};
