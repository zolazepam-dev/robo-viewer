#pragma once

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

/**
 * @enum RobotType
 * @brief Type of combat robot
 */
enum class RobotType {
    SATELLITE, ///< Satellite-based robot design
    INTERNAL_ENGINE, ///< Internal engine robot design
    AIRCRAFT ///< Aircraft with aerodynamic forces
};

/**
 * @struct SatelliteConfig
 * @brief Configuration for a single satellite component
 */
struct SatelliteConfig {
    float offsetAngle = 0.0f; 
    float elevation = 0.0f;
    float distance = 1.4f;
    float radius = 0.1f;
    float mass = 3.5f;
    float friction = 0.5f;
    float restitution = 0.2f;
    float linearDamping = 0.1f;
    float angularDamping = 0.1f;
};

/**
 * @struct SpikeConfig
 * @brief Configuration for the combat spike weapon
 */
struct SpikeConfig {
    float halfHeight = 0.2f;
    float radius = 0.02f;
    float mass = 0.5f;
    float friction = 0.0f;
    float restitution = 0.3f;
    float convexRadius = 0.01f;
};

/**
 * @struct RobotConfig
 * @brief Pure data structure containing robot morphology and parameters
 */
struct RobotConfig {
    std::string name = "unknown_robot";
    RobotType type = RobotType::SATELLITE;
    
    // Core parameters
    float coreRadius = 0.5f;
    float coreMass = 13.0f;
    float coreFriction = 0.5f;
    float coreRestitution = 0.2f;
    float coreLinearDamping = 0.1f;
    float coreAngularDamping = 0.1f;

    std::vector<SatelliteConfig> satellites;
    SpikeConfig spike;

    // Aircraft-specific parameters
    float thrustMax = 350000.0f;
    bool isUnifiedBody = false; // For compound shapes like the F-22

    // Shape parameters for general parts
    struct PartShape {
        std::string type = "sphere"; // "sphere", "box", "cylinder", "hull"
        std::vector<float> dimensions; // radii, half-extents, or points
        std::vector<float> relativePos;
        std::vector<float> relativeRot;
    };
    std::vector<PartShape> partShapes;

    // --- NEW: General Multi-Body Support ---
    struct BodyConfig {
        std::string name;
        std::string shapeType = "box";
        std::vector<float> shapeParams; // e.g. halfExtents for box
        std::vector<float> position;
        std::vector<float> rotation; // Quaternion [x,y,z,w]
        float mass = 10.0f;
        float friction = 0.5f;
        float restitution = 0.2f;
    };
    std::vector<BodyConfig> bodies;

    struct JointConfig {
        std::string type = "hinge";
        std::string name;
        std::string body1;
        std::string body2;
        std::vector<float> position;
        std::vector<float> axis;
        float minLimit = -3.14159f;
        float maxLimit = 3.14159f;
        bool hasMotor = true;
        float motorMaxTorque = 500.0f;
    };
    std::vector<JointConfig> joints;
    // ---------------------------------------

    // Joint parameters
    float jointDamping = 0.8f;
    float jointArmature = 0.5f;
    float motorTorque = 450.0f;
    float slideMin = 0.0f;
    float slideMax = 0.5f;
    float motorMinTorqueLimit = -500.0f;
    float motorMaxTorqueLimit = 500.0f;

    // Direct torque control (instead of motors)
    bool useDirectTorque = false;
    float orbiterTorqueScale = 1000.0f;

    // Sensor parameters
    int numLidarRays = 10;
    float lidarMaxDistance = 20.0f;

    // Action scaling
    int actionsPerSatellite = 4;
    int reactionWheelDim = 4;
    float rotationScale = 25.0f;
    float slideScale = 100.0f;
    float reactionTorqueScale = 5000.0f;

    // Derived dimensions
    int numSatellites = 0;
    int actionsPerRobot = 0;
    int observationDim = 256;

    static RobotConfig LoadFromJSON(const nlohmann::json& config);
    void CalculateDimensions();
};
