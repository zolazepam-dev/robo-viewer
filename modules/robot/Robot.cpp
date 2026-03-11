/**
 * @file Robot.cpp
 * @brief Implementation of Robot class
 */

#include "Robot.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// PIDController implementation
float PIDController::Compute(float setpoint, float current, float dt) {
    float error = setpoint - current;
    integral += error * dt;
    float derivative = (error - prevError) / dt;
    prevError = error;
    return kp * error + ki * integral + kd * derivative;
}

void PIDController::Reset() {
    integral = 0.0f;
    prevError = 0.0f;
}

// ForceSensorReading implementation

// RobotConfig implementation
RobotConfig RobotConfig::LoadFromJSON(const std::string& path) {
    RobotConfig config;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open robot config: " + path);
    }
    
    json j;
    file >> j;
    
    if (j.contains("core")) {
        const auto& core = j["core"];
        if (core.contains("radius")) config.coreRadius = core["radius"].get<float>();
        if (core.contains("mass")) config.coreMass = core["mass"].get<float>();
        if (core.contains("friction")) config.coreFriction = core["friction"].get<float>();
        if (core.contains("restitution")) config.coreRestitution = core["restitution"].get<float>();
    }
    
    if (j.contains("satellites")) {
        for (const auto& sat : j["satellites"]) {
            RobotConfig::Satellite s;
            if (sat.contains("offsetAngle")) s.offsetAngle = sat["offsetAngle"].get<float>();
            if (sat.contains("elevation")) s.elevation = sat["elevation"].get<float>();
            if (sat.contains("distance")) s.distance = sat["distance"].get<float>();
            if (sat.contains("radius")) s.radius = sat["radius"].get<float>();
            if (sat.contains("mass")) s.mass = sat["mass"].get<float>();
            config.satellites.push_back(s);
        }
    }
    
    if (j.contains("spike")) {
        const auto& spike = j["spike"];
        if (spike.contains("halfHeight")) config.spikeHalfHeight = spike["halfHeight"].get<float>();
        if (spike.contains("radius")) config.spikeRadius = spike["radius"].get<float>();
        if (spike.contains("mass")) config.spikeMass = spike["mass"].get<float>();
    }
    
    if (j.contains("joints")) {
        const auto& joints = j["joints"];
        if (joints.contains("damping")) config.jointDamping = joints["damping"].get<float>();
        if (joints.contains("motorTorque")) config.motorTorque = joints["motorTorque"].get<float>();
        if (joints.contains("slideMin")) config.slideMin = joints["slideMin"].get<float>();
        if (joints.contains("slideMax")) config.slideMax = joints["slideMax"].get<float>();
    }
    
    if (j.contains("sensors")) {
        const auto& sensors = j["sensors"];
        if (sensors.contains("numLidarRays")) config.numLidarRays = sensors["numLidarRays"].get<int>();
        if (sensors.contains("lidarMaxDistance")) config.lidarMaxDistance = sensors["lidarMaxDistance"].get<float>();
    }
    
    if (j.contains("actions")) {
        const auto& actions = j["actions"];
        if (actions.contains("rotationScale")) config.rotationScale = actions["rotationScale"].get<float>();
        if (actions.contains("slideScale")) config.slideScale = actions["slideScale"].get<float>();
    }
    
    config.CalculateDimensions();
    return config;
}

void RobotConfig::CalculateDimensions() {
    numSatellites = static_cast<int>(satellites.size());
    actionsPerRobot = numSatellites * actionsPerSatellite;
    // Base: 18 + 6*numSat + 3*numSat + 3*numSat + LIDAR + force sensors
    observationDim = 18 + (numSatellites * 6) + (numSatellites * 3) + 
                     (numSatellites * 3) + numLidarRays + (numSatellites * 2);
}

// Robot implementation
void Robot::ResetEpisodeStats() {
    mTotalDamageDealt = 0.0f;
    mTotalDamageTaken = 0.0f;
    mTotalEnergyUsed = 0.0f;
    mEpisodeSteps = 0;
}

void Robot::ResizeObservationBuffer(int dim) {
    mObservationBuffer.resize(dim);
}
