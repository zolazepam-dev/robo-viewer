#include "RobotConfig.h"
#include <cmath>

using json = nlohmann::json;

RobotConfig RobotConfig::LoadFromJSON(const json& config) {
    RobotConfig robotConfig;

    robotConfig.name = config.value("name", "unknown_robot");
    if (robotConfig.name == "unknown_robot" && config.contains("metadata") && config["metadata"].contains("name")) {
        robotConfig.name = config["metadata"]["name"].get<std::string>();
    }

    std::string typeStr = config.value("type", "satellite");
    if (typeStr == "aircraft") robotConfig.type = RobotType::AIRCRAFT;
    else if (typeStr == "internal_engine") robotConfig.type = RobotType::INTERNAL_ENGINE;
    else robotConfig.type = RobotType::SATELLITE;

    if (config.contains("core")) {
        const auto& core = config["core"];
        robotConfig.coreRadius = core.value("radius", 0.5f);
        robotConfig.coreMass = core.value("mass", 13.0f);
        robotConfig.coreFriction = core.value("friction", 0.5f);
        robotConfig.coreRestitution = core.value("restitution", 0.2f);
        robotConfig.coreLinearDamping = core.value("linear_damping", 0.1f);
        robotConfig.coreAngularDamping = core.value("angular_damping", 0.1f);
    }

    if (config.contains("satellites") && config["satellites"].is_array()) {
        for (const auto& satConfig : config["satellites"]) {
            SatelliteConfig satellite;
            satellite.offsetAngle = satConfig.value("offset_angle", 0.0f);
            satellite.elevation = satConfig.value("elevation", 0.0f);
            satellite.distance = satConfig.value("distance", 1.4f);
            satellite.radius = satConfig.value("radius", 0.1f);
            satellite.mass = satConfig.value("mass", 3.5f);
            satellite.friction = satConfig.value("friction", 0.5f);
            satellite.restitution = satConfig.value("restitution", 0.2f);
            satellite.linearDamping = satConfig.value("linear_damping", 0.1f);
            satellite.angularDamping = satConfig.value("angular_damping", 0.1f);
            robotConfig.satellites.push_back(satellite);
        }
    } else if (!config.contains("part_shapes") && !config.contains("bodies")) {
        // Default 6 satellites only if no other morphology defined
        std::vector<float> azimuths = {0.0f, 72.0f, 144.0f, 216.0f, 288.0f, 0.0f};
        std::vector<float> elevations = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 45.0f};
        for (size_t i = 0; i < 6; ++i) {
            SatelliteConfig satellite;
            satellite.offsetAngle = azimuths[i];
            satellite.elevation = elevations[i];
            robotConfig.satellites.push_back(satellite);
        }
    }

    if (config.contains("spike")) {
        const auto& spike = config["spike"];
        robotConfig.spike.halfHeight = spike.value("half_height", 0.2f);
        robotConfig.spike.radius = spike.value("radius", 0.02f);
        robotConfig.spike.mass = spike.value("mass", 0.5f);
        robotConfig.spike.friction = spike.value("friction", 0.0f);
        robotConfig.spike.restitution = spike.value("restitution", 0.3f);
        robotConfig.spike.convexRadius = spike.value("convex_radius", 0.01f);
    }

    if (config.contains("joints")) {
        const auto& joints = config["joints"];
        robotConfig.jointDamping = joints.value("hinge_damping", 0.8f);
        robotConfig.jointArmature = joints.value("hinge_armature", 0.5f);
        robotConfig.motorTorque = joints.value("motor_torque", 450.0f);
        if (joints.contains("slide_range") && joints["slide_range"].is_array() && joints["slide_range"].size() >= 2) {
            robotConfig.slideMin = joints["slide_range"][0].get<float>();
            robotConfig.slideMax = joints["slide_range"][1].get<float>();
        }
        robotConfig.motorMinTorqueLimit = joints.value("motor_min_torque", -500.0f);
        robotConfig.motorMaxTorqueLimit = joints.value("motor_max_torque", 500.0f);
    }

    if (config.contains("sensors")) {
        const auto& sensors = config["sensors"];
        robotConfig.numLidarRays = sensors.value("lidar_rays", 10);
        robotConfig.lidarMaxDistance = sensors.value("lidar_max_distance", 20.0f);
    }

    robotConfig.isUnifiedBody = config.value("is_unified", false);

    if (config.contains("actions")) {
        const auto& actions = config["actions"];
        robotConfig.actionsPerSatellite = actions.value("per_satellite", 4);
        robotConfig.reactionWheelDim = actions.value("reaction_wheel_dim", 4);
        robotConfig.rotationScale = actions.value("rotation_scale", 25.0f);
        robotConfig.slideScale = actions.value("slide_scale", 100.0f);
        robotConfig.reactionTorqueScale = actions.value("reaction_torque_scale", 5000.0f);
    }

    // Direct torque control for orbiters
    robotConfig.useDirectTorque = config.value("useDirectTorque", false);
    robotConfig.orbiterTorqueScale = config.value("orbiterTorqueScale", 1000.0f);

    if (config.contains("part_shapes") && config["part_shapes"].is_array()) {
        if (!config.contains("is_unified")) {
             robotConfig.isUnifiedBody = (robotConfig.type != RobotType::INTERNAL_ENGINE); 
        }
        for (const auto& shapeConfig : config["part_shapes"]) {
            PartShape part;
            part.type = shapeConfig.value("type", "sphere");
            if (shapeConfig.contains("dimensions") && shapeConfig["dimensions"].is_array())
                for (float d : shapeConfig["dimensions"]) part.dimensions.push_back(d);
            if (shapeConfig.contains("relative_pos") && shapeConfig["relative_pos"].is_array())
                for (float p : shapeConfig["relative_pos"]) part.relativePos.push_back(p);
            if (shapeConfig.contains("relative_rot") && shapeConfig["relative_rot"].is_array())
                for (float r : shapeConfig["relative_rot"]) part.relativeRot.push_back(r);
            robotConfig.partShapes.push_back(part);
        }
    }

    // --- NEW: Multi-body support ---
    if (config.contains("bodies") && config["bodies"].is_array()) {
        for (const auto& bJson : config["bodies"]) {
            BodyConfig b;
            b.name = bJson.value("name", "");
            if (bJson.contains("shape")) {
                const auto& s = bJson["shape"];
                b.shapeType = s.value("type", "box");
                if (b.shapeType == "box" && s.contains("halfExtents")) {
                    for (float v : s["halfExtents"]) b.shapeParams.push_back(v);
                } else if (s.contains("radius")) {
                    b.shapeParams.push_back(s["radius"].get<float>());
                }
            }
            if (bJson.contains("position")) for (float v : bJson["position"]) b.position.push_back(v);
            if (bJson.contains("rotation")) for (float v : bJson["rotation"]) b.rotation.push_back(v);
            b.mass = bJson.value("mass", 10.0f);
            if (bJson.contains("material")) {
                b.friction = bJson["material"].value("friction", 0.5f);
                b.restitution = bJson["material"].value("restitution", 0.2f);
            }
            robotConfig.bodies.push_back(b);
        }
    }

    if (config.contains("constraints") && config["constraints"].is_array()) {
        for (const auto& cJson : config["constraints"]) {
            JointConfig j;
            j.type = cJson.value("type", "hinge");
            j.name = cJson.value("name", "");
            j.body1 = cJson.value("body1", "");
            j.body2 = cJson.value("body2", "");
            if (cJson.contains("position")) for (float v : cJson["position"]) j.position.push_back(v);
            if (cJson.contains("hingeAxis")) for (float v : cJson["hingeAxis"]) j.axis.push_back(v);
            if (cJson.contains("limits")) {
                j.minLimit = cJson["limits"].value("min", -3.14159f);
                j.maxLimit = cJson["limits"].value("max", 3.14159f);
            }
            if (cJson.contains("motor")) {
                j.hasMotor = true;
                j.motorMaxTorque = cJson["motor"].value("maxForce", 500.0f);
            }
            robotConfig.joints.push_back(j);
        }
    }

    robotConfig.CalculateDimensions();
    return robotConfig;
}

void RobotConfig::CalculateDimensions() {
    numSatellites = static_cast<int>(satellites.size());
    int rawActionsPerRobot = 0;
    int rawObservationDim = 0;

    if (!bodies.empty()) {
        // Multi-body robot (e.g. Octopod, BouncyOrbiter) - use EXPANDED observations
        int numJoints = 0;
        for (const auto& j : joints) if (j.hasMotor) numJoints++;
        
        rawActionsPerRobot = numJoints + reactionWheelDim;
        
        // EXPANDED observation space for multi-body robots (same as satellite layout)
        // Base features: 9+9+6+4+6+6 = 40
        // Body-specific: bodies.size() * 12 (position, velocity, relative positions)
        // LIDAR + compound features: numLidarRays + 8 + 8 + 6 + 8 + 6 + 3 + 6 + 10 = numLidarRays + 55
        rawObservationDim = 40 + (static_cast<int>(bodies.size()) * 12) + numLidarRays + 55;
    } else {
        // Standard Satellite-based layout with EXPANDED observations
        rawActionsPerRobot = numSatellites * actionsPerSatellite + (type != RobotType::AIRCRAFT ? reactionWheelDim : 0);
        
        // Comprehensive observation space (all features from RobotController.cpp):
        // 1. Self body state: 9 (pos, vel, angVel)
        // 2. Opponent body state: 9 (pos, vel, angVel)
        // 3. Relative position/velocity: 6
        // 4. Distance metrics: 4 (normalized, inverse, squared, sqrt)
        // 5. Direction vectors: 6 (dirToOpponent, myForward)
        // 6. Facing angles: 6 (facingDot, oppFacingDot, relativeFacing, etc.)
        // 7. Satellites: 12 * numSatellites
        // 8. LIDAR: numLidarRays
        // 9. Compound LIDAR: 8 (min, max, avg, variance, binary flags)
        // 10. HP/energy: 8
        // 11. Damage stats: 6
        // 12. Velocity features: 8
        // 13. Arena positioning: 6
        // 14. Time/episode: 3
        // 15. Angular features: 6
        // 16. Advanced compound: 10
        
        rawObservationDim = 9 + 9 + 6 + 4 + 6 + 6 + 
                           (numSatellites * 12) + 
                           numLidarRays + 8 + 8 + 6 + 8 + 6 + 3 + 6 + 10;
        
        if (type == RobotType::AIRCRAFT) {
            rawActionsPerRobot = 12;
            rawObservationDim = 120;  // Aircraft uses simplified obs
        }
    }

    // Round up to next multiple of 8 for SIMD alignment
    actionsPerRobot = (rawActionsPerRobot + 7) & ~7;
    observationDim = (rawObservationDim + 7) & ~7;
    
    if (actionsPerRobot < 8) actionsPerRobot = 8;
    if (observationDim < 16) observationDim = 16;
}
