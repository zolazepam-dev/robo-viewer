#include "Config.h"

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

namespace {
Config gConfig{};

template <typename T>
void SetIfExists(T& field, const nlohmann::json& obj, const char* key)
{
    auto it = obj.find(key);
    if (it != obj.end()) {
        try {
            field = it->get<T>();
        } catch (const std::exception& e) {
            std::cerr << "Config parse error for key '" << key << "': " << e.what() << std::endl;
        }
    }
}
} // namespace

bool LoadConfig(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Config: unable to open " << path << ". Using defaults." << std::endl;
        return false;
    }

    try {
        nlohmann::json j;
        file >> j;

        if (j.contains("damage")) {
            const auto& d = j["damage"];
            SetIfExists(gConfig.damage.multiplier, d, "multiplier");
            SetIfExists(gConfig.damage.spikeThreshold, d, "spike_threshold");
            SetIfExists(gConfig.damage.engineThreshold, d, "engine_threshold");
            SetIfExists(gConfig.damage.spikeScale, d, "spike_scale");
            SetIfExists(gConfig.damage.engineScale, d, "engine_scale");
            SetIfExists(gConfig.damage.spikeFloor, d, "spike_floor");
            SetIfExists(gConfig.damage.engineFloor, d, "engine_floor");
        }

        if (j.contains("hp")) {
            const auto& h = j["hp"];
            SetIfExists(gConfig.hp.initial, h, "initial");
        }

        if (j.contains("koth")) {
            const auto& k = j["koth"];
            SetIfExists(gConfig.koth.radius, k, "radius");
            SetIfExists(gConfig.koth.randomXZMin, k, "random_xz_min");
            SetIfExists(gConfig.koth.randomXZMax, k, "random_xz_max");
            SetIfExists(gConfig.koth.randomYMin, k, "random_y_min");
            SetIfExists(gConfig.koth.randomYMax, k, "random_y_max");
        }

        if (j.contains("env")) {
            const auto& e = j["env"];
            SetIfExists(gConfig.env.maxSteps, e, "max_steps");
            SetIfExists(gConfig.env.spawnOffset, e, "spawn_offset");
            SetIfExists(gConfig.env.spawnHeight, e, "spawn_height");
            SetIfExists(gConfig.env.initialPushSpeed, e, "initial_push_speed");
        }

        if (j.contains("reward")) {
            const auto& r = j["reward"];
            SetIfExists(gConfig.reward.proximityRange, r, "proximity_range");
            SetIfExists(gConfig.reward.proximityScale, r, "proximity_scale");
            SetIfExists(gConfig.reward.proximityFarPenalty, r, "proximity_far_penalty");
            SetIfExists(gConfig.reward.approachScale, r, "approach_scale");
            SetIfExists(gConfig.reward.wallStart, r, "wall_start");
            SetIfExists(gConfig.reward.wallPenaltyScale, r, "wall_penalty_scale");
            SetIfExists(gConfig.reward.energyScale, r, "energy_scale");
            SetIfExists(gConfig.reward.kothWin, r, "koth_win");
            SetIfExists(gConfig.reward.altitudeScale, r, "altitude_scale");
        }

        if (j.contains("debug")) {
            const auto& d = j["debug"];
            SetIfExists(gConfig.debug.enableDamageLogs, d, "enable_damage_logs");
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Config: parse failed: " << e.what() << ". Using defaults." << std::endl;
        return false;
    }
}

const Config& GetConfig()
{
    return gConfig;
}
