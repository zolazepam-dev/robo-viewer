/**
 * @file Types.h
 * @brief Common type definitions for JOLTrl
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include "AlignedAllocator.h"
#include <cstdlib>

/** Default number of parallel environments */

/** Default observation dimension */
constexpr int DEFAULT_OBSERVATION_DIM = 256;

/** Default action dimension */
constexpr int DEFAULT_ACTION_DIM = 56;

/** Vector reward dimension */
constexpr int VECTOR_REWARD_DIM = 5;

/** Initial health points */
constexpr float INITIAL_HP = 100.0f;

/** Maximum episode steps */
constexpr int MAX_EPISODE_STEPS = 7200;

/** Arena size */
constexpr float ARENA_SIZE = 36.0f;
constexpr float ARENA_HALF = ARENA_SIZE * 0.5f;
constexpr float ROBOT_SPAWN_OFFSET = 10.0f;
constexpr float DAMAGE_MULTIPLIER = 5.0f;

/** Aligned allocator for AVX2 */



/** User commands from visualizer */
struct UserCommands {
    bool pause = false;
    bool stepOne = false;
    bool reset = false;
    bool restart = false;
    bool saveModel = false;
    bool loadModel = false;
    int renderEnvIdx = 0;
    float timeScale = 1.0f;
    int stepsPerEpisode = MAX_EPISODE_STEPS;
    std::string modelName;
};

/** Physics layers */
namespace PhysicsLayers {
    constexpr uint16_t STATIC = 0;
    constexpr uint16_t MOVING_BASE = 1;
    constexpr uint16_t GHOST_BASE = 5000;
    
    inline constexpr uint16_t GetEnvLayer(uint32_t envIndex) {
        return MOVING_BASE + envIndex;
    }
    inline constexpr uint16_t GetGhostLayer(uint32_t envIndex) {
        return GHOST_BASE + envIndex;
    }
}

/** Broad-phase layers */
namespace BroadPhaseLayers {
    constexpr uint8_t STATIC = 0;
    constexpr uint8_t DYNAMIC = 1;
    constexpr uint8_t NUM_LAYERS = 2;
}

/** Robot type */
enum class RobotType : uint8_t { SATELLITE, INTERNAL_ENGINE };

/** Training mode */
enum class TrainingMode : uint8_t { TRAIN, EVAL, SELF_PLAY, LEAGUE_PLAY };

/** Alignment assertion */

/** Vector reward structure */
struct VectorReward {
    std::array<float, VECTOR_REWARD_DIM> components{};
    
    float DamageDealt() const { return components[0]; }
    float DamageTaken() const { return components[1]; }
    float Airtime() const { return components[2]; }
    float Energy() const { return components[3]; }
    float Survival() const { return components[4]; }
    
    void SetDamageDealt(float v) { components[0] = v; }
    void SetDamageTaken(float v) { components[1] = v; }
    void SetAirtime(float v) { components[2] = v; }
    void SetEnergy(float v) { components[3] = v; }
    void SetSurvival(float v) { components[4] = v; }
    
    float Dot(const std::array<float, VECTOR_REWARD_DIM>& preferences) const {
        float result = 0.0f;
        for (int i = 0; i < VECTOR_REWARD_DIM; ++i) {
            result += components[i] * preferences[i];
        }
        return result;
    }
    
    void Reset() { components.fill(0.0f); }
};
