#pragma once

#include <string>

struct DamageConfig {
    float multiplier = 10.0f;
    float spikeThreshold = 3.5f;     // Was 0.55 - now accounts for 2m radius shells
    float engineThreshold = 4.0f;     // Was 1.0 - now accounts for shell contact
    float spikeScale = 0.002f;        // Doubled for more visible damage
    float engineScale = 0.003f;       // Doubled for more visible damage
    float spikeFloor = 0.1f;          // Minimum damage even at low velocity
    float engineFloor = 0.15f;        // Minimum damage even at low velocity
};

struct HpConfig {
    float initial = 100.0f;
};

struct KothConfig {
    float radius = 0.8f;
    float randomXZMin = -15.0f;
    float randomXZMax = 15.0f;
    float randomYMin = 2.0f;
    float randomYMax = 12.0f;
};

struct EnvConfig {
    int maxSteps = 7200;
    float spawnOffset = 2.0f;  // Reduced from 2.5 - robots now 4m apart (touching)
    float spawnHeight = 5.0f;
    float initialPushSpeed = 5.0f;  // Push robots toward each other at spawn
};

struct RewardConfig {
    float proximityRange = 15.0f;
    float proximityScale = 0.1f;
    float proximityFarPenalty = 0.05f;
    float approachScale = 0.02f;
    float wallStart = 14.0f;
    float wallPenaltyScale = 0.1f;
    float energyScale = 0.01f;
    float kothWin = 0.1f;
    float altitudeScale = 0.05f;
};

struct DebugConfig {
    bool enableDamageLogs = false;  // Set to true for combat debugging
};

struct Config {
    DamageConfig damage;
    HpConfig hp;
    KothConfig koth;
    EnvConfig env;
    RewardConfig reward;
    DebugConfig debug;
};

// Load JSON config into global cached struct. Returns false if file missing/unreadable.
bool LoadConfig(const std::string& path);

// Access cached config (thread-safe for read after LoadConfig).
const Config& GetConfig();
