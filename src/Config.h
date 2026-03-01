#pragma once

#include <string>

struct DamageConfig {
    float multiplier = 10.0f;
    float spikeThreshold = 0.55f;
    float engineThreshold = 1.0f;
    float spikeScale = 0.001f;
    float engineScale = 0.002f;
    float spikeFloor = 0.0f;
    float engineFloor = 0.0f;
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
    float spawnOffset = 2.5f;
    float spawnHeight = 5.0f;
    float initialPushSpeed = 8.0f;
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
    bool enableDamageLogs = false;
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
