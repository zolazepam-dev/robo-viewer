/**
 * @file TrainLoop.h
 * @brief Main training loop orchestrator
 * 
 * Owns and coordinates all modules: VectorEnv, Agent, Visualizer, ReplayBuffer.
 * Runs the main training loop and handles user commands.
 */

#pragma once

#include "../common/Types.h"
#include "../vectorenv/VectorEnv.h"
#include "../replay/ReplayBuffer.h"
#include <memory>
#include <string>
#include <atomic>
#include <chrono>

// Forward declarations
class PhysicsWorld;
class Visualizer;
class TD3Trainer;

/**
 * @brief Training configuration
 */
struct TrainConfig {
    int numEnvs = 64;
    int stepsPerEpisode = 1000;
    int batchSize = 16;
    int bufferCapacity = 1000000;
    int startSteps = 500;
    float physicsHz = 120.0f;
    bool trainEnabled = true;
    bool leaguePlayEnabled = true;
    bool headlessTurbo = false;
    int checkpointInterval = 50000;
    std::string checkpointDir = "checkpoints";
    std::string robotConfigPath = "robots/combat_bot.json";
    std::string loadModelPath;
};

/**
 * @brief Training statistics
 */
struct TrainStats {
    long long totalSteps = 0;
    int episodes = 0;
    float sps = 0.0f;
    float avgReward = 0.0f;
    float agent1Reward = 0.0f;
    float agent2Reward = 0.0f;
    float agent1HP = 100.0f;
    float agent2HP = 100.0f;
    int r1Wins = 0;
    int r2Wins = 0;
    int currentOpponentIdx = 0;
};

/**
 * @brief Main training loop orchestrator
 */
class TrainLoop {
public:
    explicit TrainLoop(const TrainConfig& config);
    ~TrainLoop();
    
    TrainLoop(const TrainLoop&) = delete;
    TrainLoop& operator=(const TrainLoop&) = delete;
    
    /** @brief Initialize all modules */
    bool Init();
    
    /** @brief Run the training loop */
    void Run();
    
    /** @brief Request stop */
    void Stop();
    
    /** @brief Check if running */
    bool IsRunning() const { return mRunning; }
    
    /** @brief Get statistics */
    const TrainStats& GetStats() const { return mStats; }
    
    /** @brief Get config */
    const TrainConfig& GetConfig() const { return mConfig; }
    
    /** @brief Get vectorized environments */
    VectorEnv* GetVectorEnv() { return mVectorEnv.get(); }
    
    /** @brief Get trainer */
    TD3Trainer* GetTrainer() { return mTrainer.get(); }
    
    /** @brief Get replay buffer */
    ReplayBuffer* GetReplayBuffer() { return mReplayBuffer.get(); }
    
    /** @brief Get physics world */
    PhysicsWorld* GetPhysicsWorld() { return mPhysicsWorld.get(); }
    
    /** @brief Get visualizer (nullptr in headless mode) */
    Visualizer* GetVisualizer() { return mVisualizer.get(); }

private:
    void ProcessCommands();
    void StepEnvironments();
    void TrainAgent();
    void Checkpoint();
    void UpdateStats();
    void ResetDoneEnvs();
    
    TrainConfig mConfig;
    TrainStats mStats;
    
    std::unique_ptr<PhysicsWorld> mPhysicsWorld;
    std::unique_ptr<VectorEnv> mVectorEnv;
    std::unique_ptr<TD3Trainer> mTrainer;
    std::unique_ptr<ReplayBuffer> mReplayBuffer;
    std::unique_ptr<Visualizer> mVisualizer;
    
    std::atomic<bool> mRunning;
    std::atomic<bool> mStopRequested;
    
    std::chrono::high_resolution_clock::time_point mLastTime;
    std::chrono::high_resolution_clock::time_point mLastCheckpoint;
    std::chrono::high_resolution_clock::time_point mLastStatsUpdate;
    
    long long mStepCounter = 0;
    int mStepCounterReset = 0;
    float mSpsAccumulator = 0.0f;
};
