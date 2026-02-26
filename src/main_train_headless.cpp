// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <filesystem>
#include <string>
#include <thread>

#include "VectorizedEnv.h"
#include "NeuralNetwork.h"
#include "TD3Trainer.h"

namespace fs = std::filesystem;

struct TrainingConfig {
    int numParallelEnvs = 128; 
    int checkpointInterval = 50000;
    int maxSteps = 10000000;
    std::string checkpointDir = "checkpoints";
    std::string loadCheckpoint = "";
};

void EnsureDir(const std::string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
    }
}

int main(int argc, char* argv[]) {
    TrainingConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--envs" && i + 1 < argc) {
            config.numParallelEnvs = std::stoi(argv[++i]);
        } else if (arg == "--checkpoint-interval" && i + 1 < argc) {
            config.checkpointInterval = std::stoi(argv[++i]);
        } else if (arg == "--max-steps" && i + 1 < argc) {
            config.maxSteps = std::stoi(argv[++i]);
        } else if (arg == "--checkpoint-dir" && i + 1 < argc) {
            config.checkpointDir = argv[++i];
        }
    }
    
    EnsureDir(config.checkpointDir);
    EnsureDir("saved_models");
    
    // Initialize Training Environment (Headless)
    std::cout << "[JOLTrl] Initializing headless training with " << config.numParallelEnvs << " parallel environments..." << std::endl;
    VectorizedEnv vecEnv(config.numParallelEnvs);
    vecEnv.Init();
    
    int stateDim = vecEnv.GetObservationDim();
    int actionDim = vecEnv.GetActionDim();
    int totalActionDim = actionDim * 2 * config.numParallelEnvs;
    
    TD3Config td3cfg;
    td3cfg.hiddenDim = 256;
    td3cfg.batchSize = 256;
    td3cfg.startSteps = 10000;
    
    TD3Trainer trainer(stateDim, actionDim, td3cfg);
    ReplayBuffer buffer(td3cfg.bufferSize, stateDim, actionDim);
    
    std::vector<float> actions(totalActionDim, 0.0f);
    std::mt19937 rng(42);
    std::normal_distribution<float> noiseDist(0.0f, 1.0f);
    
    int totalSteps = 0;
    int episodes = 0;
    float currentAvg = 0.0f;
    std::vector<float> avgRewards(100, 0.0f);
    int rewardIdx = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastStatsTime = startTime;
    int lastSteps = 0;
    float sps = 0.0f;

    std::cout << "[JOLTrl] Headless Training Matrix Online. Starting training loop..." << std::endl;
    std::cout << "[JOLTrl] actionDim=" << actionDim << " totalActionDim=" << totalActionDim << " stateDim=" << stateDim << std::endl;

    // The Master Loop (Headless)
    while (totalSteps < config.maxSteps) {

        // --- MACHINE LEARNING PHASE ---
        if (totalSteps < td3cfg.startSteps) {
            for (int i = 0; i < totalActionDim; ++i) actions[i] = noiseDist(rng);
        } else {
            const auto& allObs = vecEnv.GetObservations();
            for (int envIdx = 0; envIdx < config.numParallelEnvs; ++envIdx) {
                const float* obs1 = allObs.data() + envIdx * stateDim * 2;
                const float* obs2 = obs1 + stateDim;
                float* act1 = actions.data() + envIdx * actionDim * 2;
                float* act2 = act1 + actionDim;
                trainer.SelectAction(obs1, act1);
                trainer.SelectAction(obs2, act2);
            }
        }

        vecEnv.Step(actions);
        vecEnv.ResetDoneEnvs();

        const auto& allObs = vecEnv.GetObservations();
        const auto& allRewards = vecEnv.GetRewards();
        const auto& allDones = vecEnv.GetDones();
        
        for (int envIdx = 0; envIdx < config.numParallelEnvs; ++envIdx) {
            const float* obs1 = allObs.data() + envIdx * stateDim * 2;
            const float* obs2 = obs1 + stateDim;
            const float* act1 = actions.data() + envIdx * actionDim * 2;
            const float* act2 = act1 + actionDim;
            float r1 = allRewards[envIdx * 2];
            float r2 = allRewards[envIdx * 2 + 1];
            
            buffer.Add(obs1, act1, r1, obs2, allDones[envIdx]);
            buffer.Add(obs2, act2, r2, obs1, allDones[envIdx]);
            
            avgRewards[rewardIdx % 100] = (r1 + r2) / 2.0f;
            rewardIdx++;
            if (allDones[envIdx]) episodes++;
        }
        
        if (buffer.Size() >= td3cfg.startSteps) trainer.Train(buffer);
        totalSteps++;
        
        // Print statistics periodically
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastStatsTime).count();
        
        if (elapsed >= 5) { // Print stats every 5 seconds
            auto totalElapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
            sps = (totalSteps - lastSteps) / (float)elapsed;
            lastSteps = totalSteps;
            lastStatsTime = currentTime;
            
            currentAvg = 0;
            for(int i=0; i<std::min(rewardIdx, 100); i++) currentAvg += avgRewards[i];
            if (rewardIdx > 0) currentAvg /= std::min(rewardIdx, 100);
            
            std::cout << "[JOLTrl] Steps: " << totalSteps << "/" << config.maxSteps 
                      << " | SPS: " << (int)sps 
                      << " | Episodes: " << episodes 
                      << " | Avg Reward: " << currentAvg << std::endl;
        }
        
        if (totalSteps % config.checkpointInterval == 0) {
            std::string checkpointPath = config.checkpointDir + "/model_" + std::to_string(totalSteps) + ".bin";
            trainer.Save(checkpointPath);
            std::cout << "[JOLTrl] Checkpoint saved: " << checkpointPath << std::endl;
        }
    }
    
    trainer.Save("saved_models/model_final.bin");
    std::cout << "[JOLTrl] Training completed. Final model saved." << std::endl;
    
    return 0;
}