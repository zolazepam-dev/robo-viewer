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
#include <cstdio>
#include <ctime>

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

std::string GenerateCheckpointDir(const TrainingConfig& config, int stateDim, int actionDim) {
    std::string baseDir = "checkpoints";
    char timestamp[64];
    time_t now = time(nullptr);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));
    
    std::string dir = baseDir + "/obs" + std::to_string(stateDim) + 
                      "_act" + std::to_string(actionDim) + 
                      "_envs" + std::to_string(config.numParallelEnvs) + 
                      "_" + timestamp;
    return dir;
}

int main(int argc, char* argv[]) {
    TrainingConfig config;
    
    // Debug command line
    std::cout << "[JOLTrl DEBUG] Command line arguments:";
    for (int i = 0; i < argc; i++) {
        std::cout << " " << argv[i];
    }
    std::cout << std::endl;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        std::cout << "[JOLTrl DEBUG] Parsing arg[" << i << "]: \"" << arg << "\"" << std::endl;
        
        // Handle --arg=value syntax
        size_t equalsPos = arg.find('=');
        if (equalsPos != std::string::npos) {
            std::string key = arg.substr(0, equalsPos);
            std::string value = arg.substr(equalsPos + 1);
            
            if (key == "--envs") {
                config.numParallelEnvs = std::stoi(value);
                std::cout << "[JOLTrl] Command line: numParallelEnvs = " << config.numParallelEnvs << std::endl;
            } else if (key == "--checkpoint-interval") {
                config.checkpointInterval = std::stoi(value);
            } else if (key == "--max-steps") {
                config.maxSteps = std::stoi(value);
            } else if (key == "--checkpoint-dir") {
                config.checkpointDir = value;
            }
            continue;
        }
        
        // Handle --arg value syntax
        if (arg == "--envs" && i + 1 < argc) {
            config.numParallelEnvs = std::stoi(argv[++i]);
            std::cout << "[JOLTrl] Command line: numParallelEnvs = " << config.numParallelEnvs << std::endl;
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
    VectorizedEnv vecEnv(config.numParallelEnvs, 7200); // Default 7200 steps per episode
    vecEnv.Init("robots/combat_bot.json");
    
    int stateDim = vecEnv.GetObservationDim();
    int actionDim = vecEnv.GetActionDim();
    int totalActionDim = actionDim * 2 * config.numParallelEnvs;
    
    // Generate timestamped checkpoint dir based on obs/action dims
    config.checkpointDir = GenerateCheckpointDir(config, stateDim, actionDim);
    EnsureDir(config.checkpointDir);
    EnsureDir(config.checkpointDir + "/replay");
    std::cout << "[JOLTrl] Checkpoints will be saved to: " << config.checkpointDir << std::endl;
    
    TD3Config td3cfg;
    td3cfg.hiddenDim = 256;
    td3cfg.batchSize = 256;
    td3cfg.startSteps = 10000;
    
    TD3Trainer trainer(stateDim, actionDim, td3cfg);
    ReplayBuffer buffer(td3cfg.bufferSize, stateDim, actionDim);
    
    AlignedVector32<float> actions(totalActionDim, 0.0f);
    AlignedVector32<float> prevObs(config.numParallelEnvs * stateDim * 2, 0.0f);
    bool firstStep = true;
    
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

    const std::string stateFile = "/tmp/jolt_training_state.json";
    
    while (totalSteps < config.maxSteps) {
        auto loopStart = std::chrono::high_resolution_clock::now();
        
        // 1. Get Current Observations
        const auto& allObs = vecEnv.GetObservations();
        if (!firstStep) {
            // Store transitions from PREVIOUS step to CURRENT state
            const auto& allRewards = vecEnv.GetRewards();
            const auto& allDones = vecEnv.GetDones();
            
            for (int envIdx = 0; envIdx < config.numParallelEnvs; ++envIdx) {
                // Robot 1 transition
                const float* s1 = prevObs.data() + envIdx * stateDim * 2;
                const float* s1_next = allObs.data() + envIdx * stateDim * 2;
                const float* a1 = actions.data() + envIdx * actionDim * 2;
                float r1 = allRewards[envIdx * 2];
                
                // Robot 2 transition
                const float* s2 = s1 + stateDim;
                const float* s2_next = s1_next + stateDim;
                const float* a2 = a1 + actionDim;
                float r2 = allRewards[envIdx * 2 + 1];
                
                buffer.Add(s1, a1, r1, s1_next, allDones[envIdx]);
                buffer.Add(s2, a2, r2, s2_next, allDones[envIdx]);
                
                avgRewards[rewardIdx % 100] = (r1 + r2) / 2.0f;
                rewardIdx++;
                if (allDones[envIdx]) episodes++;
            }
        }
        
        // 2. Cache current obs for next step's transition
        std::copy(allObs.begin(), allObs.end(), prevObs.begin());
        firstStep = false;

        // 3. Select Next Actions
        auto actionStart = std::chrono::high_resolution_clock::now();
        if (totalSteps < td3cfg.startSteps) {
            for (int i = 0; i < totalActionDim; ++i) actions[i] = noiseDist(rng);
        } else {
            for (int envIdx = 0; envIdx < config.numParallelEnvs; ++envIdx) {
                const float* obs1 = allObs.data() + envIdx * stateDim * 2;
                const float* obs2 = obs1 + stateDim;
                float* act1 = actions.data() + envIdx * actionDim * 2;
                float* act2 = act1 + actionDim;
                trainer.SelectAction(obs1, act1);
                trainer.SelectAction(obs2, act2);
            }
        }
        auto actionEnd = std::chrono::high_resolution_clock::now();
        auto actionTime = std::chrono::duration_cast<std::chrono::microseconds>(actionEnd - actionStart).count();

        // 4. Step Environment
        auto stepStart = std::chrono::high_resolution_clock::now();
        vecEnv.Step(actions);
        vecEnv.ResetDoneEnvs();
        auto stepEnd = std::chrono::high_resolution_clock::now();
        auto stepTime = std::chrono::duration_cast<std::chrono::microseconds>(stepEnd - stepStart).count();

        // 5. Train
        auto trainStart = std::chrono::high_resolution_clock::now();
        if (buffer.Size() >= td3cfg.startSteps) trainer.Train(buffer);
        auto trainEnd = std::chrono::high_resolution_clock::now();
        auto trainTime = std::chrono::duration_cast<std::chrono::microseconds>(trainEnd - trainStart).count();

        totalSteps++;
        
        auto loopEnd = std::chrono::high_resolution_clock::now();
        auto loopTime = std::chrono::duration_cast<std::chrono::microseconds>(loopEnd - loopStart).count();
        
        if (totalSteps % 100 == 0) {
            std::cout << "[Timing] Step " << totalSteps 
                      << " | Loop: " << loopTime << "us" 
                      << " | Action: " << actionTime << "us"
                      << " | Step: " << stepTime << "us"
                      << " | Train: " << trainTime << "us" << std::endl;
        }
        
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastStatsTime).count();
        
        // Output SPS every 500 steps for testing
        if (totalSteps % 500 == 0) {
            auto totalElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() / 1000.0f;
            if (totalElapsed > 0.1f) { // At least 100ms elapsed
                sps = totalSteps / totalElapsed;
                currentAvg = 0;
                int count = std::min(rewardIdx, 100);
                for(int i=0; i<count; i++) {
                    int idx = (rewardIdx - count + i) % 100;
                    if (idx < 0) idx += 100; 
                    currentAvg += avgRewards[idx];
                }
                if (rewardIdx > 0) currentAvg /= count;
                
                std::cout << "[JOLTrl] Steps: " << totalSteps << "/" << config.maxSteps 
                          << " | SPS: " << (int)sps 
                          << " | Episodes: " << episodes 
                          << " | Avg Reward: " << currentAvg << std::endl;
            }
        }
        
        if (elapsed >= 1 && (totalSteps - lastSteps) > 0) { 
            sps = (totalSteps - lastSteps) / (float)elapsed;
            lastSteps = totalSteps;
            lastStatsTime = currentTime;
            
            currentAvg = 0;
            int count = std::min(rewardIdx, 100);
            for(int i=0; i<count; i++) {
                int idx = (rewardIdx - count + i) % 100;
                if (idx < 0) idx += 100; 
                currentAvg += avgRewards[idx];
            }
            if (rewardIdx > 0) currentAvg /= count;
            
            if (sps > 100) { // Ignore invalid SPS values
                std::cout << "[JOLTrl] Steps: " << totalSteps << "/" << config.maxSteps 
                          << " | SPS: " << (int)sps 
                          << " | Episodes: " << episodes 
                          << " | Avg Reward: " << currentAvg << std::endl;
            }
            
            // Write state for viewer (every 10 steps to avoid I/O bottleneck)
            if (totalSteps % 10 == 0) {
                float redPos[3], bluePos[3], redSat[3], blueSat[3];
                float redH = 100, blueH = 100;
                if (vecEnv.GetRenderState(redPos, bluePos, redSat, blueSat, &redH, &blueH)) {
                    FILE* f = fopen(stateFile.c_str(), "w");
                    if (f) {
                        fprintf(f, "{\"step\":%d,\"red\":[%.2f,%.2f,%.2f],\"blue\":[%.2f,%.2f,%.2f],\"red_satellite\":[%.2f,%.2f,%.2f],\"blue_satellite\":[%.2f,%.2f,%.2f],\"red_health\":%.1f,\"blue_health\":%.1f}",
                            totalSteps, redPos[0],redPos[1],redPos[2], bluePos[0],bluePos[1],bluePos[2],
                            redSat[0],redSat[1],redSat[2], blueSat[0],blueSat[1],blueSat[2], redH, blueH);
                        fclose(f);
                    }
                }
            }
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
