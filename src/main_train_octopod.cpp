/**
 * @file main_train_octopod.cpp
 * @brief Training entry point for octopod combat
 */

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <csignal>
#include <atomic>
#include <random>

#include "PhysicsCore.h"
#include "OctopodEnv.h"
#include "AlignedAllocator.h"

using namespace JPH;

constexpr int NUM_ENVS = 8;
static std::atomic<bool> gShutdownRequested(false);

void signalHandler(int signum) {
    std::cerr << "\n[OctopodTrainer] Interrupt signal received. Shutting down..." << std::endl;
    gShutdownRequested.store(true);
}

int main(int argc, char* argv[])
{
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::cout << "[OctopodTrainer] Initializing Jolt Physics..." << std::endl;
    
    PhysicsCore physicsCore;
    physicsCore.Init(NUM_ENVS);
    
    JPH::PhysicsSystem* physicsSystem = &physicsCore.GetPhysicsSystem();
    
    std::cout << "[OctopodTrainer] Physics initialized." << std::endl;
    std::cout << "[OctopodTrainer] Action dim: " << OCTOPOD_ACTION_DIM << " per robot" << std::endl;
    std::cout << "[OctopodTrainer] Observation dim: " << OCTOPOD_OBS_DIM << std::endl;
    std::cout << "[OctopodTrainer] Parallel environments: " << NUM_ENVS << std::endl;
    
    std::vector<OctopodEnv> envs(NUM_ENVS);
    for (int i = 0; i < NUM_ENVS; i++) {
        envs[i].Init(i, physicsSystem, OCTOPOD_MAX_STEPS);
    }
    
    std::cout << "[OctopodTrainer] Environments initialized." << std::endl;
    
    const int actionDim = OCTOPOD_ACTION_DIM;
    const int obsDim = OCTOPOD_OBS_DIM;
    const int totalActionDim = actionDim * 2;
    const int totalObsDim = obsDim * 2;
    
    AlignedVector32<float> observations(totalObsDim * NUM_ENVS);
    AlignedVector32<float> actions(totalActionDim * NUM_ENVS);
    AlignedVector32<float> rewards(2 * NUM_ENVS);
    AlignedVector32<float> nextObservations(totalObsDim * NUM_ENVS);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::cout << "[OctopodTrainer] Starting training loop..." << std::endl;
    
    int episodeCount = 0;
    int totalSteps = 0;
    float cumulativeReward = 0.0f;
    
    while (!gShutdownRequested.load()) {
        for (int i = 0; i < NUM_ENVS; i++) {
            if (envs[i].IsDone()) {
                envs[i].Reset();
                episodeCount++;
            }
        }
        
        for (int i = 0; i < NUM_ENVS * totalActionDim; i++) {
            actions[i] = dist(rng);
        }
        
        for (int i = 0; i < NUM_ENVS; i++) {
            float* act1 = &actions[(i * 2 + 0) * actionDim];
            float* act2 = &actions[(i * 2 + 1) * actionDim];
            envs[i].QueueActions(act1, act2);
        }
        
        physicsCore.Step(1.0f / 60.0f);
        
        for (int i = 0; i < NUM_ENVS; i++) {
            float* nextObs1 = &nextObservations[(i * 2 + 0) * obsDim];
            float* nextObs2 = &nextObservations[(i * 2 + 1) * obsDim];
            float* rew1 = &rewards[(i * 2 + 0)];
            float* rew2 = &rewards[(i * 2 + 1)];
            bool done = false;
            
            envs[i].HarvestState(nextObs1, nextObs2, rew1, rew2, done);
            
            cumulativeReward += *rew1 + *rew2;
        }
        
        totalSteps++;
        
        if (totalSteps % 100 == 0) {
            std::cout << "[OctopodTrainer] Step " << totalSteps 
                      << " | Episodes: " << episodeCount
                      << " | Avg Reward: " << (cumulativeReward / 100.0f)
                      << std::endl;
            cumulativeReward = 0.0f;
        }
        
        memcpy(observations.data(), nextObservations.data(), 
               totalObsDim * NUM_ENVS * sizeof(float));
    }
    
    std::cout << "[OctopodTrainer] Complete. Episodes: " << episodeCount << std::endl;
    
    return 0;
}
