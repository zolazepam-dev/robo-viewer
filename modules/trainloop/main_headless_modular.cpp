/**
 * @file main_headless_modular.cpp
 * @brief Headless training entry point (no visualization)
 * 
 * Optimized for maximum SPS with no rendering overhead.
 */

#include "TrainLoop.h"
#include <iostream>
#include <csignal>
#include <atomic>

std::atomic<bool> gStopRequested(false);

void SignalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", stopping...\n";
    gStopRequested = true;
}

int main(int argc, char* argv[]) {
    // Setup signal handlers
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);
    
    TrainConfig config;
    config.headlessTurbo = true;  // Force headless mode
    config.numEnvs = 128;         // Default to more envs in headless
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--envs" && i + 1 < argc) {
            config.numEnvs = std::stoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            config.stepsPerEpisode = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            config.batchSize = std::stoi(argv[++i]);
        } else if (arg == "--buffer" && i + 1 < argc) {
            config.bufferCapacity = std::stoi(argv[++i]);
        } else if (arg == "--checkpoint-dir" && i + 1 < argc) {
            config.checkpointDir = argv[++i];
        } else if (arg == "--load" && i + 1 < argc) {
            config.loadModelPath = argv[++i];
        }
    }
    
    std::cout << "=== JOLTrl Headless Training ===\n"
              << "Environments: " << config.numEnvs << "\n"
              << "Steps/Episode: " << config.stepsPerEpisode << "\n"
              << "Batch Size: " << config.batchSize << "\n"
              << "================================\n";
    
    // Create and run training loop
    TrainLoop trainLoop(config);
    
    if (!trainLoop.Init()) {
        std::cerr << "Failed to initialize training loop\n";
        return 1;
    }
    
    // Run training loop
    trainLoop.Run();
    
    std::cout << "Training complete. Total steps: " << trainLoop.GetStats().totalSteps << "\n";
    
    return 0;
}
