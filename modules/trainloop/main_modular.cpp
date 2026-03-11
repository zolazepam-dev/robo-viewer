/**
 * @file main_modular.cpp
 * @brief Main entry point for modular JOLTrl training
 * 
 * Uses the new TrainLoop orchestrator with modular architecture.
 * Supports both visualized and headless training.
 */

#include "TrainLoop.h"
#include <iostream>
#include <string>
#include <cstring>

void PrintUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "Options:\n"
              << "  --envs N           Number of parallel environments (default: 64)\n"
              << "  --steps N          Steps per episode (default: 1000)\n"
              << "  --batch N          Batch size for training (default: 16)\n"
              << "  --buffer N         Replay buffer capacity (default: 1000000)\n"
              << "  --checkpoint-dir   Directory for checkpoints (default: checkpoints)\n"
              << "  --load PATH        Load model from path\n"
              << "  --headless         Run without visualization\n"
              << "  --help             Show this help\n";
}

int main(int argc, char* argv[]) {
    TrainConfig config;
    
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
        } else if (arg == "--headless") {
            config.headlessTurbo = true;
        } else if (arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        }
    }
    
    std::cout << "=== JOLTrl Modular Training ===\n"
              << "Environments: " << config.numEnvs << "\n"
              << "Steps/Episode: " << config.stepsPerEpisode << "\n"
              << "Batch Size: " << config.batchSize << "\n"
              << "Buffer Capacity: " << config.bufferCapacity << "\n"
              << "Checkpoint Dir: " << config.checkpointDir << "\n"
              << "Headless: " << (config.headlessTurbo ? "yes" : "no") << "\n"
              << "===============================\n";
    
    // Create and run training loop
    TrainLoop trainLoop(config);
    
    if (!trainLoop.Init()) {
        std::cerr << "Failed to initialize training loop\n";
        return 1;
    }
    
    // Run training loop
    trainLoop.Run();
    
    return 0;
}
