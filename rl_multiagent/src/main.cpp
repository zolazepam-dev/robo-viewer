/**
 * @file main.cpp
 * @brief Multi-Agent RL Pipeline - Main Entry Point
 * 
 * Usage:
 *   ./marl_train [options]
 * 
 * Options:
 *   --envs N        Number of parallel environments (default: 4)
 *   --episodes N    Number of training episodes (default: 100)
 *   --steps N       Max steps per episode (default: 1000)
 *   --actor-lr F    Actor learning rate (default: 0.01)
 *   --critic-lr F   Critic learning rate (default: 0.01)
 *   --gamma F       Discount factor (default: 0.99)
 *   --help          Show this help message
 */

#include <iostream>
#include <string>
#include <cstring>
#include "parallel.hpp"

using namespace marl;

void print_help() {
    std::cout << "\nMulti-Agent RL Pipeline - Training\n";
    std::cout << "===================================\n\n";
    std::cout << "Usage: ./marl_train [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --envs N        Number of parallel environments (default: 4)\n";
    std::cout << "  --episodes N    Number of training episodes (default: 100)\n";
    std::cout << "  --steps N       Max steps per episode (default: 1000)\n";
    std::cout << "  --actor-lr F    Actor learning rate (default: 0.01)\n";
    std::cout << "  --critic-lr F   Critic learning rate (default: 0.01)\n";
    std::cout << "  --gamma F       Discount factor (default: 0.99)\n";
    std::cout << "  --help          Show this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  ./marl_train --envs 8 --episodes 200 --steps 500\n\n";
}

ParallelConfig parse_args(int argc, char* argv[]) {
    ParallelConfig config;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_help();
            exit(0);
        } else if (strcmp(argv[i], "--envs") == 0 && i + 1 < argc) {
            config.num_envs = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) {
            config.num_episodes = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            config.max_steps = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--actor-lr") == 0 && i + 1 < argc) {
            config.actor_lr = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--critic-lr") == 0 && i + 1 < argc) {
            config.critic_lr = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) {
            config.gamma = std::stof(argv[++i]);
        }
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "\n╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Multi-Agent RL Pipeline v1.0          ║" << std::endl;
    std::cout << "║  4 Agents Collaborating on Dev Tasks   ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝" << std::endl;
    
    // Parse command line arguments
    ParallelConfig config = parse_args(argc, argv);
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Parallel Environments: " << config.num_envs << std::endl;
    std::cout << "  Episodes: " << config.num_episodes << std::endl;
    std::cout << "  Max Steps: " << config.max_steps << std::endl;
    std::cout << "  Actor LR: " << config.actor_lr << std::endl;
    std::cout << "  Critic LR: " << config.critic_lr << std::endl;
    std::cout << "  Gamma: " << config.gamma << std::endl;
    std::cout << std::endl;
    
    // Create and run trainer
    ParallelTrainer trainer(config);
    auto results = trainer.train();
    
    // Print summary statistics
    std::cout << "\nTraining Summary:" << std::endl;
    
    double total_reward = 0.0;
    int total_steps = 0;
    double min_reward = results[0].reward;
    double max_reward = results[0].reward;
    
    for (const auto& result : results) {
        total_reward += result.reward;
        total_steps += result.steps;
        if (result.reward < min_reward) min_reward = result.reward;
        if (result.reward > max_reward) max_reward = result.reward;
    }
    
    double avg_reward = total_reward / results.size();
    double avg_steps = static_cast<double>(total_steps) / results.size();
    
    std::cout << "  Average Reward: " << avg_reward << std::endl;
    std::cout << "  Min Reward: " << min_reward << std::endl;
    std::cout << "  Max Reward: " << max_reward << std::endl;
    std::cout << "  Average Steps: " << avg_steps << std::endl;
    std::cout << "  Total Episodes: " << results.size() << std::endl;
    
    // Check for learning improvement
    if (results.size() >= 10) {
        double first_avg = 0.0, last_avg = 0.0;
        for (size_t i = 0; i < 5; i++) first_avg += results[i].reward;
        for (size_t i = results.size() - 5; i < results.size(); i++) last_avg += results[i].reward;
        first_avg /= 5;
        last_avg /= 5;
        
        std::cout << "\nLearning Progress:" << std::endl;
        std::cout << "  First 5 Episodes Avg: " << first_avg << std::endl;
        std::cout << "  Last 5 Episodes Avg: " << last_avg << std::endl;
        
        if (last_avg > first_avg) {
            std::cout << "  ✓ Reward improved by " << (last_avg - first_avg) << std::endl;
        } else {
            std::cout << "  ⚠ Reward decreased by " << (first_avg - last_avg) << std::endl;
            std::cout << "  (Try adjusting learning rates or running more episodes)" << std::endl;
        }
    }
    
    std::cout << "\nMetrics saved to: logs/metrics.csv" << std::endl;
    std::cout << "Learning curve: logs/learning_curve.csv" << std::endl;
    std::cout << "\nTraining complete!" << std::endl;
    
    return 0;
}
