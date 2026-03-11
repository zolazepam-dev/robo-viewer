/**
 * @file rl_algo.hpp
 * @brief Reinforcement Learning Algorithms for Multi-Agent System
 * 
 * This module provides:
 * - Trajectory buffer for storing experience
 * - REINFORCE with baseline (A2C-style)
 * - Training loop
 * - Multi-agent coordination
 */

#ifndef MARL_RL_ALGO_HPP
#define MARL_RL_ALGO_HPP

#include "neural_net.hpp"
#include "environment.hpp"
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <random>

namespace marl {

// ============================================================================
// Trajectory Buffer
// ============================================================================

/**
 * @brief Stores trajectory data for policy gradient updates
 */
struct TrajectoryBuffer {
    using Vector = Math::Vector;
    
    std::vector<Vector> states;
    std::vector<int> actions;
    std::vector<double> rewards;
    std::vector<Vector> next_states;
    std::vector<bool> dones;
    std::vector<double> values;  // Critic estimates
    
    size_t capacity;
    
    TrajectoryBuffer(size_t cap = 1000) : capacity(cap) {}
    
    /**
     * @brief Add transition to buffer
     */
    void add(const Vector& state, int action, double reward, 
             const Vector& next_state, bool done, double value) {
        if (states.size() >= capacity) {
            clear();
        }
        
        states.push_back(state);
        actions.push_back(action);
        rewards.push_back(reward);
        next_states.push_back(next_state);
        dones.push_back(done);
        values.push_back(value);
    }
    
    /**
     * @brief Clear buffer
     */
    void clear() {
        states.clear();
        actions.clear();
        rewards.clear();
        next_states.clear();
        dones.clear();
        values.clear();
    }
    
    /**
     * @brief Get buffer size
     */
    size_t size() const {
        return states.size();
    }
    
    /**
     * @brief Check if buffer is empty
     */
    bool empty() const {
        return states.empty();
    }
    
    /**
     * @brief Calculate discounted returns
     * @param gamma Discount factor
     * @return Vector of returns
     */
    std::vector<double> calculate_returns(double gamma) const {
        std::vector<double> returns(rewards.size());
        double G = 0.0;
        
        // Backward pass to calculate returns
        for (int t = static_cast<int>(rewards.size()) - 1; t >= 0; t--) {
            G = rewards[t] + gamma * G * (dones[t] ? 0.0 : 1.0);
            returns[t] = G;
        }
        
        return returns;
    }
    
    /**
     * @brief Calculate advantages (returns - baseline)
     * @param gamma Discount factor
     * @return Vector of advantages
     */
    std::vector<double> calculate_advantages(double gamma) const {
        auto returns = calculate_returns(gamma);
        std::vector<double> advantages(returns.size());
        
        for (size_t t = 0; t < returns.size(); t++) {
            advantages[t] = returns[t] - values[t];
        }
        
        return advantages;
    }
};

// ============================================================================
// RL Agent (Combines Actor + Critic)
// ============================================================================

/**
 * @brief RL Agent with Actor-Critic architecture
 */
class RLAgent {
public:
    using Vector = Math::Vector;
    
    /**
     * @brief Initialize agent
     * @param state_dim Observation dimension
     * @param action_dim Number of actions
     * @param config Configuration parameters
     */
    RLAgent(size_t state_dim, size_t action_dim, 
            size_t hidden_dim = 64,
            double actor_lr = 0.01,
            double critic_lr = 0.01,
            double gamma = 0.99)
        : m_state_dim(state_dim)
        , m_action_dim(action_dim)
        , m_gamma(gamma)
        , m_actor_lr(actor_lr)
        , m_critic_lr(critic_lr)
        , m_actor(state_dim, action_dim, hidden_dim)
        , m_critic(state_dim, hidden_dim)
        , m_buffer(1000) {
    }
    
    /**
     * @brief Select action given state
     * @param state Current state
     * @param explore Whether to explore (sample) or exploit (argmax)
     * @return Selected action
     */
    int select_action(const Vector& state, bool explore = true) {
        if (explore) {
            return m_actor.sample_action(state);
        } else {
            auto probs = m_actor.forward(state);
            return static_cast<int>(std::distance(
                probs.begin(), 
                std::max_element(probs.begin(), probs.end())
            ));
        }
    }
    
    /**
     * @brief Store transition in buffer
     */
    void store_transition(const Vector& state, int action, double reward,
                         const Vector& next_state, bool done) {
        double value = m_critic.forward(state);
        m_buffer.add(state, action, reward, next_state, done, value);
    }
    
    /**
     * @brief Update agent using policy gradient
     * @return Training loss (approximate)
     */
    double update() {
        if (m_buffer.empty()) return 0.0;
        
        // Calculate advantages
        auto advantages = m_buffer.calculate_advantages(m_gamma);
        
        // Normalize advantages (reduce variance)
        normalize(advantages);
        
        double total_loss = 0.0;
        
        // Update for each transition
        for (size_t t = 0; t < m_buffer.states.size(); t++) {
            // Actor update (policy gradient)
            m_actor.backward(m_buffer.states[t], m_buffer.actions[t], advantages[t]);
            
            // Critic update (TD error)
            double td_error = m_buffer.rewards[t] + m_gamma * m_buffer.values[t] * (m_buffer.dones[t] ? 0.0 : 1.0) - m_buffer.values[t];
            m_critic.backward(m_buffer.states[t], td_error);
            
            total_loss += std::abs(td_error) + std::abs(advantages[t]);
        }
        
        // Apply updates
        m_actor.update(m_actor_lr);
        m_critic.update(m_critic_lr);
        
        // Clear buffer
        m_buffer.clear();
        
        return total_loss / m_buffer.capacity;
    }
    
    /**
     * @brief Get actor network
     */
    ActorNetwork& get_actor() { return m_actor; }
    
    /**
     * @brief Get critic network
     */
    CriticNetwork& get_critic() { return m_critic; }
    
    /**
     * @brief Get trajectory buffer
     */
    TrajectoryBuffer& get_buffer() { return m_buffer; }

private:
    size_t m_state_dim;
    size_t m_action_dim;
    double m_gamma;
    double m_actor_lr;
    double m_critic_lr;
    
    ActorNetwork m_actor;
    CriticNetwork m_critic;
    TrajectoryBuffer m_buffer;
    
    void normalize(std::vector<double>& vec) {
        if (vec.empty()) return;
        
        double mean = 0.0;
        for (double v : vec) mean += v;
        mean /= vec.size();
        
        double var = 0.0;
        for (double v : vec) var += (v - mean) * (v - mean);
        var /= vec.size();
        
        double std = std::sqrt(var + 1e-8);
        
        for (double& v : vec) {
            v = (v - mean) / std;
        }
    }
};

// ============================================================================
// REINFORCE with Baseline Trainer
// ============================================================================

/**
 * @brief Training configuration
 */
struct RLConfig {
    size_t hidden_dim = 64;
    double actor_lr = 0.01;
    double critic_lr = 0.01;
    double gamma = 0.99;
    int max_steps = 1000;
    int num_episodes = 100;
};

/**
 * @brief REINFORCE with baseline training loop
 */
class REINFORCETrainer {
public:
    using Vector = Math::Vector;
    
    /**
     * @brief Initialize trainer
     * @param env Environment reference
     * @param config Training configuration
     */
    REINFORCETrainer(FeatureDevEnv& env, const RLConfig& config)
        : m_env(env)
        , m_config(config)
        , m_agent(5,  // State dimension (our feature vector)
                  7,  // 7 action types
                  config.hidden_dim,
                  config.actor_lr,
                  config.critic_lr,
                  config.gamma) {
    }
    
    /**
     * @brief Run single episode
     * @param episode_num Episode number
     * @return Total episode reward
     */
    double run_episode(int episode_num) {
        auto state = m_env.reset();
        Vector state_vec = state_to_vector(state);
        
        double episode_reward = 0.0;
        int step = 0;
        
        while (step < m_config.max_steps) {
            // Select action
            int action_idx = m_agent.select_action(state_vec, true);
            ActionType action_type = int_to_action_type(action_idx);
            
            // Create and execute action
            Action action = create_action(action_type, 0, state);
            auto result = m_env.step(action);
            
            // Store transition
            Vector next_state_vec = state_to_vector(result.next_state);
            m_agent.store_transition(state_vec, action_idx, result.reward, 
                                    next_state_vec, result.done);
            
            episode_reward += result.reward;
            state_vec = next_state_vec;
            step++;
            
            if (result.done) {
                break;
            }
        }
        
        return episode_reward;
    }
    
    /**
     * @brief Train for multiple episodes
     * @param callback Optional callback after each episode
     * @return Vector of episode rewards
     */
    std::vector<double> train(int num_episodes, 
                              std::function<void(int, double)> callback = nullptr) {
        std::vector<double> rewards;
        
        for (int ep = 0; ep < num_episodes; ep++) {
            double ep_reward = run_episode(ep);
            rewards.push_back(ep_reward);
            
            // Update agent
            m_agent.update();
            
            if (callback) {
                callback(ep, ep_reward);
            }
        }
        
        return rewards;
    }
    
    /**
     * @brief Get agent
     */
    RLAgent& get_agent() { return m_agent; }

private:
    FeatureDevEnv& m_env;
    RLConfig m_config;
    RLAgent m_agent;
    
    Vector state_to_vector(const EnvironmentState& state) {
        // Convert environment state to feature vector
        // Features: num_files, num_tasks, num_complete, step_count, episode
        
        Vector features(5);
        features[0] = static_cast<double>(state.files.size()) / 100.0;  // Normalize
        features[1] = static_cast<double>(state.tasks.size()) / 50.0;
        features[2] = static_cast<double>(state.completed_task_count()) / 50.0;
        features[3] = static_cast<double>(state.step_count) / 1000.0;
        features[4] = static_cast<double>(state.episode) / 100.0;
        
        return features;
    }
    
    ActionType int_to_action_type(int idx) {
        switch (idx) {
            case 0: return ActionType::READ;
            case 1: return ActionType::EDIT;
            case 2: return ActionType::CLAIM;
            case 3: return ActionType::COMPLETE;
            case 4: return ActionType::RUN_COMMAND;
            case 5: return ActionType::BLOCK;
            default: return ActionType::NOOP;
        }
    }
    
    Action create_action(ActionType type, int agent_id, const EnvironmentState& state) {
        Action action(type, agent_id, "");
        
        // Simple heuristic for target selection
        switch (type) {
            case ActionType::EDIT:
                action.target = "/src/main.cpp";
                action.parameter = "// New code";
                break;
            case ActionType::CLAIM:
                // Find unclaimed task
                for (const auto& [id, task] : state.tasks) {
                    if (task.status == TaskStatus::UNCLAIMED) {
                        action.target = id;
                        break;
                    }
                }
                break;
            case ActionType::COMPLETE:
                // Find claimed task
                for (const auto& [id, task] : state.tasks) {
                    if (task.status == TaskStatus::CLAIMED) {
                        action.target = id;
                        break;
                    }
                }
                break;
            case ActionType::RUN_COMMAND:
                action.parameter = "g++ -std=c++20 main.cpp";
                break;
            default:
                break;
        }
        
        return action;
    }
};

} // namespace marl

#endif // MARL_RL_ALGO_HPP
