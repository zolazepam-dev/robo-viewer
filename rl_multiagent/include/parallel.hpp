/**
 * @file parallel.hpp
 * @brief Parallel Infrastructure for Multi-Agent RL
 * 
 * This module provides:
 * - Thread pool for parallel environments
 * - Worker function for episode execution
 * - Trajectory aggregator
 * - Console renderer for real-time monitoring
 * - Metrics logger to CSV
 */

#ifndef MARL_PARALLEL_HPP
#define MARL_PARALLEL_HPP

#include "rl_algo.hpp"
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace marl {

// ============================================================================
// Thread Pool
// ============================================================================

/**
 * @brief Simple thread pool for parallel execution
 */
class ThreadPool {
public:
    /**
     * @brief Initialize thread pool
     * @param num_threads Number of worker threads
     */
    explicit ThreadPool(size_t num_threads) : m_stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            m_workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(m_queue_mutex);
                        m_condition.wait(lock, [this] {
                            return m_stop || !m_tasks.empty();
                        });
                        
                        if (m_stop && m_tasks.empty()) {
                            return;
                        }
                        
                        task = std::move(m_tasks.front());
                        m_tasks.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    /**
     * @brief Enqueue a task
     * @param f Function to execute
     */
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_tasks.emplace(std::forward<F>(f));
        }
        m_condition.notify_one();
    }
    
    /**
     * @brief Wait for all tasks to complete
     */
    void wait_all() {
        // Simple busy-wait implementation
        while (true) {
            {
                std::unique_lock<std::mutex> lock(m_queue_mutex);
                if (m_tasks.empty()) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    /**
     * @brief Destructor - joins all threads
     */
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_stop = true;
        }
        m_condition.notify_all();
        
        for (std::thread& worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    /**
     * @brief Get number of threads
     */
    size_t size() const {
        return m_workers.size();
    }

private:
    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_queue_mutex;
    std::condition_variable m_condition;
    bool m_stop;
};

// ============================================================================
// Parallel Environment Worker
// ============================================================================

/**
 * @brief Configuration for parallel training
 */
struct ParallelConfig {
    int num_envs = 4;           // Number of parallel environments
    int num_episodes = 100;     // Total episodes to run
    int max_steps = 1000;       // Max steps per episode
    double actor_lr = 0.01;     // Actor learning rate
    double critic_lr = 0.01;    // Critic learning rate
    double gamma = 0.99;        // Discount factor
    bool render = false;        // Whether to render one environment
};

/**
 * @brief Result from a single episode
 */
struct EpisodeResult {
    int episode;
    double reward;
    int steps;
    double duration_ms;
    int tasks_completed;
};

/**
 * @brief Worker that runs episodes in parallel environments
 */
class ParallelWorker {
public:
    using Vector = Math::Vector;
    
    /**
     * @brief Initialize worker
     * @param config Parallel configuration
     */
    explicit ParallelWorker(const ParallelConfig& config)
        : m_config(config)
        , m_env(config.num_envs)
        , m_agent(5, 7, 64, config.actor_lr, config.critic_lr, config.gamma)
        , m_episode_count(0)
        , m_total_reward(0.0) {
    }
    
    /**
     * @brief Run single episode
     * @param env_id Environment ID
     * @param episode_num Episode number
     * @return Episode result
     */
    EpisodeResult run_episode(int env_id, int episode_num) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto state = m_env.reset();
        Vector state_vec = state_to_vector(state);
        
        double episode_reward = 0.0;
        int steps = 0;
        
        while (steps < m_config.max_steps) {
            // Select action
            int action_idx = m_agent.select_action(state_vec, true);
            ActionType action_type = int_to_action_type(action_idx);
            
            // Create action
            Action action = create_action(action_type, env_id, state);
            auto result = m_env.step(action);
            
            // Store transition
            Vector next_state_vec = state_to_vector(result.next_state);
            m_agent.store_transition(state_vec, action_idx, result.reward,
                                    next_state_vec, result.done);
            
            episode_reward += result.reward;
            state_vec = next_state_vec;
            steps++;
            
            if (result.done) {
                break;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        m_episode_count++;
        m_total_reward += episode_reward;
        
        EpisodeResult ep_result;
        ep_result.episode = episode_num;
        ep_result.reward = episode_reward;
        ep_result.steps = steps;
        ep_result.duration_ms = static_cast<double>(duration);
        ep_result.tasks_completed = m_env.get_state().completed_task_count();
        
        return ep_result;
    }
    
    /**
     * @brief Update agent from accumulated trajectories
     */
    void update_agent() {
        m_agent.update();
    }
    
    /**
     * @brief Get agent
     */
    RLAgent& get_agent() { return m_agent; }
    
    /**
     * @brief Get episode count
     */
    int get_episode_count() const { return m_episode_count; }
    
    /**
     * @brief Get average reward
     */
    double get_average_reward() const {
        return m_episode_count > 0 ? m_total_reward / m_episode_count : 0.0;
    }

private:
    ParallelConfig m_config;
    FeatureDevEnv m_env;
    RLAgent m_agent;
    std::atomic<int> m_episode_count;
    std::atomic<double> m_total_reward;
    
    Vector state_to_vector(const EnvironmentState& state) {
        Vector features(5);
        features[0] = static_cast<double>(state.files.size()) / 100.0;
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
        
        switch (type) {
            case ActionType::EDIT:
                action.target = "/src/main.cpp";
                action.parameter = "// New code";
                break;
            case ActionType::CLAIM:
                for (const auto& [id, task] : state.tasks) {
                    if (task.status == TaskStatus::UNCLAIMED) {
                        action.target = id;
                        break;
                    }
                }
                break;
            case ActionType::COMPLETE:
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

// ============================================================================
// Trajectory Aggregator
// ============================================================================

/**
 * @brief Aggregates trajectories from multiple parallel workers
 */
class TrajectoryAggregator {
public:
    using Vector = Math::Vector;
    
    /**
     * @brief Add trajectory from worker
     * @param states States
     * @param actions Actions
     * @param rewards Rewards
     * @param dones Done flags
     */
    void add_trajectory(const std::vector<Vector>& states,
                       const std::vector<int>& actions,
                       const std::vector<double>& rewards,
                       const std::vector<bool>& dones) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        for (size_t i = 0; i < states.size(); i++) {
            m_states.push_back(states[i]);
            m_actions.push_back(actions[i]);
            m_rewards.push_back(rewards[i]);
            m_dones.push_back(dones[i]);
        }
    }
    
    /**
     * @brief Get all aggregated states
     */
    const std::vector<Vector>& get_states() const { return m_states; }
    
    /**
     * @brief Get all aggregated actions
     */
    const std::vector<int>& get_actions() const { return m_actions; }
    
    /**
     * @brief Get all aggregated rewards
     */
    const std::vector<double>& get_rewards() const { return m_rewards; }
    
    /**
     * @brief Get total count
     */
    size_t size() const { return m_states.size(); }
    
    /**
     * @brief Clear aggregated data
     */
    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_states.clear();
        m_actions.clear();
        m_rewards.clear();
        m_dones.clear();
    }

private:
    std::vector<Vector> m_states;
    std::vector<int> m_actions;
    std::vector<double> m_rewards;
    std::vector<bool> m_dones;
    std::mutex m_mutex;
};

// ============================================================================
// Console Renderer
// ============================================================================

/**
 * @brief Real-time console renderer for training progress
 */
class ConsoleRenderer {
public:
    /**
     * @brief Initialize renderer
     * @param log_interval Log every N episodes
     */
    explicit ConsoleRenderer(int log_interval = 10)
        : m_log_interval(log_interval)
        , m_start_time(std::chrono::high_resolution_clock::now()) {
    }
    
    /**
     * @brief Render episode result
     * @param result Episode result
     * @param avg_reward Rolling average reward
     */
    void render(const EpisodeResult& result, double avg_reward) {
        if (result.episode % m_log_interval != 0) return;
        
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - m_start_time).count();
        
        std::cout << "\n=== Episode " << result.episode << " ===" << std::endl;
        std::cout << "  Reward: " << std::fixed << std::setprecision(2) << result.reward << std::endl;
        std::cout << "  Steps: " << result.steps << std::endl;
        std::cout << "  Avg Reward: " << avg_reward << std::endl;
        std::cout << "  Tasks Done: " << result.tasks_completed << std::endl;
        std::cout << "  Time: " << elapsed << "s" << std::endl;
        std::cout << "========================" << std::endl;
    }
    
    /**
     * @brief Render final summary
     * @param total_episodes Total episodes run
     * @param final_avg_reward Final average reward
     */
    void render_summary(int total_episodes, double final_avg_reward) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - m_start_time).count();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "TRAINING COMPLETE" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total Episodes: " << total_episodes << std::endl;
        std::cout << "Final Avg Reward: " << std::fixed << std::setprecision(3) << final_avg_reward << std::endl;
        std::cout << "Total Time: " << elapsed << "s" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }

private:
    int m_log_interval;
    std::chrono::high_resolution_clock::time_point m_start_time;
};

// ============================================================================
// Metrics Logger
// ============================================================================

/**
 * @brief Logs training metrics to CSV file
 */
class MetricsLogger {
public:
    /**
     * @brief Initialize logger
     * @param filename Output CSV filename
     */
    explicit MetricsLogger(const std::string& filename)
        : m_file(filename) {
        
        if (m_file.is_open()) {
            // Write header
            m_file << "episode,reward,steps,duration_ms,tasks_completed,avg_reward\n";
        }
    }
    
    /**
     * @brief Log episode metrics
     * @param result Episode result
     * @param avg_reward Rolling average reward
     */
    void log(const EpisodeResult& result, double avg_reward) {
        if (!m_file.is_open()) return;
        
        m_file << result.episode << ","
               << std::fixed << std::setprecision(4) << result.reward << ","
               << result.steps << ","
               << std::setprecision(2) << result.duration_ms << ","
               << result.tasks_completed << ","
               << std::setprecision(4) << avg_reward << "\n";
        
        m_file.flush();  // Ensure written immediately
    }
    
    /**
     * @brief Close the file
     */
    void close() {
        if (m_file.is_open()) {
            m_file.close();
        }
    }
    
    /**
     * @brief Check if file is open
     */
    bool is_open() const {
        return m_file.is_open();
    }

private:
    std::ofstream m_file;
};

// ============================================================================
// Parallel Trainer
// ============================================================================

/**
 * @brief Main parallel training orchestrator
 */
class ParallelTrainer {
public:
    /**
     * @brief Initialize parallel trainer
     * @param config Parallel configuration
     */
    explicit ParallelTrainer(const ParallelConfig& config)
        : m_config(config)
        , m_worker(config)
        , m_renderer(10)
        , m_logger("logs/metrics.csv")
        , m_pool(config.num_envs) {
    }
    
    /**
     * @brief Run training
     * @return Vector of episode results
     */
    std::vector<EpisodeResult> train() {
        std::vector<EpisodeResult> all_results;
        std::vector<double> reward_window;
        const size_t window_size = 20;
        
        std::cout << "\nStarting parallel training with " 
                  << m_config.num_envs << " environments..." << std::endl;
        
        for (int ep = 0; ep < m_config.num_episodes; ep++) {
            // Run episode in thread pool
            EpisodeResult result;
            
            std::mutex result_mutex;
            std::condition_variable result_cv;
            bool done = false;
            
            m_pool.enqueue([this, ep, &result, &result_mutex, &result_cv, &done] {
                auto ep_result = m_worker.run_episode(0, ep);
                
                {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    result = ep_result;
                    done = true;
                }
                result_cv.notify_one();
            });
            
            // Wait for completion
            std::unique_lock<std::mutex> lock(result_mutex);
            result_cv.wait(lock, [&done] { return done; });
            
            // Update agent
            m_worker.update_agent();
            
            // Track rewards
            reward_window.push_back(result.reward);
            if (reward_window.size() > window_size) {
                reward_window.erase(reward_window.begin());
            }
            
            double avg_reward = 0.0;
            for (double r : reward_window) avg_reward += r;
            avg_reward /= reward_window.size();
            
            // Render and log
            m_renderer.render(result, avg_reward);
            m_logger.log(result, avg_reward);
            
            all_results.push_back(result);
        }
        
        m_pool.wait_all();
        
        // Final summary
        double final_avg = m_worker.get_average_reward();
        m_renderer.render_summary(m_config.num_episodes, final_avg);
        
        m_logger.close();
        
        return all_results;
    }
    
    /**
     * @brief Get worker
     */
    ParallelWorker& get_worker() { return m_worker; }

private:
    ParallelConfig m_config;
    ParallelWorker m_worker;
    ConsoleRenderer m_renderer;
    MetricsLogger m_logger;
    ThreadPool m_pool;
};

} // namespace marl

#endif // MARL_PARALLEL_HPP
