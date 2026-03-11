/**
 * @file environment.hpp
 * @brief Multi-Agent Development Environment - State Structures and Core Environment
 * 
 * This module provides the simulated development environment where multiple RL agents
 * collaborate on feature development. It includes:
 * - State structures for files, tasks, and environment
 * - Action types and role definitions
 * - In-memory file system simulation
 * - TODO.md parser and updater
 * - Reward function for task completion and coordination
 */

#ifndef MARL_ENVIRONMENT_HPP
#define MARL_ENVIRONMENT_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace marl {

// ============================================================================
// Core Enums
// ============================================================================

/**
 * @brief Actions available to agents in the development environment
 */
enum class ActionType {
    READ,          // Read a file's contents
    EDIT,          // Edit a file's contents
    RUN_COMMAND,   // Execute a shell command (compile, test, etc.)
    CLAIM,         // Claim a task from TODO.md
    COMPLETE,      // Mark a task as complete
    BLOCK,         // Mark a task as blocked
    NOOP           // No operation (wait)
};

/**
 * @brief Agent roles in the multi-agent system
 */
enum class Role {
    ARCHITECT,     // Environment Architect: defines structures, file system
    CORE_RL,       // Core RL Implementer: neural networks, algorithms
    PARALLEL,      // Parallelization Engineer: threading, concurrency
    TESTING        // Testing Specialist: unit tests, integration
};

/**
 * @brief Task status in the workflow
 */
enum class TaskStatus {
    UNCLAIMED,     // Available for any agent
    CLAIMED,       // Claimed by an agent
    IN_PROGRESS,   // Currently being worked on
    COMPLETE,      // Finished and verified
    BLOCKED        // Blocked by dependencies
};

// ============================================================================
// State Structures
// ============================================================================

/**
 * @brief Represents the state of a file in the simulated file system
 */
struct FileState {
    std::string path;        // File path
    std::string content;     // File contents
    int version;             // Version number (increments on edit)
    bool exists;             // Whether file exists
    
    FileState() : version(0), exists(false) {}
    
    FileState(const std::string& p, const std::string& c) 
        : path(p), content(c), version(1), exists(true) {}
    
    /**
     * @brief Update file content and increment version
     */
    void update(const std::string& new_content) {
        content = new_content;
        version++;
        exists = true;
    }
};

/**
 * @brief Represents a task from TODO.md
 */
struct TaskState {
    std::string task_id;     // e.g., "ENV-001"
    Role role;               // Responsible role
    std::string description; // Task description
    TaskStatus status;       // Current status
    std::string claimed_by;  // Agent ID if claimed
    std::string blocked_by;  // Dependency if blocked
    
    TaskState() : role(Role::ARCHITECT), status(TaskStatus::UNCLAIMED) {}
    
    /**
     * @brief Check if task can be claimed by a role
     */
    bool can_claim_by(Role r) const {
        return status == TaskStatus::UNCLAIMED && role == r;
    }
    
    /**
     * @brief Check if task is complete
     */
    bool is_complete() const {
        return status == TaskStatus::COMPLETE;
    }
};

/**
 * @brief Complete environment state observable by agents
 */
struct EnvironmentState {
    std::map<std::string, FileState> files;      // In-memory file system
    std::map<std::string, TaskState> tasks;      // TODO.md tasks
    int step_count;                               // Steps in current episode
    int episode;                                  // Current episode number
    std::map<std::string, int> agent_stats;      // Per-agent statistics
    
    EnvironmentState() : step_count(0), episode(0) {}
    
    /**
     * @brief Get number of completed tasks
     */
    int completed_task_count() const {
        int count = 0;
        for (const auto& [id, task] : tasks) {
            if (task.is_complete()) count++;
        }
        return count;
    }
    
    /**
     * @brief Get number of tasks for a specific role
     */
    int role_task_count(Role r) const {
        int count = 0;
        for (const auto& [id, task] : tasks) {
            if (task.role == r) count++;
        }
        return count;
    }
};

/**
 * @brief Action taken by an agent
 */
struct Action {
    ActionType type;           // Type of action
    int agent_id;              // Which agent took this action
    std::string target;        // File path or task ID
    std::string parameter;     // Content for EDIT, command for RUN_COMMAND
    int priority;              // Priority level (for scheduling)
    
    Action() : type(ActionType::NOOP), agent_id(0), priority(0) {}
    
    Action(ActionType t, int aid, const std::string& tgt) 
        : type(t), agent_id(aid), target(tgt), priority(0) {}
    
    Action(ActionType t, int aid, const std::string& tgt, const std::string& param)
        : type(t), agent_id(aid), target(tgt), parameter(param), priority(0) {}
};

/**
 * @brief Result of taking a step in the environment
 */
struct StepResult {
    double reward;                    // Reward signal
    bool done;                        // Episode termination
    EnvironmentState next_state;      // New environment state
    std::string message;              // Feedback message
    
    StepResult() : reward(0.0), done(false) {}
};

// ============================================================================
// TODO.md Parser
// ============================================================================

/**
 * @brief Parser and updater for TODO.md files
 */
class TodoParser {
public:
    /**
     * @brief Parse TODO.md content into TaskState objects
     * @param content Raw markdown content
     * @return Map of task_id -> TaskState
     */
    static std::map<std::string, TaskState> parse(const std::string& content) {
        std::map<std::string, TaskState> tasks;
        std::istringstream stream(content);
        std::string line;
        
        // Skip header lines until we find the table header
        bool found_header = false;
        while (std::getline(stream, line)) {
            if (line.find("| Task ID") != std::string::npos) {
                found_header = true;
                // Skip separator line
                std::getline(stream, line);
                break;
            }
        }
        
        if (!found_header) {
            return tasks;  // Empty or invalid format
        }
        
        // Parse table rows
        while (std::getline(stream, line)) {
            if (line.empty() || line[0] != '|') break;
            
            TaskState task = parse_line(line);
            if (!task.task_id.empty()) {
                tasks[task.task_id] = task;
            }
        }
        
        return tasks;
    }
    
    /**
     * @brief Generate TODO.md content from TaskState map
     * @param tasks Map of task_id -> TaskState
     * @return Formatted markdown string
     */
    static std::string generate(const std::map<std::string, TaskState>& tasks) {
        std::ostringstream oss;
        
        oss << "# Multi-Agent RL Pipeline - Task Tracker\n\n";
        oss << "| Task ID | Role      | Description                          | Status    | Claimed By | Blocked By |\n";
        oss << "|---------|-----------|--------------------------------------|-----------|------------|------------|\n";
        
        // Sort tasks by ID for consistent output
        std::vector<std::pair<std::string, TaskState>> sorted_tasks(tasks.begin(), tasks.end());
        std::sort(sorted_tasks.begin(), sorted_tasks.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        
        for (const auto& [id, task] : sorted_tasks) {
            oss << "| " << std::left << std::setw(9) << task.task_id << " ";
            oss << "| " << std::left << std::setw(9) << role_to_string(task.role) << " ";
            oss << "| " << std::left << std::setw(36) << truncate(task.description, 36) << " ";
            oss << "| " << std::left << std::setw(9) << status_to_string(task.status) << " ";
            oss << "| " << std::left << std::setw(10) << task.claimed_by << " ";
            oss << "| " << std::left << std::setw(10) << task.blocked_by << " ";
            oss << "|\n";
        }
        
        return oss.str();
    }
    
private:
    /**
     * @brief Parse a single table row
     */
    static TaskState parse_line(const std::string& line) {
        TaskState task;
        std::vector<std::string> fields;
        
        // Split by '|'
        std::istringstream stream(line);
        std::string field;
        while (std::getline(stream, field, '|')) {
            // Trim whitespace
            size_t start = field.find_first_not_of(" \t");
            size_t end = field.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos) {
                fields.push_back(field.substr(start, end - start + 1));
            }
        }
        
        // Need at least 6 fields
        if (fields.size() < 6) {
            return task;
        }
        
        task.task_id = fields[0];
        task.role = string_to_role(fields[1]);
        task.description = fields[2];
        task.status = string_to_status(fields[3]);
        task.claimed_by = fields[4];
        task.blocked_by = fields[5];
        
        return task;
    }
    
    static std::string role_to_string(Role r) {
        switch (r) {
            case Role::ARCHITECT: return "Architect";
            case Role::CORE_RL: return "Core RL";
            case Role::PARALLEL: return "Parallel";
            case Role::TESTING: return "Testing";
            default: return "Unknown";
        }
    }
    
    static Role string_to_role(const std::string& s) {
        if (s == "Architect" || s == "ARCHITECT") return Role::ARCHITECT;
        if (s == "Core RL" || s == "CORE_RL") return Role::CORE_RL;
        if (s == "Parallel" || s == "PARALLEL") return Role::PARALLEL;
        if (s == "Testing" || s == "TESTING") return Role::TESTING;
        return Role::ARCHITECT;  // Default
    }
    
    static std::string status_to_string(TaskStatus s) {
        switch (s) {
            case TaskStatus::UNCLAIMED: return "unclaimed";
            case TaskStatus::CLAIMED: return "claimed";
            case TaskStatus::IN_PROGRESS: return "in_progress";
            case TaskStatus::COMPLETE: return "complete";
            case TaskStatus::BLOCKED: return "blocked";
            default: return "unknown";
        }
    }
    
    static TaskStatus string_to_status(const std::string& s) {
        if (s == "unclaimed" || s == "UNCLAIMED") return TaskStatus::UNCLAIMED;
        if (s == "claimed" || s == "CLAIMED") return TaskStatus::CLAIMED;
        if (s == "in_progress" || s == "IN_PROGRESS") return TaskStatus::IN_PROGRESS;
        if (s == "complete" || s == "COMPLETE") return TaskStatus::COMPLETE;
        if (s == "blocked" || s == "BLOCKED") return TaskStatus::BLOCKED;
        return TaskStatus::UNCLAIMED;  // Default
    }
    
    static std::string truncate(const std::string& s, size_t max_len) {
        if (s.length() <= max_len) return s;
        return s.substr(0, max_len - 3) + "...";
    }
};

// ============================================================================
// Reward Function
// ============================================================================

/**
 * @brief Calculates rewards for agent actions
 */
class RewardFunction {
public:
    // Reward coefficients
    static constexpr double TASK_CLAIM_REWARD = 0.1;
    static constexpr double TASK_COMPLETE_REWARD = 1.0;
    static constexpr double FILE_EDIT_REWARD = 0.05;
    static constexpr double COMMAND_SUCCESS_REWARD = 0.2;
    static constexpr double COORDINATION_BONUS = 0.3;
    static constexpr double STEP_PENALTY = -0.01;
    static constexpr double INVALID_ACTION_PENALTY = -0.1;
    
    /**
     * @brief Calculate reward for an action
     * @param action The action taken
     * @param old_state State before action
     * @param new_state State after action
     * @param success Whether action succeeded
     * @return Reward value
     */
    static double calculate(const Action& action, 
                           const EnvironmentState& old_state,
                           const EnvironmentState& new_state,
                           bool success) {
        if (!success) {
            return INVALID_ACTION_PENALTY;
        }
        
        double reward = STEP_PENALTY;  // Small penalty for each step (encourages efficiency)
        
        switch (action.type) {
            case ActionType::CLAIM:
                reward += calculate_claim_reward(action, old_state, new_state);
                break;
            case ActionType::COMPLETE:
                reward += calculate_complete_reward(action, old_state, new_state);
                break;
            case ActionType::EDIT:
                reward += calculate_edit_reward(action, old_state, new_state);
                break;
            case ActionType::RUN_COMMAND:
                reward += calculate_command_reward(action, success);
                break;
            default:
                break;
        }
        
        return reward;
    }
    
private:
    static double calculate_claim_reward(const Action& action,
                                         const EnvironmentState& old_state,
                                         const EnvironmentState& new_state) {
        // Check if a task was successfully claimed
        auto it = new_state.tasks.find(action.target);
        if (it != new_state.tasks.end() && it->second.status == TaskStatus::CLAIMED) {
            return TASK_CLAIM_REWARD;
        }
        return 0.0;
    }

    static double calculate_complete_reward(const Action& action,
                                            const EnvironmentState& old_state,
                                            const EnvironmentState& new_state) {
        // Check if a task was completed
        auto it = new_state.tasks.find(action.target);
        if (it != new_state.tasks.end() && it->second.status == TaskStatus::COMPLETE) {
            // Bonus for completing your own role's tasks
            if (it->second.role == get_role_for_agent(action.agent_id)) {
                return TASK_COMPLETE_REWARD * 1.5;
            }
            return TASK_COMPLETE_REWARD;
        }
        return 0.0;
    }

    static double calculate_edit_reward(const Action& action,
                                        const EnvironmentState& old_state,
                                        const EnvironmentState& new_state) {
        // Check if file was modified
        auto it = new_state.files.find(action.target);
        auto old_it = old_state.files.find(action.target);

        if (it != new_state.files.end() && old_it != old_state.files.end()) {
            if (it->second.version > old_it->second.version) {
                return FILE_EDIT_REWARD;
            }
        }
        return 0.0;
    }
    
    static double calculate_command_reward(const Action& action, bool success) {
        if (success) {
            return COMMAND_SUCCESS_REWARD;
        }
        return 0.0;
    }
    
    static Role get_role_for_agent(int agent_id) {
        // Round-robin role assignment
        switch (agent_id % 4) {
            case 0: return Role::ARCHITECT;
            case 1: return Role::CORE_RL;
            case 2: return Role::PARALLEL;
            case 3: return Role::TESTING;
            default: return Role::ARCHITECT;
        }
    }
};

// ============================================================================
// FeatureDevEnv - Main Environment Class
// ============================================================================

/**
 * @brief Simulated development environment for multi-agent RL
 * 
 * This class provides:
 * - In-memory file system
 * - Task management via TODO.md
 * - Action execution and validation
 * - Reward calculation
 * - Episode management
 */
class FeatureDevEnv {
public:
    /**
     * @brief Initialize environment with given number of agents
     * @param num_agents Number of collaborating agents
     */
    explicit FeatureDevEnv(int num_agents = 4) 
        : m_num_agents(num_agents), m_current_episode(0) {
        reset();
    }
    
    /**
     * @brief Reset environment for new episode
     * @return Initial environment state
     */
    EnvironmentState reset() {
        m_state = EnvironmentState();
        m_state.episode = m_current_episode++;
        m_state.step_count = 0;
        
        // Initialize with basic TODO.md structure
        initialize_tasks();
        
        return m_state;
    }
    
    /**
     * @brief Execute action and return result
     * @param action Action to execute
     * @return StepResult with reward, next state, done flag
     */
    StepResult step(const Action& action) {
        StepResult result;
        result.next_state = m_state;
        
        // Validate action
        bool valid = validate_action(action);
        if (!valid) {
            result.reward = RewardFunction::INVALID_ACTION_PENALTY;
            result.message = "Invalid action";
            result.next_state.step_count++;
            m_state = result.next_state;
            return result;
        }
        
        // Execute action
        bool success = execute_action(action, result);
        
        // Calculate reward
        result.reward = RewardFunction::calculate(action, m_state, result.next_state, success);
        
        // Update state
        result.next_state.step_count = m_state.step_count + 1;
        m_state = result.next_state;
        
        // Check termination
        result.done = check_done();
        
        return result;
    }
    
    /**
     * @brief Get current state
     */
    const EnvironmentState& get_state() const {
        return m_state;
    }
    
    /**
     * @brief Get number of agents
     */
    int get_num_agents() const {
        return m_num_agents;
    }
    
    /**
     * @brief Render environment state to console
     */
    void render() const {
        std::cout << "\n=== Environment State (Episode " << m_state.episode 
                  << ", Step " << m_state.step_count << ") ===" << std::endl;
        
        std::cout << "\nTasks:" << std::endl;
        int complete = 0;
        int in_progress = 0;
        int unclaimed = 0;
        
        for (const auto& [id, task] : m_state.tasks) {
            if (task.status == TaskStatus::COMPLETE) complete++;
            else if (task.status == TaskStatus::IN_PROGRESS) in_progress++;
            else if (task.status == TaskStatus::UNCLAIMED) unclaimed++;
        }
        
        std::cout << "  Complete: " << complete << std::endl;
        std::cout << "  In Progress: " << in_progress << std::endl;
        std::cout << "  Unclaimed: " << unclaimed << std::endl;
        std::cout << "  Total: " << m_state.tasks.size() << std::endl;
        
        std::cout << "\nFiles: " << m_state.files.size() << std::endl;
        std::cout << "==========================================\n" << std::endl;
    }
    
private:
    int m_num_agents;
    int m_current_episode;
    EnvironmentState m_state;
    
    void initialize_tasks() {
        // Initial tasks from project requirements
        std::vector<std::tuple<std::string, Role, std::string>> initial_tasks = {
            {"ENV-001", Role::ARCHITECT, "Define state structs and enums"},
            {"ENV-002", Role::ARCHITECT, "Implement file system simulation"},
            {"ENV-003", Role::ARCHITECT, "Implement TODO.md parser/updater"},
            {"ENV-004", Role::ARCHITECT, "Write reward function logic"},
            {"RL-001", Role::CORE_RL, "Implement matrix/vector operations"},
            {"RL-002", Role::CORE_RL, "Build Actor class with forward pass"},
            {"RL-003", Role::CORE_RL, "Build Critic class"},
            {"PAR-001", Role::PARALLEL, "Set up thread pool"},
            {"PAR-002", Role::PARALLEL, "Implement worker function"},
            {"TST-001", Role::TESTING, "Write unit tests for environment"},
            {"TST-002", Role::TESTING, "Write unit tests for neural net"},
        };
        
        for (const auto& [id, role, desc] : initial_tasks) {
            TaskState task;
            task.task_id = id;
            task.role = role;
            task.description = desc;
            task.status = TaskStatus::UNCLAIMED;
            m_state.tasks[id] = task;
        }
    }
    
    bool validate_action(const Action& action) const {
        // Check agent ID is valid
        if (action.agent_id < 0 || action.agent_id >= m_num_agents) {
            return false;
        }
        
        // Check action-specific validation
        switch (action.type) {
            case ActionType::READ:
            case ActionType::EDIT:
                // Target should be a valid file path
                return !action.target.empty();
                
            case ActionType::CLAIM:
            case ActionType::COMPLETE:
            case ActionType::BLOCK:
                // Target should be a valid task ID
                return m_state.tasks.count(action.target) > 0;
                
            case ActionType::RUN_COMMAND:
                // Should have a command
                return !action.parameter.empty();
                
            case ActionType::NOOP:
                return true;
                
            default:
                return false;
        }
    }
    
    bool execute_action(const Action& action, StepResult& result) {
        switch (action.type) {
            case ActionType::READ:
                return execute_read(action, result);
                
            case ActionType::EDIT:
                return execute_edit(action, result);
                
            case ActionType::CLAIM:
                return execute_claim(action, result);
                
            case ActionType::COMPLETE:
                return execute_complete(action, result);
                
            case ActionType::RUN_COMMAND:
                return execute_command(action, result);
                
            default:
                result.message = "Noop or unknown action";
                return true;
        }
    }
    
    bool execute_read(const Action& action, StepResult& result) {
        auto it = m_state.files.find(action.target);
        if (it != m_state.files.end()) {
            result.message = "Read " + std::to_string(it->second.content.size()) + " bytes";
            return true;
        }
        result.message = "File not found: " + action.target;
        return false;
    }
    
    bool execute_edit(const Action& action, StepResult& result) {
        // Create or update file
        auto it = m_state.files.find(action.target);
        if (it != m_state.files.end()) {
            it->second.update(action.parameter);
        } else {
            m_state.files[action.target] = FileState(action.target, action.parameter);
        }
        result.message = "Edited " + action.target;
        return true;
    }
    
    bool execute_claim(const Action& action, StepResult& result) {
        auto it = m_state.tasks.find(action.target);
        if (it != m_state.tasks.end() && it->second.status == TaskStatus::UNCLAIMED) {
            it->second.status = TaskStatus::CLAIMED;
            it->second.claimed_by = "agent_" + std::to_string(action.agent_id);
            result.message = "Claimed " + action.target;
            return true;
        }
        result.message = "Cannot claim " + action.target;
        return false;
    }
    
    bool execute_complete(const Action& action, StepResult& result) {
        auto it = m_state.tasks.find(action.target);
        if (it != m_state.tasks.end() && it->second.status == TaskStatus::CLAIMED) {
            it->second.status = TaskStatus::COMPLETE;
            result.message = "Completed " + action.target;
            return true;
        }
        result.message = "Cannot complete " + action.target;
        return false;
    }
    
    bool execute_command(const Action& action, StepResult& result) {
        // Simulate command execution (in real impl, would use popen or similar)
        result.message = "Executed: " + action.parameter;
        
        // Simulate success for compile/test commands
        if (action.parameter.find("g++") != std::string::npos ||
            action.parameter.find("cmake") != std::string::npos) {
            return true;
        }
        
        return true;  // Assume success for simulation
    }
    
    bool check_done() const {
        // Episode ends when all tasks are complete or max steps reached
        int max_steps = 1000;
        
        if (m_state.step_count >= max_steps) {
            return true;
        }
        
        int complete = 0;
        for (const auto& [id, task] : m_state.tasks) {
            if (task.status == TaskStatus::COMPLETE) {
                complete++;
            }
        }
        
        return complete == static_cast<int>(m_state.tasks.size());
    }
};

} // namespace marl

#endif // MARL_ENVIRONMENT_HPP
