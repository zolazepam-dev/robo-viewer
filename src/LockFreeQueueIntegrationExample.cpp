/**
 * @file LockFreeQueueIntegrationExample.cpp
 * @brief Example integration of LockFreeQueue into JOLTrl training loop
 * 
 * This file demonstrates how to replace mutex-protected queues with
 * lock-free data structures for improved SPS performance.
 * 
 * Key Integration Points:
 * 1. Transition transfer from SimulationLoop to TrainingLoop
 * 2. Action transfer from TrainingLoop to SimulationLoop  
 * 3. Elimination of gSimMutex contention
 * 
 * Performance Benefits:
 * - Zero mutex contention overhead
 * - Better cache locality
 * - Reduced thread synchronization latency
 * - Scalable to 128+ environments
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <cstring>
#include <queue>

#include "LockFreeQueue.h"
#include "AlignedAllocator.h"

// Example transition structure (simplified from actual ReplayBuffer)
struct Transition {
    float state[256];
    float action[56];
    float reward;
    float nextState[256];
    bool done;
    float latentPos[24];
    float latentVel[24];
};

/**
 * @class LockFreeTransitionQueue
 * @brief Specialized lock-free queue for transition transfer
 * 
 * Wraps LockFreeQueue<Transition> with batch operations for efficiency.
 * Used to transfer experiences from simulation environments to training buffer.
 */
class LockFreeTransitionQueue {
public:
    explicit LockFreeTransitionQueue(size_t capacity)
        : mQueue(capacity)
    {
    }
    
    /**
     * @brief Batch push multiple transitions
     * 
     * More efficient than individual pushes for parallel environment data.
     * 
     * @param transitions Array of transitions to push
     * @param count Number of transitions
     * @return Number of successfully pushed transitions
     */
    size_t pushBatch(const Transition* transitions, size_t count) {
        size_t pushed = 0;
        for (size_t i = 0; i < count; ++i) {
            if (mQueue.push(transitions[i])) {
                pushed++;
            } else {
                break;  // Queue full
            }
        }
        return pushed;
    }
    
    /**
     * @brief Batch pop multiple transitions
     * 
     * More efficient than individual pops for training batch collection.
     * 
     * @param transitions Output array for popped transitions
     * @param maxCount Maximum number to pop
     * @return Number of successfully popped transitions
     */
    size_t popBatch(Transition* transitions, size_t maxCount) {
        size_t popped = 0;
        for (size_t i = 0; i < maxCount; ++i) {
            if (mQueue.pop(transitions[i])) {
                popped++;
            } else {
                break;  // Queue empty
            }
        }
        return popped;
    }
    
    /**
     * @brief Single transition push
     */
    bool push(const Transition& t) {
        return mQueue.push(t);
    }
    
    /**
     * @brief Single transition pop
     */
    bool pop(Transition& t) {
        return mQueue.pop(t);
    }
    
    /**
     * @brief Check if queue is empty
     */
    bool empty() const {
        return mQueue.empty();
    }
    
    /**
     * @brief Get approximate size
     */
    size_t size() const {
        return mQueue.size();
    }

private:
    LockFreeQueue<Transition> mQueue;
};

// ============================================================================
// Example: Parallel Simulation with Lock-Free Transfer
// ============================================================================

void exampleParallelSimulation() {
    std::cout << "\n=== Example: Parallel Simulation with Lock-Free Transfer ===\n\n";
    
    const int numEnvironments = 128;
    const int transitionsPerEnv = 100;
    
    // Create lock-free queue for transition transfer
    // Capacity sized for 1 second of transitions from all environments
    LockFreeTransitionQueue transitionQueue(numEnvironments * 1000);
    
    std::atomic<long long> totalTransitions{0};
    std::atomic<bool> simulationRunning{true};
    
    // ========================================================================
    // Simulation Thread (Producer)
    // ========================================================================
    auto simulationThread = [&]() {
        std::cout << "[Simulation] Starting " << numEnvironments << " environments...\n";
        
        // Pre-allocate transition buffer (zero allocation in hot loop)
        std::vector<Transition> envTransitions(numEnvironments);
        
        for (int step = 0; step < transitionsPerEnv && simulationRunning; ++step) {
            // Each environment generates a transition
            for (int env = 0; env < numEnvironments; ++env) {
                // Simulate environment step (placeholder)
                Transition& t = envTransitions[env];
                t.reward = 1.0f;
                t.done = (step == transitionsPerEnv - 1);
                
                // Fill with dummy data
                std::memset(t.state, 0, sizeof(t.state));
                std::memset(t.action, 0, sizeof(t.action));
                std::memset(t.nextState, 0, sizeof(t.nextState));
                std::memset(t.latentPos, 0, sizeof(t.latentPos));
                std::memset(t.latentVel, 0, sizeof(t.latentVel));
                t.state[0] = static_cast<float>(env);
                t.nextState[0] = static_cast<float>(env + step);
            }
            
            // Batch push all transitions (lock-free, no mutex!)
            size_t pushed = transitionQueue.pushBatch(envTransitions.data(), numEnvironments);
            totalTransitions += pushed;
            
            if (step % 100 == 0) {
                std::cout << "[Simulation] Step " << step << ", pushed " << pushed 
                          << " transitions, queue size: " << transitionQueue.size() << "\n";
            }
        }
        
        std::cout << "[Simulation] Complete. Total transitions: " << totalTransitions << "\n";
    };
    
    // ========================================================================
    // Training Thread (Consumer)
    // ========================================================================
    auto trainingThread = [&]() {
        std::cout << "[Training] Starting training loop...\n";
        
        // Pre-allocate batch buffer (zero allocation in hot loop)
        std::vector<Transition> batch(256);
        
        int trainingSteps = 0;
        size_t totalConsumed = 0;
        
        while (totalTransitions < numEnvironments * transitionsPerEnv || !transitionQueue.empty()) {
            // Pop a batch of transitions for training
            size_t popped = transitionQueue.popBatch(batch.data(), batch.size());
            
            if (popped > 0) {
                totalConsumed += popped;
                
                // Simulate training on batch (placeholder)
                // In real implementation: compute gradients, update network
                trainingSteps++;
                
                if (trainingSteps % 10 == 0) {
                    std::cout << "[Training] Step " << trainingSteps << ", consumed " 
                              << popped << " transitions, total: " << totalConsumed << "\n";
                }
            } else {
                // Queue empty, yield to avoid busy waiting
                std::this_thread::yield();
            }
        }
        
        std::cout << "[Training] Complete. Training steps: " << trainingSteps 
                  << ", Total consumed: " << totalConsumed << "\n";
    };
    
    // ========================================================================
    // Run Simulation and Training in Parallel
    // ========================================================================
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::thread sim(simulationThread);
    std::thread train(trainingThread);
    
    sim.join();
    train.join();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedSec = std::chrono::duration<double>(endTime - startTime).count();
    
    std::cout << "\n=== Performance Summary ===\n";
    std::cout << "Total time: " << elapsedSec << " seconds\n";
    std::cout << "Total transitions: " << totalTransitions << "\n";
    std::cout << "Throughput: " << (totalTransitions / elapsedSec) << " transitions/sec\n";
    std::cout << "Zero mutex contention: YES\n";
    std::cout << "==========================\n";
}

// ============================================================================
// Example: Comparison with Mutex-Based Queue
// ============================================================================

void exampleMutexComparison() {
    std::cout << "\n=== Example: Mutex vs Lock-Free Comparison ===\n\n";
    
    const int numOperations = 100000;
    
    // ========================================================================
    // Test 1: Mutex-protected queue
    // ========================================================================
    {
        std::queue<int> mutexQueue;
        std::mutex queueMutex;
        std::atomic<int> pushCount{0};
        std::atomic<int> popCount{0};
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Single thread test (to avoid deadlock issues in example)
        for (int i = 0; i < numOperations; ++i) {
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                mutexQueue.push(i);
            }
            pushCount++;
            
            int value;
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                if (!mutexQueue.empty()) {
                    value = mutexQueue.front();
                    mutexQueue.pop();
                    popCount++;
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::cout << "Mutex Queue (single thread):\n";
        std::cout << "  Operations: " << numOperations << "\n";
        std::cout << "  Time: " << elapsed << " ms\n";
        std::cout << "  Throughput: " << (numOperations / elapsed * 1000.0) << " ops/sec\n";
    }
    
    // ========================================================================
    // Test 2: Lock-free queue
    // ========================================================================
    {
        LockFreeQueue<int> lockFreeQueue(numOperations);
        std::atomic<int> pushCount{0};
        std::atomic<int> popCount{0};
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Single thread test (for fair comparison)
        for (int i = 0; i < numOperations; ++i) {
            if (lockFreeQueue.push(i)) {
                pushCount++;
            }
            
            int value;
            if (lockFreeQueue.pop(value)) {
                popCount++;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::cout << "\nLock-Free Queue (single thread):\n";
        std::cout << "  Operations: " << numOperations << "\n";
        std::cout << "  Time: " << elapsed << " ms\n";
        std::cout << "  Throughput: " << (numOperations / elapsed * 1000.0) << " ops/sec\n";
        std::cout << "  Speedup: " << (elapsed > 0 ? (numOperations / elapsed * 1000.0) / 
                                        (numOperations / elapsed * 1000.0) : 0) << "x\n";
    }
    
    std::cout << "\nNote: Lock-free queue shows greater benefits under high contention\n";
    std::cout << "      (multiple producers/consumers) where mutex would cause blocking.\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "JOLTrl Lock-Free Queue Integration Examples\n";
    std::cout << "============================================\n\n";
    
    // Run examples
    exampleParallelSimulation();
    exampleMutexComparison();
    
    std::cout << "\n=== Integration Complete ===\n";
    std::cout << "Next Steps:\n";
    std::cout << "1. Replace gSimMutex protected queues in main_train.cpp\n";
    std::cout << "2. Use LockFreeTransitionQueue for experience transfer\n";
    std::cout << "3. Measure SPS improvement (target: 25,000+ SPS)\n";
    std::cout << "============================\n";
    
    return 0;
}
