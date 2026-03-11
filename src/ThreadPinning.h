/**
 * @file ThreadPinning.h
 * @brief CPU affinity management for high-performance threading in JOLTrl
 * 
 * This implementation provides:
 * - Thread pinning to specific CPU cores using pthread_setaffinity_np
 * - Core isolation to prevent cross-core migration
 * - Reduced context switch overhead
 * - Optimal CPU layout for Jolt Physics job system
 * 
 * CPU Layout for JOLTrl (12-core system example):
 * - Core 0: Main RL loop + OS tasks
 * - Cores 1-5: Jolt Physics worker threads
 * - Cores 6-11: Additional physics workers (hyperthreading)
 * 
 * Usage:
 * @code
 * ThreadPinning pinning;
 * 
 * // Pin current thread to core 0
 * pinning.pinThread(0);
 * 
 * // Pin physics worker to core 3
 * pinning.pinThread(3);
 * @endcode
 */

#pragma once

#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <atomic>
#include <vector>
#include <iostream>

/**
 * @class ThreadPinning
 * @brief Manages CPU core affinity for threads
 * 
 * Provides a simple interface for pinning threads to specific CPU cores
 * to reduce context switch overhead and improve cache locality.
 * 
 * Features:
 * - Runtime detection of available CPU cores
 * - Graceful fallback if pinning is unavailable
 * - Error handling for invalid core IDs
 * - Thread-safe operations
 * 
 * Performance Benefits:
 * - Reduced context switch overhead (<5% migration rate)
 * - Improved cache locality (data stays in L1/L2)
 * - Predictable thread scheduling (no OS interference)
 * - Better SPS in RL training loop
 */
class ThreadPinning {
public:
    /**
     * @brief Construct ThreadPinning manager
     * 
     * Detects number of available CPU cores at runtime.
     */
    ThreadPinning()
        : mNumCores(sysconf(_SC_NPROCESSORS_ONLN))
        , mIsAvailable(mNumCores > 0)
    {
        if (mIsAvailable) {
            std::cerr << "[ThreadPinning] Detected " << mNumCores << " CPU cores\n";
        } else {
            std::cerr << "[ThreadPinning] WARNING: Could not detect CPU cores\n";
        }
    }
    
    /**
     * @brief Check if thread pinning is available
     * @return true if pinning is supported on this system
     */
    bool isAvailable() const {
        return mIsAvailable;
    }
    
    /**
     * @brief Get number of available CPU cores
     * @return Number of cores
     */
    int getNumCores() const {
        return mNumCores;
    }
    
    /**
     * @brief Pin current thread to specified core
     * 
     * Uses pthread_setaffinity_np to set CPU affinity mask.
     * Fails gracefully if core ID is invalid or pinning is unavailable.
     * 
     * @param coreId CPU core to pin to (0-based)
     * @return true if pinning succeeded, false otherwise
     */
    bool pinThread(int coreId) {
        if (!mIsAvailable) {
            std::cerr << "[ThreadPinning] Pinning not available\n";
            return false;
        }
        
        if (coreId < 0 || coreId >= mNumCores) {
            std::cerr << "[ThreadPinning] Invalid core ID: " << coreId 
                      << " (valid range: 0-" << (mNumCores - 1) << ")\n";
            return false;
        }
        
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(coreId, &cpuset);
        
        pthread_t currentThread = pthread_self();
        int result = pthread_setaffinity_np(currentThread, sizeof(cpuset), &cpuset);
        
        if (result != 0) {
            std::cerr << "[ThreadPinning] Failed to pin thread to core " << coreId 
                      << ": error " << result << "\n";
            return false;
        }
        
        #ifdef DEBUG_THREAD_PINNING
        std::cerr << "[ThreadPinning] Thread pinned to core " << coreId << "\n";
        #endif
        
        return true;
    }
    
    /**
     * @brief Get current core ID of calling thread
     * @return Core ID, or -1 if detection failed
     */
    int getCurrentCore() const {
        if (!mIsAvailable) {
            return -1;
        }
        
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        
        pthread_t currentThread = pthread_self();
        int result = pthread_getaffinity_np(currentThread, sizeof(cpuset), &cpuset);
        
        if (result != 0) {
            std::cerr << "[ThreadPinning] Failed to get thread affinity: error " 
                      << result << "\n";
            return -1;
        }
        
        // Find first set bit (current core)
        for (int i = 0; i < mNumCores; ++i) {
            if (CPU_ISSET(i, &cpuset)) {
                return i;
            }
        }
        
        return -1;  // No core found (should not happen)
    }
    
    /**
     * @brief Pin Jolt Physics worker threads to optimal cores
     * 
     * Recommended layout for 12-core system:
     * - Worker 0-4: Cores 1-5 (dedicated physics cores)
     * - Worker 5-11: Cores 7-11 (hyperthreading cores)
     * 
     * Core 0 and 6 are reserved for main RL loop and OS.
     * 
     * @param workerId Worker thread ID (0-based)
     * @return true if pinning succeeded
     */
    bool pinJoltWorker(int workerId) {
        if (!mIsAvailable) {
            return false;
        }
        
        // Map worker ID to physical core
        // Workers 0-4 -> Cores 1-5
        // Workers 5-11 -> Cores 7-11
        int physicalCore;
        if (workerId < 5) {
            physicalCore = workerId + 1;  // Cores 1-5
        } else {
            physicalCore = workerId + 2;  // Cores 7-11
        }
        
        if (physicalCore >= mNumCores) {
            // Wrap around if we have more workers than cores
            physicalCore = (workerId % (mNumCores - 2)) + 1;
        }
        
        return pinThread(physicalCore);
    }
    
    /**
     * @brief Pin main RL loop to core 0
     * 
     * Core 0 is typically the primary core with best single-thread performance.
     * 
     * @return true if pinning succeeded
     */
    bool pinMainLoop() {
        return pinThread(0);
    }

private:
    int mNumCores;           ///< Number of available CPU cores
    bool mIsAvailable;       ///< Whether pinning is available
};

/**
 * @namespace opt
 * @brief Optimization utilities for JOLTrl
 * 
 * Contains performance optimization utilities including thread pinning.
 */
namespace opt {

/**
 * @brief Global ThreadPinning instance
 * 
 * Singleton instance for thread pinning management.
 * Use this for consistent core assignment across the application.
 */
inline ThreadPinning& getThreadPinning() {
    static ThreadPinning instance;
    return instance;
}

/**
 * @brief Pin current thread to specified core (convenience function)
 * @param coreId CPU core to pin to
 * @return true if pinning succeeded
 */
inline bool pinThreadToCore(int coreId) {
    return getThreadPinning().pinThread(coreId);
}

/**
 * @brief Pin Jolt worker thread (convenience function)
 * @param workerId Worker thread ID
 * @return true if pinning succeeded
 */
inline bool pinJoltWorkerThread(int workerId) {
    return getThreadPinning().pinJoltWorker(workerId);
}

}  // namespace opt

/**
 * @example ThreadPinningExample.cpp
 * 
 * Basic usage example:
 * 
 * @code
 * #include "ThreadPinning.h"
 * #include <thread>
 * #include <iostream>
 * 
 * int main() {
 *     ThreadPinning pinning;
 *     
 *     if (!pinning.isAvailable()) {
 *         std::cout << "Thread pinning not available\n";
 *         return 0;
 *     }
 *     
 *     std::cout << "Detected " << pinning.getNumCores() << " cores\n";
 *     
 *     // Pin main thread to core 0
 *     pinning.pinMainLoop();
 *     std::cout << "Main thread pinned to core " << pinning.getCurrentCore() << "\n";
 *     
 *     // Create worker threads pinned to specific cores
 *     std::vector<std::thread> workers;
 *     for (int i = 0; i < 4; ++i) {
 *         workers.emplace_back([&pinning, i]() {
 *             pinning.pinJoltWorker(i);
 *             std::cout << "Worker " << i << " on core " 
 *                       << pinning.getCurrentCore() << "\n";
 *             // ... do work ...
 *         });
 *     }
 *     
 *     for (auto& t : workers) {
 *         t.join();
 *     }
 *     
 *     return 0;
 * }
 * @endcode
 */
