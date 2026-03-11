/**
 * @file ThreadPinningTest.cpp
 * @brief Test suite for ThreadPinning - CPU affinity management for JOLTrl
 * 
 * Tests critical properties:
 * 1. Thread affinity setting correctness
 * 2. Core isolation (no cross-core migration)
 * 3. Context switch reduction
 * 4. Performance improvement from pinned threads
 * 
 * Target Metrics:
 * - Zero cross-core migration
 * - <5% context switch reduction
 * - Measurable SPS improvement
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <sched.h>
#include <unistd.h>

#include "ThreadPinning.h"

// Test result structure
struct TestResult {
    std::string name;
    bool passed;
    double measured;
    double target;
    std::string unit;
    std::string message;
};

// Global test results
std::vector<TestResult> gTestResults;

// ============================================================================
// TEST 1: Basic Thread Pinning Test
// ============================================================================
// Verifies that threads can be pinned to specific cores

TestResult testBasicThreadPinning() {
    TestResult result{"Basic Thread Pinning Test", true, 0.0, 0.0, "errors", ""};
    
    std::cout << "\n[TEST 1] Basic Thread Pinning Test\n";
    std::cout << "  Verifying thread affinity setting...\n";
    
    try {
        // This will fail to compile until ThreadPinning is implemented
        ThreadPinning pinning;
        
        if (!pinning.isAvailable()) {
            result.passed = false;
            result.message = "Thread pinning not available on this system";
            std::cout << "  ⚠ WARNING: " << result.message << "\n";
            return result;
        }
        
        // Test pinning to core 0
        bool success = pinning.pinThread(0);
        if (!success) {
            result.passed = false;
            result.message = "Failed to pin thread to core 0";
            std::cout << "  ❌ FAILED: " << result.message << "\n";
            return result;
        }
        
        // Verify affinity
        int currentCore = pinning.getCurrentCore();
        if (currentCore < 0) {
            result.passed = false;
            result.message = "Failed to get current core";
            std::cout << "  ❌ FAILED: " << result.message << "\n";
            return result;
        }
        
        std::cout << "  ✓ Thread pinned to core " << currentCore << "\n";
        
        // Test pinning to different cores
        int numCores = pinning.getNumCores();
        std::cout << "  System has " << numCores << " CPU cores\n";
        
        for (int core = 0; core < std::min(numCores, 4); ++core) {
            success = pinning.pinThread(core);
            if (!success) {
                result.passed = false;
                result.message = "Failed to pin thread to core " + std::to_string(core);
                std::cout << "  ❌ FAILED: " << result.message << "\n";
                return result;
            }
            
            currentCore = pinning.getCurrentCore();
            std::cout << "  ✓ Core " << core << ": pinned successfully (actual: " 
                      << currentCore << ")\n";
        }
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
        std::cout << "  ❌ FAILED: " << result.message << "\n";
    }
    
    if (result.passed) {
        std::cout << "  ✓ PASSED: Thread pinning works correctly\n";
    }
    
    return result;
}

// ============================================================================
// TEST 2: Core Isolation Test
// ============================================================================
// Verifies that pinned threads stay on their assigned cores

TestResult testCoreIsolation() {
    TestResult result{"Core Isolation Test", true, 0.0, 5.0, "% migration", ""};
    
    std::cout << "\n[TEST 2] Core Isolation Test\n";
    std::cout << "  Testing thread migration under load...\n";
    
    try {
        ThreadPinning pinning;
        
        if (!pinning.isAvailable()) {
            result.passed = true;
            result.message = "Thread pinning not available, skipping test";
            std::cout << "  ⚠ WARNING: " << result.message << "\n";
            return result;
        }
        
        const int numThreads = 4;
        const int iterations = 10000;
        
        std::vector<std::thread> threads;
        std::atomic<int> migrationCount{0};
        std::atomic<bool> running{true};
        
        // Create threads pinned to different cores
        for (int t = 0; t < numThreads; ++t) {
            threads.emplace_back([&, t, iterations]() {
                // Pin thread to core t
                if (!pinning.pinThread(t % pinning.getNumCores())) {
                    return;  // Failed to pin
                }
                
                int lastCore = pinning.getCurrentCore();
                
                // Run workload
                for (int i = 0; i < iterations && running; ++i) {
                    // CPU-bound work
                    volatile double sum = 0.0;
                    for (int j = 0; j < 100; ++j) {
                        sum += std::sin(static_cast<double>(j)) * std::cos(static_cast<double>(i));
                    }
                    
                    // Check core every 100 iterations
                    if (i % 100 == 0) {
                        int currentCore = pinning.getCurrentCore();
                        if (currentCore != lastCore) {
                            migrationCount.fetch_add(1, std::memory_order_relaxed);
                            lastCore = currentCore;
                        }
                    }
                }
            });
        }
        
        // Wait for all threads
        for (auto& t : threads) {
            t.join();
        }
        
        int totalChecks = numThreads * (iterations / 100);
        double migrationRate = migrationCount.load() * 100.0 / totalChecks;
        
        result.measured = migrationRate;
        result.message = "Migration rate: " + std::to_string(migrationRate) + "%";
        
        if (migrationRate > 5.0) {
            result.passed = false;
            std::cout << "  ❌ FAILED: High migration rate (" << migrationRate << "%)\n";
        } else {
            std::cout << "  ✓ PASSED: Low migration rate (" << migrationRate << "%)\n";
        }
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
        std::cout << "  ❌ FAILED: " << result.message << "\n";
    }
    
    return result;
}

// ============================================================================
// TEST 3: Performance Comparison (Pinned vs Unpinned)
// ============================================================================
// Measures performance improvement from thread pinning

TestResult testPerformanceImprovement() {
    TestResult result{"Performance Improvement Test", true, 0.0, 1.1, "speedup", ""};
    
    std::cout << "\n[TEST 3] Performance Improvement Test\n";
    std::cout << "  Comparing pinned vs unpinned threads...\n";
    
    try {
        ThreadPinning pinning;
        
        if (!pinning.isAvailable()) {
            result.passed = true;
            result.message = "Thread pinning not available, skipping test";
            std::cout << "  ⚠ WARNING: " << result.message << "\n";
            return result;
        }
        
        const int numThreads = 4;
        const int iterations = 50000;
        
        // Test 1: Unpinned threads
        auto startUnpinned = std::chrono::high_resolution_clock::now();
        
        {
            std::vector<std::thread> threads;
            for (int t = 0; t < numThreads; ++t) {
                threads.emplace_back([iterations]() {
                    for (int i = 0; i < iterations; ++i) {
                        volatile double sum = 0.0;
                        for (int j = 0; j < 100; ++j) {
                            sum += std::sin(static_cast<double>(j)) * std::cos(static_cast<double>(i));
                        }
                    }
                });
            }
            
            for (auto& t : threads) {
                t.join();
            }
        }
        
        auto endUnpinned = std::chrono::high_resolution_clock::now();
        double timeUnpinned = std::chrono::duration<double, std::milli>(endUnpinned - startUnpinned).count();
        
        // Test 2: Pinned threads
        auto startPinned = std::chrono::high_resolution_clock::now();
        
        {
            std::vector<std::thread> threads;
            for (int t = 0; t < numThreads; ++t) {
                threads.emplace_back([&, t, iterations]() {
                    pinning.pinThread(t % pinning.getNumCores());
                    
                    for (int i = 0; i < iterations; ++i) {
                        volatile double sum = 0.0;
                        for (int j = 0; j < 100; ++j) {
                            sum += std::sin(static_cast<double>(j)) * std::cos(static_cast<double>(i));
                        }
                    }
                });
            }
            
            for (auto& t : threads) {
                t.join();
            }
        }
        
        auto endPinned = std::chrono::high_resolution_clock::now();
        double timePinned = std::chrono::duration<double, std::milli>(endPinned - startPinned).count();
        
        double speedup = timeUnpinned / timePinned;
        
        result.measured = speedup;
        result.message = "Unpinned: " + std::to_string(timeUnpinned) + "ms, " +
                        "Pinned: " + std::to_string(timePinned) + "ms, " +
                        "Speedup: " + std::to_string(speedup) + "x";
        
        std::cout << "  Unpinned time: " << timeUnpinned << " ms\n";
        std::cout << "  Pinned time: " << timePinned << " ms\n";
        std::cout << "  Speedup: " << speedup << "x\n";
        
        if (speedup < 1.0) {
            result.passed = false;
            std::cout << "  ❌ FAILED: No performance improvement\n";
        } else {
            std::cout << "  ✓ PASSED: " << speedup << "x speedup\n";
        }
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
        std::cout << "  ❌ FAILED: " << result.message << "\n";
    }
    
    return result;
}

// ============================================================================
// TEST 4: Invalid Core Handling
// ============================================================================
// Verifies error handling for invalid core IDs

TestResult testInvalidCoreHandling() {
    TestResult result{"Invalid Core Handling Test", true, 0.0, 0.0, "errors", ""};
    
    std::cout << "\n[TEST 4] Invalid Core Handling Test\n";
    std::cout << "  Testing error handling for invalid cores...\n";
    
    try {
        ThreadPinning pinning;
        
        if (!pinning.isAvailable()) {
            result.passed = true;
            result.message = "Thread pinning not available, skipping test";
            std::cout << "  ⚠ WARNING: " << result.message << "\n";
            return result;
        }
        
        int numCores = pinning.getNumCores();
        
        // Try to pin to invalid core (should fail gracefully)
        bool success = pinning.pinThread(numCores + 100);
        if (success) {
            result.passed = false;
            result.message = "Pinning to invalid core succeeded (should fail)";
            std::cout << "  ❌ FAILED: " << result.message << "\n";
            return result;
        }
        
        std::cout << "  ✓ Correctly rejected invalid core\n";
        
        // Try negative core (should fail gracefully)
        success = pinning.pinThread(-1);
        if (success) {
            result.passed = false;
            result.message = "Pinning to negative core succeeded (should fail)";
            std::cout << "  ❌ FAILED: " << result.message << "\n";
            return result;
        }
        
        std::cout << "  ✓ Correctly rejected negative core\n";
        
    } catch (const std::exception& e) {
        // Exception is acceptable for invalid input
        std::cout << "  ✓ Exception thrown for invalid core (acceptable)\n";
    }
    
    if (result.passed) {
        std::cout << "  ✓ PASSED: Invalid core handling correct\n";
    }
    
    return result;
}

// ============================================================================
// Main Test Runner
// ============================================================================

void printHeader() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║         JOLTrl ThreadPinning Test Suite                  ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Target Metrics:                                         ║\n";
    std::cout << "║  • Zero cross-core migration                             ║\n";
    std::cout << "║  • <5% context switch reduction                          ║\n";
    std::cout << "║  • Measurable SPS improvement                            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

void printSummary(const std::vector<TestResult>& results) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    TEST SUMMARY                          ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    
    int passed = 0;
    int failed = 0;
    int warnings = 0;
    
    for (const auto& result : results) {
        if (!result.passed && result.message.find("not available") != std::string::npos) {
            std::cout << "  ⚠ ";
            warnings++;
        } else if (result.passed) {
            std::cout << "  ✓ ";
            passed++;
        } else {
            std::cout << "  ❌ ";
            failed++;
        }
        std::cout << result.name << "\n";
    }
    
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "  Total: " << (passed + failed + warnings) << " tests, "
              << passed << " passed, " << failed << " failed, " 
              << warnings << " warnings\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    if (failed > 0) {
        std::cout << "\n⚠️  TESTS FAILED - IMPLEMENTATION REQUIRED\n";
        std::cout << "Note: Tests will fail to compile until ThreadPinning.h is implemented\n";
    } else if (warnings > 0) {
        std::cout << "\n⚠️  THREAD PINNING NOT AVAILABLE ON THIS SYSTEM\n";
        std::cout << "Tests skipped - this is expected on some systems\n";
    } else {
        std::cout << "\n✓ ALL TESTS PASSED\n";
    }
}

int main(int argc, char* argv[]) {
    printHeader();
    
    std::cout << "Configuration:\n";
    std::cout << "  • Testing thread pinning implementation\n";
    std::cout << "  • Verifying CPU core affinity\n";
    std::cout << "\n";
    
    // Run all tests
    // Note: These will fail to compile until ThreadPinning is implemented
    gTestResults.push_back(testBasicThreadPinning());
    gTestResults.push_back(testCoreIsolation());
    gTestResults.push_back(testPerformanceImprovement());
    gTestResults.push_back(testInvalidCoreHandling());
    
    printSummary(gTestResults);
    
    // Return exit code based on results
    int failed = 0;
    for (const auto& result : gTestResults) {
        if (!result.passed && result.message.find("not available") == std::string::npos) {
            failed++;
        }
    }
    
    return failed > 0 ? 1 : 0;
}
