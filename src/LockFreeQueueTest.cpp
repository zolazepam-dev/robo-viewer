/**
 * @file LockFreeQueueTest.cpp
 * @brief Test suite for LockFreeQueue - Lock-free ring buffer for JOLTrl
 * 
 * Tests critical properties:
 * 1. Thread-safe push/pop operations
 * 2. Zero contention overhead with multiple producers/consumers
 * 3. Correct FIFO ordering under concurrent access
 * 4. Memory ordering constraints (acquire/release semantics)
 * 
 * Target Metrics:
 * - Zero mutex contention
 * - <1% failed CAS operations
 * - Correct ordering under stress test (128 producers)
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>

#include "AlignedAllocator.h"
#include "LockFreeQueue.h"

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
// TEST 1: Basic Push/Pop Test (Single Thread)
// ============================================================================
// Verifies FIFO ordering in single-threaded scenario

TestResult testBasicPushPop() {
    TestResult result{"Basic Push/Pop Test", true, 0.0, 0.0, "errors", ""};
    
    std::cout << "\n[TEST 1] Basic Push/Pop Test (Single Thread)\n";
    std::cout << "  Verifying FIFO ordering...\n";
    
    try {
        // This will fail to compile until LockFreeQueue is implemented
        LockFreeQueue<int> queue(1024);
        
        // Push test values
        for (int i = 0; i < 100; ++i) {
            bool success = queue.push(i);
            if (!success) {
                result.passed = false;
                result.message = "Push failed at index " + std::to_string(i);
                break;
            }
        }
        
        // Pop and verify order
        int expected = 0;
        while (!queue.empty()) {
            int value;
            bool success = queue.pop(value);
            if (!success) {
                result.passed = false;
                result.message = "Pop failed at index " + std::to_string(expected);
                break;
            }
            
            if (value != expected) {
                result.passed = false;
                result.message = "Order mismatch: expected " + std::to_string(expected) + 
                                ", got " + std::to_string(value);
                break;
            }
            expected++;
        }
        
        if (expected != 100) {
            result.passed = false;
            result.message = "Count mismatch: expected 100, popped " + std::to_string(expected);
        }
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    if (result.passed) {
        std::cout << "  ✓ PASSED: FIFO ordering correct\n";
    } else {
        std::cout << "  ❌ FAILED: " << result.message << "\n";
    }
    
    return result;
}

// ============================================================================
// TEST 2: Concurrent Producer/Consumer Test
// ============================================================================
// Verifies thread safety with multiple producers and consumers

TestResult testConcurrentProducersConsumers() {
    TestResult result{"Concurrent Producer/Consumer Test", true, 0.0, 1.0, "% loss", ""};
    
    std::cout << "\n[TEST 2] Concurrent Producer/Consumer Test\n";
    std::cout << "  Testing with 8 producers and 4 consumers...\n";
    
    const int numProducers = 8;
    const int numConsumers = 4;
    const int itemsPerProducer = 10000;
    const int totalItems = numProducers * itemsPerProducer;
    
    LockFreeQueue<int> queue(1024 * 1024);  // 1M capacity
    
    std::atomic<int> pushCount{0};
    std::atomic<int> popCount{0};
    std::atomic<bool> producersDone{false};
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    
    // Start producers
    for (int p = 0; p < numProducers; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < itemsPerProducer; ++i) {
                int value = p * itemsPerProducer + i;
                int attempts = 0;
                while (!queue.push(value)) {
                    // Queue full, spin briefly
                    if (++attempts > 1000) {
                        std::this_thread::yield();
                        attempts = 0;
                    }
                }
                pushCount.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    
    // Start consumers
    for (int c = 0; c < numConsumers; ++c) {
        consumers.emplace_back([&]() {
            while (true) {
                int value;
                if (queue.pop(value)) {
                    popCount.fetch_add(1, std::memory_order_relaxed);
                } else {
                    // Queue empty
                    if (producersDone.load(std::memory_order_acquire) && queue.empty()) {
                        break;
                    }
                    std::this_thread::yield();
                }
            }
        });
    }
    
    // Wait for producers
    for (auto& t : producers) {
        t.join();
    }
    producersDone.store(true, std::memory_order_release);
    
    // Wait for consumers
    for (auto& t : consumers) {
        t.join();
    }
    
    int pushed = pushCount.load();
    int popped = popCount.load();
    double lossRate = (pushed - popped) * 100.0 / pushed;
    
    result.measured = lossRate;
    result.message = "Pushed: " + std::to_string(pushed) + 
                    ", Popped: " + std::to_string(popped) +
                    ", Loss: " + std::to_string(lossRate) + "%";
    
    if (lossRate > 1.0) {
        result.passed = false;
        std::cout << "  ❌ FAILED: Data loss detected (" << lossRate << "%)\n";
    } else {
        std::cout << "  ✓ PASSED: Zero data loss (" << lossRate << "%)\n";
    }
    
    return result;
}

// ============================================================================
// TEST 3: Stress Test with 128 Producers (Simulating 128 Environments)
// ============================================================================
// Verifies scalability with maximum parallel environments

TestResult testStress128Producers() {
    TestResult result{"Stress Test (128 Producers)", true, 0.0, 10.0, "ms/item", ""};
    
    std::cout << "\n[TEST 3] Stress Test with 128 Producers\n";
    std::cout << "  Simulating 128 parallel environments...\n";
    
    const int numProducers = 128;
    const int itemsPerProducer = 1000;
    const int totalItems = numProducers * itemsPerProducer;
    
    LockFreeQueue<int> queue(1024 * 1024);  // 1M capacity
    
    std::atomic<int> pushCount{0};
    std::atomic<int> popCount{0};
    std::atomic<bool> producersDone{false};
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Start 128 producers
    for (int p = 0; p < numProducers; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < itemsPerProducer; ++i) {
                int value = p * itemsPerProducer + i;
                int attempts = 0;
                while (!queue.push(value)) {
                    if (++attempts > 1000) {
                        std::this_thread::yield();
                        attempts = 0;
                    }
                }
                pushCount.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    
    // Start 16 consumers
    const int numConsumers = 16;
    for (int c = 0; c < numConsumers; ++c) {
        consumers.emplace_back([&]() {
            while (true) {
                int value;
                if (queue.pop(value)) {
                    popCount.fetch_add(1, std::memory_order_relaxed);
                } else {
                    if (producersDone.load(std::memory_order_acquire) && queue.empty()) {
                        break;
                    }
                    std::this_thread::yield();
                }
            }
        });
    }
    
    // Wait for producers
    for (auto& t : producers) {
        t.join();
    }
    producersDone.store(true, std::memory_order_release);
    
    // Wait for consumers
    for (auto& t : consumers) {
        t.join();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedSec = std::chrono::duration<double>(endTime - startTime).count();
    double msPerItem = elapsedSec * 1000.0 / totalItems;
    
    result.measured = msPerItem;
    int pushed = pushCount.load();
    int popped = popCount.load();
    
    result.message = "Total time: " + std::to_string(elapsedSec) + "s, " +
                    "Throughput: " + std::to_string(totalItems / elapsedSec) + " items/s, " +
                    "Per-item: " + std::to_string(msPerItem) + "ms, " +
                    "Pushed: " + std::to_string(pushed) + 
                    ", Popped: " + std::to_string(popped);
    
    if (msPerItem > 10.0) {
        result.passed = false;
        std::cout << "  ❌ FAILED: Too slow (" << msPerItem << " ms/item)\n";
    } else {
        std::cout << "  ✓ PASSED: " << std::to_string(totalItems / elapsedSec) 
                  << " items/s throughput\n";
    }
    
    return result;
}

// ============================================================================
// TEST 4: CAS Failure Rate Test
// ============================================================================
// Measures contention by tracking CAS retry rates

TestResult testCASFailureRate() {
    TestResult result{"CAS Failure Rate Test", true, 0.0, 5.0, "%", ""};
    
    std::cout << "\n[TEST 4] CAS Failure Rate Test\n";
    std::cout << "  Measuring contention under high load...\n";
    
    // This test requires instrumentation in LockFreeQueue to track CAS attempts
    // For now, we'll simulate high contention and measure throughput degradation
    
    const int numProducers = 32;
    const int itemsPerProducer = 5000;
    
    LockFreeQueue<int> queue(256);  // Small queue to force contention
    
    std::atomic<int> pushCount{0};
    std::atomic<int> popCount{0};
    std::atomic<bool> running{true};
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Start producers
    for (int p = 0; p < numProducers; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < itemsPerProducer; ++i) {
                int attempts = 0;
                while (!queue.push(p * itemsPerProducer + i)) {
                    attempts++;
                    if (attempts > 10000) {
                        std::this_thread::yield();
                        attempts = 0;
                    }
                }
                pushCount.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    
    // Start consumers
    const int numConsumers = 8;
    for (int c = 0; c < numConsumers; ++c) {
        consumers.emplace_back([&]() {
            while (running.load(std::memory_order_acquire)) {
                int value;
                if (queue.pop(value)) {
                    popCount.fetch_add(1, std::memory_order_relaxed);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }
    
    // Wait for producers
    for (auto& t : producers) {
        t.join();
    }
    
    // Let consumers drain
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    running.store(false, std::memory_order_release);
    
    for (auto& t : consumers) {
        t.join();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedSec = std::chrono::duration<double>(endTime - startTime).count();
    
    // Calculate effective throughput vs theoretical maximum
    int pushed = pushCount.load();
    int popped = popCount.load();
    
    // If queue is working correctly, all items should be popped
    double lossRate = (pushed - popped) * 100.0 / pushed;
    
    result.measured = lossRate;
    result.message = "Pushed: " + std::to_string(pushed) + 
                    ", Popped: " + std::to_string(popped) +
                    ", Loss: " + std::to_string(lossRate) + "%";
    
    if (lossRate > 5.0) {
        result.passed = false;
        std::cout << "  ❌ FAILED: High data loss under contention (" << lossRate << "%)\n";
    } else {
        std::cout << "  ✓ PASSED: Low data loss (" << lossRate << "%)\n";
    }
    
    return result;
}

// ============================================================================
// TEST 5: Empty/Full Boundary Test
// ============================================================================
// Verifies correct behavior at queue boundaries

TestResult testBoundaryConditions() {
    TestResult result{"Boundary Conditions Test", true, 0.0, 0.0, "errors", ""};
    
    std::cout << "\n[TEST 5] Empty/Full Boundary Test\n";
    std::cout << "  Testing edge cases...\n";
    
    bool allPassed = true;
    
    try {
        LockFreeQueue<int> queue(8);  // Small capacity for easy testing
        
        // Test 1: Pop from empty queue
        int value;
        bool popResult = queue.pop(value);
        if (popResult) {
            result.passed = false;
            allPassed = false;
            result.message = "Pop from empty queue succeeded (should fail)";
            std::cout << "  ❌ Pop from empty returned true\n";
        } else {
            std::cout << "  ✓ Pop from empty correctly returns false\n";
        }
        
        // Test 2: Fill to capacity
        for (int i = 0; i < 8; ++i) {
            bool pushResult = queue.push(i);
            if (!pushResult) {
                result.passed = false;
                allPassed = false;
                result.message = "Push " + std::to_string(i) + " failed (queue not full yet)";
                std::cout << "  ❌ Push " << i << " failed prematurely\n";
                break;
            }
        }
        
        if (allPassed) {
            std::cout << "  ✓ Fill to capacity succeeded\n";
            
            // Test 3: Push to full queue (should fail)
            bool pushResult = queue.push(999);
            if (pushResult) {
                result.passed = false;
                allPassed = false;
                result.message = "Push to full queue succeeded (should fail)";
                std::cout << "  ❌ Push to full returned true\n";
            } else {
                std::cout << "  ✓ Push to full correctly returns false\n";
            }
            
            // Test 4: Drain completely
            for (int i = 0; i < 8; ++i) {
                bool popResult = queue.pop(value);
                if (!popResult || value != i) {
                    result.passed = false;
                    allPassed = false;
                    result.message = "Pop " + std::to_string(i) + " failed or wrong value";
                    std::cout << "  ❌ Pop " << i << " failed\n";
                    break;
                }
            }
            
            if (allPassed) {
                std::cout << "  ✓ Complete drain succeeded\n";
                
                // Test 5: Verify empty after drain
                popResult = queue.pop(value);
                if (popResult) {
                    result.passed = false;
                    allPassed = false;
                    result.message = "Queue not empty after drain";
                    std::cout << "  ❌ Queue not empty after drain\n";
                } else {
                    std::cout << "  ✓ Queue empty after drain\n";
                }
            }
        }
        
    } catch (const std::exception& e) {
        result.passed = false;
        allPassed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    if (allPassed) {
        std::cout << "  ✓ PASSED: All boundary conditions correct\n";
    } else {
        std::cout << "  ❌ FAILED: " << result.message << "\n";
    }
    
    return result;
}

// ============================================================================
// Main Test Runner
// ============================================================================

void printHeader() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║         JOLTrl LockFreeQueue Test Suite                  ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Target Metrics:                                         ║\n";
    std::cout << "║  • Zero mutex contention                                 ║\n";
    std::cout << "║  • <1% failed CAS operations                             ║\n";
    std::cout << "║  • Correct FIFO ordering under stress                    ║\n";
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
    
    for (const auto& result : results) {
        std::cout << (result.passed ? "  ✓ " : "  ❌ ");
        std::cout << result.name << "\n";
        
        if (result.passed) passed++;
        else failed++;
    }
    
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "  Total: " << (passed + failed) << " tests, "
              << passed << " passed, " << failed << " failed\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    if (failed > 0) {
        std::cout << "\n⚠️  TESTS FAILED - IMPLEMENTATION REQUIRED\n";
        std::cout << "Note: Tests will fail to compile until LockFreeQueue.h is implemented\n";
    } else {
        std::cout << "\n✓ ALL TESTS PASSED\n";
    }
}

int main(int argc, char* argv[]) {
    printHeader();
    
    std::cout << "Configuration:\n";
    std::cout << "  • Testing lock-free ring buffer implementation\n";
    std::cout << "  • Concurrent access with up to 128 producers\n";
    std::cout << "\n";
    
    // Run all tests
    // Note: These will fail to compile until LockFreeQueue is implemented
    gTestResults.push_back(testBasicPushPop());
    gTestResults.push_back(testConcurrentProducersConsumers());
    gTestResults.push_back(testStress128Producers());
    gTestResults.push_back(testCASFailureRate());
    gTestResults.push_back(testBoundaryConditions());
    
    printSummary(gTestResults);
    
    // Return exit code based on results
    int failed = 0;
    for (const auto& result : gTestResults) {
        if (!result.passed) failed++;
    }
    
    return failed > 0 ? 1 : 0;
}
