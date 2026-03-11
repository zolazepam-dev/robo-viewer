/**
 * @file IntegrationTest.cpp
 * @brief Comprehensive integration tests for SPS optimization phases 1-4
 *
 * This test suite validates that all optimizations work together correctly:
 * - Phase 1: Lock-Free Queues (thread-safe data transfer)
 * - Phase 2: Thread Pinning (CPU affinity management)
 * - Phase 3: SIMD Vectorization (AVX2 operations)
 * - Phase 4: SoA Memory Pool (cache-efficient layout)
 *
 * Test Categories:
 * 1. Concurrency Tests: Deadlock detection, race condition testing
 * 2. Correctness Tests: Verify optimized ops match scalar baseline
 * 3. Stress Tests: High-load scenarios with 128+ environments
 * 4. Stability Tests: Long-running validation (500+ steps)
 *
 * Usage:
 * @code
 * # Run all tests
 * bazel run //:IntegrationTest
 *
 * # Run specific test
 * bazel run //:IntegrationTest -- --test deadlock
 *
 * # Run with custom environment count
 * bazel run //:IntegrationTest -- --envs 256 --steps 1000
 * @endcode
 *
 * @author JOLTrl Team
 * @date March 2026
 * @version 1.0 (Phase 5 Integration)
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <random>
#include <functional>
#include <sstream>
#include <cstring>

#include "src/LockFreeQueue.h"
#include "src/ThreadPinning.h"
#include "src/NeuralMath.h"
#include "src/OptimizedMath.h"
#include "src/OptimizedBatchOps.h"
#include "src/SoAEnvironment.h"
#include "src/AlignedAllocator.h"
#include "src/VectorizedEnv.h"
#include "src/TD3Trainer.h"

// ============================================================================
// TEST CONFIGURATION
// ============================================================================

struct TestConfig {
    int numEnvs = 128;              // Number of parallel environments
    int numSteps = 500;             // Number of test steps
    int numThreads = 8;             // Number of concurrent threads
    int timeoutSec = 30;            // Timeout for deadlock detection
    std::string testFilter = "";    // Run specific test (empty = all)
    bool verbose = false;           // Print detailed output
    int stressIterations = 10;      // Iterations for stress tests
};

// ============================================================================
// TEST RESULT STRUCTURE
// ============================================================================

struct TestResult {
    std::string testName;
    std::string category;  // Concurrency, Correctness, Stress, Stability
    bool passed;
    double executionTimeMs;
    std::string message;
    std::string errorDetails;
    
    TestResult() : passed(false), executionTimeMs(0.0) {}
    
    static TestResult Pass(const std::string& name, const std::string& category, 
                           double timeMs, const std::string& msg = "") {
        TestResult r;
        r.testName = name;
        r.category = category;
        r.passed = true;
        r.executionTimeMs = timeMs;
        r.message = msg;
        return r;
    }
    
    static TestResult Fail(const std::string& name, const std::string& category,
                           const std::string& error) {
        TestResult r;
        r.testName = name;
        r.category = category;
        r.passed = false;
        r.errorDetails = error;
        return r;
    }
};

// ============================================================================
// TEST UTILITIES
// ============================================================================

/**
 * @brief Run a test with timeout protection (deadlock detection)
 */
template<typename Func>
TestResult runWithTimeout(const std::string& name, const std::string& category,
                          Func testFunc, int timeoutSec) {
    std::atomic<bool> completed{false};
    std::atomic<bool> error{false};
    std::string errorMsg;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::thread testThread([&]() {
        try {
            testFunc();
            completed = true;
        } catch (const std::exception& e) {
            error = true;
            errorMsg = e.what();
        } catch (...) {
            error = true;
            errorMsg = "Unknown exception";
        }
    });
    
    // Wait for completion or timeout
    std::chrono::seconds timeout(timeoutSec);
    if (testThread.joinable()) {
        if (testThread.join_for(timeout)) {
            auto end = std::chrono::high_resolution_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
            
            if (error) {
                return TestResult::Fail(name, category, errorMsg);
            }
            return TestResult::Pass(name, category, elapsedMs);
        } else {
            // Timeout - likely deadlock
            std::ostringstream oss;
            oss << "TIMEOUT: Test exceeded " << timeoutSec << "s (possible deadlock)";
            auto result = TestResult::Fail(name, category, oss.str());
            result.testName = name;
            result.category = category;
            
            // Detach thread (let it run, we can't safely kill it)
            testThread.detach();
            return result;
        }
    }
    
    return TestResult::Fail(name, category, "Failed to start test thread");
}

/**
 * @brief Print test result
 */
void printTestResult(const TestResult& result) {
    std::cout << (result.passed ? "  ✓ " : "  ❌ ");
    std::cout << std::left << std::setw(45) << result.testName;
    std::cout << std::fixed << std::setprecision(2) << std::setw(8) << result.executionTimeMs << "ms";
    
    if (!result.passed) {
        std::cout << "\n      Error: " << result.errorDetails;
    } else if (!result.message.empty()) {
        std::cout << " - " << result.message;
    }
    std::cout << "\n";
}

// ============================================================================
// PHASE 1: LOCK-FREE QUEUE TESTS
// ============================================================================

/**
 * @brief Test lock-free queue with single producer, single consumer
 */
TestResult testLockFreeQueueSingleProducerConsumer(const TestConfig& config) {
    const std::string name = "LockFreeQueue: Single Producer/Consumer";
    const std::string category = "Concurrency";
    
    return runWithTimeout(name, category, [&]() {
        LockFreeQueue<int> queue(1024);
        std::atomic<int> produced{0};
        std::atomic<int> consumed{0};
        const int numItems = 10000;
        
        // Producer thread
        std::thread producer([&]() {
            for (int i = 0; i < numItems; ++i) {
                while (!queue.push(i)) {
                    std::this_thread::yield();
                }
                produced++;
            }
        });
        
        // Consumer thread
        std::thread consumer([&]() {
            int value;
            while (consumed < numItems) {
                if (queue.pop(value)) {
                    consumed++;
                } else {
                    std::this_thread::yield();
                }
            }
        });
        
        producer.join();
        consumer.join();
        
        assert(produced == numItems);
        assert(consumed == numItems);
        assert(queue.empty());
        
    }, config.timeoutSec);
}

/**
 * @brief Test lock-free queue with multiple producers (stress test)
 */
TestResult testLockFreeQueueMultiProducer(const TestConfig& config) {
    const std::string name = "LockFreeQueue: Multi-Producer Stress";
    const std::string category = "Stress";
    
    return runWithTimeout(name, category, [&]() {
        LockFreeQueue<int> queue(8192);
        std::atomic<int> produced{0};
        std::atomic<int> consumed{0};
        const int numProducers = config.numThreads;
        const int itemsPerProducer = 1000;
        
        // Multiple producer threads
        std::vector<std::thread> producers;
        for (int t = 0; t < numProducers; ++t) {
            producers.emplace_back([&, t]() {
                for (int i = 0; i < itemsPerProducer; ++i) {
                    while (!queue.push(t * itemsPerProducer + i)) {
                        std::this_thread::yield();
                    }
                    produced++;
                }
            });
        }
        
        // Single consumer thread
        std::thread consumer([&]() {
            const int totalItems = numProducers * itemsPerProducer;
            int value;
            while (consumed < totalItems) {
                if (queue.pop(value)) {
                    consumed++;
                } else {
                    std::this_thread::yield();
                }
            }
        });
        
        for (auto& t : producers) t.join();
        consumer.join();
        
        assert(produced == numProducers * itemsPerProducer);
        assert(consumed == numProducers * itemsPerProducer);
        
    }, config.timeoutSec * 2);  // Longer timeout for stress test
}

/**
 * @brief Test lock-free queue for race conditions
 */
TestResult testLockFreeQueueRaceCondition(const TestConfig& config) {
    const std::string name = "LockFreeQueue: Race Condition Test";
    const std::string category = "Concurrency";
    
    return runWithTimeout(name, category, [&]() {
        LockFreeQueue<uint64_t> queue(16384);
        std::atomic<bool> stop{false};
        std::atomic<uint64_t> checksum{0};
        const int numProducers = 4;
        const int itemsPerProducer = 10000;
        
        // Producers push sequential values
        std::vector<std::thread> producers;
        for (int t = 0; t < numProducers; ++t) {
            producers.emplace_back([&, t]() {
                uint64_t base = t * itemsPerProducer;
                for (int i = 0; i < itemsPerProducer; ++i) {
                    uint64_t value = base + i;
                    while (!queue.push(value)) {
                        std::this_thread::yield();
                    }
                }
            });
        }
        
        // Consumer calculates checksum
        std::thread consumer([&]() {
            const uint64_t totalItems = numProducers * itemsPerProducer;
            uint64_t value;
            while (checksum < totalItems) {
                if (queue.pop(value)) {
                    checksum++;
                } else {
                    std::this_thread::yield();
                }
            }
        });
        
        for (auto& t : producers) t.join();
        consumer.join();
        
        assert(checksum == numProducers * itemsPerProducer);
        assert(queue.empty());
        
    }, config.timeoutSec * 2);
}

// ============================================================================
// PHASE 2: THREAD PINNING TESTS
// ============================================================================

/**
 * @brief Test thread pinning functionality
 */
TestResult testThreadPinningBasic(const TestConfig& config) {
    const std::string name = "ThreadPinning: Basic Pinning";
    const std::string category = "Concurrency";
    
    return runWithTimeout(name, category, [&]() {
        ThreadPinning pinning;
        
        if (!pinning.isAvailable()) {
            std::cout << "  [SKIP] Thread pinning not available on this system\n";
            return;
        }
        
        int numCores = pinning.getNumCores();
        assert(numCores > 0);
        
        // Test pinning to each available core
        for (int coreId = 0; coreId < std::min(numCores, 4); ++coreId) {
            bool success = pinning.pinThread(coreId);
            assert(success || !"Failed to pin thread");
            
            int currentCore = pinning.getCurrentCore();
            assert(currentCore == coreId || !"Thread not pinned to expected core");
        }
        
    }, config.timeoutSec);
}

/**
 * @brief Test thread pinning with multiple threads
 */
TestResult testThreadPinningMultiThread(const TestConfig& config) {
    const std::string name = "ThreadPinning: Multi-Thread";
    const std::string category = "Concurrency";
    
    return runWithTimeout(name, category, [&]() {
        ThreadPinning pinning;
        
        if (!pinning.isAvailable()) {
            std::cout << "  [SKIP] Thread pinning not available\n";
            return;
        }
        
        int numCores = pinning.getNumCores();
        std::atomic<int> successCount{0};
        
        std::vector<std::thread> threads;
        for (int i = 0; i < std::min(numCores, 4); ++i) {
            threads.emplace_back([&, i]() {
                bool success = pinning.pinThread(i);
                if (success) {
                    int core = pinning.getCurrentCore();
                    if (core == i) {
                        successCount++;
                    }
                }
            });
        }
        
        for (auto& t : threads) t.join();
        
        // At least some threads should be pinned successfully
        assert(successCount > 0 || !"No threads pinned successfully");
        
    }, config.timeoutSec);
}

// ============================================================================
// PHASE 3: SIMD VECTORIZATION TESTS
// ============================================================================

/**
 * @brief Test SIMD observation normalization correctness
 */
TestResult testSIMDNormalizationCorrectness(const TestConfig& config) {
    const std::string name = "SIMD: Normalization Correctness";
    const std::string category = "Correctness";
    
    return runWithTimeout(name, category, [&]() {
        const size_t batchSize = 128;
        const size_t obsDim = 256;
        const size_t totalSize = batchSize * obsDim;
        
        AlignedVector32<float> inputData(totalSize);
        AlignedVector32<float> scalarOutput(totalSize);
        AlignedVector32<float> avx2Output(totalSize);
        
        // Generate random input
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 10.0f);
        for (size_t i = 0; i < totalSize; ++i) {
            inputData[i] = dist(rng);
        }
        
        // Scalar reference implementation
        auto normalizeScalar = [&](const float* input, float* output) {
            constexpr float epsilon = 1e-5f;
            for (size_t batch = 0; batch < batchSize; ++batch) {
                const float* obs = input + batch * obsDim;
                float* normObs = output + batch * obsDim;
                
                float mean = 0.0f;
                for (size_t i = 0; i < obsDim; ++i) mean += obs[i];
                mean /= obsDim;
                
                float variance = 0.0f;
                for (size_t i = 0; i < obsDim; ++i) {
                    float diff = obs[i] - mean;
                    variance += diff * diff;
                }
                variance /= obsDim;
                
                float stddev = std::sqrt(variance + epsilon);
                float invStddev = 1.0f / stddev;
                
                for (size_t i = 0; i < obsDim; ++i) {
                    normObs[i] = (obs[i] - mean) * invStddev;
                }
            }
        };
        
        normalizeScalar(inputData.data(), scalarOutput.data());
        opt::NormalizeObservations_AVX2(inputData.data(), avx2Output.data(), batchSize, obsDim);
        
        // Verify correctness (within epsilon tolerance)
        constexpr float epsilon = 0.001f;
        double maxError = 0.0;
        for (size_t i = 0; i < totalSize; ++i) {
            double error = std::abs(scalarOutput[i] - avx2Output[i]);
            if (error > maxError) maxError = error;
        }
        
        assert(maxError < epsilon || !"AVX2 output differs from scalar by more than epsilon");
        
    }, config.timeoutSec);
}

/**
 * @brief Test SIMD scaling correctness
 */
TestResult testSIMDScalingCorrectness(const TestConfig& config) {
    const std::string name = "SIMD: Scaling Correctness";
    const std::string category = "Correctness";
    
    return runWithTimeout(name, category, [&]() {
        const size_t testSize = 4096;
        const float scale = 2.5f;
        const float offset = -1.0f;
        
        AlignedVector32<float> inputData(testSize);
        AlignedVector32<float> scalarOutput(testSize);
        AlignedVector32<float> avx2Output(testSize);
        
        // Generate random input
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (size_t i = 0; i < testSize; ++i) {
            inputData[i] = dist(rng);
        }
        
        // Scalar reference
        for (size_t i = 0; i < testSize; ++i) {
            scalarOutput[i] = inputData[i] * scale + offset;
        }
        
        // AVX2 implementation
        opt::ScaleObservations_AVX2(inputData.data(), avx2Output.data(), testSize, scale, offset);
        
        // Verify correctness
        constexpr float epsilon = 0.001f;
        double maxError = 0.0;
        for (size_t i = 0; i < testSize; ++i) {
            double error = std::abs(scalarOutput[i] - avx2Output[i]);
            if (error > maxError) maxError = error;
        }
        
        assert(maxError < epsilon || !"AVX2 scaling output differs from scalar");
        
    }, config.timeoutSec);
}

/**
 * @brief Test SIMD alignment requirements
 */
TestResult testSIMDAlignment(const TestConfig& config) {
    const std::string name = "SIMD: Memory Alignment";
    const std::string category = "Correctness";
    
    return runWithTimeout(name, category, [&]() {
        std::vector<size_t> testSizes = {64, 128, 256, 512, 1024, 2048};
        
        for (size_t size : testSizes) {
            AlignedVector32<float> buffer(size);
            uintptr_t addr = reinterpret_cast<uintptr_t>(buffer.data());
            bool aligned = (addr % 32 == 0);
            assert(aligned || !"Buffer not 32-byte aligned");
        }
        
    }, config.timeoutSec);
}

// ============================================================================
// PHASE 4: SOA MEMORY POOL TESTS
// ============================================================================

/**
 * @brief Test SoA environment batch initialization
 */
TestResult testSoAInitialization(const TestConfig& config) {
    const std::string name = "SoA: Initialization";
    const std::string category = "Correctness";
    
    return runWithTimeout(name, category, [&]() {
        opt::SoAEnvironmentBatch batch;
        bool success = batch.Initialize(config.numEnvs, 256, 56);
        
        assert(success || !"SoA initialization failed");
        assert(batch.numEnvs == config.numEnvs);
        assert(batch.obsDim == 256);
        assert(batch.actionDim == 56);
        
        // Verify alignment
        assert(batch.CheckAlignment() || !"SoA buffers not properly aligned");
        
    }, config.timeoutSec);
}

/**
 * @brief Test SoA reward calculation correctness
 */
TestResult testSoARewardCalculation(const TestConfig& config) {
    const std::string name = "SoA: Reward Calculation";
    const std::string category = "Correctness";
    
    return runWithTimeout(name, category, [&]() {
        opt::SoAEnvironmentBatch batch;
        batch.Initialize(config.numEnvs, 256, 56);
        
        // Set up test data
        for (size_t i = 0; i < batch.numEnvs; ++i) {
            batch.observations[i] = 10.0f;  // damage_dealt
            batch.observations[batch.numEnvs + i] = 5.0f;  // damage_taken
            batch.observations[2 * batch.numEnvs + i] = 1.0f;  // alive
            batch.observations[3 * batch.numEnvs + i] = 0.5f;  // air_time
            batch.observations[4 * batch.numEnvs + i] = 100.0f;  // energy
        }
        
        // Calculate rewards with SIMD
        batch.CalculateRewardsSIMD();
        
        // Verify rewards are calculated (non-zero)
        for (size_t i = 0; i < batch.numEnvs; ++i) {
            assert(batch.rewards[i * 2] != 0.0f || !"Reward calculation produced zero");
        }
        
    }, config.timeoutSec);
}

/**
 * @brief Test SoA batch operations with multiple threads
 */
TestResult testSoAMultiThreadAccess(const TestConfig& config) {
    const std::string name = "SoA: Multi-Thread Access";
    const std::string category = "Stress";
    
    return runWithTimeout(name, category, [&]() {
        opt::SoAEnvironmentBatch batch;
        batch.Initialize(config.numEnvs, 256, 56);
        
        std::atomic<int> successCount{0};
        
        // Multiple threads accessing different parts of SoA batch
        std::vector<std::thread> threads;
        for (int t = 0; t < config.numThreads; ++t) {
            threads.emplace_back([&, t]() {
                int startEnv = t * (config.numEnvs / config.numThreads);
                int endEnv = (t + 1) * (config.numEnvs / config.numThreads);
                
                for (int env = startEnv; env < endEnv; ++env) {
                    batch.SetObservationBatch(env, batch.observations.data() + env);
                    batch.GetObservationBatch(env, batch.observations.data() + env);
                }
                successCount += (endEnv - startEnv);
            });
        }
        
        for (auto& t : threads) t.join();
        
        assert(successCount == config.numEnvs || !"Not all environments processed");
        
    }, config.timeoutSec * 2);
}

// ============================================================================
// INTEGRATION TESTS (ALL PHASES COMBINED)
// ============================================================================

/**
 * @brief Test all optimizations working together (128 envs, 500 steps)
 */
TestResult testFullIntegration128Envs(const TestConfig& config) {
    const std::string name = "Integration: 128 Environments (500 steps)";
    const std::string category = "Stability";
    
    return runWithTimeout(name, category, [&]() {
        // Initialize environment
        VectorizedEnv vecEnv(128, 7200);
        vecEnv.Init("robots/combat_bot.json", false);  // Skip robot init for speed
        
        int actionDim = vecEnv.GetActionDim();
        AlignedVector32<float> actions(128 * 2 * actionDim);
        
        // Run 500 steps
        for (int step = 0; step < 500; ++step) {
            // Generate random actions
            std::mt19937 rng(step);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < actions.size(); ++i) {
                actions[i] = dist(rng);
            }
            
            // Step environment
            vecEnv.Step(actions);
            vecEnv.ResetDoneEnvs();
        }
        
    }, config.timeoutSec * 5);  // Longer timeout for full integration
}

/**
 * @brief Test deadlock detection with concurrent access
 */
TestResult testDeadlockDetection(const TestConfig& config) {
    const std::string name = "Deadlock: Concurrent Access Test";
    const std::string category = "Concurrency";
    
    return runWithTimeout(name, category, [&]() {
        std::mutex testMutex;
        std::atomic<int> counter{0};
        const int target = 10000;
        const int numThreads = config.numThreads;
        
        std::vector<std::thread> threads;
        for (int t = 0; t < numThreads; ++t) {
            threads.emplace_back([&]() {
                for (int i = 0; i < target / numThreads; ++i) {
                    std::lock_guard<std::mutex> lock(testMutex);
                    counter++;
                }
            });
        }
        
        for (auto& t : threads) t.join();
        
        assert(counter == target || !"Counter mismatch");
        
    }, config.timeoutSec);
}

/**
 * @brief Stress test with high environment count
 */
TestResult testStressHighEnvCount(const TestConfig& config) {
    const std::string name = "Stress: High Environment Count (256 envs)";
    const std::string category = "Stress";
    
    return runWithTimeout(name, category, [&]() {
        VectorizedEnv vecEnv(256, 7200);
        vecEnv.Init("robots/combat_bot.json", false);
        
        int actionDim = vecEnv.GetActionDim();
        AlignedVector32<float> actions(256 * 2 * actionDim);
        
        // Run 200 steps with 256 environments
        for (int step = 0; step < 200; ++step) {
            std::mt19937 rng(step);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < actions.size(); ++i) {
                actions[i] = dist(rng);
            }
            
            vecEnv.Step(actions);
            vecEnv.ResetDoneEnvs();
        }
        
    }, config.timeoutSec * 3);
}

// ============================================================================
// TEST RUNNER
// ============================================================================

class IntegrationTestRunner {
public:
    IntegrationTestRunner(const TestConfig& config) : mConfig(config) {}
    
    void addTest(std::function<TestResult()> testFunc) {
        mTests.push_back(testFunc);
    }
    
    void runAllTests() {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║         JOLTrl Phase 1-4 Integration Test Suite          ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Configuration:                                          ║\n";
        std::cout << "║  • Environments: " << std::setw(6) << mConfig.numEnvs << "                        ║\n";
        std::cout << "║  • Steps: " << std::setw(11) << mConfig.numSteps << "                        ║\n";
        std::cout << "║  • Threads: " << std::setw(9) << mConfig.numThreads << "                        ║\n";
        std::cout << "║  • Timeout: " << std::setw(9) << mConfig.timeoutSec << "s                       ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
        int passed = 0;
        int failed = 0;
        int skipped = 0;
        
        for (const auto& testFunc : mTests) {
            TestResult result = testFunc();
            printTestResult(result);
            
            if (result.passed) {
                passed++;
            } else {
                if (result.errorDetails.find("SKIP") != std::string::npos) {
                    skipped++;
                } else {
                    failed++;
                }
            }
        }
        
        // Print summary
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║                      TEST SUMMARY                        ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Total: " << (passed + failed + skipped) << " tests: " 
                  << passed << " passed, " << failed << " failed, " << skipped << " skipped\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n";
        
        if (failed > 0) {
            std::cout << "\n⚠️  INTEGRATION TESTS FAILED - Review errors above\n";
        } else {
            std::cout << "\n✓ ALL INTEGRATION TESTS PASSED\n";
        }
    }
    
private:
    TestConfig mConfig;
    std::vector<std::function<TestResult()>> mTests;
};

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    TestConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--envs" && i + 1 < argc) {
            config.numEnvs = std::stoi(argv[++i]);
        }
        else if (arg == "--steps" && i + 1 < argc) {
            config.numSteps = std::stoi(argv[++i]);
        }
        else if (arg == "--threads" && i + 1 < argc) {
            config.numThreads = std::stoi(argv[++i]);
        }
        else if (arg == "--timeout" && i + 1 < argc) {
            config.timeoutSec = std::stoi(argv[++i]);
        }
        else if (arg == "--test" && i + 1 < argc) {
            config.testFilter = argv[++i];
        }
        else if (arg == "--verbose" || arg == "-v") {
            config.verbose = true;
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --envs N       Number of environments (default: 128)\n";
            std::cout << "  --steps N      Number of steps (default: 500)\n";
            std::cout << "  --threads N    Number of threads (default: 8)\n";
            std::cout << "  --timeout N    Timeout in seconds (default: 30)\n";
            std::cout << "  --test NAME    Run specific test\n";
            std::cout << "  --verbose, -v  Verbose output\n";
            std::cout << "  --help, -h     Show this help\n";
            return 0;
        }
    }
    
    IntegrationTestRunner runner(config);
    
    // Phase 1: Lock-Free Queue Tests
    runner.addTest([&]() { return testLockFreeQueueSingleProducerConsumer(config); });
    runner.addTest([&]() { return testLockFreeQueueMultiProducer(config); });
    runner.addTest([&]() { return testLockFreeQueueRaceCondition(config); });
    
    // Phase 2: Thread Pinning Tests
    runner.addTest([&]() { return testThreadPinningBasic(config); });
    runner.addTest([&]() { return testThreadPinningMultiThread(config); });
    
    // Phase 3: SIMD Vectorization Tests
    runner.addTest([&]() { return testSIMDNormalizationCorrectness(config); });
    runner.addTest([&]() { return testSIMDScalingCorrectness(config); });
    runner.addTest([&]() { return testSIMDAlignment(config); });
    
    // Phase 4: SoA Memory Pool Tests
    runner.addTest([&]() { return testSoAInitialization(config); });
    runner.addTest([&]() { return testSoARewardCalculation(config); });
    runner.addTest([&]() { return testSoAMultiThreadAccess(config); });
    
    // Integration Tests
    runner.addTest([&]() { return testDeadlockDetection(config); });
    runner.addTest([&]() { return testFullIntegration128Envs(config); });
    runner.addTest([&]() { return testStressHighEnvCount(config); });
    
    runner.runAllTests();
    
    return 0;
}
