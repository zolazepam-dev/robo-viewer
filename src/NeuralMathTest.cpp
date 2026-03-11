// Test-Driven Refactoring Example: Fix AVX2 Tanh Implementation
// 
// This test file demonstrates the test-driven approach to fixing
// the _mm256_tanh_ps compilation error in OptimizedBatchOps.h
//
// Usage:
//   bazel test //:NeuralMathTest
//
// Refactoring Goal:
//   Replace _mm256_tanh_ps (AVX-512) with AVX2-compatible implementation
//
// Success Criteria:
//   1. Code compiles with AVX2 flags (-mavx2)
//   2. Tanh function produces correct results (within epsilon)
//   3. Performance is maintained or improved
//   4. All existing tests continue to pass

#include <gtest/gtest.h>
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>

// Include the actual implementation
#include "NeuralMath.h"
#include "OptimizedBatchOps.h"

// Test constants
constexpr size_t TEST_SIZE = 1024;
constexpr float EPSILON = 1e-5f;

/**
 * Test Suite: AVX2 Tanh Implementation
 * 
 * These tests verify the correctness and performance of the
 * AVX2-compatible tanh implementation.
 */
class AVX2TanhTest : public ::testing::Test {
protected:
    std::vector<float> input_data;
    std::vector<float> output_data;
    std::vector<float> expected_data;
    
    void SetUp() override {
        // Allocate aligned memory for AVX2 operations
        input_data.resize(TEST_SIZE);
        output_data.resize(TEST_SIZE);
        expected_data.resize(TEST_SIZE);
        
        // Initialize test data with various ranges
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            // Test values across different ranges
            float t = static_cast<float>(i) / TEST_SIZE * 10.0f - 5.0f;
            input_data[i] = t;
            expected_data[i] = std::tanh(t);
        }
    }
    
    void TearDown() override {
        // Cleanup
    }
    
    // Helper to verify results
    void VerifyResults(const std::vector<float>& actual, float tolerance = EPSILON) {
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            float diff = std::abs(actual[i] - expected_data[i]);
            EXPECT_LT(diff, tolerance) 
                << "Mismatch at index " << i 
                << ": expected " << expected_data[i] 
                << ", got " << actual[i];
        }
    }
};

/**
 * Test: Baseline - Current Behavior
 * 
 * This test documents the current behavior before refactoring.
 * It should pass even with the broken implementation (if it compiles).
 */
TEST_F(AVX2TanhTest, Baseline_CurrentImplementation) {
    std::cout << "\n=== Baseline Test: Current Implementation ===" << std::endl;
    
    // Copy input data
    std::copy(input_data.begin(), input_data.end(), output_data.begin());
    
    // Call current implementation (may not compile)
    // Note: This will fail to compile until we fix _mm256_tanh_ps
    // BatchedTanh_AVX2(output_data.data(), TEST_SIZE);
    
    // For now, just verify we can compute expected values
    bool can_compute = true;
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        if (std::isnan(expected_data[i])) {
            can_compute = false;
            break;
        }
    }
    
    EXPECT_TRUE(can_compute) << "Cannot compute expected tanh values";
    std::cout << "Baseline: Can compute " << TEST_SIZE << " tanh values" << std::endl;
}

/**
 * Test: Correctness - AVX2 Implementation
 * 
 * This test verifies the AVX2 implementation produces correct results.
 * It will fail initially, then pass after refactoring.
 */
TEST_F(AVX2TanhTest, Correctness_AVX2Implementation) {
    std::cout << "\n=== Correctness Test: AVX2 Implementation ===" << std::endl;
    
    // Copy input data
    std::copy(input_data.begin(), input_data.end(), output_data.begin());
    
    // Call AVX2 implementation
    BatchedTanh_AVX2(output_data.data(), TEST_SIZE);
    
    // Verify results
    VerifyResults(output_data);
    
    std::cout << "✓ AVX2 implementation produces correct results" << std::endl;
}

/**
 * Test: Edge Cases - Extreme Values
 * 
 * Test tanh behavior at extreme values where numerical stability matters.
 */
TEST_F(AVX2TanhTest, EdgeCases_ExtremeValues) {
    std::cout << "\n=== Edge Cases: Extreme Values ===" << std::endl;
    
    std::vector<float> extreme_inputs = {
        -100.0f, -50.0f, -20.0f, -10.0f,
        -1.0f, -0.5f, 0.0f, 0.5f, 1.0f,
        10.0f, 20.0f, 50.0f, 100.0f
    };
    
    std::vector<float> extreme_outputs(extreme_inputs.size());
    
    // Compute tanh for extreme values
    std::copy(extreme_inputs.begin(), extreme_inputs.end(), extreme_outputs.begin());
    BatchedTanh_AVX2(extreme_outputs.data(), extreme_inputs.size());
    
    // Verify extreme values approach ±1
    for (size_t i = 0; i < extreme_inputs.size(); ++i) {
        float expected = std::tanh(extreme_inputs[i]);
        float actual = extreme_outputs[i];
        
        // For large |x|, tanh(x) should approach ±1
        if (std::abs(extreme_inputs[i]) > 10.0f) {
            EXPECT_NEAR(std::abs(actual), 1.0f, 0.01f)
                << "Extreme value test failed for input " << extreme_inputs[i];
        } else {
            EXPECT_NEAR(actual, expected, EPSILON)
                << "Mismatch for input " << extreme_inputs[i];
        }
    }
    
    std::cout << "✓ Edge cases handled correctly" << std::endl;
}

/**
 * Test: Performance - AVX2 vs Scalar
 * 
 * Verify that AVX2 implementation is faster than scalar version.
 */
TEST_F(AVX2TanhTest, Performance_AVX2VsScalar) {
    std::cout << "\n=== Performance: AVX2 vs Scalar ===" << std::endl;
    
    const int iterations = 1000;
    
    // Time AVX2 implementation
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::copy(input_data.begin(), input_data.end(), output_data.begin());
        BatchedTanh_AVX2(output_data.data(), TEST_SIZE);
    }
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    auto duration_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(
        end_avx2 - start_avx2
    ).count();
    
    // Time scalar implementation
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        for (size_t j = 0; j < TEST_SIZE; ++j) {
            output_data[j] = std::tanh(input_data[j]);
        }
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    auto duration_scalar = std::chrono::duration_cast<std::chrono::microseconds>(
        end_scalar - start_scalar
    ).count();
    
    // Report results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "AVX2 time:    " << duration_avx2 << " μs (" 
              << (100.0 * iterations / duration_avx2) << " M steps/s)" << std::endl;
    std::cout << "Scalar time:  " << duration_scalar << " μs (" 
              << (100.0 * iterations / duration_scalar) << " M steps/s)" << std::endl;
    std::cout << "Speedup:      " << (static_cast<float>(duration_scalar) / duration_avx2) 
              << "x" << std::endl;
    
    // AVX2 should be at least as fast as scalar
    EXPECT_LT(duration_avx2, duration_scalar * 1.5) 
        << "AVX2 implementation is too slow compared to scalar";
    
    std::cout << "✓ Performance test completed" << std::endl;
}

/**
 * Test: Alignment - 32-byte Memory Alignment
 * 
 * Verify that the implementation correctly handles aligned memory.
 */
TEST_F(AVX2TanhTest, Alignment_32ByteAlignment) {
    std::cout << "\n=== Alignment: 32-byte Memory ===" << std::endl;
    
    // Allocate aligned memory
    void* aligned_ptr = aligned_alloc(32, TEST_SIZE * sizeof(float));
    ASSERT_NE(aligned_ptr, nullptr) << "Failed to allocate aligned memory";
    
    float* aligned_data = static_cast<float*>(aligned_ptr);
    
    // Copy test data
    std::copy(input_data.begin(), input_data.end(), aligned_data);
    
    // Verify alignment
    bool is_aligned = (reinterpret_cast<uintptr_t>(aligned_data) % 32) == 0;
    EXPECT_TRUE(is_aligned) << "Memory is not 32-byte aligned";
    
    // Call AVX2 implementation (should use aligned loads)
    BatchedTanh_AVX2(aligned_data, TEST_SIZE);
    
    // Verify results
    std::vector<float> result(aligned_data, aligned_data + TEST_SIZE);
    VerifyResults(result);
    
    // Cleanup
    free(aligned_ptr);
    
    std::cout << "✓ 32-byte alignment test passed" << std::endl;
}

/**
 * Test: SIMD Width - Non-Multiple-of-8 Sizes
 * 
 * Test that the implementation handles sizes that aren't multiples of 8.
 */
TEST_F(AVX2TanhTest, SIMDWidth_NonMultipleOf8) {
    std::cout << "\n=== SIMD Width: Non-Multiple-of-8 Sizes ===" << std::endl;
    
    std::vector<size_t> test_sizes = {1, 7, 9, 15, 17, 127, 129, 1000};
    
    for (size_t size : test_sizes) {
        std::vector<float> test_input(size);
        std::vector<float> test_output(size);
        
        // Initialize
        for (size_t i = 0; i < size; ++i) {
            test_input[i] = static_cast<float>(i) / size * 10.0f - 5.0f;
        }
        
        // Compute
        std::copy(test_input.begin(), test_input.end(), test_output.begin());
        BatchedTanh_AVX2(test_output.data(), size);
        
        // Verify
        for (size_t i = 0; i < size; ++i) {
            float expected = std::tanh(test_input[i]);
            float actual = test_output[i];
            EXPECT_NEAR(actual, expected, EPSILON)
                << "Mismatch at index " << i << " for size " << size;
        }
        
        std::cout << "  Size " << size << ": ✓" << std::endl;
    }
    
    std::cout << "✓ Non-multiple-of-8 sizes handled correctly" << std::endl;
}

/**
 * Test: Regression - No Performance Degradation
 * 
 * Ensure the refactoring doesn't degrade performance.
 */
TEST_F(AVX2TanhTest, Regression_NoPerformanceDegradation) {
    std::cout << "\n=== Regression: Performance Check ===" << std::endl;
    
    // Baseline performance threshold (adjust based on your hardware)
    const float MIN_STEPS_PER_SECOND = 50.0f; // Million steps per second
    
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        std::copy(input_data.begin(), input_data.end(), output_data.begin());
        BatchedTanh_AVX2(output_data.data(), TEST_SIZE);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start
    ).count();
    
    float steps_per_second = (100.0f * iterations) / duration;
    
    std::cout << "Performance: " << steps_per_second << " M steps/s" << std::endl;
    
    EXPECT_GT(steps_per_second, MIN_STEPS_PER_SECOND)
        << "Performance below threshold: " << steps_per_second 
        << " < " << MIN_STEPS_PER_SECOND;
    
    std::cout << "✓ No performance regression detected" << std::endl;
}

// Main entry point for test runner
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     AVX2 Tanh Implementation - Test Suite                ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    
    return RUN_ALL_TESTS();
}
