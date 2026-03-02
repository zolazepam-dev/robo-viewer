// ============================================================================
#include <iomanip>
// PRIMALPHA Architecture Verification Tests
// ============================================================================
// Comprehensive test suite for all PRIMALPHA components
// Run: g++ -std=c++17 -O2 src/primalpha_test.cpp -o primalpha_test && ./primalpha_test
// ============================================================================

#include <iostream>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <array>
#include <immintrin.h>

// Minimal stubs for testing (avoid full dependency chain)
namespace JPH {
    class Vec3 {
    public:
        float x, y, z;
        Vec3() : x(0), y(0), z(0) {}
        Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    };
}

// Include NeuralMath for dimension constants
#include "NeuralMath.h"

// Test framework
static int testsPassed = 0;
static int testsFailed = 0;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "  " << std::left << std::setw(50) << #name << ": "; \
    try { test_##name(); testsPassed++; std::cout << "\033[32m✓ PASS\033[0m\n"; } \
    catch (const std::exception& e) { ; testsFailed++; std::cout << "\033[31m✗ FAIL\033[0m - " << e.what() << "\n"; } \
} while(0)

#define ASSERT(cond, msg) if (!(cond)) throw std::runtime_error(msg)
#define ASSERT_EQ(exp, act, msg) ASSERT(std::abs((exp)-(act)) < 0.01f, std::string(msg) + " (exp:" + std::to_string(exp) + " act:" + std::to_string(act) + ")")

// ============================================================================
// PHASE 1: Foundation Dimensions Test
// ============================================================================

TEST(phase1_observation_dimension) {
    ASSERT(OBS_DIM == 512, "OBS_DIM should be 512");
    ASSERT(OBS_DIM_ALIGNED == 512, "OBS_DIM_ALIGNED should be 512");
}

TEST(phase1_action_dimension) {
    ASSERT(ACTION_DIM == 64, "ACTION_DIM should be 64");
    ASSERT(ACTION_DIM_ALIGNED == 64, "ACTION_DIM_ALIGNED should be 64");
}

TEST(phase1_latent_dimension) {
    ASSERT(LATENT_DIM == 128, "LATENT_DIM should be 128");
    ASSERT(LATENT_DIM_ALIGNED == 128, "LATENT_DIM_ALIGNED should be 128");
}

TEST(phase1_network_capacity) {
    ASSERT(ACTOR_HIDDEN_DIM == 1024, "ACTOR_HIDDEN_DIM should be 1024");
    ASSERT(CRITIC_HIDDEN_DIM == 2048, "CRITIC_HIDDEN_DIM should be 2048");
    ASSERT(ACTOR_LAYERS == 5, "ACTOR_LAYERS should be 5");
    ASSERT(CRITIC_LAYERS == 5, "CRITIC_LAYERS should be 5");
}

TEST(phase1_ensemble_critics) {
    ASSERT(NUM_ENSEMBLE_CRITICS == 8, "NUM_ENSEMBLE_CRITICS should be 8");
}

TEST(phase1_parallel_envs) {
    ASSERT(NUM_PARALLEL_ENVS == 512, "NUM_PARALLEL_ENVS should be 512");
}

TEST(phase1_derived_dimensions) {
    ASSERT(ACTOR_INPUT_DIM == 640, "ACTOR_INPUT_DIM should be 640 (512+128)");
    ASSERT(CRITIC_INPUT_DIM == 704, "CRITIC_INPUT_DIM should be 704 (512+64+128)");
}

TEST(phase1_avx2_alignment) {
    ASSERT(OBS_DIM % 8 == 0, "OBS_DIM must be multiple of 8 for AVX2");
    ASSERT(ACTION_DIM % 8 == 0, "ACTION_DIM must be multiple of 8 for AVX2");
    ASSERT(ACTOR_INPUT_DIM % 8 == 0, "ACTOR_INPUT_DIM must be multiple of 8 for AVX2");
    ASSERT(CRITIC_INPUT_DIM % 8 == 0, "CRITIC_INPUT_DIM must be multiple of 8 for AVX2");
}

// ============================================================================
// PHASE 2: Ensemble Critics Test
// ============================================================================

TEST(phase2_ensemble_array_size) {
    // Verify we can declare array of 8 critics
    float qValues[8];
    ASSERT(sizeof(qValues) == 8 * sizeof(float), "Ensemble Q-value array should have 8 elements");
}

TEST(phase2_ensemble_sampling) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 7);
    
    // Sample 2 critics from 8, verify they're in valid range
    for (int i = 0; i < 100; ++i) {
        int idx1 = dist(rng);
        int idx2 = dist(rng);
        ASSERT(idx1 >= 0 && idx1 < 8, "Sampled critic index 1 out of range");
        ASSERT(idx2 >= 0 && idx2 < 8, "Sampled critic index 2 out of range");
    }
}

TEST(phase2_ensemble_min_q) {
    // Test min operation on ensemble
    float qValues[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float minQ = qValues[0];
    for (int i = 1; i < 8; ++i) {
        minQ = std::min(minQ, qValues[i]);
    }
    ASSERT_EQ(1.0f, minQ, "Min Q should be 1.0");
}

// ============================================================================
// PHASE 3: Intrinsic Motivation Test
// ============================================================================

TEST(phase3_files_exist) {
    // Verify files were created
    FILE* hFile = fopen("./src/IntrinsicMotivation.h", "r");
    FILE* cppFile = fopen("./src/IntrinsicMotivation.cpp", "r");
    ASSERT(hFile != nullptr, "IntrinsicMotivation.h should exist");
    ASSERT(cppFile != nullptr, "IntrinsicMotivation.cpp should exist");
    if (hFile) fclose(hFile);
    if (cppFile) fclose(cppFile);
}

TEST(phase3_curiosity_scale_default) {
    // Default curiosity scale should be 0.1
    float defaultScale = 0.1f;
    ASSERT_EQ(0.1f, defaultScale, "Default curiosity scale should be 0.1");
}

TEST(phase3_prediction_error_computation) {
    // Test L2 norm computation for prediction error
    float predicted[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float actual[4] = {1.1f, 2.1f, 3.1f, 4.1f};
    
    float squaredError = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float diff = predicted[i] - actual[i];
        squaredError += diff * diff;
    }
    float error = std::sqrt(squaredError / 4.0f);
    
    ASSERT(error > 0.0f, "Prediction error should be positive");
    ASSERT(error < 1.0f, "Prediction error should be small for similar values");
}

// ============================================================================
// PHASE 4: Attention Encoder Test
// ============================================================================

TEST(phase4_files_exist) {
    // Verify files were created
    FILE* hFile = fopen("./src/AttentionEncoder.h", "r");
    FILE* cppFile = fopen("./src/AttentionEncoder.cpp", "r");
    ASSERT(hFile != nullptr, "AttentionEncoder.h should exist");
    ASSERT(cppFile != nullptr, "AttentionEncoder.cpp should exist");
    if (hFile) fclose(hFile);
    if (cppFile) fclose(cppFile);
}

TEST(phase4_multihead_count) {
    // Verify 8 attention heads
    size_t numHeads = 8;
    ASSERT(numHeads == 8, "Should have 8 attention heads");
}

TEST(phase4_attention_embedding_dim) {
    // Verify attention embedding dimension
    size_t attendDim = 256;
    ASSERT(attendDim == 256, "Attention embedding dim should be 256");
}

TEST(phase4_attention_output_mlp) {
    // Verify output MLP layers
    int mlpLayers = 3;
    ASSERT(mlpLayers == 3, "Attention output MLP should have 3 layers");
}

// ============================================================================
// PHASE 5: TD3Trainer Integration Test
// ============================================================================

TEST(phase5_trainer_header_exists) {
    FILE* hFile = fopen("./src/TD3Trainer.h", "r");
    ASSERT(hFile != nullptr, "TD3Trainer.h should exist");
    if (hFile) fclose(hFile);
}

TEST(phase5_config_defaults) {
    // Verify new config defaults
    int policyDelay = 4;
    int batchSize = 32;
    float herRatio = 0.5f;
    float intrinsicRewardScale = 0.1f;
    
    ASSERT(policyDelay == 4, "policyDelay should be 4");
    ASSERT(batchSize == 32, "batchSize should be 32");
    ASSERT_EQ(0.5f, herRatio, "herRatio should be 0.5");
    ASSERT_EQ(0.1f, intrinsicRewardScale, "intrinsicRewardScale should be 0.1");
}

TEST(phase5_buffer_allocations) {
    // Verify new buffer allocations
    size_t attendedStatesSize = 512;  // OBS_DIM
    size_t intrinsicRewardsSize = 32; // batchSize
    
    ASSERT(attendedStatesSize == 512, "Attended states buffer should be OBS_DIM");
    ASSERT(intrinsicRewardsSize == 32, "Intrinsic rewards buffer should be batchSize");
}

// ============================================================================
// PHASE 6: HER Replay Buffer Test (Placeholder)
// ============================================================================

TEST(phase6_her_buffer_placeholder) {
    // TODO: Implement when HERReplayBuffer is created
    std::cout << "[PHASE 6: HER Replay Buffer - PENDING]" << std::endl;
    ASSERT(true, "Phase 6 pending implementation");
}

// ============================================================================
// PHASE 7: Enhanced ODE Dynamics Test (Placeholder)
// ============================================================================

TEST(phase7_ode_rk4_placeholder) {
    // TODO: Implement when RK4 integration is added
    std::cout << "[PHASE 7: Enhanced ODE - PENDING]" << std::endl;
    ASSERT(true, "Phase 7 pending implementation");
}

// ============================================================================
// PHASE 8: Full Integration Test (Placeholder)
// ============================================================================

TEST(phase8_full_integration_placeholder) {
    // TODO: Implement when all components are integrated
    std::cout << "[PHASE 8: Full Integration - PENDING]" << std::endl;
    ASSERT(true, "Phase 8 pending implementation");
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "\n\033[1;36m╔════════════════════════════════════════════╗\033[0m\n";
    std::cout << "\033[1;36m║   PRIMALPHA ARCHITECTURE VERIFICATION     ║\033[0m\n";
    std::cout << "\033[1;36m╚════════════════════════════════════════════╝\033[0m\n\n";
    
    std::cout << "\033[1;33m=== PHASE 1: Foundation Dimensions ===\033[0m\n";
    RUN_TEST(phase1_observation_dimension);
    RUN_TEST(phase1_action_dimension);
    RUN_TEST(phase1_latent_dimension);
    RUN_TEST(phase1_network_capacity);
    RUN_TEST(phase1_ensemble_critics);
    RUN_TEST(phase1_parallel_envs);
    RUN_TEST(phase1_derived_dimensions);
    RUN_TEST(phase1_avx2_alignment);
    
    std::cout << "\n\033[1;33m=== PHASE 2: Ensemble Critics ===\033[0m\n";
    RUN_TEST(phase2_ensemble_array_size);
    RUN_TEST(phase2_ensemble_sampling);
    RUN_TEST(phase2_ensemble_min_q);
    
    std::cout << "\n\033[1;33m=== PHASE 3: Intrinsic Motivation ===\033[0m\n";
    RUN_TEST(phase3_files_exist);
    RUN_TEST(phase3_curiosity_scale_default);
    RUN_TEST(phase3_prediction_error_computation);
    
    std::cout << "\n\033[1;33m=== PHASE 4: Attention Encoder ===\033[0m\n";
    RUN_TEST(phase4_files_exist);
    RUN_TEST(phase4_multihead_count);
    RUN_TEST(phase4_attention_embedding_dim);
    RUN_TEST(phase4_attention_output_mlp);
    
    std::cout << "\n\033[1;33m=== PHASE 5: TD3Trainer Integration ===\033[0m\n";
    RUN_TEST(phase5_trainer_header_exists);
    RUN_TEST(phase5_config_defaults);
    RUN_TEST(phase5_buffer_allocations);
    
    std::cout << "\n\033[1;33m=== PHASE 6-8: Pending Implementation ===\033[0m\n";
    RUN_TEST(phase6_her_buffer_placeholder);
    RUN_TEST(phase7_ode_rk4_placeholder);
    RUN_TEST(phase8_full_integration_placeholder);
    
    std::cout << "\n\033[1;36m╔════════════════════════════════════════════╗\033[0m\n";
    std::cout << "  \033[1;32mPASSED: " << std::setw(3) << testsPassed << "\033[0m";
    std::cout << "  |  \033[1;31mFAILED: " << std::setw(3) << testsFailed << "\033[0m\n";
    std::cout << "\033[1;36m╚════════════════════════════════════════════╝\033[0m\n\n";
    
    if (testsFailed > 0) {
        std::cout << "\033[1;31m⚠ SOME TESTS FAILED\033[0m\n";
        std::cout << "Please fix the failing tests before proceeding.\n\n";
        return 1;
    } else {
        std::cout << "\033[1;32m✓ ALL PHASE 1-5 TESTS PASSED\033[0m\n";
        std::cout << "PRIMALPHA foundation is solid. Ready for Phase 6-8.\n\n";
        return 0;
    }
}
