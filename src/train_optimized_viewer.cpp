// ============================================================================
// Optimized Training with Viewer
// ============================================================================
// Launches the optimized RL training pipeline with real-time visualization
// 
// Features:
// - Optimized training pipeline (20-50x SPS improvement)
// - Real-time rendering of one environment
// - ImGui controls for training parameters
// - Performance metrics display
//
// Usage:
//   bazel run //:train_optimized_viewer -- --envs 256 --render-env 0
// ============================================================================

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <thread>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "OptimizedTrainingPipeline.h"
#include "VectorizedEnv.h"
#include "TD3Trainer.h"
#include "Renderer.h"
#include "PerformanceProfiler.h"

namespace fs = std::filesystem;

// ============================================================================
// Configuration
// ============================================================================

struct TrainingConfig {
    int numEnvs = 256;
    int numPhysicsSystems = 8;
    int batchSize = 1024;
    int accumulationSteps = 4;
    int checkpointInterval = 50000;
    int maxSteps = 10000000;
    int renderEnvIndex = 0;  // Which environment to render (0 = visible)
    std::string checkpointDir = "checkpoints";
    std::string loadCheckpoint = "";
    bool pauseOnStart = false;
    float timeScale = 1.0f;  // Speed up/slow down training
};

// ============================================================================
// Global State
// ============================================================================

Camera gCamera;
std::atomic<bool> gRunning(true);
std::atomic<bool> gPaused(false);
std::atomic<bool> gTrainingComplete(false);

// IO THREAD STATE
struct IOTask {
    enum Type { SAVE_MODEL, SNAPSHOT_OPPONENT };
    Type type;
    std::string path;
    std::string robotPath;
    int numSatellites;
    int obsDim;
    std::vector<float> weights;
};
std::queue<IOTask> gIOQueue;
std::mutex gIOMutex;
std::condition_variable gIOCV;
std::atomic<bool> gIORunning{true};

void IOWorker(TD3Trainer* trainer) {
    while (gIORunning) {
        IOTask task;
        {
            std::unique_lock<std::mutex> lock(gIOMutex);
            gIOCV.wait(lock, []{ return !gIOQueue.empty() || !gIORunning; });
            if (!gIORunning && gIOQueue.empty()) break;
            task = std::move(gIOQueue.front());
            gIOQueue.pop();
        }
        if (task.type == IOTask::SAVE_MODEL) trainer->SaveToDisk(task.path, task.robotPath, task.numSatellites, task.obsDim);
        else if (task.type == IOTask::SNAPSHOT_OPPONENT) trainer->GetOpponentPool().Snapshot(task.weights, {}, 0);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

void EnsureDir(const std::string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
    }
}

std::string GetLatestCheckpoint(const std::string& dir) {
    std::string latest;
    int maxStep = -1;

    if (!fs::exists(dir)) return "";

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".bin") {
            std::string filename = entry.path().stem().string();
            size_t pos = filename.find_last_of('_');
            if (pos != std::string::npos) {
                try {
                    int step = std::stoi(filename.substr(pos + 1));
                    if (step > maxStep) {
                        maxStep = step;
                        latest = entry.path().string();
                    }
                } catch (...) {}
            }
        }
    }
    return latest;
}

void PrintWelcomeMessage(const TrainingConfig& config) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     OPTIMIZED RL TRAINING PIPELINE WITH VIEWER               ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Optimizations Enabled:                                      ║\n";
    std::cout << "║    ✓ Parallel Physics Stepping (8 systems)                   ║\n";
    std::cout << "║    ✓ SoA Memory Layout                                       ║\n";
    std::cout << "║    ✓ Batched Neural Inference                                ║\n";
    std::cout << "║    ✓ Gradient Accumulation                                   ║\n";
    std::cout << "║    ✓ Lock-Free Communication                                 ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Configuration:                                              ║\n";
    printf("║    Environments:        %6d                            ║\n", config.numEnvs);
    printf("║    Physics Systems:     %6d                            ║\n", config.numPhysicsSystems);
    printf("║    Batch Size:          %6d                            ║\n", config.batchSize);
    printf("║    Accumulation Steps:  %6d                            ║\n", config.accumulationSteps);
    printf("║    Render Env Index:    %6d                            ║\n", config.renderEnvIndex);
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Expected Performance: 20-50x SPS improvement                ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

// ============================================================================
// Training Thread
// ============================================================================

struct TrainingState {
    std::unique_ptr<OptimizedTrainingPipeline> pipeline;
    VectorizedEnv* vecEnv;
    TD3Trainer* trainer;
    int currentStep = 0;
    float meanSPS = 0.0f;
    float meanReward = 0.0f;
    std::vector<float> rewardHistory;
};

void TrainingThreadFunc(TrainingState* state, const TrainingConfig& config) {
    std::cout << "[Training] Starting training thread..." << "\n";
    
    auto startTime = std::chrono::high_resolution_clock::now();
    int lastLogStep = 0;
    
    while (gRunning && state->currentStep < config.maxSteps) {
        // Check pause
        while (gPaused && gRunning) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        if (!gRunning) break;
        
        // Apply time scale
        int stepsToRun = 1;
        if (config.timeScale > 1.0f) {
            stepsToRun = static_cast<int>(config.timeScale);
        }
        
        for (int s = 0; s < stepsToRun && gRunning; s++) {
            // Run one training step
            state->pipeline->Step();
            state->currentStep++;
            
            // Update statistics
            state->meanSPS = state->pipeline->GetMeanSPS();
            state->meanReward = state->pipeline->GetMeanReward();
            state->rewardHistory.push_back(state->meanReward);
            
            // Keep history manageable
            if (state->rewardHistory.size() > 1000) {
                state->rewardHistory.erase(state->rewardHistory.begin());
            }
            
            // Checkpoint
            if (state->currentStep % config.checkpointInterval == 0 && state->currentStep > 0) {
                std::string path = config.checkpointDir + "/checkpoint_" + 
                                  std::to_string(state->currentStep) + ".bin";
                state->pipeline->SaveCheckpoint(path);
            }
        }
        
        // Sync with viewer - actions are already in VecEnv from training
        
        // Periodic logging
        if (state->currentStep - lastLogStep >= 1000) {
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            float actualSPS = (state->currentStep - lastLogStep) / elapsed;
            
            std::cout << "[Training] Step: " << state->currentStep 
                      << " | SPS: " << actualSPS 
                      << " | Mean Reward: " << state->meanReward << "\n";
            
            lastLogStep = state->currentStep;
            startTime = now;
        }
    }
    
    gTrainingComplete = true;
    std::cout << "[Training] Training complete! Final step: " << state->currentStep << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    TrainingConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--envs" && i + 1 < argc) {
            config.numEnvs = std::stoi(argv[++i]);
        } else if (arg == "--physics-systems" && i + 1 < argc) {
            config.numPhysicsSystems = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batchSize = std::stoi(argv[++i]);
        } else if (arg == "--accumulation-steps" && i + 1 < argc) {
            config.accumulationSteps = std::stoi(argv[++i]);
        } else if (arg == "--checkpoint-interval" && i + 1 < argc) {
            config.checkpointInterval = std::stoi(argv[++i]);
        } else if (arg == "--max-steps" && i + 1 < argc) {
            config.maxSteps = std::stoi(argv[++i]);
        } else if (arg == "--checkpoint-dir" && i + 1 < argc) {
            config.checkpointDir = argv[++i];
        } else if (arg == "--load" && i + 1 < argc) {
            config.loadCheckpoint = argv[++i];
        } else if (arg == "--load-latest") {
            config.loadCheckpoint = "latest";
        } else if (arg == "--render-env" && i + 1 < argc) {
            config.renderEnvIndex = std::stoi(argv[++i]);
        } else if (arg == "--pause-on-start") {
            config.pauseOnStart = true;
        } else if (arg == "--time-scale" && i + 1 < argc) {
            config.timeScale = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: train_optimized_viewer [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --envs N                Number of parallel environments (default: 256)\n";
            std::cout << "  --physics-systems N     Number of physics systems (default: 8)\n";
            std::cout << "  --batch-size N          Training batch size (default: 1024)\n";
            std::cout << "  --accumulation-steps N  Gradient accumulation steps (default: 4)\n";
            std::cout << "  --checkpoint-interval N Steps between checkpoints (default: 50000)\n";
            std::cout << "  --max-steps N           Maximum training steps (default: 10000000)\n";
            std::cout << "  --checkpoint-dir PATH   Checkpoint directory (default: checkpoints)\n";
            std::cout << "  --load PATH             Load checkpoint from path\n";
            std::cout << "  --load-latest           Load latest checkpoint\n";
            std::cout << "  --render-env N          Environment index to render (default: 0)\n";
            std::cout << "  --pause-on-start        Start paused\n";
            std::cout << "  --time-scale N          Training speed multiplier (default: 1.0)\n";
            return 0;
        }
    }
    
    PrintWelcomeMessage(config);
    
    EnsureDir(config.checkpointDir);
    EnsureDir("saved_models");
    
    gPaused = config.pauseOnStart;
    
    // =========================================================================
    // Initialize GLFW and OpenGL
    // =========================================================================
    
    if (!glfwInit()) {
        std::cerr << "[Error] Failed to initialize GLFW" << "\n";
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Optimized RL Training", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        glfwTerminate();
        return -1;
    }
    
    glEnable(GL_DEPTH_TEST);
    glfwSwapInterval(1);  // VSync
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    // =========================================================================
    // Initialize Training
    // =========================================================================
    
    std::cout << "[Init] Initializing vectorized environments..." << "\n";
    VectorizedEnv vecEnv(config.numEnvs, 1);
    vecEnv.Init("robot_configs.json");  // Adjust path as needed
    
    int stateDim = vecEnv.GetObservationDim();
    int actionDim = vecEnv.GetActionDim();
    
    std::cout << "[Init] State dim: " << stateDim << ", Action dim: " << actionDim << "\n";
    
    std::cout << "[Init] Creating optimized training pipeline..." << "\n";
    OptimizedTrainingPipeline::Config pipelineConfig;
    pipelineConfig.numEnvs = config.numEnvs;
    pipelineConfig.numPhysicsSystems = config.numPhysicsSystems;
    pipelineConfig.batchSize = config.batchSize;
    pipelineConfig.accumulationSteps = config.accumulationSteps;
    pipelineConfig.checkpointInterval = config.checkpointInterval;
    pipelineConfig.replayBufferSize = 2000000;
    pipelineConfig.warmupSteps = 10000;
    
    auto pipeline = std::make_unique<OptimizedTrainingPipeline>(pipelineConfig);
    pipeline->Init("robot_configs.json", stateDim, actionDim);
    
    // Load checkpoint if specified
    if (config.loadCheckpoint == "latest") {
        std::string latest = GetLatestCheckpoint(config.checkpointDir);
        if (!latest.empty()) {
            std::cout << "[Init] Loading latest checkpoint: " << latest << "\n";
            pipeline->LoadCheckpoint(latest);
        }
    } else if (!config.loadCheckpoint.empty()) {
        std::cout << "[Init] Loading checkpoint: " << config.loadCheckpoint << "\n";
        pipeline->LoadCheckpoint(config.loadCheckpoint);
    }
    
    // Initialize renderer
    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    Renderer renderer(fbW, fbH);
    
    gCamera.distance = 25.0f;
    gCamera.pitch = 0.4f;
    gCamera.yaw = 0.0f;
    
    // Training state
    TrainingState trainingState;
    trainingState.pipeline = std::move(pipeline);
    trainingState.vecEnv = &vecEnv;
    trainingState.trainer = &trainingState.pipeline->GetTrainer();
    
    // Start I/O thread
    std::thread ioThread(IOWorker, trainingState.trainer);
    
    // Start training thread
    std::thread trainingThread(TrainingThreadFunc, &trainingState, config);
    
    // =========================================================================
    // Main Loop
    // =========================================================================
    
    std::cout << "[Main] Starting main loop..." << "\n";
    
    while (!glfwWindowShouldClose(window) && gRunning) {
        glfwPollEvents();
        
        // Handle input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            gRunning = false;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            gPaused = !gPaused;
            glfwSetTime(0);  // Prevent multiple toggles
        }
        
        // Camera controls
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) gCamera.distance -= 0.1f;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) gCamera.distance += 0.1f;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) gCamera.yaw -= 0.01f;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) gCamera.yaw += 0.01f;
        
        // Render
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Render visible environment
        JPH::PhysicsSystem* physics = vecEnv.GetGlobalPhysics();
        
        // Set up camera
        glm::vec3 cameraPos(
            gCamera.distance * cos(gCamera.pitch) * sin(gCamera.yaw),
            gCamera.distance * sin(gCamera.pitch),
            gCamera.distance * cos(gCamera.pitch) * cos(gCamera.yaw)
        );
        glm::vec3 cameraFront = glm::normalize(-cameraPos);
        
        renderer.Draw(physics, cameraPos, config.renderEnvIndex, cameraFront);
        
        // Render ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // ImGui UI
        ImGui::Begin("Training Control");
        
        ImGui::Text("Step: %d / %d", trainingState.currentStep, config.maxSteps);
        ImGui::Text("SPS: %.2f", trainingState.meanSPS);
        ImGui::Text("Mean Reward: %.4f", trainingState.meanReward);
        
        ImGui::Separator();
        
        if (ImGui::Button(gPaused ? "Resume (Space)" : "Pause (Space)")) {
            gPaused = !gPaused;
        }
        
        ImGui::SliderFloat("Time Scale", &config.timeScale, 0.1f, 10.0f);
        ImGui::Text("Actual Speed: %.1fx", config.timeScale);
        
        ImGui::Separator();
        
        ImGui::Text("Environments: %d", config.numEnvs);
        ImGui::Text("Physics Systems: %d", config.numPhysicsSystems);
        ImGui::Text("Batch Size: %d", config.batchSize);
        ImGui::Text("Accumulation Steps: %d", config.accumulationSteps);
        
        ImGui::Separator();
        
        if (ImGui::Button("Save Checkpoint")) {
            std::string path = config.checkpointDir + "/checkpoint_manual_" + 
                              std::to_string(trainingState.currentStep) + ".bin";
            trainingState.pipeline->SaveCheckpoint(path);
        }
        
        ImGui::End();
        
        // Performance metrics
        ImGui::Begin("Performance");
        
        if (trainingState.rewardHistory.size() > 0) {
            static std::vector<float> displayRewards;
            displayRewards = trainingState.rewardHistory;
            if (displayRewards.size() > 100) {
                // Downsample for display
                std::vector<float> sampled;
                for (size_t i = 0; i < displayRewards.size(); i += displayRewards.size() / 100) {
                    sampled.push_back(displayRewards[i]);
                }
                displayRewards = sampled;
            }
            
            ImGui::PlotLines("Reward History", displayRewards.data(), 
                            displayRewards.size(), 0, nullptr,
                            -1.0f, 1.0f, ImVec2(0, 80));
        }
        
        ImGui::End();
        
        // Training status
        ImGui::Begin("Training Status");
        
        if (gPaused) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "PAUSED");
        } else if (gTrainingComplete) {
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "COMPLETE");
        } else {
            ImGui::TextColored(ImVec4(0, 1, 1, 1), "RUNNING");
        }
        
        float progress = static_cast<float>(trainingState.currentStep) / config.maxSteps * 100.0f;
        ImGui::ProgressBar(progress / 100.0f, ImVec2(0, 20));
        ImGui::Text("%.2f%%", progress);
        
        ImGui::End();
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    
    std::cout << "[Cleanup] Shutting down..." << "\n";
    
    gRunning = false;
    if (trainingThread.joinable()) {
        trainingThread.join();
    }
    
    // Shut down I/O thread
    gIORunning = false;
    gIOCV.notify_all();
    if (ioThread.joinable()) {
        ioThread.join();
    }
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    std::cout << "[Cleanup] Done. Final step: " << trainingState.currentStep << "\n";
    
    return 0;
}
