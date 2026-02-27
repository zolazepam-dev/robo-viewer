#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <filesystem>
#include <fstream>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "CombatEnv.h"
#include "VectorizedEnv.h"
#include "NeuralNetwork.h"
#include "TD3Trainer.h"
#include "Renderer.h"

namespace fs = std::filesystem;

Camera gCamera;

struct TrainingConfig {
    int numParallelEnvs = 8;
    int checkpointInterval = 50000;
    int maxSteps = 10000000;
    std::string checkpointDir = "checkpoints";
    std::string loadCheckpoint = "";
};

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
                int step = std::stoi(filename.substr(pos + 1));
                if (step > maxStep) {
                    maxStep = step;
                    latest = entry.path().string();
                }
            }
        }
    }
    return latest;
}

int main(int argc, char* argv[]) {
    TrainingConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--envs" && i + 1 < argc) {
            config.numParallelEnvs = std::stoi(argv[++i]);
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
        }
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Parallel Combat Self-Play Training    " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Parallel envs: " << config.numParallelEnvs << std::endl;
    std::cout << "Checkpoint interval: " << config.checkpointInterval << std::endl;
    std::cout << "Max steps: " << config.maxSteps << std::endl;
    std::cout << "Checkpoint dir: " << config.checkpointDir << std::endl;
    
    EnsureDir(config.checkpointDir);
    EnsureDir("saved_models");
    
    if (!glfwInit()) return 1;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Parallel Self-Play Training", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { glfwTerminate(); return 1; }
    
    glEnable(GL_DEPTH_TEST);
    glfwSwapInterval(1);
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    std::cout << "Initializing vectorized environments..." << std::endl;
    VectorizedEnv vecEnv(config.numParallelEnvs, 1);
    vecEnv.Init();
    
    CombatEnv& visibleEnv = vecEnv.GetEnv(0);
    
    int stateDim = vecEnv.GetObservationDim();
    int actionDim = vecEnv.GetActionDim();
    int totalActionDim = actionDim * 2 * config.numParallelEnvs;
    
    std::cout << "State dim: " << stateDim << ", Action dim: " << actionDim << std::endl;
    
    TD3Config td3cfg;
    td3cfg.hiddenDim = 256;
    td3cfg.batchSize = 256;
    td3cfg.startSteps = 10000;
    
    std::cout << "Creating trainer..." << std::endl;
    TD3Trainer trainer(stateDim, actionDim, td3cfg);
    
    int startStep = 0;
    if (config.loadCheckpoint == "latest") {
        std::string latest = GetLatestCheckpoint(config.checkpointDir);
        if (!latest.empty()) {
            std::cout << "Loading latest checkpoint: " << latest << std::endl;
            trainer.Load(latest);
            size_t pos = latest.find_last_of('_');
            if (pos != std::string::npos) {
                startStep = std::stoi(latest.substr(pos + 1, latest.find(".bin") - pos - 1));
            }
            std::cout << "Resumed from step " << startStep << std::endl;
        }
    } else if (!config.loadCheckpoint.empty()) {
        std::cout << "Loading checkpoint: " << config.loadCheckpoint << std::endl;
        trainer.Load(config.loadCheckpoint);
    }
    
    std::cout << "Creating replay buffer..." << std::endl;
    ReplayBuffer buffer(td3cfg.bufferSize, stateDim, actionDim);
    
    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    Renderer renderer(fbW, fbH);
    
    gCamera.distance = 25.0f;
    gCamera.pitch = 0.4f;
    
    std::vector<float> actions(totalActionDim, 0.0f);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> noiseDist(0.0f, 1.0f);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    int totalSteps = startStep;
    int episodes = 0;
    int lastCheckpointStep = startStep;
    float bestAvgReward = -999999.0f;
    
    bool paused = false;
    float camSpeed = 0.02f;
    
    std::vector<float> avgRewards(100, 0.0f);
    int rewardIdx = 0;
    
    std::cout << "Starting training loop..." << std::endl;
    
    while (!glfwWindowShouldClose(window) && totalSteps < config.maxSteps) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        
        if (!paused) {
            if (totalSteps < td3cfg.startSteps) {
                for (int i = 0; i < totalActionDim; ++i) {
                    actions[i] = noiseDist(rng);
                }
            } else {
                const auto& allObs = vecEnv.GetObservations();
                for (int envIdx = 0; envIdx < config.numParallelEnvs; ++envIdx) {
                    const float* obs1 = allObs.data() + envIdx * stateDim * 2;
                    const float* obs2 = obs1 + stateDim;
                    float* act1 = actions.data() + envIdx * actionDim * 2;
                    float* act2 = act1 + actionDim;
                    trainer.SelectAction(obs1, act1);
                    trainer.SelectAction(obs2, act2);
                }
            }
            
            vecEnv.Step(actions);
            vecEnv.ResetDoneEnvs();
            
            const auto& allObs = vecEnv.GetObservations();
            const auto& allRewards = vecEnv.GetRewards();
            const auto& allDones = vecEnv.GetDones();
            
            for (int envIdx = 0; envIdx < config.numParallelEnvs; ++envIdx) {
                const float* obs1 = allObs.data() + envIdx * stateDim * 2;
                const float* obs2 = obs1 + stateDim;
                const float* act1 = actions.data() + envIdx * actionDim * 2;
                const float* act2 = act1 + actionDim;
                float r1 = allRewards[envIdx * 2];
                float r2 = allRewards[envIdx * 2 + 1];
                
                buffer.Add(obs1, act1, r1, obs1, allDones[envIdx]);
                buffer.Add(obs2, act2, r2, obs2, allDones[envIdx]);
                
                avgRewards[rewardIdx % 100] = (r1 + r2) / 2.0f;
                rewardIdx++;
                
                if (allDones[envIdx]) {
                    episodes++;
                }
            }
            
            if (buffer.Size() >= td3cfg.startSteps) {
                trainer.Train(buffer);
            }
            
            totalSteps++;
            
            if (totalSteps - lastCheckpointStep >= config.checkpointInterval) {
                std::string checkpointPath = config.checkpointDir + "/model_" + std::to_string(totalSteps) + ".bin";
                trainer.Save(checkpointPath);
                std::cout << "Checkpoint saved: " << checkpointPath << std::endl;
                lastCheckpointStep = totalSteps;
                
                float currentAvg = 0.0f;
                int count = std::min(rewardIdx, 100);
                for (int i = 0; i < count; ++i) {
                    currentAvg += avgRewards[i];
                }
                currentAvg /= count;
                
                if (currentAvg > bestAvgReward) {
                    bestAvgReward = currentAvg;
                    trainer.Save("saved_models/best_policy.bin");
                    std::cout << "New best policy saved! Avg reward: " << currentAvg << std::endl;
                }
            }
        }
        
        float camX = gCamera.distance * std::cos(gCamera.pitch) * std::sin(gCamera.yaw);
        float camY = gCamera.distance * std::sin(gCamera.pitch);
        float camZ = gCamera.distance * std::cos(gCamera.pitch) * std::cos(gCamera.yaw);
        
        glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        std::vector<JPH::BodyID> bodies;
        const auto& r1 = visibleEnv.GetRobot1();
        const auto& r2 = visibleEnv.GetRobot2();
        bodies.push_back(r1.mainBodyId);
        bodies.push_back(r2.mainBodyId);
        for (int i = 0; i < NUM_SATELLITES; ++i) {
            bodies.push_back(r1.satellites[i].coreBodyId);
            bodies.push_back(r2.satellites[i].coreBodyId);
        }
        
        renderer.Draw(&visibleEnv.GetPhysicsSystem(), bodies, glm::vec3(camX, camY, camZ));
        
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
        ImGui::Begin("Parallel Training Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
        int seconds = elapsed % 60;
        int minutes = (elapsed / 60) % 60;
        int hours = elapsed / 3600;
        
        ImGui::Text("Time: %02d:%02d:%02d", hours, minutes, seconds);
        ImGui::Separator();
        ImGui::Text("Steps: %d / %d", totalSteps, config.maxSteps);
        ImGui::Text("Episodes: %d", episodes);
        
        float progress = (float)totalSteps / config.maxSteps;
        ImGui::ProgressBar(progress, ImVec2(-1, 0));
        
        // Calculate recent SPS (last 1 second)
        static auto lastSPSCheck = startTime;
        static int lastSPSCount = startStep;
        auto now = std::chrono::high_resolution_clock::now();
        auto spsElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastSPSCheck).count();
        
        float sps = 0.0f;
        if (spsElapsed >= 1) {
            sps = (totalSteps - lastSPSCount) / (float)spsElapsed;
            lastSPSCheck = now;
            lastSPSCount = totalSteps;
        }
        
        ImGui::Text("SPS: %.1f", sps);
        ImGui::Text("Parallel Envs: %d", config.numParallelEnvs);
        ImGui::Text("Buffer: %d / %d", buffer.Size(), td3cfg.bufferSize);
        
        ImGui::Separator();
        
        float currentAvg = 0.0f;
        int count = std::min(rewardIdx, 100);
        for (int i = 0; i < count; ++i) {
            int idx = (rewardIdx - count + i) % 100;
            if (idx < 0) idx += 100; // Handle negative indices
            currentAvg += avgRewards[idx];
        }
        currentAvg /= count > 0 ? count : 1;
        ImGui::Text("Avg Reward (100): %.2f", currentAvg);
        ImGui::Text("Best Avg Reward: %.2f", bestAvgReward);
        
        ImGui::Separator();
        if (ImGui::Button(paused ? "Resume" : "Pause")) {
            paused = !paused;
        }
        ImGui::SameLine();
        if (ImGui::Button("Save Now")) {
            std::string path = config.checkpointDir + "/model_manual_" + std::to_string(totalSteps) + ".bin";
            trainer.Save(path);
            std::cout << "Manual save: " << path << std::endl;
        }
        
        ImGui::End();
        
        ImGui::SetNextWindowPos(ImVec2(10, 280), ImGuiCond_Always);
        ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::SliderFloat("Distance", &gCamera.distance, 5.0f, 50.0f);
        ImGui::SliderFloat("Pitch", &gCamera.pitch, -1.5f, 1.5f);
        ImGui::SliderFloat("Yaw", &gCamera.yaw, -3.14f, 3.14f);
        ImGui::SliderFloat("Speed", &camSpeed, 0.01f, 0.1f);
        ImGui::End();
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    std::string finalPath = "saved_models/model_final.bin";
    trainer.Save(finalPath);
    std::cout << "Final model saved: " << finalPath << std::endl;
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    
    std::cout << "Training complete. Total steps: " << totalSteps << std::endl;
    return 0;
}
