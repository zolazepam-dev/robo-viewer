// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <filesystem>

#include "VectorizedEnv.h"
#include "NeuralNetwork.h"
#include "TD3Trainer.h"
#include "Renderer.h"

namespace fs = std::filesystem;

Camera gCamera;

struct TrainingConfig {
    int numParallelEnvs = 128; 
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

int main(int argc, char* argv[]) {
    TrainingConfig config;
    
    // 1. Initialize Window and Context
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "JOLTrl Hybrid Overseer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    
    // UNLOCK FRAMERATE: Do not let Vsync block our training matrix
    glfwSwapInterval(0); 

    if (glewInit() != GLEW_OK) return -1;

    // --- THE "UNGODLY ABOMINATION" FIX ---
    // Enforces 3D depth, but we MUST leave culling off because 
    // the sphere meshes are wound clockwise!
    glEnable(GL_DEPTH_TEST);
    // glDisable(GL_CULL_FACE); // Make sure this is completely gone or disabled!
    // -------------------------------------

    // 2. Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    EnsureDir(config.checkpointDir);
    EnsureDir("saved_models");
    
    // 3. Initialize Training Environment
    VectorizedEnv vecEnv(config.numParallelEnvs);
    vecEnv.Init();
    Renderer renderer(1280, 720);
    
    int stateDim = vecEnv.GetObservationDim();
    int actionDim = vecEnv.GetActionDim();
    int totalActionDim = actionDim * 2 * config.numParallelEnvs;
    
    TD3Config td3cfg;
    td3cfg.hiddenDim = 256;
    td3cfg.batchSize = 256;
    td3cfg.startSteps = 10000;
    
    TD3Trainer trainer(stateDim, actionDim, td3cfg);
    ReplayBuffer buffer(td3cfg.bufferSize, stateDim, actionDim);
    
    std::vector<float> actions(totalActionDim, 0.0f);
    std::mt19937 rng(42);
    std::normal_distribution<float> noiseDist(0.0f, 1.0f);
    
    int totalSteps = 0;
    int episodes = 0;
    float currentAvg = 0.0f;
    std::vector<float> avgRewards(100, 0.0f);
    int rewardIdx = 0;
    
    double lastRenderTime = glfwGetTime();
    double lastUiTime = glfwGetTime();
    int lastSteps = 0;
    float sps = 0.0f;
    int selectedEnvIdx = 0;
    bool renderEnabled = true;

    std::cout << "[JOLTrl] Hybrid Overseer Online. Igniting Training Matrix..." << std::endl;
    
    // 4. The Master Loop
    while (totalSteps < config.maxSteps && !glfwWindowShouldClose(window)) {
        
        // --- MACHINE LEARNING PHASE ---
        if (totalSteps < td3cfg.startSteps) {
            for (int i = 0; i < totalActionDim; ++i) actions[i] = noiseDist(rng);
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
            
            buffer.Add(obs1, act1, r1, obs2, allDones[envIdx]);
            buffer.Add(obs2, act2, r2, obs1, allDones[envIdx]);
            
            avgRewards[rewardIdx % 100] = (r1 + r2) / 2.0f;
            rewardIdx++;
            if (allDones[envIdx]) episodes++;
        }
        
        if (buffer.Size() >= td3cfg.startSteps) trainer.Train(buffer);
        totalSteps++;
        
        // --- HYBRID OVERSEER RENDER SIPHON (Capped at 60 FPS) ---
        double currentTime = glfwGetTime();
        if (currentTime - lastRenderTime >= (1.0 / 60.0)) {
            lastRenderTime = currentTime;
            glfwPollEvents();

            // Simple orbital camera
            gCamera.yaw += 0.005f;
            glm::vec3 camPos(
                gCamera.distance * cos(gCamera.pitch) * sin(gCamera.yaw),
                gCamera.distance * sin(gCamera.pitch),
                gCamera.distance * cos(gCamera.pitch) * cos(gCamera.yaw)
            );

            // Draw exactly 1 environment via the Dimensional Filter
            if (renderEnabled) {
                renderer.Draw(vecEnv.GetGlobalPhysics(), camPos, selectedEnvIdx);
            } else {
                glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            }

            // Draw ImGui Overlay
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(300, 180), ImGuiCond_FirstUseEver);
            ImGui::Begin("JOLTrl Hybrid Overseer");
            ImGui::Text("Silicon: Intel i5-10500");
            ImGui::Text("Graphics: Intel HD 630");
            ImGui::Separator();
            ImGui::Text("Parallel Envs: %d", config.numParallelEnvs);
            ImGui::Text("Total Steps: %d", totalSteps);
            ImGui::Text("Steps Per Second: %.0f", sps);
            ImGui::Text("Episodes: %d", episodes);
            ImGui::Text("Average Reward: %.2f", currentAvg);
            ImGui::Separator();
            ImGui::Checkbox("Enable Rendering", &renderEnabled);
            ImGui::SliderInt("Watch Env", &selectedEnvIdx, 0, config.numParallelEnvs - 1);
            ImGui::Text("FPS: %.1f", sps);
            ImGui::End();

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }

        // Calculate SPS once per second
        if (currentTime - lastUiTime >= 1.0) {
            sps = (totalSteps - lastSteps) / (currentTime - lastUiTime);
            lastSteps = totalSteps;
            lastUiTime = currentTime;
            
            currentAvg = 0;
            for(int i=0; i<std::min(rewardIdx, 100); i++) currentAvg += avgRewards[i];
            if (rewardIdx > 0) currentAvg /= std::min(rewardIdx, 100);
        }
        
        if (totalSteps % config.checkpointInterval == 0) {
            std::string checkpointPath = config.checkpointDir + "/model_" + std::to_string(totalSteps) + ".bin";
            trainer.Save(checkpointPath);
        }
    }
    
    trainer.Save("saved_models/model_final.bin");
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    
    return 0;
}