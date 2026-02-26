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
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>

#include "VectorizedEnv.h"
#include "NeuralNetwork.h"
#include "TD3Trainer.h"
#include "Renderer.h"
#include "LockFreeQueue.h"

// Include nlohmann JSON library for telemetry serialization
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

struct TelemetryData
{
    alignas(32) float envStates[128 * 240 * 2];  // 128 envs * 240 obs dim * 2 robots
    alignas(32) float globalReward[128 * 2];      // 128 envs * 2 rewards
    alignas(32) float jointStress[128 * 13 * 2];  // 128 envs * 13 satellites * 2 robots
    int numEnvs = 128;
    int stepCount = 0;
};

using TelemetryQueue = LockFreeSPSCQueue<TelemetryData, 8>;

// Global telemetry queue for communication between training and telemetry threads
TelemetryQueue gTelemetryQueue;
std::atomic<bool> gTelemetryRunning(false);

void TelemetryThreadFunction()
{
    gTelemetryRunning = true;
    
    while (gTelemetryRunning)
    {
        TelemetryData data;
        if (gTelemetryQueue.Pop(data))
        {
            // Convert telemetry data to JSON
            nlohmann::json jsonData;
            jsonData["stepCount"] = data.stepCount;
            jsonData["numEnvs"] = data.numEnvs;
            
            // Add environment states (first 10 values for brevity)
            nlohmann::json envStatesJson = nlohmann::json::array();
            for (int i = 0; i < std::min(10, data.numEnvs * 240 * 2); ++i)
            {
                envStatesJson.push_back(data.envStates[i]);
            }
            jsonData["envStates"] = envStatesJson;
            
            // Add rewards (first 10 values)
            nlohmann::json rewardsJson = nlohmann::json::array();
            for (int i = 0; i < std::min(10, data.numEnvs * 2); ++i)
            {
                rewardsJson.push_back(data.globalReward[i]);
            }
            jsonData["rewards"] = rewardsJson;
            
            // Add joint stress (first 10 values)
            nlohmann::json jointStressJson = nlohmann::json::array();
            for (int i = 0; i < std::min(10, data.numEnvs * 13 * 2); ++i)
            {
                jointStressJson.push_back(data.jointStress[i]);
            }
            jsonData["jointStress"] = jointStressJson;
            
            // In a real implementation, we would send this JSON data over a WebSocket
            // For now, we'll just print a message to indicate the data is ready
            // std::cout << "[Telemetry] Data prepared for step " << data.stepCount << std::endl;
        }
        
        // Small delay to control telemetry frequency
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 Hz
    }
}

struct TrainingConfig {
    int numParallelEnvs = 2;
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
    
    // Start telemetry thread
    std::thread telemetryThread(TelemetryThreadFunction);
    
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
    
    AlignedVector32<float> actions(totalActionDim, 0.0f);
    std::mt19937 rng(42);
    std::normal_distribution<float> noiseDist(0.0f, 1.0f);
    
     auto last_time = std::chrono::high_resolution_clock::now();
     int step_counter = 0;
     double current_sps = 0.0;
     int selectedEnvIdx = 0;
     bool renderEnabled = true;
     Camera camera;

     std::cout << "[JOLTrl] Visualizer Matrix Engaged. Suppressing per-frame debug spam." << std::endl;

     while (!glfwWindowShouldClose(window)) {
         
         // 1. Generate kinetic jitter actions
         AlignedVector32<float> actions(config.numParallelEnvs * 2 * 56);
         for (float& a : actions) {
             // Apply massive random forces to make them jump/move visibly
             a = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
         }

         // 2. Step the physics matrix silently
         vecEnv.Step(actions);
         vecEnv.ResetDoneEnvs();

         // 3. SPS Calculation & Telemetry (1-second sliding window)
         step_counter += config.numParallelEnvs;
         auto now = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> elapsed = now - last_time;
         
         // Only update UI and Terminal once per second to prevent I/O choking
         if (elapsed.count() >= 1.0) {
             current_sps = step_counter / elapsed.count();
             
             // THE RADAR PING: Let's see if Robot 0 is actually moving
             const auto& obs = vecEnv.GetObservations();
             if (obs.size() >= 3) {
                 // Assuming obs[1] is the Y-axis (Altitude) of the chassis
                 std::cout << "[TELEMETRY] SPS: " << (int)current_sps 
                           << " | Env[0] Robot[0] Altitude (Y): " << obs[1] 
                           << std::endl;
             } else {
                 std::cout << "[TELEMETRY] SPS: " << (int)current_sps << std::endl;
             }
             
             step_counter = 0;
             last_time = now;
         }

         // 4. Render Frame
         // (If the screen is still grey, your camera origin is trapped inside the floor mesh!)
         double currentTime = glfwGetTime();
         static double lastRenderTime = currentTime;
         if (currentTime - lastRenderTime >= (1.0 / 60.0)) {
             lastRenderTime = currentTime;
             glfwPollEvents();

             // Simple orbital camera
             camera.yaw += 0.005f;
             glm::vec3 camPos(
                 camera.distance * cos(camera.pitch) * sin(camera.yaw),
                 camera.distance * sin(camera.pitch),
                 camera.distance * cos(camera.pitch) * cos(camera.yaw)
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
             ImGui::Text("Total Steps: %d", step_counter);
             ImGui::Text("Steps Per Second: %.0f", current_sps);
             ImGui::Separator();
             ImGui::Checkbox("Enable Rendering", &renderEnabled);
             ImGui::SliderInt("Watch Env", &selectedEnvIdx, 0, config.numParallelEnvs - 1);
             ImGui::End();

             // Camera controls window
             ImGui::SetNextWindowPos(ImVec2(10, 250), ImGuiCond_FirstUseEver);
             ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
             ImGui::SliderFloat("Distance", &camera.distance, 5.0f, 50.0f);
             ImGui::SliderFloat("Pitch", &camera.pitch, -1.5f, 1.5f);
             ImGui::SliderFloat("Yaw", &camera.yaw, -3.14f, 3.14f);
             ImGui::End();

             ImGui::Render();
             ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
             glfwSwapBuffers(window);
         }
     }
    
    trainer.Save("saved_models/model_final.bin");
    
    // Stop telemetry thread
    gTelemetryRunning = false;
    if (telemetryThread.joinable()) {
        telemetryThread.join();
    }
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    
    return 0;
}