#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <deque>
#include <thread>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "CombatEnv.h"
#include "NeuralNetwork.h"
#include "TD3Trainer.h"
#include "Renderer.h"
#include "PhysicsCore.h"
#include "CombatRobot.h"

Camera gCamera;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   Combat Self-Play Training           " << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Init GLFW
    if (!glfwInit()) return 1;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Self-Play Training", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { glfwTerminate(); return 1; }
    
    glEnable(GL_DEPTH_TEST);
    glfwSwapInterval(1); // HARDWARE V-SYNC LOCK: Force the GPU to wait for 60Hz monitor
    
    // Init ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    // Config
    struct {
        int numParallelEnvs = 1;
    } config;

    // Create environment
    CombatEnv env;
    PhysicsCore physicsCore;
    physicsCore.Init(config.numParallelEnvs);  
    
    CombatRobotLoader robotLoader;
    
    // Initialize the environment with required parameters
    env.Init(0, &physicsCore.GetPhysicsSystem(), &robotLoader);
    std::cout << "CombatEnv initialized successfully" << std::endl;
    
    int stateDim = env.GetObservationDim();
    int actionDim = ACTIONS_PER_ROBOT;
    std::cout << "State: " << stateDim << ", Action: " << actionDim << std::endl;
    
    // TD3 trainer
    TD3Config td3cfg;
    td3cfg.hiddenDim = 128;
    td3cfg.batchSize = 64;
    td3cfg.startSteps = 1000;
    
    std::cout << "Creating trainer..." << std::endl;
    TD3Trainer trainer(stateDim, actionDim, td3cfg);
    std::cout << "Creating buffer..." << std::endl;
    ReplayBuffer buffer(100000, stateDim, actionDim);
    
    // Renderer
    std::cout << "Creating renderer..." << std::endl;
    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    Renderer renderer(fbW, fbH);
    std::cout << "Renderer created!" << std::endl;
    
    gCamera.distance = 20.0f;
    gCamera.pitch = 0.5f;
    
    // Buffers
    std::vector<float> actions(actionDim * 2, 0.0f);
    std::vector<float> state1(stateDim, 0.0f);
    std::vector<float> state2(stateDim, 0.0f);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> noiseDist(0.0f, 1.0f);
    
    // Stats
    auto startTime = std::chrono::high_resolution_clock::now();
    int totalSteps = 0;
    int episodes = 0;
    float lastReward1 = 0, lastReward2 = 0;
    float hp1 = 100, hp2 = 100;

    // Time tracking for FPS/SPS
    auto last_time = std::chrono::high_resolution_clock::now();
    int step_counter = 0;
    float currentSPS = 0.0f;
    
    // UI state
    bool paused = false;
    float camSpeed = 0.02f;

    // Get initial observations
    std::cout << "Getting initial observations..." << std::endl;
    env.Reset();
    float r1_dummy, r2_dummy;
    bool done_dummy;
    env.HarvestState(state1.data(), state2.data(), &r1_dummy, &r2_dummy, done_dummy);
    std::cout << "Initial observations obtained!" << std::endl;

    std::cout << "[JOLTrl] Visualizer Matrix Engaged. Suppressing per-frame debug spam." << std::endl;

    while (!glfwWindowShouldClose(window) && totalSteps < 10000000) {
        // --- FRAME DELTA TRACKING START ---
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        
        if (!paused) {
            // Actions
            if (totalSteps < td3cfg.startSteps) {
                for (int i = 0; i < actionDim * 2; ++i) {
                    actions[i] = noiseDist(rng);
                }
            } else {
                trainer.SelectAction(state1.data(), actions.data());
                trainer.SelectAction(state2.data(), actions.data() + actionDim);
            }
            
            // Step
            env.QueueActions(actions.data(), actions.data() + actionDim);
            float reward1, reward2;
            bool done;
            env.HarvestState(state1.data(), state2.data(), &reward1, &reward2, done);
            
            lastReward1 = reward1;
            lastReward2 = reward2;
            hp1 = state1[stateDim - 3];
            hp2 = state2[stateDim - 3];
            
            // Note: buffer.Add expects VectorReward, creating a dummy one for now or passing scalar if supported
            VectorReward vr1; vr1.damage_dealt = reward1; // Approximation
            buffer.Add(state1.data(), actions.data(), vr1, state1.data(), done);
            
            if (buffer.Size() >= td3cfg.startSteps) {
                trainer.Train(buffer);
            }
            
            if (done) {
                env.Reset();
                episodes++;
            }
            
            totalSteps++;
            step_counter += config.numParallelEnvs;
        }

        // --- SPS Calculation & Telemetry (1-second sliding window) ---
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - last_time;
        
        if (elapsed.count() >= 1.0) {
            currentSPS = step_counter / elapsed.count();
            
            // THE RADAR PING: Watch the altitude drop as they fall into frame
            // Robot 1 Y is typically at index 1 of the observation vector
            if (state1.size() >= 2) {
                std::cout << "[TELEMETRY] SPS: " << (int)currentSPS 
                          << " | Robot[0] Altitude (Y): " << state1[1] 
                          << " | HP: " << hp1 << " vs " << hp2
                          << std::endl;
            } else {
                std::cout << "[TELEMETRY] SPS: " << (int)currentSPS << std::endl;
            }
            
            step_counter = 0;
            last_time = now;
        }

        // Render 3D
        float camX = gCamera.distance * std::cos(gCamera.pitch) * std::sin(gCamera.yaw);
        float camY = gCamera.distance * std::sin(gCamera.pitch);
        float camZ = gCamera.distance * std::cos(gCamera.pitch) * std::cos(gCamera.yaw);
        
        glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        renderer.Draw(&physicsCore.GetPhysicsSystem(), glm::vec3(camX, camY, camZ), 0);
        
        // ImGui UI
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
        ImGui::Begin("Training Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        
        auto up_now = std::chrono::high_resolution_clock::now();
        auto up_elapsed = std::chrono::duration_cast<std::chrono::seconds>(up_now - startTime).count();
        int seconds = up_elapsed % 60;
        int minutes = (up_elapsed / 60) % 60;
        int hours = up_elapsed / 3600;
        
        ImGui::Text("Time: %02d:%02d:%02d", hours, minutes, seconds);
        ImGui::Separator();
        ImGui::Text("Steps: %d", totalSteps);
        ImGui::Text("Episodes: %d", episodes);
        ImGui::Text("SPS: %.1f", currentSPS);
        ImGui::Separator();
        ImGui::Text("Robot 1 HP: %.1f", hp1);
        ImGui::Text("Robot 2 HP: %.1f", hp2);
        
        if (ImGui::Button(paused ? "Resume" : "Pause")) paused = !paused;
        ImGui::End();
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);

        // --- THE FRAME LIMITER (60 FPS) ---
        auto frame_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> frame_time = frame_end - frame_start;
        double target_time = 1.0 / 60.0; // 60 Hz Target
        if (frame_time.count() < target_time) {
            std::this_thread::sleep_for(std::chrono::duration<double>(target_time - frame_time.count()));
        }
    }
    
    trainer.Save("saved_models/model_final.bin");
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    
    std::cout << "Done. Model saved." << std::endl;
    return 0;
}
