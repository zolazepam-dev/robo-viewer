#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <deque>

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
    glfwSwapInterval(1);
    
    // Init ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    // Create environment
    CombatEnv env;
    PhysicsCore physicsCore;
    physicsCore.Init(1);  // Initialize for 1 environment
    
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
    std::cout << "Framebuffer size: " << fbW << "x" << fbH << std::endl;
    std::cout << "About to create Renderer..." << std::endl;
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
    float avgSPS = 0;
    
    // Sliding window SPS tracking (1-second window)
    std::deque<int> stepHistory;
    std::deque<double> timeHistory;
    double lastSpsUpdate = 0.0;
    float currentSPS = 0.0f;
    
    // UI state
    bool paused = false;
    float camSpeed = 0.02f;

    // Get initial observations
    std::cout << "Getting initial observations..." << std::endl;
    // Don't call Reset() here - it was already called in Init()
    // Use the correct method to get initial observations
    env.Reset(); // Reset the environment to get initial state
    StepResult initialResult = env.HarvestState(); // Get initial state
    // Convert AlignedVector32 to std::vector
    state1.assign(initialResult.obs_robot1.begin(), initialResult.obs_robot1.end());
    state2.assign(initialResult.obs_robot2.begin(), initialResult.obs_robot2.end());
    std::cout << "Initial observations obtained!" << std::endl;

    while (!glfwWindowShouldClose(window) && totalSteps < 10000000) {
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
            // Queue actions and harvest state
            env.QueueActions(actions.data(), actions.data() + actionDim);
            StepResult result = env.HarvestState();
            
            // Convert AlignedVector32 to std::vector
            state1.assign(result.obs_robot1.begin(), result.obs_robot1.end());
            state2.assign(result.obs_robot2.begin(), result.obs_robot2.end());
            // For rewards, we need to extract a scalar value from VectorReward
            // Using the first component of the reward for now
            lastReward1 = result.reward1.damage_dealt; // or another component
            lastReward2 = result.reward2.damage_dealt; // or another component
            hp1 = result.obs_robot1[stateDim - 3];  // Second to last is robot hp
            hp2 = result.obs_robot2[stateDim - 3];
            
            buffer.Add(state1.data(), actions.data(), result.reward1, state1.data(), result.done);
            
            if (buffer.Size() >= td3cfg.startSteps) {
                trainer.Train(buffer);
            }
            
            if (result.done) {
                env.Reset();
                episodes++;
            }
            
            totalSteps++;
            
            // Update sliding window SPS
            double currentTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
            stepHistory.push_back(totalSteps);
            timeHistory.push_back(currentTime);
            
            // Remove entries older than 1 second
            while (!timeHistory.empty() && currentTime - timeHistory.front() > 1.0) {
                stepHistory.pop_front();
                timeHistory.pop_front();
            }
            
            // Calculate SPS from sliding window
            if (stepHistory.size() >= 2) {
                int stepsInWindow = stepHistory.back() - stepHistory.front();
                double timeWindow = timeHistory.back() - timeHistory.front();
                currentSPS = timeWindow > 0.0 ? static_cast<float>(stepsInWindow / timeWindow) : 0.0f;
            }
        }

        // Render 3D
        float camX = gCamera.distance * std::cos(gCamera.pitch) * std::sin(gCamera.yaw);
        float camY = gCamera.distance * std::sin(gCamera.pitch);
        float camZ = gCamera.distance * std::cos(gCamera.pitch) * std::cos(gCamera.yaw);
        
        glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Use the envIndex parameter for the renderer to draw the correct environment
        renderer.Draw(&physicsCore.GetPhysicsSystem(), glm::vec3(camX, camY, camZ), 0);
        
        // ImGui UI
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Main stats window
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
        ImGui::Begin("Training Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
        int seconds = elapsed % 60;
        int minutes = (elapsed / 60) % 60;
        int hours = elapsed / 3600;
        
        ImGui::Text("Time: %02d:%02d:%02d", hours, minutes, seconds);
        ImGui::Separator();
        ImGui::Text("Steps: %d", totalSteps);
        ImGui::Text("Episodes: %d", episodes);
        
        ImGui::Text("SPS: %.1f", currentSPS);
        ImGui::Text("Buffer: %d", buffer.Size());
        ImGui::Separator();
        ImGui::Text("Robot 1 HP: %.1f", hp1);
        ImGui::Text("Robot 2 HP: %.1f", hp2);
        ImGui::Text("Reward 1: %.2f", lastReward1);
        ImGui::Text("Reward 2: %.2f", lastReward2);
        
        ImGui::Separator();
        if (ImGui::Button(paused ? "Resume" : "Pause")) {
            paused = !paused;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            env.Reset();
            totalSteps = 0;
            episodes = 0;
        }
        
        ImGui::End();
        
        // Camera controls window
        ImGui::SetNextWindowPos(ImVec2(10, 250), ImGuiCond_Always);
        ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::SliderFloat("Distance", &gCamera.distance, 5.0f, 50.0f);
        ImGui::SliderFloat("Pitch", &gCamera.pitch, -1.5f, 1.5f);
        ImGui::SliderFloat("Yaw", &gCamera.yaw, -3.14f, 3.14f);
        ImGui::SliderFloat("Speed", &camSpeed, 0.01f, 0.1f);
        ImGui::End();
        
        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    trainer.Save("saved_models/model_final.bin");
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    
    std::cout << "Done. Model saved." << std::endl;
    return 0;
}
