#include <iostream>
#include <chrono>
#include <vector>
#include <random>

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
    CombatEnv env(true);
    if (!env.Init()) { std::cerr << "CombatEnv Init failed" << std::endl; glfwTerminate(); return 1; } else { std::cout << "CombatEnv Init succeeded" << std::endl; }
    
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
    
    // UI state
    bool paused = false;
    float camSpeed = 0.02f;

    // Get initial observations
    std::cout << "Getting initial observations..." << std::endl;
    // Don't call Reset() here - it was already called in Init()
    StepResult initialResult = env.Step(actions.data(), actions.data() + actionDim);
    state1 = initialResult.obs_robot1;
    state2 = initialResult.obs_robot2;
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
            StepResult result = env.Step(actions.data(), actions.data() + actionDim);
            
            state1 = result.obs_robot1;
            state2 = result.obs_robot2;
            lastReward1 = result.reward1;
            lastReward2 = result.reward2;
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
        }

        // Render 3D
        float camX = gCamera.distance * std::cos(gCamera.pitch) * std::sin(gCamera.yaw);
        float camY = gCamera.distance * std::sin(gCamera.pitch);
        float camZ = gCamera.distance * std::cos(gCamera.pitch) * std::cos(gCamera.yaw);
        
        glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        std::vector<JPH::BodyID> bodies;
        const auto& r1 = env.GetRobot1();
        const auto& r2 = env.GetRobot2();
        bodies.push_back(r1.mainBodyId);
        bodies.push_back(r2.mainBodyId);
        for (int i = 0; i < NUM_SATELLITES; ++i) {
            bodies.push_back(r1.satellites[i].coreBodyId);
            bodies.push_back(r2.satellites[i].coreBodyId);
        }
        
        renderer.Draw(&env.GetPhysicsSystem(), bodies, glm::vec3(camX, camY, camZ));
        
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
        
        float sps = elapsed > 0 ? (float)totalSteps / elapsed : 0;
        ImGui::Text("SPS: %.1f", sps);
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
