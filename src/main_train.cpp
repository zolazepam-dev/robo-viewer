// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <filesystem>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>

#include "src/VectorizedEnv.h"
#include "src/NeuralNetwork.h"
#include "src/TD3Trainer.h"
#include "src/Renderer.h"
#include "src/OverlayUI.h"

namespace fs = std::filesystem;

struct FreeCamera {
    glm::vec3 position{0.0f, 15.0f, 40.0f};
    glm::vec3 front{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float yaw = -90.0f;
    float pitch = -20.0f;
    float speed = 30.0f;
    float sensitivity = 0.1f;
    bool active = false;
};

// GLOBAL CAMERA - NO SHADOWING
FreeCamera gCam;
double gLastX, gLastY;
bool gFirstMouse = true;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!gCam.active) {
        gLastX = xpos; gLastY = ypos;
        return;
    }
    if (gFirstMouse) { gLastX = xpos; gLastY = ypos; gFirstMouse = false; }
    float xoff = (float)(xpos - gLastX) * gCam.sensitivity;
    float yoff = (float)(gLastY - ypos) * gCam.sensitivity;
    gLastX = xpos; gLastY = ypos;
    gCam.yaw += xoff; gCam.pitch += yoff;
    gCam.pitch = std::clamp(gCam.pitch, -89.0f, 89.0f);
    glm::vec3 dir;
    dir.x = cos(glm::radians(gCam.yaw)) * cos(glm::radians(gCam.pitch));
    dir.y = sin(glm::radians(gCam.pitch));
    dir.z = sin(glm::radians(gCam.yaw)) * cos(glm::radians(gCam.pitch));
    gCam.front = glm::normalize(dir);
}

void process_input(GLFWwindow* window, float dt) {
    if (!gCam.active) return;
    float vel = gCam.speed * dt;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) gCam.position += gCam.front * vel;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) gCam.position -= gCam.front * vel;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) gCam.position -= glm::normalize(glm::cross(gCam.front, gCam.up)) * vel;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) gCam.position += glm::normalize(glm::cross(gCam.front, gCam.up)) * vel;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) gCam.position += gCam.up * vel;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) gCam.position -= gCam.up * vel;
}

void EnsureDir(const std::string& path) { if (!fs::exists(path)) fs::create_directories(path); }

int main(int argc, char* argv[]) {
    int numEnvs = 1;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--envs" && i + 1 < argc) {
            numEnvs = std::stoi(argv[++i]);
        }
    }

    const char* home = std::getenv("HOME");
    std::string checkpointDir = (home ? std::string(home) : ".") + "/.joltrl/checkpoints";
    EnsureDir(checkpointDir);

    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "JOLTrl - Pro Training Suite", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); 
    glewInit();
    glEnable(GL_DEPTH_TEST);
    glfwSetCursorPosCallback(window, mouse_callback);

    CyberpunkUI ui;
    ui.Init(window);

    // Initializing vectorized environments
    VectorizedEnv vecEnv(numEnvs);
    vecEnv.Init();
    
    Renderer renderer(1280, 720);
    int stateDim = vecEnv.GetObservationDim(); // Should be 256
    int actionDim = vecEnv.GetActionDim();      // Should be 56
    
    TD3Config td3cfg;
    TD3Trainer trainer(stateDim, actionDim, td3cfg);
    ReplayBuffer buffer(td3cfg.bufferSize, stateDim, actionDim);
    
    // Auto-load weights if available
    std::string finalModelPath = checkpointDir + "/model_final.bin";
    if (fs::exists(finalModelPath)) {
        trainer.Load(finalModelPath);
        std::cout << "[main_train] Auto-loaded weights from: " << finalModelPath << std::endl;
    }
    
    float timeScale = 1.0f;
    float physicsHz = 120.0f;
    bool trainEnabled = false; // Start paused
    bool renderEnabled = true;
    bool headlessTurbo = false;
    
    auto last_time = std::chrono::high_resolution_clock::now();
    long long totalSteps = 0;
    float sps = 0;
    int step_counter = 0;
    int renderEnvIdx = 0;
    float currentRew1 = 0.0f;
    float currentRew2 = 0.0f;
    VectorReward lastVR1, lastVR2;

    // Fixed: Ensure global cam is used
    gCam.front = glm::normalize(glm::vec3(0, 2, 0) - gCam.position);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!gCam.active) { 
                gCam.active = true; 
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); 
                gFirstMouse = true; 
            }
        } else {
            if (gCam.active) { 
                gCam.active = false; 
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); 
            }
        }
        process_input(window, dt);

        if (trainEnabled && timeScale > 0.01f) {
            const auto& obs = vecEnv.GetObservations();
            int numEnvs = vecEnv.GetNumEnvs();
            AlignedVector32<float> robotActions(numEnvs * 2 * actionDim);
            
            // Generate actions for all robots in all envs
            for (int i = 0; i < numEnvs; ++i) {
                // Each environment has 2 robots. 
                // We use envIdx * 2 and envIdx * 2 + 1 as latent slots to keep them distinct.
                trainer.SelectActionWithLatent((float*)obs.data() + (i * 2 * stateDim), robotActions.data() + (i * 2 * actionDim), i * 2);
                trainer.SelectActionWithLatent((float*)obs.data() + (i * 2 * stateDim + stateDim), robotActions.data() + (i * 2 * actionDim + actionDim), i * 2 + 1);
            }
            
            for (int i = 0; i < numEnvs; ++i) {
                vecEnv.GetEnv(i).QueueActions(robotActions.data() + (i * 2 * actionDim), robotActions.data() + (i * 2 * actionDim + actionDim));
            }
            
            PhysicsCore* core = vecEnv.GetPhysicsCore();
            core->GetPhysicsSystem().Update(1.0f / physicsHz, 1, core->GetTempAllocator(), core->GetJobSystem());
            
            for (int i = 0; i < numEnvs; ++i) {
                StepResult res = vecEnv.GetEnv(i).HarvestState();
                
                if (i == 0) {
                    currentRew1 = res.reward1.Scalar();
                    currentRew2 = res.reward2.Scalar();
                    lastVR1 = res.reward1;
                    lastVR2 = res.reward2;
                }

                // Transition 1
                buffer.Add((float*)obs.data() + (i*2*stateDim), robotActions.data() + (i*2*actionDim), res.reward1.Scalar(), res.obs_robot1.data(), res.done);
                // Transition 2
                buffer.Add((float*)obs.data() + (i*2*stateDim+stateDim), robotActions.data() + (i*2*actionDim+actionDim), res.reward2.Scalar(), res.obs_robot2.data(), res.done);
                
                if (res.done) vecEnv.Reset(i);
            }
            
            if (buffer.Size() > td3cfg.batchSize) trainer.Train(buffer);
            
            if (totalSteps > 0 && totalSteps % 18000 == 0) {
                trainer.Save(checkpointDir + "/model_step_" + std::to_string(totalSteps) + ".bin");
            }
            
            totalSteps += 1;
            step_counter += 1;
        }

        static auto lastSpsTime = now;
        std::chrono::duration<float> spsElapsed = now - lastSpsTime;
        if (spsElapsed.count() >= 1.0f) { sps = step_counter / spsElapsed.count(); step_counter = 0; lastSpsTime = now; }

        if (renderEnabled && !headlessTurbo) {
            renderer.Draw(vecEnv.GetGlobalPhysics(), gCam.position, renderEnvIdx, gCam.front);
        } else {
            glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        ui.BeginFrame();
        ImGui::Begin("Full Training Overseer");
        ImGui::Text("Silicon: TD3 + 2nd Order Latent");
        ImGui::Separator();
        ImGui::Text("STATUS: %s", trainEnabled ? "TRAINING" : "PAUSED");
        ImGui::Text("SPS: %.0f | Total Steps: %lld", sps, totalSteps);
        
        if (numEnvs > 1) {
            ImGui::SliderInt("View Env", &renderEnvIdx, 0, numEnvs - 1);
        }

        {
            auto& r1 = vecEnv.GetEnv(renderEnvIdx).GetRobot1();
            auto& r2 = vecEnv.GetEnv(renderEnvIdx).GetRobot2();
            
            ImGui::TextColored(ImVec4(0, 1, 1, 1), "Robot 1 HP: %.1f", r1.hp);
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Dmg+: %.3f | Dmg-: %.3f", lastVR1.damage_dealt, lastVR1.damage_taken);
                ImGui::Text("KOTH: %.3f | Energy: %.3f", lastVR1.koth, lastVR1.energy_used);
                ImGui::Text("Alt: %.3f", lastVR1.altitude);
                ImGui::EndTooltip();
            }

            ImGui::TextColored(ImVec4(1, 0, 1, 1), "Robot 2 HP: %.1f | Scalar Rew: %.3f", r2.hp, currentRew2);
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Dmg+: %.3f | Dmg-: %.3f", lastVR2.damage_dealt, lastVR2.damage_taken);
                ImGui::Text("KOTH: %.3f | Energy: %.3f", lastVR2.koth, lastVR2.energy_used);
                ImGui::Text("Alt: %.3f", lastVR2.altitude);
                ImGui::EndTooltip();
            }
        }
        ImGui::Separator();

        ImGui::Checkbox("Enable Training", &trainEnabled);
        ImGui::Checkbox("HEADLESS TURBO", &headlessTurbo);
        
        if (ImGui::Button("Load Final Checkpoint")) {
            trainer.Load(checkpointDir + "/model_final.bin");
        }
        
        ImGui::SliderFloat("Time Scale", &timeScale, 0.0f, 5.0f);
        
        if (ImGui::CollapsingHeader("Incremental Power Tuner")) {
            auto& r1 = vecEnv.GetEnv(0).GetRobot1Ref();
            auto& r2 = vecEnv.GetEnv(0).GetRobot2Ref();
            ImGui::SliderFloat("Rotation Speed", &r1.actionScale.rotationScale, 0.0f, 500.0f);
            ImGui::SliderFloat("Stab Speed", &r1.actionScale.slideScale, 0.0f, 2000.0f);
            r2.actionScale.rotationScale = r1.actionScale.rotationScale;
            r2.actionScale.slideScale = r1.actionScale.slideScale;
        }

        if (ImGui::CollapsingHeader("Physics Tuner (Expansive)")) {
            JPH::PhysicsSettings settings = vecEnv.GetGlobalPhysics()->GetPhysicsSettings();
            bool changed = false;
            ImGui::SliderFloat("Simulation Hz", &physicsHz, 60.0f, 1000.0f, "%.0f Hz");
            if (ImGui::SliderInt("Velocity Steps", (int*)&settings.mNumVelocitySteps, 1, 50)) changed = true;
            if (ImGui::SliderInt("Position Steps", (int*)&settings.mNumPositionSteps, 1, 50)) changed = true;
            if (ImGui::SliderFloat("Baumgarte", &settings.mBaumgarte, 0.0f, 1.0f)) changed = true;
            if (ImGui::SliderFloat("Penetration Slop", &settings.mPenetrationSlop, 0.0f, 0.1f)) changed = true;
            if (ImGui::SliderFloat("Speculative Dist", &settings.mSpeculativeContactDistance, 0.0f, 0.1f)) changed = true;
            if (changed) vecEnv.GetGlobalPhysics()->SetPhysicsSettings(settings);
        }

        if (ImGui::Button("Manual Reset All")) vecEnv.Reset();
        ImGui::End();
        ui.EndFrame();
        glfwSwapBuffers(window);
    }

    trainer.Save(checkpointDir + "/model_final.bin");
    ui.Shutdown();
    glfwTerminate();
    return 0;
}
