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
    TD3Trainer opponentTrainer(stateDim, actionDim, td3cfg); // League Play opponent
    ReplayBuffer buffer(td3cfg.bufferSize, stateDim, actionDim);
    
    // Auto-load weights if available
    std::string finalModelPath = checkpointDir + "/model_final.bin";
    if (fs::exists(finalModelPath)) {
        trainer.Load(finalModelPath);
        std::cout << "[main_train] Auto-loaded weights from: " << finalModelPath << std::endl;
    }
    
    // Sync opponent to start identical to main agent
    opponentTrainer.GetModel().GetActor().SetAllWeights(trainer.GetModel().GetActor().GetAllWeights());

    float timeScale = 1.0f;
    float physicsHz = 120.0f;
    bool trainEnabled = true; // Default to enabled
    bool leaguePlayEnabled = true; // Default to Fictitious Self-Play
    bool renderEnabled = true;
    bool headlessTurbo = false;
    
    auto last_time = std::chrono::high_resolution_clock::now();
    long long totalSteps = 0;
    float sps = 0;
    int step_counter = 0;
    int renderEnvIdx = 0;
    int previousRenderEnvIdx = renderEnvIdx;
    float sliderConfirmationTime = 0.0f;
    const float SLIDER_CONFIRMATION_DURATION = 2.0f;
    int r1Wins = 0;
    int r2Wins = 0;
    float currentRew1 = 0.0f;
    float currentRew2 = 0.0f;
    VectorReward lastVR1, lastVR2;
    

    std::mt19937 leagueRng(std::random_device{}());

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
            
            // Collect all robot 1 observations and environment indices
            static AlignedVector32<float> obs1Batch;
            static std::vector<int> indices1;
            obs1Batch.resize(numEnvs * stateDim);
            indices1.resize(numEnvs);
            
            for (int i = 0; i < numEnvs; ++i) {
                std::memcpy(obs1Batch.data() + i * stateDim, (float*)obs.data() + (i * 2 * stateDim), stateDim * sizeof(float));
                indices1[i] = i * 2;
            }
            trainer.SelectActionBatchWithLatent(obs1Batch.data(), robotActions.data(), numEnvs, indices1);
            
            // Collect all robot 2 observations and environment indices
            static AlignedVector32<float> obs2Batch;
            static std::vector<int> indices2;
            obs2Batch.resize(numEnvs * stateDim);
            indices2.resize(numEnvs);
            
            for (int i = 0; i < numEnvs; ++i) {
                std::memcpy(obs2Batch.data() + i * stateDim, (float*)obs.data() + (i * 2 * stateDim + stateDim), stateDim * sizeof(float));
                indices2[i] = i * 2 + 1;
            }
            
            if (leaguePlayEnabled) {
                opponentTrainer.SelectActionBatchWithLatent(obs2Batch.data(), robotActions.data() + (numEnvs * actionDim), numEnvs, indices2);
            } else {
                trainer.SelectActionBatchWithLatent(obs2Batch.data(), robotActions.data() + (numEnvs * actionDim), numEnvs, indices2);
            }
            
            for (int i = 0; i < numEnvs; ++i) {
                // Adjust indexing for robotActions because we batched them separately
                vecEnv.GetEnv(i).QueueActions(robotActions.data() + (i * actionDim), robotActions.data() + (numEnvs * actionDim + i * actionDim));
            }
            
            PhysicsCore* core = vecEnv.GetPhysicsCore();
            core->GetPhysicsSystem().Update(1.0f / physicsHz, 1, core->GetTempAllocator(), core->GetJobSystem());
            
             for (int i = 0; i < numEnvs; ++i) {
                 int obsOffset = i * 2 * stateDim;
                 float* obs1_dest = (float*)obs.data() + obsOffset;
                 float* obs2_dest = (float*)obs.data() + obsOffset + stateDim;
                 float r1_val, r2_val;
                 bool done_val;
                 
                 vecEnv.GetEnv(i).HarvestState(obs1_dest, obs2_dest, &r1_val, &r2_val, done_val);
                 
                 float scalar1 = r1_val;
                 float scalar2 = r2_val;
                
                if (i == renderEnvIdx) {
                    currentRew1 = scalar1;
                    currentRew2 = scalar2;
                }

                // Transition 1 (Note: simplification, usually needs next_state)
                buffer.Add(obs1_dest, robotActions.data() + (i*2*actionDim), scalar1, obs1_dest, done_val);
                // Transition 2
                buffer.Add(obs2_dest, robotActions.data() + (i*2*actionDim+actionDim), scalar2, obs2_dest, done_val);
                
                if (done_val) {
                    // ELO Tracking: Determine winner based on HP
                    auto& robot1 = vecEnv.GetEnv(i).GetRobot1();
                    auto& robot2 = vecEnv.GetEnv(i).GetRobot2();
                    if (robot1.hp > robot2.hp) r1Wins++;
                    else if (robot2.hp > robot1.hp) r2Wins++;

                    vecEnv.Reset(i);
                    
                     // League Play: Rotate the opponent policy from the pool
                    if (leaguePlayEnabled && trainer.GetOpponentPool().Size() > 0) {
                        if (trainer.SampleOpponent()) {
                            opponentTrainer.GetModel().GetActor().SetAllWeights(trainer.GetModel().GetActor().GetAllWeights());
                        }
                    }
                }
            }
            
            if (totalSteps % 4 == 0 && buffer.Size() > td3cfg.batchSize) {
                for (int update = 0; update < 2; ++update) {
                    trainer.Train(buffer);
                }
            }
            


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
            if (ImGui::SliderInt("View Env", &renderEnvIdx, 0, numEnvs - 1)) {
                if (renderEnvIdx != previousRenderEnvIdx) {
                    sliderConfirmationTime = SLIDER_CONFIRMATION_DURATION;
                    previousRenderEnvIdx = renderEnvIdx;
                }
            }
        }

        // Slider confirmation text
        if (sliderConfirmationTime > 0.0f) {
            sliderConfirmationTime -= dt;
            float alpha = std::clamp(sliderConfirmationTime / SLIDER_CONFIRMATION_DURATION, 0.0f, 1.0f);
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, alpha), "Watching Environment %d", renderEnvIdx);
        }

        {
            auto& r1 = vecEnv.GetEnv(renderEnvIdx).GetRobot1();
            auto& r2 = vecEnv.GetEnv(renderEnvIdx).GetRobot2();
            
            ImGui::TextColored(ImVec4(0, 1, 1, 1), "Robot 1 HP: %.1f | Scalar Rew: %.3f", r1.hp, currentRew1);
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
        ImGui::SameLine();
        if (ImGui::Checkbox("League Play (FSP)", &leaguePlayEnabled)) {
            opponentTrainer.GetModel().GetActor().SetAllWeights(trainer.GetModel().GetActor().GetAllWeights());
        }
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
