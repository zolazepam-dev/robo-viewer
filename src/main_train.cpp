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
    glm::vec3 position{15.0f, 10.0f, 15.0f};
    glm::vec3 front{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float yaw = -135.0f;
    float pitch = -30.0f;
    float speed = 20.0f;
    float sensitivity = 0.1f;
    bool active = false;
};

FreeCamera gCam;
double lastX, lastY;
bool firstMouse = true;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!gCam.active) return;
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoff = (float)(xpos - lastX) * gCam.sensitivity;
    float yoff = (float)(lastY - ypos) * gCam.sensitivity;
    lastX = xpos; lastY = ypos;
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
    // 0. Persistent Checkpoint Setup
    const char* home = std::getenv("HOME");
    std::string checkpointDir = (home ? std::string(home) : ".") + "/.joltrl/checkpoints";
    EnsureDir(checkpointDir);
    std::cout << "[JOLTrl] Safe Checkpoint Zone: " << checkpointDir << std::endl;

    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "JOLTrl - Full Training Suite", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); 
    glewInit();
    glEnable(GL_DEPTH_TEST);
    glfwSetCursorPosCallback(window, mouse_callback);

    CyberpunkUI ui;
    ui.Init(window);

    int numEnvs = 1;
    VectorizedEnv vecEnv(numEnvs);
    vecEnv.Init();
    
    Renderer renderer(1280, 720);
    int stateDim = vecEnv.GetObservationDim();
    int actionDim = vecEnv.GetActionDim();
    
    TD3Config td3cfg;
    TD3Trainer trainer(stateDim, actionDim, td3cfg);
    ReplayBuffer buffer(td3cfg.bufferSize, stateDim, actionDim);
    
    float timeScale = 1.0f;
    bool trainEnabled = true;
    bool renderEnabled = true;
    bool headlessTurbo = false;
    
    auto last_time = std::chrono::high_resolution_clock::now();
    long long totalSteps = 0;
    float sps = 0;
    int step_counter = 0;

    std::cout << "[JOLTrl] SINGLE-ENV TRAINING MATRIX ENGAGED." << std::endl;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!gCam.active) { gCam.active = true; glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); firstMouse = true; }
        } else {
            if (gCam.active) { gCam.active = false; glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); }
        }
        process_input(window, dt);

        // --- PHYSICS & TRAINING ---
        if (trainEnabled && timeScale > 0.01f) {
            const auto& obs = vecEnv.GetObservations();
            
            AlignedVector32<float> robotActions(numEnvs * 2 * actionDim);
            
            // Generate actions for both robots in the env
            for (int e = 0; e < numEnvs; ++e) {
                float* obs1 = (float*)obs.data() + (e * 2 * stateDim);
                float* obs2 = (float*)obs.data() + (e * 2 * stateDim + stateDim);
                float* act1 = robotActions.data() + (e * 2 * actionDim);
                float* act2 = robotActions.data() + (e * 2 * actionDim + actionDim);
                
                trainer.SelectAction(obs1, act1);
                trainer.SelectAction(obs2, act2);
            }
            
            vecEnv.Step(robotActions);
            
            const auto& nextObs = vecEnv.GetObservations();
            const auto& rewards = vecEnv.GetRewards();
            const auto& dones = vecEnv.GetDones();
            
            // Store transitions
            for (int e = 0; e < numEnvs; ++e) {
                float* o1 = (float*)obs.data() + (e * 2 * stateDim);
                float* no1 = (float*)nextObs.data() + (e * 2 * stateDim);
                float* a1 = robotActions.data() + (e * 2 * actionDim);
                buffer.Add(o1, a1, rewards[e * 2], no1, dones[e]);
                
                float* o2 = (float*)obs.data() + (e * 2 * stateDim + stateDim);
                float* no2 = (float*)nextObs.data() + (e * 2 * stateDim + stateDim);
                float* a2 = robotActions.data() + (e * 2 * actionDim + actionDim);
                buffer.Add(o2, a2, rewards[e * 2 + 1], no2, dones[e]);
            }
            
            if (buffer.Size() > td3cfg.batchSize) {
                trainer.Train(buffer);
            }
            
            // Periodic Save (Every 5 Minutes / 18000 steps)
            if (totalSteps > 0 && totalSteps % 18000 == 0) {
                std::string path = checkpointDir + "/model_step_" + std::to_string(totalSteps) + ".bin";
                trainer.Save(path);
                std::cout << "[JOLTrl] Safe Checkpoint Saved: " << path << std::endl;
            }
            
            vecEnv.ResetDoneEnvs();
            totalSteps += numEnvs;
            step_counter += numEnvs;
        }

        // SPS Calculation
        static auto lastSpsTime = now;
        std::chrono::duration<float> spsElapsed = now - lastSpsTime;
        if (spsElapsed.count() >= 1.0f) {
            sps = step_counter / spsElapsed.count();
            step_counter = 0;
            lastSpsTime = now;
        }

        // --- RENDER ---
        if (renderEnabled && !headlessTurbo) {
            renderer.Draw(vecEnv.GetGlobalPhysics(), gCam.position, 0, gCam.front);
        } else {
            glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        // --- UI ---
        ui.BeginFrame();
        ImGui::Begin("Training Matrix - Overseer");
        ImGui::Text("Silicon: TD3 + SPAN Tech");
        ImGui::Text("Episode Length: 2 Minutes (7200 steps)");
        ImGui::Separator();
        ImGui::Text("Total Steps: %lld", totalSteps);
        ImGui::Text("SPS: %.0f", sps);
        
        ImGui::Checkbox("Enable Training", &trainEnabled);
        ImGui::Checkbox("Enable Rendering", &renderEnabled);
        ImGui::Checkbox("HEADLESS TURBO (No Render/Sync)", &headlessTurbo);
        
        ImGui::SliderFloat("Time Scale", &timeScale, 0.0f, 5.0f);
        
        if (ImGui::CollapsingHeader("Incremental Power Tuner")) {
            auto& r1 = vecEnv.GetEnv(0).GetRobot1Ref();
            auto& r2 = vecEnv.GetEnv(0).GetRobot2Ref();
            ImGui::SliderFloat("Rotation Speed", &r1.actionScale.rotationScale, 0.0f, 500.0f);
            ImGui::SliderFloat("Stab Speed", &r1.actionScale.slideScale, 0.0f, 2000.0f);
            r2.actionScale.rotationScale = r1.actionScale.rotationScale;
            r2.actionScale.slideScale = r1.actionScale.slideScale;
        }

        if (ImGui::CollapsingHeader("Physics Tuner")) {
            JPH::PhysicsSettings settings = vecEnv.GetGlobalPhysics()->GetPhysicsSettings();
            bool changed = false;
            if (ImGui::SliderInt("Vel Steps", (int*)&settings.mNumVelocitySteps, 1, 20)) changed = true;
            if (ImGui::SliderInt("Pos Steps", (int*)&settings.mNumPositionSteps, 1, 20)) changed = true;
            if (ImGui::SliderFloat("Baumgarte", &settings.mBaumgarte, 0.0f, 1.0f)) changed = true;
            if (changed) vecEnv.GetGlobalPhysics()->SetPhysicsSettings(settings);
        }

        if (ImGui::Button("Manual Reset Scene")) vecEnv.Reset();
        
        ImGui::End();
        ui.EndFrame();
        glfwSwapBuffers(window);
        
        // If not turbo, cap framerate to ~60Hz for UI/Render sanity
        if (!headlessTurbo && timeScale <= 1.0f) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    trainer.Save(checkpointDir + "/model_final.bin");
    
    ui.Shutdown();
    glfwTerminate();
    return 0;
}
