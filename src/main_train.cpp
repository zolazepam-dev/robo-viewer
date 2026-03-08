// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <random>
#include <filesystem>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <omp.h>

#include "src/VectorizedEnv.h"
#include "src/NeuralNetwork.h"
#include "src/TD3Trainer.h"
#include "src/Renderer.h"
#include "src/OverlayUI_refactor.h"
#include "src/EigenUtils.h"
#include "VisualState.h"

// TRIPLE BUFFERING STATE
std::vector<EnvVisualState> gVisualBuffers[3];
std::atomic<int> gWriteBufferIdx{0};
std::atomic<int> gReadBufferIdx{1};
std::atomic<int> gIntermediateBufferIdx{2};
std::atomic<bool> gNewFrameReady{false};

// GLOBAL STATE FOR DECOUPLING
std::atomic<bool> gSimRunning{true};
std::atomic<bool> gSimPaused{false};
std::atomic<float> gSPS{0.0f};
std::atomic<long long> gTotalSteps{0};
std::atomic<int> gEpisodes{0};
std::atomic<float> gAvgReward{0.0f};
std::atomic<float> gAgent1HP{100.0f};
std::atomic<float> gAgent2HP{100.0f};
std::atomic<float> gAgent1Reward{0.0f};
std::atomic<float> gAgent2Reward{0.0f};
std::mutex gSimMutex;

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
Renderer* gRenderer = nullptr;
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

void window_size_callback(GLFWwindow* window, int width, int height) {
    if (gRenderer) {
        glViewport(0, 0, width, height);
        gRenderer->Resize(width, height);
    }
}

void SimulationLoop(VectorizedEnv* vecEnv, TD3Trainer* trainer, TD3Trainer* opponentTrainer, ReplayBuffer* buffer, OverlayUIRefactored* ui, int stateDim, int actionDim)
{
    long long localSteps = 0;
    int mEpisodes = 0;
    float currentRew1 = 0.0f;
    float currentRew2 = 0.0f;
    bool leaguePlayEnabled = true;
    int currentOpponentIdx = 0;
    
    auto lastSpsTime = std::chrono::high_resolution_clock::now();
    int totalEnvStepsAccum = 0;

    AlignedVector32<float> robotActions(vecEnv->GetNumEnvs() * 2 * actionDim);
    AlignedVector32<float> prevObs(vecEnv->GetNumEnvs() * stateDim * 2, 0.0f);
    bool firstStep = true;

    while (gSimRunning) {
        if (gSimPaused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        int numEnvs = vecEnv->GetNumEnvs();
        
        // 1. Select Actions (No lock needed for trainer/buffer)
        const auto& obs = vecEnv->GetObservations();
        static AlignedVector32<float> obs1Batch;
        static std::vector<int> indices1;
        obs1Batch.resize(numEnvs * stateDim);
        indices1.resize(numEnvs);
        for (int i = 0; i < numEnvs; ++i) {
            std::memcpy(obs1Batch.data() + i * stateDim, (float*)obs.data() + (i * 2 * stateDim), stateDim * sizeof(float));
            indices1[i] = i * 2;
        }
        trainer->SelectActionBatchWithLatent(obs1Batch.data(), robotActions.data(), numEnvs, indices1);
        
        static AlignedVector32<float> obs2Batch;
        static std::vector<int> indices2;
        obs2Batch.resize(numEnvs * stateDim);
        indices2.resize(numEnvs);
        for (int i = 0; i < numEnvs; ++i) {
            std::memcpy(obs2Batch.data() + i * stateDim, (float*)obs.data() + (i * 2 * stateDim + stateDim), stateDim * sizeof(float));
            indices2[i] = i * 2 + 1;
        }
        
        if (leaguePlayEnabled) {
            opponentTrainer->SelectActionBatchWithLatent(obs2Batch.data(), robotActions.data() + (numEnvs * actionDim), numEnvs, indices2);
        } else {
            trainer->SelectActionBatchWithLatent(obs2Batch.data(), robotActions.data() + (numEnvs * actionDim), numEnvs, indices2);
        }
        
        // 2. Queue Actions (Completely outside lock - individual envs are thread-safe for this)
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < numEnvs; ++i) {
            vecEnv->GetEnv(i).QueueActions(robotActions.data() + (i * actionDim), robotActions.data() + (numEnvs * actionDim + i * actionDim));
        }
        
        {
            std::lock_guard<std::mutex> lock(gSimMutex);
            // 3. Physics Step
            PhysicsCore* core = vecEnv->GetPhysicsCore();
            float physicsHz = 120.0f;
            core->GetPhysicsSystem().Update(1.0f / physicsHz, 1, core->GetTempAllocator(), core->GetJobSystem());
            
            // 4. Harvest States
            vecEnv->HarvestStates();
        }

        // 4.1 Update Visual Triple Buffer (Outside lock for speed)
        int writeIdx = gWriteBufferIdx.load();
        if (gVisualBuffers[writeIdx].size() != (size_t)numEnvs) gVisualBuffers[writeIdx].resize(numEnvs);
        
        PhysicsCore* physCore = vecEnv->GetPhysicsCore();
        auto& bi = physCore->GetPhysicsSystem().GetBodyInterface();

        for (int i = 0; i < numEnvs; ++i) {
            auto& env = vecEnv->GetEnv(i);
            auto& r1 = env.GetRobot1();
            auto& r2 = env.GetRobot2();
            
            if (r1.IsValid()) {
                auto p = bi.GetPosition(r1.mainBodyId);
                auto q = bi.GetRotation(r1.mainBodyId);
                gVisualBuffers[writeIdx][i].r1.x = p.GetX();
                gVisualBuffers[writeIdx][i].r1.y = p.GetY();
                gVisualBuffers[writeIdx][i].r1.z = p.GetZ();
                gVisualBuffers[writeIdx][i].r1.rx = q.GetX();
                gVisualBuffers[writeIdx][i].r1.ry = q.GetY();
                gVisualBuffers[writeIdx][i].r1.rz = q.GetZ();
                gVisualBuffers[writeIdx][i].r1.rw = q.GetW();
                gVisualBuffers[writeIdx][i].r1.hp = r1.hp;
            }

            if (r2.IsValid()) {
                auto p = bi.GetPosition(r2.mainBodyId);
                auto q = bi.GetRotation(r2.mainBodyId);
                gVisualBuffers[writeIdx][i].r2.x = p.GetX();
                gVisualBuffers[writeIdx][i].r2.y = p.GetY();
                gVisualBuffers[writeIdx][i].r2.z = p.GetZ();
                gVisualBuffers[writeIdx][i].r2.rx = q.GetX();
                gVisualBuffers[writeIdx][i].r2.ry = q.GetY();
                gVisualBuffers[writeIdx][i].r2.rz = q.GetZ();
                gVisualBuffers[writeIdx][i].r2.rw = q.GetW();
                gVisualBuffers[writeIdx][i].r2.hp = r2.hp;
            }
        }
        
        // SWAP Write and Intermediate
        int oldIntermediate = gIntermediateBufferIdx.exchange(writeIdx);
        gWriteBufferIdx.store(oldIntermediate);
        gNewFrameReady = true;

        const auto& allObs = vecEnv->GetObservations();
        const auto& allRewards = vecEnv->GetRewards();
        const auto& allDones = vecEnv->GetDones();

        // 5. Add transitions to buffer - PARALLEL
        if (!firstStep) {
            #pragma omp parallel for num_threads(8)
            for (int i = 0; i < numEnvs; ++i) {
                const float* s1 = prevObs.data() + i * 2 * stateDim;
                const float* s1_next = allObs.data() + i * 2 * stateDim;
                const float* s2 = s1 + stateDim;
                const float* s2_next = s1_next + stateDim;
                float r1 = allRewards[i * 2];
                float r2 = allRewards[i * 2 + 1];
                bool done = allDones[i];
                
                buffer->Add(s1, robotActions.data() + i * actionDim, r1, s1_next, done);
                buffer->Add(s2, robotActions.data() + numEnvs * actionDim + i * actionDim, r2, s2_next, done);
                
                if (done) {
                    mEpisodes++;
                    {
                        std::lock_guard<std::mutex> lock(gSimMutex);
                        vecEnv->Reset(i);
                    }
                }
            }
        }
        std::memcpy(prevObs.data(), allObs.data(), allObs.size() * sizeof(float));
        firstStep = false;

        // 6. Train (Reduced frequency for MAX throughput)
        if (localSteps % 16 == 0 && buffer->Size() >= 256) {
            for (int u = 0; u < 2; ++u) trainer->Train(*buffer);
        }
        
        // 7. Stats Update
        currentRew1 = allRewards[0];
        currentRew2 = allRewards[1];
        if (numEnvs > 0) {
            // We can read HP without lock if we're careful, 
            // but for stats it's fine even if slightly jittery
            auto& envHP = vecEnv->GetEnv(0);
            gAgent1HP = envHP.GetRobot1().hp;
            gAgent2HP = envHP.GetRobot2().hp;
        }

        localSteps++;
        gTotalSteps = localSteps;
        gEpisodes = mEpisodes;
        gAgent1Reward = currentRew1;
        gAgent2Reward = currentRew2;
        gAvgReward = (currentRew1 + currentRew2) * 0.5f;
        
        totalEnvStepsAccum += numEnvs;
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = now - lastSpsTime;
        if (elapsed.count() >= 1.0f) {
            gSPS = totalEnvStepsAccum / elapsed.count();
            totalEnvStepsAccum = 0;
            lastSpsTime = now;
        }
    }
}

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
    glfwSetWindowSizeCallback(window, window_size_callback);
    glEnable(GL_DEPTH_TEST);
    glfwSetCursorPosCallback(window, mouse_callback);

    std::cerr << "[main] Creating UI..." << std::endl;
    OverlayUIRefactored ui;
    std::cerr << "[main] UI Init start..." << std::endl;
    ui.Init(window);
    ui.LoadSettings(); // Force load from disk now

    // Command line overrides config, otherwise use config value
    int targetNumEnvs = ui.GetConfig().numEnvs;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--envs" && i + 1 < argc) {
            targetNumEnvs = std::stoi(argv[++i]);
        }
    }

    // Initializing vectorized environments
    std::cerr << "[main] Creating VectorizedEnv with " << targetNumEnvs << " envs..." << std::endl;
    VectorizedEnv* vecEnv = new VectorizedEnv(targetNumEnvs, ui.GetStepsPerEpisode());
    std::string robotConfigPath = ui.GetConfig().robotConfigPath;
    std::cerr << "[main] vecEnv->Init start with " << robotConfigPath << "..." << std::endl;
    if (vecEnv) vecEnv->Init(robotConfigPath);

    // DYNAMIC DIMENSIONS & ROBOT HANDLING
    int stateDim = vecEnv->GetObservationDim();
    int actionDim = vecEnv->GetActionDim();
    
    std::string robotName = vecEnv->GetEnv(0).GetRobot1().config.name;
    if (robotName.empty()) robotName = "unknown_bot";
    
    std::string robotCheckpointDir = checkpointDir + "/" + robotName;
    EnsureDir(robotCheckpointDir);

    // [REMOVED] std::cout << "[main] Robot: " << robotName << " (Dims: " << stateDim << "x" << actionDim << ")" << std::endl;
    // [REMOVED] std::cout << "[main] Checkpoint Dir: " << robotCheckpointDir << std::endl;

    // Prepare Telemetry File for micro_board
    std::string telemetryPath = robotCheckpointDir + "/telemetry.csv";
    std::ofstream telemetryFile(telemetryPath, std::ios::trunc);
    if (telemetryFile.is_open()) {
        telemetryFile << "Step,Tag,Value\n"; // Header
    }
    
    gRenderer = new Renderer(1280, 720);
    
    TD3Config td3cfg;
    TD3Trainer trainer(stateDim, actionDim, td3cfg);
    TD3Trainer opponentTrainer(stateDim, actionDim, td3cfg); // League Play opponent
    ReplayBuffer buffer(td3cfg.bufferSize, stateDim, actionDim);
    
    // Auto-load weights if available
    std::string finalModelPath = robotCheckpointDir + "/model_final.bin";
    if (fs::exists(finalModelPath)) {
        try {
            trainer.Load(finalModelPath);
            std::cout << "[main_train] Auto-loaded weights from: " << finalModelPath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[main_train] Failed to load checkpoint (likely architecture mismatch): " << e.what() << std::endl;
            std::cerr << "[main_train] Starting with fresh weights." << std::endl;
        }
    }
    
    // Sync opponent to start identical to main agent
    opponentTrainer.GetModel().GetActor().SetAllWeights(trainer.GetModel().GetActor().GetAllWeights());

    // LAUNCH SIMULATION THREAD
    std::thread simThread(SimulationLoop, vecEnv, &trainer, &opponentTrainer, &buffer, &ui, stateDim, actionDim);

    auto last_time = std::chrono::high_resolution_clock::now();
    int renderEnvIdx = 0;
    int frameCount = 0;
    int renderSkip = 1;

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

        // SYNC UI STATE WITH ATOMICS
        gSimPaused = ui.IsPaused();
        ui.UpdateStats((int)gTotalSteps, gEpisodes, gSPS, gAvgReward, renderEnvIdx, vecEnv->GetNumEnvs());
        
        // TRIPLE BUFFER SWAP (Read <-> Intermediate)
        if (gNewFrameReady.exchange(false)) {
            int oldRead = gReadBufferIdx.load();
            int newRead = gIntermediateBufferIdx.exchange(oldRead);
            gReadBufferIdx.store(newRead);
        }

        int readIdx = gReadBufferIdx.load();
        if (readIdx < 3 && !gVisualBuffers[readIdx].empty() && renderEnvIdx < (int)gVisualBuffers[readIdx].size()) {
            const auto& visual = gVisualBuffers[readIdx][renderEnvIdx];
            ui.UpdateAgentHP(visual.r1.hp, visual.r2.hp);
            
            // DRAW using buffered state
            const auto& graphics = ui.GetGraphics();
            // Need to update Renderer::Draw to take buffer or use core only for static geometry
            gRenderer->Draw(vecEnv->GetPhysicsCore(), gCam.position, renderEnvIdx, gCam.front, glm::vec3(0.0f, 1.0f, 0.0f),
                            graphics.showCollisionShapes, graphics.showAABBs, graphics.showContactPoints,
                            graphics.showRobot1, graphics.showRobot2, &visual);
        } else {
            // Fallback for static world if no buffer yet
            gRenderer->Draw(vecEnv->GetPhysicsCore(), gCam.position, renderEnvIdx, gCam.front, glm::vec3(0.0f, 1.0f, 0.0f),
                            false, false, false, false, false);
        }

        // Handle restart/config requests (Stop thread, reinit, restart thread)
        if (ui.ShouldRestartSim()) {
            gSimRunning = false;
            if (simThread.joinable()) simThread.join();
            
            {
                std::lock_guard<std::mutex> lock(gSimMutex);
                delete vecEnv;
                int newNumEnvs = ui.GetConfig().numEnvs;
                vecEnv = new VectorizedEnv(newNumEnvs, ui.GetStepsPerEpisode());
                vecEnv->Init(ui.GetConfig().robotConfigPath);
                
                // Re-init trainer if dims changed? (Keeping current for now)
                trainer.GetModel().UpdateTargets(1.0f);
            }
            
            gSimRunning = true;
            simThread = std::thread(SimulationLoop, vecEnv, &trainer, &opponentTrainer, &buffer, &ui, stateDim, actionDim);
            ui.ClearRestartRequest();
        }

        ui.NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(450, 700), ImGuiCond_FirstUseEver);
        ImGui::Begin("JOLTrl Control Center", nullptr, ImGuiWindowFlags_NoCollapse);
        ui.DrawAllTabs();
        ImGui::End();
        ui.Render();

        glfwSwapBuffers(window);
    }

    gSimRunning = false;
    if (simThread.joinable()) simThread.join();

    std::string finalRobotPath = "robots/" + robotName + ".json";
    int numSatellites = vecEnv->GetEnv(0).GetRobot1().config.numSatellites;
    trainer.Save(robotCheckpointDir + "/model_final.bin", finalRobotPath, numSatellites, stateDim);
    
    // [REMOVED] std::cout << "[main] Final cleanup start..." << std::endl;
    if (gRenderer) {
        delete gRenderer;
        gRenderer = nullptr;
    }
    
    if (vecEnv) {
        delete vecEnv;
        vecEnv = nullptr;
    }
    
    ui.Shutdown();
    glfwTerminate();
    // [REMOVED] std::cout << "[main] Cleanup complete. Exiting." << std::endl;
    return 0;
}
