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

    // Initializing vectorized environments
    std::cerr << "[main] Creating VectorizedEnv..." << std::endl;
    VectorizedEnv* vecEnv = new VectorizedEnv(numEnvs, ui.GetStepsPerEpisode());
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

    bool trainEnabled = true; // Default to enabled
    bool leaguePlayEnabled = true; // Default to Fictitious Self-Play
    bool renderEnabled = true;
    bool headlessTurbo = false;
    float physicsHz = 120.0f;
    
    auto last_time = std::chrono::high_resolution_clock::now();
    long long totalSteps = 0;
    const long long MAX_STEPS = 100000000;
    bool reachedMaxSteps = false;
    float sps = 0;
    int step_counter = 0;
    int renderEnvIdx = 0;
    int previousRenderEnvIdx = renderEnvIdx;
    float sliderConfirmationTime = 0.0f;
    const float SLIDER_CONFIRMATION_DURATION = 2.0f;
    int r1Wins = 0;
    int r2Wins = 0;
    int mEpisodes = 0;
    float currentRew1 = 0.0f;
    float currentRew2 = 0.0f;
    int currentOpponentIdx = 0;
    VectorReward lastVR1, lastVR2;
    int frameCount = 0;
    int renderSkip = 1;
    

    std::mt19937 leagueRng(std::random_device{}());

    // Fixed: Ensure global cam is used
    gCam.front = glm::normalize(glm::vec3(0, 2, 0) - gCam.position);

    while (!glfwWindowShouldClose(window) && !reachedMaxSteps) {
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

        float timeScale = ui.GetTimeScale();

        // --- BROADCAST ROBOT TUNABLES ---
        {
            const auto& robotTune = ui.GetRobots();
            for (int i = 0; i < vecEnv->GetNumEnvs(); ++i) {
                auto& r1 = vecEnv->GetEnv(i).GetRobot1Ref();
                auto& r2 = vecEnv->GetEnv(i).GetRobot2Ref();
                r1.actionScale.slideScale = robotTune.enginePower;
                r1.actionScale.rotationScale = robotTune.reactionWheelPower;
                r2.actionScale.slideScale = robotTune.enginePower;
                r2.actionScale.rotationScale = robotTune.reactionWheelPower;
            }
        }

        if (!ui.IsPaused()) {
            // [REMOVED] std::cout << "[main] Step start" << std::endl;
            const auto& obs = vecEnv->GetObservations();
            int numEnvs = vecEnv->GetNumEnvs();
            AlignedVector32<float> robotActions(numEnvs * 2 * actionDim);
            
            // [REMOVED] std::cout << "[main] Batching observations" << std::endl;
            // Collect all robot 1 observations and environment indices
            static AlignedVector32<float> obs1Batch;
            static std::vector<int> indices1;
            obs1Batch.resize(numEnvs * stateDim);
            indices1.resize(numEnvs);
            
            for (int i = 0; i < numEnvs; ++i) {
                std::memcpy(obs1Batch.data() + i * stateDim, (float*)obs.data() + (i * 2 * stateDim), stateDim * sizeof(float));
                indices1[i] = i * 2;
            }
            
            // [REMOVED] std::cout << "[main] Selecting actions for agent 1" << std::endl;
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
            
            // [REMOVED] std::cout << "[main] Selecting actions for agent 2" << std::endl;
            if (leaguePlayEnabled) {
                opponentTrainer.SelectActionBatchWithLatent(obs2Batch.data(), robotActions.data() + (numEnvs * actionDim), numEnvs, indices2);
            } else {
                trainer.SelectActionBatchWithLatent(obs2Batch.data(), robotActions.data() + (numEnvs * actionDim), numEnvs, indices2);
            }
            
            // [REMOVED] std::cout << "[main] Queuing actions" << std::endl;
            // PARALLEL: Queue actions for all environments in parallel
            #pragma omp parallel for num_threads(8)
            for (int i = 0; i < numEnvs; ++i) {
                // Adjust indexing for robotActions because we batched them separately
                vecEnv->GetEnv(i).QueueActions(robotActions.data() + (i * actionDim), robotActions.data() + (numEnvs * actionDim + i * actionDim));
            }
            
            // [REMOVED] std::cout << "[main] Physics step" << std::endl;
            PhysicsCore* core = vecEnv->GetPhysicsCore();
            core->GetPhysicsSystem().Update(1.0f / physicsHz * timeScale, 1, core->GetTempAllocator(), core->GetJobSystem());
            
            // [REMOVED] std::cout << "[main] Harvesting states" << std::endl;
            vecEnv->HarvestStates();
            
            // ... (rest of the physics settings)
            {
                const auto& phys = ui.GetPhysics();
                JPH::PhysicsSettings settings;
                settings.mNumVelocitySteps = phys.velocitySteps;
                settings.mNumPositionSteps = phys.positionSteps;
                settings.mBaumgarte = phys.Baumgarte;
                settings.mPenetrationSlop = phys.penetrationSlop;
                settings.mSpeculativeContactDistance = phys.speculativeContactDistance;
                settings.mAllowSleeping = phys.allowSleep;
                core->SetSettings(settings);
                core->GetPhysicsSystem().SetGravity(JPH::Vec3(0.0f, phys.gravityY, 0.0f));
            }
            
            // Use batch observations from VectorizedEnv (already harvested in Step())
            const auto& allObs = vecEnv->GetObservations();
            const auto& allRewards = vecEnv->GetRewards();
            const auto& allDones = vecEnv->GetDones();
            const auto& allVectorRewards = vecEnv->GetVectorRewards();
            
            // Render only one environment to avoid drawing the rest
            currentRew1 = allRewards[renderEnvIdx * 2];
            currentRew2 = allRewards[renderEnvIdx * 2 + 1];
            
            // Add transitions to replay buffer
            for (int i = 0; i < numEnvs; ++i) {
                const float* obs1 = allObs.data() + i * 2 * stateDim;
                const float* obs2 = obs1 + stateDim;
                float r1 = allRewards[i * 2];
                float r2 = allRewards[i * 2 + 1];
                bool done = allDones[i];
                
                // CORRECT action indexing: Robot1 first, then Robot2
                buffer.Add(obs1, robotActions.data() + i * actionDim, r1, obs1, done);
                buffer.Add(obs2, robotActions.data() + numEnvs * actionDim + i * actionDim, r2, obs2, done);
                
                if (done) {
                    // Get vector rewards for UI display
                    const auto& vr = allVectorRewards[i];
                    float dmgDealt = vr.damage_dealt;
                    float dmgTaken = vr.damage_taken;
                    float energy = vr.energy_used;
                    
                    if (i == renderEnvIdx) {
                        ui.PushRewardData(dmgDealt, dmgTaken, 0.0f, energy, dmgDealt - dmgTaken * 0.5f);
                    }
                    
                    mEpisodes++;
                    auto& robot1 = vecEnv->GetEnv(i).GetRobot1();
                    auto& robot2 = vecEnv->GetEnv(i).GetRobot2();
                    if (robot1.hp > robot2.hp) r1Wins++;
                    else if (robot2.hp > robot1.hp) r2Wins++;

                    vecEnv->Reset(i);
                    
                    // League Play
                    if (leaguePlayEnabled && trainer.GetOpponentPool().Size() > 0) {
                        if (trainer.SampleOpponent()) {
                            opponentTrainer.GetModel().GetActor().SetAllWeights(trainer.GetModel().GetActor().GetAllWeights());
                            currentOpponentIdx++;
                            ui.SetOpponentIndex(currentOpponentIdx);
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
            
            if (totalSteps >= MAX_STEPS) {
                reachedMaxSteps = true;
                // [REMOVED] std::cout << "[main] Reached MAX_STEPS = " << MAX_STEPS << ", exiting..." << std::endl;
                break;
            }
            
            totalSteps += 1;
            step_counter += 1;
        }

     // Handle robot configuration and checkpoint folder requests
         if (ui.GetAndClearLoadConfigRequest()) {
             std::string robotType = ui.GetSelectedRobotType();
             std::cout << "[main_train] Loading robot configuration: " << robotType << std::endl;
             
             // Create robot-specific checkpoint directory
             checkpointDir = (home ? std::string(home) : ".") + "/.joltrl/checkpoints/" + robotType;
             EnsureDir(checkpointDir);
             
             // Reset the environment with new robot configuration
             if(vecEnv) { vecEnv->Shutdown(); delete vecEnv; vecEnv = nullptr; }
             vecEnv = new VectorizedEnv(numEnvs, ui.GetStepsPerEpisode());
             std::cerr << "[main] vecEnv->Init start..." << std::endl;
    if (vecEnv) vecEnv->Init(robotConfigPath);
             
             // Initialize fresh brain model
             int stateDim = vecEnv->GetObservationDim();
             int actionDim = vecEnv->GetActionDim();
             TD3Config td3cfg;
             trainer = TD3Trainer(stateDim, actionDim, td3cfg);
             opponentTrainer = TD3Trainer(stateDim, actionDim, td3cfg);
             buffer = ReplayBuffer(td3cfg.bufferSize, stateDim, actionDim);
             
             // Sync opponent to start identical to main agent
             opponentTrainer.GetModel().GetActor().SetAllWeights(trainer.GetModel().GetActor().GetAllWeights());
             
             // Reset stats
             totalSteps = 0;
             mEpisodes = 0;
             r1Wins = 0;
             r2Wins = 0;
             
             // Update telemetry file path
             telemetryPath = checkpointDir + "/telemetry.csv";
             telemetryFile.close();
             telemetryFile.open(telemetryPath, std::ios::trunc);
             if (telemetryFile.is_open()) {
                 telemetryFile << "Step,Tag,Value\n";
             }
             
             std::cout << "[main_train] Robot configuration loaded successfully. Checkpoints will be saved to: " << checkpointDir << std::endl;
         }
         
         std::string newCheckpointFolderName;
         if (ui.GetAndClearCreateCheckpointFolderRequest(newCheckpointFolderName)) {
             std::string newCheckpointDir = checkpointDir + "/" + newCheckpointFolderName;
             EnsureDir(newCheckpointDir);
             std::cout << "[main_train] Created new checkpoint folder: " << newCheckpointDir << std::endl;
         }
         
         std::string saveName;
         if (ui.GetAndClearSaveRequest(saveName)) {
             trainer.Save(checkpointDir + "/" + saveName + ".bin");
         }
         std::string loadName;
         if (ui.GetAndClearLoadRequest(loadName)) {
             trainer.Load(checkpointDir + "/" + loadName + ".bin");
         }

        if (ui.GetAndClearGraphRequest()) {
            std::string cmd = "./micro_board_gui " + telemetryPath + " &";
            std::cout << "[main_train] Current working directory: " << fs::current_path() << std::endl;
            std::cout << "[main_train] Launching 3D Graph: " << cmd << std::endl;
            system(cmd.c_str());
        }

        if (ui.GetManualOverride()) {
            // use zero/random actions instead of policy
        }

        static auto lastSpsTime = now;
        static int totalEnvStepsAccum = 0;
        
        // Count actual environment steps completed (sum across all envs)
        totalEnvStepsAccum += vecEnv->GetNumEnvs();
        
        std::chrono::duration<float> spsElapsed = now - lastSpsTime;
        if (spsElapsed.count() >= 1.0f) { 
            // SPS = total environment steps / elapsed time
            // This counts actual steps across ALL environments, not multiplication
            sps = static_cast<float>(totalEnvStepsAccum) / spsElapsed.count(); 
            totalEnvStepsAccum = 0;
            lastSpsTime = now; 
            
            // CSV Telemetry output for micro_board
            if (!ui.IsPaused()) {
                std::cout << totalSteps << ",SPS," << sps << "\n";
                std::cout << totalSteps << ",Reward1," << currentRew1 << "\n";
                std::cout << totalSteps << ",Reward2," << currentRew2 << "\n";
                std::cout << totalSteps << ",Buffer_Size," << buffer.Size() << "\n";

                if (telemetryFile.is_open()) {
                    telemetryFile << totalSteps << ",SPS," << sps << "\n";
                    telemetryFile << totalSteps << ",Reward1," << currentRew1 << "\n";
                    telemetryFile << totalSteps << ",Reward2," << currentRew2 << "\n";
                    telemetryFile << totalSteps << ",Buffer_Size," << buffer.Size() << "\n";
                    telemetryFile.flush();
                }
            }
        }

        // Update HP display - use actual HP from rendered environment
        if (!vecEnv) continue;
        auto& envHP = vecEnv->GetEnv(renderEnvIdx);
        float hp1 = envHP.GetRobot1().hp;
        float hp2 = envHP.GetRobot2().hp;
        
        // Clamp HP to valid range to prevent display issues
        hp1 = std::max(0.0f, std::min(100.0f, hp1));
        hp2 = std::max(0.0f, std::min(100.0f, hp2));
        ui.UpdateAgentHP(hp1, hp2);

        // Update UI stats every frame - compute average reward from currentRew (render env only)
        // For true multi-env avg reward, would need to accumulate during step processing
        float avgReward = (currentRew1 + currentRew2) * 0.5f;
        ui.UpdateStats(totalSteps, mEpisodes, sps, avgReward, renderEnvIdx, vecEnv ? vecEnv->GetNumEnvs() : 0);

        // Check for restart request from UI
        if (ui.ShouldRestartSim()) {
            // [REMOVED] std::cout << "[main] CRITICAL: Sim restart requested." << std::endl;
            if (vecEnv) {
                // [REMOVED] std::cout << "[main] Deleting vecEnv..." << std::endl;
                delete vecEnv;
                vecEnv = nullptr;
            }
            int newNumEnvs = ui.GetConfig().numEnvs;
            int newSteps = ui.GetStepsPerEpisode();
            std::string newRobotPath = ui.GetConfig().robotConfigPath;
            
            // [REMOVED] std::cout << "[main] Creating new vecEnv (" << newNumEnvs << " envs) with " << newRobotPath << "..." << std::endl;
            numEnvs = newNumEnvs;
            vecEnv = new VectorizedEnv(numEnvs, newSteps);
            vecEnv->Init(newRobotPath);            
            
            // Re-initialize trainer dimensions
            stateDim = vecEnv->GetObservationDim();
            actionDim = vecEnv->GetActionDim();
            robotName = vecEnv->GetEnv(0).GetRobot1Ref().config.name;
            robotCheckpointDir = checkpointDir + "/" + robotName;
            EnsureDir(robotCheckpointDir);
            
            TD3Config td3cfg;
            trainer = TD3Trainer(stateDim, actionDim, td3cfg);
            opponentTrainer = TD3Trainer(stateDim, actionDim, td3cfg);
            buffer = ReplayBuffer(td3cfg.bufferSize, stateDim, actionDim);
            
            std::string modelPath = robotCheckpointDir + "/model_final.bin";
            if (fs::exists(modelPath)) {
                try { trainer.Load(modelPath); } catch(...) {}
            }

            totalSteps = 0;
            mEpisodes = 0;
            // [REMOVED] std::cout << "[main] Restart successful. New Dims: " << stateDim << "x" << actionDim << std::endl;
            ui.ClearRestartRequest();
        }

        // Update per-agent rewards for UI display
        ui.UpdateAgentRewards(currentRew1, currentRew2);

        // Apply graphics settings from UI to renderer
        const auto& graphics = ui.GetGraphics();

        // Render skip optimization: only render every N frames for max SPS
        frameCount++;
        bool shouldRender = renderEnabled && !headlessTurbo && (frameCount % renderSkip == 0);
        
        if (shouldRender) {
            gRenderer->Draw(vecEnv->GetPhysicsCore(), gCam.position, renderEnvIdx, gCam.front, glm::vec3(0.0f, 1.0f, 0.0f),
                            graphics.showCollisionShapes, graphics.showAABBs, graphics.showContactPoints,
                            graphics.showRobot1, graphics.showRobot2);
        } else {
            glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        // Draw debug visualizations if enabled
        if (graphics.showCollisionShapes || graphics.showAABBs || graphics.showContactPoints) {
            // These would need to be implemented in Renderer
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
