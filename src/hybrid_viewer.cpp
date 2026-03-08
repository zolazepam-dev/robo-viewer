// Hybrid Viewer - 1 environment visualized, rest headless parallel training
// Supports runtime restart with new environment count
// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <filesystem>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdlib>
#include <cmath>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "src/VectorizedEnv.h"
#include "src/NeuralNetwork.h"
#include "src/TD3Trainer.h"
#include "src/Renderer.h"
#include "src/OverlayUI_refactor.h"

using namespace std;
namespace fs = std::filesystem;

#include "src/VectorizedEnv.h"
#include "src/NeuralNetwork.h"
#include "src/TD3Trainer.h"
#include "src/Renderer.h"
#include "src/OverlayUI_refactor.h"

struct FreeCamera {
    glm::vec3 position{0.0f, 20.0f, 50.0f};
    glm::vec3 front{0.0f, -0.3f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float yaw = -90.0f;
    float pitch = -15.0f;
    float speed = 30.0f;
    float sensitivity = 0.1f;
    bool active = false;
};

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
    gCam.pitch = glm::clamp(gCam.pitch, -89.0f, 89.0f);
    glm::vec3 dir;
    dir.x = glm::cos(glm::radians(gCam.yaw)) * glm::cos(glm::radians(gCam.pitch));
    dir.y = glm::sin(glm::radians(gCam.pitch));
    dir.z = glm::sin(glm::radians(gCam.yaw)) * glm::cos(glm::radians(gCam.pitch));
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
    int numEnvs = 64;
    int renderEnvIdx = 0;
    std::string robotConfigPath = "robots/bouncy_orbiter.json";  // Default to bouncy orbiter

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--envs" && i + 1 < argc) {
            numEnvs = std::stoi(argv[++i]);
        } else if (arg == "--render-idx" && i + 1 < argc) {
            renderEnvIdx = std::stoi(argv[++i]);
        } else if (arg == "--robot-config" && i + 1 < argc) {
            robotConfigPath = argv[++i];
        }
    }
    
    std::cout << "[Hybrid Viewer] Using robot config: " << robotConfigPath << std::endl;

    const char* home = std::getenv("HOME");
    std::string checkpointDir = (home ? std::string(home) : ".") + "/.joltrl/checkpoints";
    EnsureDir(checkpointDir);

    if (!glfwInit()) return -1;
    std::string title = "JOLTrl - Hybrid (1+ " + std::to_string(numEnvs-1) + ")";
    GLFWwindow* window = glfwCreateWindow(1280, 720, title.c_str(), nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    glewInit();
    glfwSetWindowSizeCallback(window, window_size_callback);
    glEnable(GL_DEPTH_TEST);
    glfwSetCursorPosCallback(window, mouse_callback);

    TD3Config td3cfg;
    
    // Massive network for exhaustive observations
    td3cfg.hiddenDim = 512;     // Huge hidden layer
    td3cfg.latentDim = 128;     // Large latent memory (ODE)
    td3cfg.actorLR = 1e-4f;    // Slower learning for bigger network
    td3cfg.criticLR = 1e-4f;
    td3cfg.batchSize = 256;    // Bigger batch for stable training
    
    std::cout << "[Hybrid Viewer] Network config: hidden=" << td3cfg.hiddenDim 
              << " latent=" << td3cfg.latentDim 
              << " batch=" << td3cfg.batchSize << std::endl;
    float physicsHz = 120.0f;
    long long totalSteps = 0;
    const long long MAX_STEPS = 100000;
    bool reachedMaxSteps = false;
    float sps = 0;
    int mEpisodes = 0;
    int r1Wins = 0, r2Wins = 0;
    int currentOpponentIdx = 0;

    gCam.front = glm::normalize(glm::vec3(0, -5, 0) - gCam.position);
    gCam.yaw = 0.0f;
    gCam.pitch = -20.0f;

    std::ofstream telemetryFile;
    std::string telemetryPath;

    std::cout << "[Hybrid Viewer] Starting with " << numEnvs << " environments\n";
    std::cout << "[Hybrid Viewer] Rendering env index: " << renderEnvIdx << " (rest headless)\n";

    // Initialize UI
    OverlayUIRefactored ui;
    ui.Init(window);
    
    // Check if user changed env count in config
    if (ui.GetConfig().numEnvs != numEnvs && ui.GetConfig().numEnvs > 0) {
        numEnvs = ui.GetConfig().numEnvs;
        std::cout << "[Hybrid Viewer] Updated env count from config: " << numEnvs << std::endl;
    }

    // Initialize environment
    VectorizedEnv* vecEnv = new VectorizedEnv(numEnvs, ui.GetStepsPerEpisode());
    vecEnv->Init(robotConfigPath);

        telemetryPath = checkpointDir + "/telemetry.csv";
        telemetryFile.open(telemetryPath, std::ios::trunc);
        if (telemetryFile.is_open()) {
            telemetryFile << "Step,Tag,Value\n";
        }

        gRenderer = new Renderer(1280, 720);
        int stateDim = vecEnv->GetObservationDim();
        int actionDim = vecEnv->GetActionDim();

        TD3Trainer* trainer = new TD3Trainer(stateDim, actionDim, td3cfg);
        TD3Trainer* opponentTrainer = new TD3Trainer(stateDim, actionDim, td3cfg);
        ReplayBuffer* buffer = new ReplayBuffer(td3cfg.bufferSize, stateDim, actionDim);
        
        // Fast mode disabled - causes memory corruption during weight copies
        // trainer->SetFastMode(true);
        // opponentTrainer->SetFastMode(true);
        std::cout << "[Hybrid Viewer] Using B-spline networks (fast mode disabled due to memory issues)" << std::endl;

        std::string finalModelPath = checkpointDir + "/model_final.bin";
        if (fs::exists(finalModelPath)) {
            try {
                trainer->Load(finalModelPath);
                std::cout << "[Hybrid Viewer] Auto-loaded weights from: " << finalModelPath << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[Hybrid Viewer] Warning: Could not load model (dimension mismatch). Starting fresh training. " << e.what() << std::endl;
            }
        } else {
            std::cout << "[Hybrid Viewer] No existing model found. Starting fresh training." << std::endl;
        }

        opponentTrainer->GetModel().GetActor().SetAllWeights(trainer->GetModel().GetActor().GetAllWeights());

        bool leaguePlayEnabled = true;
        auto last_time = std::chrono::high_resolution_clock::now();

        // Warmup steps
        std::cout << "[Hybrid Viewer] Running warmup steps..." << std::endl;
        {
            AlignedVector32<float> warmupActions(numEnvs * 2 * actionDim, 0.0f);
            PhysicsCore* core = vecEnv->GetPhysicsCore();
            for (int warmup = 0; warmup < 5; warmup++) {
                for (int i = 0; i < numEnvs; ++i) {
                    vecEnv->GetEnv(i).QueueActions(warmupActions.data() + (i * actionDim),
                                                    warmupActions.data() + (numEnvs * actionDim + i * actionDim));
                }
                core->GetPhysicsSystem().Update(1.0f / physicsHz, 1, core->GetTempAllocator(), core->GetJobSystem());
                vecEnv->Step(warmupActions);
            }
        }
        std::cout << "[Hybrid Viewer] Warmup completed, starting main loop..." << std::endl;

        while (!glfwWindowShouldClose(window) && !reachedMaxSteps) {
            // Check for restart request - HOT RESTART without closing window
            if (ui.ShouldRestartSim()) {
                int newNumEnvs = ui.GetConfig().numEnvs;
                std::cout << "[Hybrid Viewer] HOT RESTART with " << newNumEnvs << " envs (keeping window open)" << std::endl;
                
                // Save model before restart
                trainer->Save(checkpointDir + "/model_restart.bin");
                std::cout << "[Hybrid Viewer] Model saved to model_restart.bin" << std::endl;
                
                // Cleanup old environment (but keep window and renderer)
                vecEnv->Shutdown();
                delete vecEnv;
                
                // Create new environment with more/fewer environments
                vecEnv = new VectorizedEnv(newNumEnvs, ui.GetStepsPerEpisode());
                vecEnv->Init(robotConfigPath);
                std::cout << "[Hybrid Viewer] New VectorizedEnv created with " << newNumEnvs << " environments" << std::endl;
                
                // Update dims and recreate trainers/buffers
                stateDim = vecEnv->GetObservationDim();
                actionDim = vecEnv->GetActionDim();
                
                delete trainer;
                delete opponentTrainer;
                delete buffer;
                
                trainer = new TD3Trainer(stateDim, actionDim, td3cfg);
                opponentTrainer = new TD3Trainer(stateDim, actionDim, td3cfg);
                buffer = new ReplayBuffer(td3cfg.bufferSize, stateDim, actionDim);
                
                // Try to load the saved model
                if (fs::exists(checkpointDir + "/model_restart.bin")) {
                    try {
                        trainer->Load(checkpointDir + "/model_restart.bin");
                        opponentTrainer->GetModel().GetActor().SetAllWeights(trainer->GetModel().GetActor().GetAllWeights());
                        std::cout << "[Hybrid Viewer] Loaded model from restart checkpoint" << std::endl;
                    } catch (const std::exception& e) {
                        std::cout << "[Hybrid Viewer] Could not load restart model: " << e.what() << std::endl;
                    }
                }
                
                // Reset counters
                totalSteps = 0;
                reachedMaxSteps = false;
                mEpisodes = 0;
                r1Wins = 0;
                r2Wins = 0;
                
                // Clear restart request
                ui.ClearRestartRequest();
                
                // Re-warmup
                std::cout << "[Hybrid Viewer] Running warmup after hot restart..." << std::endl;
                {
                    AlignedVector32<float> warmupActions(newNumEnvs * 2 * actionDim, 0.0f);
                    for (int warmup = 0; warmup < 5; warmup++) {
                        vecEnv->Step(warmupActions);
                    }
                }
                std::cout << "[Hybrid Viewer] Hot restart complete!" << std::endl;
            }

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
                const auto& obs = vecEnv->GetObservations();
                int numEnvs = vecEnv->GetNumEnvs();
                AlignedVector32<float> robotActions(numEnvs * 2 * actionDim);

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

                for (int i = 0; i < numEnvs; ++i) {
                    vecEnv->GetEnv(i).QueueActions(robotActions.data() + (i * actionDim), robotActions.data() + (numEnvs * actionDim + i * actionDim));
                }

                PhysicsCore* core = vecEnv->GetPhysicsCore();
                core->GetPhysicsSystem().Update(1.0f / physicsHz * timeScale, 1, core->GetTempAllocator(), core->GetJobSystem());

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

                const auto& allObs = vecEnv->GetObservations();
                const auto& allRewards = vecEnv->GetRewards();
                const auto& allDones = vecEnv->GetDones();
                const auto& allVectorRewards = vecEnv->GetVectorRewards();

                for (int i = 0; i < numEnvs; ++i) {
                    const float* obs1 = allObs.data() + i * 2 * stateDim;
                    const float* obs2 = obs1 + stateDim;
                    float r1 = allRewards[i * 2];
                    float r2 = allRewards[i * 2 + 1];
                    bool done = allDones[i];

                    buffer->Add(obs1, robotActions.data() + i * actionDim, r1, obs1, done);
                    buffer->Add(obs2, robotActions.data() + numEnvs * actionDim + i * actionDim, r2, obs2, done);

                    if (done) {
                        const auto& vr = allVectorRewards[i];
                        if (i == renderEnvIdx) {
                            ui.PushRewardData(vr.damage_dealt, vr.damage_taken, 0.0f, vr.energy_used, vr.damage_dealt - vr.damage_taken * 0.5f);
                        }

                        mEpisodes++;
                        auto& robot1 = vecEnv->GetEnv(i).GetRobot1();
                        auto& robot2 = vecEnv->GetEnv(i).GetRobot2();
                        if (robot1.hp > robot2.hp) r1Wins++;
                        else if (robot2.hp > robot1.hp) r2Wins++;

                        vecEnv->Reset(i);

                        if (leaguePlayEnabled && trainer->GetOpponentPool().Size() > 0) {
                            if (trainer->SampleOpponent()) {
                                opponentTrainer->GetModel().GetActor().SetAllWeights(trainer->GetModel().GetActor().GetAllWeights());
                                currentOpponentIdx++;
                                ui.SetOpponentIndex(currentOpponentIdx);
                            }
                        }
                    }
                }

                if (totalSteps % 4 == 0 && buffer->Size() > td3cfg.batchSize) {
                    for (int update = 0; update < 2; ++update) {
                        trainer->Train(*buffer);
                    }
                }

                totalSteps++;
            }

            // Print progress every 100 steps
            if (totalSteps % 100 == 0) {
                std::cout << "[Hybrid Viewer] Step " << totalSteps << " / " << MAX_STEPS << std::endl;
            }
            
            std::cout << "[HYBRID] About to RENDER..." << std::endl;

            // RENDER
            const auto& graphics = ui.GetGraphics();
            std::cout << "[HYBRID] Calling gRenderer->Draw..." << std::endl;
            if (gRenderer && vecEnv) {
                gRenderer->Draw(vecEnv->GetPhysicsCore(), gCam.position, renderEnvIdx, gCam.front, gCam.up,
                                graphics.showCollisionShapes, graphics.showAABBs, graphics.showContactPoints,
                                graphics.showRobot1, graphics.showRobot2);
            }
            std::cout << "[HYBRID] Draw COMPLETE, calling ui.NewFrame..." << std::endl;

            // UI Overlay
            ui.NewFrame();
            ui.UpdateStats(totalSteps, mEpisodes, sps, 0.0f, renderEnvIdx, numEnvs);

            if (vecEnv) {
                const auto& allRewards = vecEnv->GetRewards();
                if (renderEnvIdx < numEnvs) {
                    CombatEnv& renderEnv = vecEnv->GetEnv(renderEnvIdx);
                    ui.UpdateAgentRewards(allRewards[renderEnvIdx * 2], allRewards[renderEnvIdx * 2 + 1]);
                    ui.UpdateAgentHP(renderEnv.GetRobot1().hp, renderEnv.GetRobot2().hp);
                }
            }

            ui.DrawAllTabs();

            if (ui.ShouldReset()) {
                std::cout << "[Hybrid Viewer] Reset requested" << std::endl;
            }

            if (ui.ShouldStepOne()) {
                // Step one frame
            }

            std::string saveName;
            if (ui.GetAndClearSaveRequest(saveName)) {
                if (trainer) {
                    trainer->Save(checkpointDir + "/" + saveName + ".bin");
                    std::cout << "[Hybrid Viewer] Saved: " << saveName << std::endl;
                }
            }

            std::string loadName;
            if (ui.GetAndClearLoadRequest(loadName)) {
                if (trainer) {
                    trainer->Load(checkpointDir + "/" + loadName + ".bin");
                    std::cout << "[Hybrid Viewer] Loaded: " << loadName << std::endl;
                }
            }

            if (ui.GetAndClearGraphRequest()) {
                std::string cmd = "./micro_board_gui " + telemetryPath + " &";
                system(cmd.c_str());
            }

            // Cycle environments with arrow keys
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
                renderEnvIdx = (renderEnvIdx + 1) % numEnvs;
                glfwWaitEventsTimeout(0.3);
                std::cout << "[Hybrid Viewer] Now rendering env #" << renderEnvIdx << std::endl;
            }
            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
                renderEnvIdx = (renderEnvIdx - 1 + numEnvs) % numEnvs;
                glfwWaitEventsTimeout(0.3);
                std::cout << "[Hybrid Viewer] Now rendering env #" << renderEnvIdx << std::endl;
            }

            ui.Render();

            auto now2 = std::chrono::high_resolution_clock::now();
            float dt2 = std::chrono::duration<float>(now2 - now).count();
            if (dt2 > 0) sps = (1.0f / dt2) * numEnvs;  // Total SPS across all environments

            glfwSwapBuffers(window);

            if (totalSteps >= MAX_STEPS) reachedMaxSteps = true;
        }

        // Save final model
        if (trainer && !checkpointDir.empty()) {
            std::string finalPath = checkpointDir + "/model_final.bin";
            trainer->Save(finalPath);
            std::cout << "[Hybrid Viewer] Final model saved to: " << finalPath << std::endl;
        }

        // Cleanup
        if (trainer) delete trainer;
        if (opponentTrainer) delete opponentTrainer;
        if (buffer) delete buffer;
        if (gRenderer) delete gRenderer;
        if (vecEnv) {
            vecEnv->Shutdown();
            delete vecEnv;
        }
        if (telemetryFile.is_open()) {
            telemetryFile.close();
        }
        ui.Shutdown();

        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
}
