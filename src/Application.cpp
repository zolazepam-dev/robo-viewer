#include "Application.h"
#include "VectorizedEnv.h"
#include "TD3Trainer.h"
#include "Renderer.h"
#include "OverlayUI_refactor.h"
#include "ConfigManager.h"
#include "VisualState.h"
#include "NeuralNetwork.h"
#include "PerformanceDiagnoser.h"

#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================================
// Application Implementation
// ============================================================================

Application::Application() {}

Application::~Application() {
    Shutdown();
}

bool Application::Init(int argc, char* argv[]) {
    try {
        // Parse command line
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--envs" && i + 1 < argc) {
                mTargetNumEnvs = std::stoi(argv[++i]);
            } else if (arg == "--render-env" && i + 1 < argc) {
                mRenderEnvIdx = std::stoi(argv[++i]);
            }
        }
        
        // Setup checkpoint directory
        const char* home = std::getenv("HOME");
        mCheckpointDir = (home ? std::string(home) : ".") + "/.joltrl/checkpoints";
        if (!fs::exists(mCheckpointDir)) {
            fs::create_directories(mCheckpointDir);
        }
        
        // Initialize subsystems
        InitWindow(argc, argv);
        InitEnvironment();
        InitTrainer();
        CreateThreads();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Application] Init failed: " << e.what() << std::endl;
        return false;
    }
}

void Application::InitWindow(int argc, char* argv[]) {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    
    mWindow = glfwCreateWindow(mWindowWidth, mWindowHeight, "JOLTrl - Physics Mastery Suite", nullptr, nullptr);
    if (!mWindow) {
        glfwTerminate();
        throw std::runtime_error("Failed to create window");
    }
    
    glfwMakeContextCurrent(mWindow);
    glfwSwapInterval(0);
    glewInit();
    
    // Setup callbacks
    glfwSetWindowSizeCallback((GLFWwindow*)mWindow, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
        if (gApp && gApp->GetRenderer()) {
            gApp->GetRenderer()->Resize(width, height);
        }
    });
    
    glfwSetCursorPosCallback((GLFWwindow*)mWindow, [](GLFWwindow* window, double xpos, double ypos) {
        if (!gApp->mCameraActive) {
            gApp->mState.lastX = xpos;
            gApp->mState.lastY = ypos;
            return;
        }
        if (gApp->mState.firstMouse) {
            gApp->mState.lastX = xpos;
            gApp->mState.lastY = ypos;
            gApp->mState.firstMouse = false;
        }
        float xoff = (float)(xpos - gApp->mState.lastX) * gApp->mCameraSensitivity;
        float yoff = (float)(gApp->mState.lastY - ypos) * gApp->mCameraSensitivity;
        gApp->mState.lastX = xpos;
        gApp->mState.lastY = ypos;
        gApp->mCameraYaw += xoff;
        gApp->mCameraPitch += yoff;
        gApp->mCameraPitch = std::clamp(gApp->mCameraPitch, -89.0f, 89.0f);
        
        glm::vec3 dir;
        dir.x = cos(glm::radians(gApp->mCameraYaw)) * cos(glm::radians(gApp->mCameraPitch));
        dir.y = sin(glm::radians(gApp->mCameraPitch));
        dir.z = sin(glm::radians(gApp->mCameraYaw)) * cos(glm::radians(gApp->mCameraPitch));
        gApp->mCameraFront = glm::normalize(dir);
    });
    
    glEnable(GL_DEPTH_TEST);
    
    // Create UI and Renderer
    mUI = std::make_unique<OverlayUIRefactored>();
    mUI->Init(mWindow);
    mUI->LoadSettings();
    
    mRenderer = std::make_unique<Renderer>(mWindowWidth, mWindowHeight);
}

void Application::InitEnvironment() {
    int numEnvs = (mTargetNumEnvs > 0) ? mTargetNumEnvs : mUI->GetConfig().numEnvs;
    mVecEnv = std::make_unique<VectorizedEnv>(numEnvs, mUI->GetStepsPerEpisode());
    mVecEnv->Init(mUI->GetConfig().robotConfigPath);
    
    mStateDim = mVecEnv->GetObservationDim();
    mActionDim = mVecEnv->GetActionDim();
    mRobotName = mVecEnv->GetEnv(0).GetRobot1().config.name;
    
    mRobotCheckpointDir = mCheckpointDir + "/" + (mRobotName.empty() ? "unknown_bot" : mRobotName);
    if (!fs::exists(mRobotCheckpointDir)) {
        fs::create_directories(mRobotCheckpointDir);
    }
    
    mVisualBuffer = &mState.visualBuffer;
}

void Application::InitTrainer() {
    TD3Config config;
    mTrainer = std::make_unique<TD3Trainer>(mStateDim, mActionDim, config);
    mOpponentTrainer = std::make_unique<TD3Trainer>(mStateDim, mActionDim, config);
    mBuffer = std::make_unique<ReplayBuffer>(1000000, mStateDim, mActionDim, config.latentDim);
    
    // Load checkpoint if exists
    LoadCheckpoint();
    
    // Copy weights to opponent
    mOpponentTrainer->GetModel().GetActor().SetAllWeights(
        mTrainer->GetModel().GetActor().GetAllWeights()
    );
}

void Application::LoadCheckpoint() {
    std::string modelPath = mRobotCheckpointDir + "/model_final.bin";
    if (fs::exists(modelPath)) {
        fprintf(stderr, "[Application] Loading model from: %s\n", modelPath.c_str());
        try {
            mTrainer->Load(modelPath);
            fprintf(stderr, "[Application] Model loaded successfully\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[Application] Failed to load model: %s\n", e.what());
        }
    }
}

void Application::SaveCheckpoint() {
    std::string modelPath = mRobotCheckpointDir + "/model_final.bin";
    mTrainer->Save(modelPath, "robots/" + mRobotName + ".json",
                   mVecEnv->GetEnv(0).GetRobot1().config.numSatellites, mStateDim);
}

void Application::CreateThreads() {
    mIOThread = std::thread(IOWorker, this);
    mTrainingThread = std::thread(TrainingLoop, this);
    mSimThread = std::thread(SimulationLoop, this);
}

void Application::JoinThreads() {
    mState.simRunning = false;
    mState.trainingRunning = false;
    mState.ioRunning = false;
    
    if (mSimThread.joinable()) mSimThread.join();
    if (mTrainingThread.joinable()) mTrainingThread.join();
    
    if (mIOThread.joinable()) mIOThread.join();
}

void Application::Shutdown() {
    JoinThreads();
    
    if (mRenderer) mRenderer.reset();
    if (mVecEnv) mVecEnv.reset();
    if (mUI) {
        mUI->Shutdown();
        mUI.reset();
    }
    
    glfwTerminate();
}

void Application::Run() {
    auto lastTime = std::chrono::high_resolution_clock::now();
    auto lastDiagReport = std::chrono::high_resolution_clock::now();
    
    while (!glfwWindowShouldClose((GLFWwindow*)mWindow)) {
        DIAGNOSE_SCOPE("UILoop: TotalFrame");
        
        glfwPollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        
        // Periodic diagnostic report (every 10s)
        if (std::chrono::duration<float>(now - lastDiagReport).count() > 10.0f) {
            PerformanceDiagnoser::Get().PrintReport();
            lastDiagReport = now;
        }
        
        // Camera controls
        if (glfwGetMouseButton((GLFWwindow*)mWindow, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!gApp->mCameraActive) {
                gApp->mCameraActive = true;
                glfwSetInputMode((GLFWwindow*)mWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                gApp->mState.firstMouse = true;
            }
        } else if (gApp->mCameraActive) {
            gApp->mCameraActive = false;
            glfwSetInputMode((GLFWwindow*)mWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        
        // Process camera movement
        float vel = gApp->mCameraSpeed * dt;
        if (glfwGetKey((GLFWwindow*)mWindow, GLFW_KEY_W) == GLFW_PRESS)
            gApp->mCameraPosition += gApp->mCameraFront * vel;
        if (glfwGetKey((GLFWwindow*)mWindow, GLFW_KEY_S) == GLFW_PRESS)
            gApp->mCameraPosition -= gApp->mCameraFront * vel;
        if (glfwGetKey((GLFWwindow*)mWindow, GLFW_KEY_A) == GLFW_PRESS)
            gApp->mCameraPosition -= glm::normalize(glm::cross(gApp->mCameraFront, gApp->mCameraUp)) * vel;
        if (glfwGetKey((GLFWwindow*)mWindow, GLFW_KEY_D) == GLFW_PRESS)
            gApp->mCameraPosition += glm::normalize(glm::cross(gApp->mCameraFront, gApp->mCameraUp)) * vel;
        
        // Update UI state
        gApp->mState.simPaused = gApp->mUI->IsPaused();
        gApp->mUI->UpdateStats((int)gApp->mState.totalSteps, gApp->mState.episodes, gApp->mState.sps,
                        gApp->mState.avgReward, gApp->mRenderEnvIdx, gApp->mVecEnv->GetNumEnvs());
        
        // Update domain randomization
        gApp->mVecEnv->SetDomainRandomization(gApp->mUI->GetConfig().dr);
        
        // Render
        if (gApp->mState.visualBuffer.newFrameReady.exchange(false)) {
            gApp->mState.visualBuffer.readBufferIdx.store(gApp->mState.visualBuffer.intermediateBufferIdx.exchange(gApp->mState.visualBuffer.readBufferIdx.load()));
        }
        
        int rIdx = gApp->mState.visualBuffer.readBufferIdx.load();
        if (rIdx < 3 && gApp->mRenderEnvIdx < gApp->mVecEnv->GetNumEnvs()) {
            const auto& v = gApp->mState.visualBuffer.visualBuffers[rIdx][gApp->mRenderEnvIdx];
            gApp->mUI->UpdateAgentHP(v.r1.hp, v.r2.hp);
            gApp->mUI->UpdateAgentRewards(gApp->mState.agent1Reward, gApp->mState.agent2Reward);
            
            {
                DIAGNOSE_MUTEX_LOCK(gApp->mState.simMutex, "UILoop: RenderDraw");
                gApp->mRenderer->Draw(&gApp->mVecEnv->GetPhysicsCore()->GetPhysicsSystem(),
                              gApp->mCameraPosition, gApp->mRenderEnvIdx, gApp->mCameraFront,
                              gApp->mUI->GetGraphics().showCollisionShapes,
                              gApp->mUI->GetGraphics().showAABBs, false,
                              gApp->mUI->GetGraphics().showRobot1,
                              gApp->mUI->GetGraphics().showRobot2, &v);
            }
        }
        
        // Handle restart
        if (gApp->mUI->ShouldRestartSim()) {
            gApp->mState.simRunning = false;
            gApp->mState.trainingRunning = false;
            gApp->JoinThreads();
            
            {
                DIAGNOSE_MUTEX_LOCK(gApp->mState.simMutex, "MainLoop: SimRestart");
                gApp->mVecEnv = std::make_unique<VectorizedEnv>(gApp->mUI->GetConfig().numEnvs,
                                                        gApp->mUI->GetStepsPerEpisode());
                gApp->mVecEnv->Init(gApp->mUI->GetConfig().robotConfigPath);
                gApp->mTargetNumEnvs = gApp->mUI->GetConfig().numEnvs;
            }
            
            gApp->mState.simRunning = true;
            gApp->mState.trainingRunning = true;
            gApp->CreateThreads();
            gApp->mUI->ClearRestartRequest();
        }
        
        // Render UI
        gApp->mUI->NewFrame();
        ImGui::Begin("JOLTrl Control Center");
        gApp->mUI->DrawAllTabs();
        ImGui::End();
        gApp->mUI->Render();
        
        glfwSwapBuffers((GLFWwindow*)mWindow);
    }
    
    // Save on exit
    SaveCheckpoint();
}

// ============================================================================
// Thread Functions
// ============================================================================

void IOWorker(Application* app) {
    fprintf(stderr, "[IOWorker] STARTING\n");
    
    while (app->mState.ioRunning) {
        IOTask task;
        {
            std::unique_lock<std::mutex> lock(app->mTrainer->GetIOMutex());
            app->mTrainer->GetIOCV().wait(lock, [&]{
                return !app->mTrainer->GetIOQueue()->empty() || !app->mState.ioRunning;
            });
            if (!app->mState.ioRunning && app->mTrainer->GetIOQueue()->empty()) break;
            task = std::move(app->mTrainer->GetIOQueue()->front());
            app->mTrainer->GetIOQueue()->pop();
        }
        
        if (task.type == IOTask::SAVE_MODEL) {
            app->mTrainer->SaveToDisk(task.path, task.robotPath, task.numSatellites, task.obsDim);
        } else if (task.type == IOTask::SNAPSHOT_OPPONENT) {
            app->mTrainer->GetOpponentPool().Snapshot(task.weights, {}, (int)app->mState.totalSteps);
        }
    }
}

void TrainingLoop(Application* app) {
    fprintf(stderr, "[TrainingLoop] STARTING\n");
    
    // Disable Eigen's internal multi-threading
    Eigen::setNbThreads(1);
    
    int trainCounter = 0;
    const int TRAIN_EVERY_N_STEPS = 2;
    
    while (app->mState.trainingRunning) {
        if (app->mState.simPaused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        if (app->mBuffer->Size() >= 512 && ++trainCounter % TRAIN_EVERY_N_STEPS == 0) {
            app->mTrainer->Train(*app->mBuffer);
        } else {
            std::this_thread::yield();
        }
    }
}

void SimulationLoop(Application* app) {
    fprintf(stderr, "[SimulationLoop] STARTING\n");
    
    VectorizedEnv* vecEnv = app->mVecEnv.get();
    TD3Trainer* trainer = app->mTrainer.get();
    TD3Trainer* opponentTrainer = app->mOpponentTrainer.get();
    ReplayBuffer* buffer = app->mBuffer.get();
    OverlayUIRefactored* ui = app->mUI.get();
    
    int latentDim = trainer->GetModel().GetLatentDim();
    int stateDim = app->mStateDim;
    int actionDim = app->mActionDim;
    
    long long localSteps = 0;
    int mEpisodes = 0;
    auto lastSpsTime = std::chrono::high_resolution_clock::now();
    int totalEnvStepsAccum = 0;
    
    AlignedVector32<float> robotActions(vecEnv->GetNumEnvs() * 2 * actionDim);
    AlignedVector32<float> prevObs(vecEnv->GetNumEnvs() * stateDim * 2, 0.0f);
    
    int numRobots = vecEnv->GetNumEnvs() * 2;
    AlignedVector32<float> latentPosBuffer(numRobots * latentDim);
    AlignedVector32<float> latentVelBuffer(numRobots * latentDim);
    AlignedVector32<float> prevLatentPos(numRobots * latentDim, 0.0f);
    AlignedVector32<float> prevLatentVel(numRobots * latentDim, 0.0f);
    
    // Pre-allocate batch buffers
    int numEnvs = vecEnv->GetNumEnvs();
    static AlignedVector32<float> obs1Batch, obs2Batch;
    static std::vector<int> indices1, indices2;
    static std::vector<float> batchRewards, batchActions, batchNextStates, batchLatentPos, batchLatentVel;
    static std::vector<char> batchDones;
    
    if (obs1Batch.capacity() < numEnvs * stateDim) {
        obs1Batch.reserve(numEnvs * stateDim * 2);
        obs2Batch.reserve(numEnvs * stateDim * 2);
        indices1.reserve(numEnvs * 2);
        indices2.reserve(numEnvs * 2);
        batchRewards.reserve(numEnvs * 2);
        batchActions.reserve(numEnvs * 2 * actionDim);
        batchNextStates.reserve(numEnvs * 2 * stateDim);
        batchDones.reserve(numEnvs * 2);
        batchLatentPos.reserve(numEnvs * 2 * latentDim);
        batchLatentVel.reserve(numEnvs * 2 * latentDim);
    }
    
    bool firstStep = true;
    int ompThreads = std::max(1, omp_get_max_threads() - 2);
    
    while (app->mState.simRunning) {
        if (app->mState.simPaused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        numEnvs = vecEnv->GetNumEnvs();
        const auto& obs = vecEnv->GetObservations();
        
        obs1Batch.resize(numEnvs * stateDim);
        obs2Batch.resize(numEnvs * stateDim);
        indices1.resize(numEnvs);
        indices2.resize(numEnvs);
        
        #pragma omp parallel for num_threads(ompThreads) schedule(static)
        for (int i = 0; i < numEnvs; ++i) {
            std::memcpy(obs1Batch.data() + i * stateDim, (float*)obs.data() + (i * 2 * stateDim), stateDim * sizeof(float));
            indices1[i] = i * 2;
            std::memcpy(obs2Batch.data() + i * stateDim, (float*)obs.data() + (i * 2 * stateDim + stateDim), stateDim * sizeof(float));
            indices2[i] = i * 2 + 1;
        }
        
        // Action selection
        trainer->SelectActionBatchWithLatent(obs1Batch.data(), robotActions.data(), numEnvs, indices1);
        opponentTrainer->SelectActionBatchWithLatent(obs2Batch.data(), robotActions.data() + (numEnvs * actionDim), numEnvs, indices2);
        
        // Capture latent states
        #pragma omp parallel for num_threads(ompThreads) schedule(static)
        for (int i = 0; i < numEnvs; ++i) {
            trainer->GetModel().GetLatentMemory().GetLatentStates(
                latentPosBuffer.data() + (i * 2) * latentDim,
                latentVelBuffer.data() + (i * 2) * latentDim,
                indices1[i]);
            opponentTrainer->GetModel().GetLatentMemory().GetLatentStates(
                latentPosBuffer.data() + (i * 2 + 1) * latentDim,
                latentVelBuffer.data() + (i * 2 + 1) * latentDim,
                indices2[i]);
        }
        
        // Step simulation
        vecEnv->Step(robotActions);
        
        // Capture visual state
        int writeIdx = app->mState.visualBuffer.writeBufferIdx.load();
        auto& bi = vecEnv->GetPhysicsCore()->GetPhysicsSystem().GetBodyInterface();
        
        int renderIdx = ui->GetRenderEnvIdx();
        if (renderIdx >= numEnvs) renderIdx = 0;
        
        auto& env = vecEnv->GetEnv(renderIdx);
        auto& r1 = env.GetRobot1();
        auto& r2 = env.GetRobot2();
        
        if (r1.IsValid()) {
            auto p = bi.GetPosition(r1.mainBodyId);
            auto q = bi.GetRotation(r1.mainBodyId);
            app->mState.visualBuffer.visualBuffers[writeIdx][renderIdx].r1 = {p.GetX(), p.GetY(), p.GetZ(), q.GetX(), q.GetY(), q.GetZ(), q.GetW(), r1.hp};
        }
        if (r2.IsValid()) {
            auto p = bi.GetPosition(r2.mainBodyId);
            auto q = bi.GetRotation(r2.mainBodyId);
            app->mState.visualBuffer.visualBuffers[writeIdx][renderIdx].r2 = {p.GetX(), p.GetY(), p.GetZ(), q.GetX(), q.GetY(), q.GetZ(), q.GetW(), r2.hp};
        }
        
        app->mState.visualBuffer.intermediateBufferIdx.store(app->mState.visualBuffer.writeBufferIdx.exchange(app->mState.visualBuffer.intermediateBufferIdx.load()));
        app->mState.visualBuffer.newFrameReady = true;
        
        const auto& allObs = vecEnv->GetObservations();
        const auto& allRewards = vecEnv->GetRewards();
        const auto& allDones = vecEnv->GetDones();
        
        if (!firstStep) {
            batchRewards.resize(numEnvs * 2);
            batchActions.resize(numEnvs * 2 * actionDim);
            batchNextStates.resize(numEnvs * 2 * stateDim);
            batchDones.resize(numEnvs * 2);
            batchLatentPos.resize(numEnvs * 2 * latentDim);
            batchLatentVel.resize(numEnvs * 2 * latentDim);
            
            for (int i = 0; i < numEnvs; ++i) {
                batchRewards[i * 2] = allRewards[i * 2];
                batchRewards[i * 2 + 1] = allRewards[i * 2 + 1];
                std::memcpy(batchActions.data() + i * 2 * actionDim, robotActions.data() + i * actionDim, actionDim * sizeof(float));
                std::memcpy(batchActions.data() + (i * 2 + 1) * actionDim, robotActions.data() + numEnvs * actionDim + i * actionDim, actionDim * sizeof(float));
                std::memcpy(batchNextStates.data() + i * 2 * stateDim, allObs.data() + i * 2 * stateDim, stateDim * 2 * sizeof(float));
                batchDones[i * 2] = allDones[i] ? 1.0f : 0.0f;
                batchDones[i * 2 + 1] = allDones[i] ? 1.0f : 0.0f;
                std::memcpy(batchLatentPos.data() + i * 2 * latentDim, prevLatentPos.data() + (i * 2) * latentDim, latentDim * 2 * sizeof(float));
                std::memcpy(batchLatentVel.data() + i * 2 * latentDim, prevLatentVel.data() + (i * 2) * latentDim, latentDim * 2 * sizeof(float));
            }
            
            buffer->AddBatch(prevObs.data(), batchActions.data(), batchRewards.data(),
                           batchNextStates.data(), batchDones.data(),
                           batchLatentPos.data(), batchLatentVel.data(), numEnvs * 2);
            
            for (int i = 0; i < numEnvs; ++i) if (allDones[i]) mEpisodes++;
            vecEnv->ResetDoneEnvs();
        }
        
        std::memcpy(prevObs.data(), allObs.data(), allObs.size() * sizeof(float));
        std::memcpy(prevLatentPos.data(), latentPosBuffer.data(), latentPosBuffer.size() * sizeof(float));
        std::memcpy(prevLatentVel.data(), latentVelBuffer.data(), latentVelBuffer.size() * sizeof(float));
        firstStep = false;
        
        // Update statistics
        app->mState.totalSteps = ++localSteps;
        app->mState.episodes = mEpisodes;
        app->mState.agent1Reward = allRewards[0];
        app->mState.agent2Reward = allRewards[1];
        app->mState.avgReward = (allRewards[0] + allRewards[1]) * 0.5f;
        
        auto& env0 = vecEnv->GetEnv(0);
        app->mState.agent1HP = env0.GetRobot1().hp;
        app->mState.agent2HP = env0.GetRobot2().hp;
        
        totalEnvStepsAccum += numEnvs;
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = now - lastSpsTime;
        if (elapsed.count() >= 1.0f) {
            app->mState.sps = totalEnvStepsAccum / elapsed.count();
            totalEnvStepsAccum = 0;
            lastSpsTime = now;
        }
    }
}

// Global application instance
Application* gApp = nullptr;
