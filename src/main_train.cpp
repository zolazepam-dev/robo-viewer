#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <queue>
#include <condition_variable>

#include "src/VectorizedEnv.h"
#include "src/NeuralNetwork.h"
#include "src/TD3Trainer.h"
#include "src/Renderer.h"
#include "src/OverlayUI_refactor.h"
#include "src/ConfigManager.h"
#include "src/VisualState.h"
#include "modules/common/PerformanceDiagnoser.h"

// GLOBAL STATE FOR SYNC
std::atomic<bool> gSimRunning{true};
std::atomic<bool> gSimPaused{false};
std::atomic<bool> gTrainingRunning{true};
std::atomic<long long> gTotalSteps{0};
std::atomic<int> gEpisodes{0};
std::atomic<float> gAvgReward{0.0f};
std::atomic<float> gSPS{0.0f};
std::atomic<float> gAgent1HP{100.0f};
std::atomic<float> gAgent2HP{100.0f};
std::atomic<float> gAgent1Reward{0.0f};
std::atomic<float> gAgent2Reward{0.0f};
std::mutex gSimMutex;

// IO THREAD STATE (now defined in TD3Trainer.cpp)
extern std::queue<IOTask> gIOQueue;
extern std::mutex gIOMutex;
extern std::condition_variable gIOCV;
std::atomic<bool> gIORunning{true};

void IOWorker(TD3Trainer* trainer) {
    fprintf(stderr, "[IOWorker] STARTING\n");
    fflush(stderr);
    
    while (gIORunning) {
        IOTask task;
        {
            std::unique_lock<std::mutex> lock(gIOMutex);
            gIOCV.wait(lock, []{ return !gIOQueue.empty() || !gIORunning; });
            if (!gIORunning && gIOQueue.empty()) break;
            task = std::move(gIOQueue.front());
            gIOQueue.pop();
        }
        if (task.type == IOTask::SAVE_MODEL) trainer->SaveToDisk(task.path, task.robotPath, task.numSatellites, task.obsDim);
        else if (task.type == IOTask::SNAPSHOT_OPPONENT) trainer->GetOpponentPool().Snapshot(task.weights, {}, (int)gTotalSteps);
    }
}

namespace fs = std::filesystem;

struct FreeCamera {
    glm::vec3 position{0.0f, 15.0f, 40.0f};
    glm::vec3 front{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float yaw = -90.0f; float pitch = -20.0f; float speed = 30.0f; float sensitivity = 0.1f; bool active = false;
};
FreeCamera gCam;
Renderer* gRenderer = nullptr;
double gLastX, gLastY;
bool gFirstMouse = true;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!gCam.active) { gLastX = xpos; gLastY = ypos; return; }
    if (gFirstMouse) { gLastX = xpos; gLastY = ypos; gFirstMouse = false; }
    float xoff = (float)(xpos - gLastX) * gCam.sensitivity; float yoff = (float)(gLastY - ypos) * gCam.sensitivity;
    gLastX = xpos; gLastY = ypos; gCam.yaw += xoff; gCam.pitch += yoff; gCam.pitch = std::clamp(gCam.pitch, -89.0f, 89.0f);
    glm::vec3 dir; dir.x = cos(glm::radians(gCam.yaw)) * cos(glm::radians(gCam.pitch)); dir.y = sin(glm::radians(gCam.pitch));
    dir.z = sin(glm::radians(gCam.yaw)) * cos(glm::radians(gCam.pitch)); gCam.front = glm::normalize(dir);
}

void process_input(GLFWwindow* window, float dt) {
    float vel = gCam.speed * dt;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) gCam.position += gCam.front * vel;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) gCam.position -= gCam.front * vel;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) gCam.position -= glm::normalize(glm::cross(gCam.front, gCam.up)) * vel;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) gCam.position += glm::normalize(glm::cross(gCam.front, gCam.up)) * vel;
}

void EnsureDir(const std::string& path) { if (!fs::exists(path)) fs::create_directories(path); }
void window_size_callback(GLFWwindow* window, int width, int height) { if (gRenderer) { glViewport(0, 0, width, height); gRenderer->Resize(width, height); } }

// Visual Triple Buffering
EnvVisualState gVisualBuffers[3][NUM_PARALLEL_ENVS];
std::atomic<int> gReadBufferIdx{0};
std::atomic<int> gWriteBufferIdx{1};
std::atomic<int> gIntermediateBufferIdx{2};
std::atomic<bool> gNewFrameReady{false};

void TrainingLoop(TD3Trainer* trainer, ReplayBuffer* buffer) {
    fprintf(stderr, "[TrainingLoop] STARTING\n");
    fflush(stderr);

    // Disable Eigen's internal multi-threading
    Eigen::setNbThreads(1);

    while (gTrainingRunning) {
        if (gSimPaused) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }

        // Train when we have enough data (lower threshold for faster learning)
        if (buffer->Size() >= 512) {
            trainer->Train(*buffer);
        } else {
            std::this_thread::yield();  // Yield instead of sleep for lower latency
        }
    }
}

void SimulationLoop(VectorizedEnv* vecEnv, TD3Trainer* trainer, TD3Trainer* opponentTrainer, ReplayBuffer* buffer, OverlayUIRefactored* ui, int stateDim, int actionDim)
{
    int latentDim = trainer->GetModel().GetLatentDim();
    fprintf(stderr, "[SimulationLoop] STARTING with %d envs, stateDim=%d, actionDim=%d, latentDim=%d\n", 
            vecEnv->GetNumEnvs(), stateDim, actionDim, latentDim);
    fflush(stderr);
    
    long long localSteps = 0; int mEpisodes = 0;
    auto lastSpsTime = std::chrono::high_resolution_clock::now(); int totalEnvStepsAccum = 0;
    AlignedVector32<float> robotActions(vecEnv->GetNumEnvs() * 2 * actionDim);
    AlignedVector32<float> prevObs(vecEnv->GetNumEnvs() * stateDim * 2, 0.0f);
    
    // Buffers to store latent states for replay buffer
    int numRobots = vecEnv->GetNumEnvs() * 2;
    AlignedVector32<float> latentPosBuffer(numRobots * latentDim);
    AlignedVector32<float> latentVelBuffer(numRobots * latentDim);
    AlignedVector32<float> prevLatentPos(numRobots * latentDim, 0.0f);
    AlignedVector32<float> prevLatentVel(numRobots * latentDim, 0.0f);

    bool firstStep = true;
    
    fprintf(stderr, "[SimulationLoop] Buffers allocated, entering loop\n");
    fflush(stderr);

    while (gSimRunning) {
        if (gSimPaused) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue; }
        
        const PhysicsTunables& phys = ui->GetPhysics();
        int numEnvs = vecEnv->GetNumEnvs();
        
        if (localSteps % 100 == 0) {
            fprintf(stderr, "[SimulationLoop] Step %lld, numEnvs=%d\n", localSteps, numEnvs);
            fflush(stderr);
        }
        const auto& obs = vecEnv->GetObservations();
        
        static AlignedVector32<float> obs1Batch, obs2Batch; 
        static std::vector<int> indices1, indices2;
        obs1Batch.resize(numEnvs * stateDim); 
        obs2Batch.resize(numEnvs * stateDim);
        indices1.resize(numEnvs); 
        indices2.resize(numEnvs);
        
        // Parallel observation batching
        #pragma omp parallel for num_threads(8) schedule(static)
        for (int i = 0; i < numEnvs; ++i) {
            std::memcpy(obs1Batch.data() + i * stateDim, (float*)obs.data() + (i * 2 * stateDim), stateDim * sizeof(float));
            indices1[i] = i * 2;
            std::memcpy(obs2Batch.data() + i * stateDim, (float*)obs.data() + (i * 2 * stateDim + stateDim), stateDim * sizeof(float));
            indices2[i] = i * 2 + 1;
        }

        // Action selection (removed profiling overhead)
        trainer->SelectActionBatchWithLatent(obs1Batch.data(), robotActions.data(), numEnvs, indices1);
        opponentTrainer->SelectActionBatchWithLatent(obs2Batch.data(), robotActions.data() + (numEnvs * actionDim), numEnvs, indices2);

        // Capture updated latent states for replay buffer
        #pragma omp parallel for num_threads(8)
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
        
        // NEW: vecEnv->Step now handles parallel action queuing, physics step, and parallel harvesting
        {
            DIAGNOSE_MUTEX_LOCK(gSimMutex, "SimulationLoop: PhysicsUpdate");
            vecEnv->Step(robotActions);
        }

        int writeIdx = gWriteBufferIdx.load();
        auto& bi = vecEnv->GetPhysicsCore()->GetPhysicsSystem().GetBodyInterface();
        
        int renderIdx = ui->GetRenderEnvIdx();
        if (renderIdx >= numEnvs) renderIdx = 0;
        
        auto& env = vecEnv->GetEnv(renderIdx); auto& r1 = env.GetRobot1(); auto& r2 = env.GetRobot2();
        if (r1.IsValid()) { auto p = bi.GetPosition(r1.mainBodyId); auto q = bi.GetRotation(r1.mainBodyId); gVisualBuffers[writeIdx][renderIdx].r1 = {p.GetX(), p.GetY(), p.GetZ(), q.GetX(), q.GetY(), q.GetZ(), q.GetW(), r1.hp}; }
        if (r2.IsValid()) { auto p = bi.GetPosition(r2.mainBodyId); auto q = bi.GetRotation(r2.mainBodyId); gVisualBuffers[writeIdx][renderIdx].r2 = {p.GetX(), p.GetY(), p.GetZ(), q.GetX(), q.GetY(), q.GetZ(), q.GetW(), r2.hp}; }
        
        gIntermediateBufferIdx.store(gWriteBufferIdx.exchange(gIntermediateBufferIdx.load()));
        gNewFrameReady = true;

        const auto& allObs = vecEnv->GetObservations(); 
        const auto& allRewards = vecEnv->GetRewards(); 
        const auto& allDones = vecEnv->GetDones();
        
        if (!firstStep) {
            DIAGNOSE_SCOPE("SimulationLoop: ReplayBufferAdd");
            for (int i = 0; i < numEnvs; ++i) {
                buffer->Add(prevObs.data() + i * 2 * stateDim, 
                            robotActions.data() + i * actionDim, 
                            allRewards[i * 2], 
                            allObs.data() + i * 2 * stateDim, 
                            allDones[i],
                            prevLatentPos.data() + (i * 2) * latentDim,
                            prevLatentVel.data() + (i * 2) * latentDim);
                            
                buffer->Add(prevObs.data() + i * 2 * stateDim + stateDim, 
                            robotActions.data() + numEnvs * actionDim + i * actionDim, 
                            allRewards[i * 2 + 1], 
                            allObs.data() + i * 2 * stateDim + stateDim, 
                            allDones[i],
                            prevLatentPos.data() + (i * 2 + 1) * latentDim,
                            prevLatentVel.data() + (i * 2 + 1) * latentDim);
            }
            
            for (int i = 0; i < numEnvs; ++i) if (allDones[i]) mEpisodes++;
            vecEnv->ResetDoneEnvs();
        }
        std::memcpy(prevObs.data(), allObs.data(), allObs.size() * sizeof(float));
        std::memcpy(prevLatentPos.data(), latentPosBuffer.data(), latentPosBuffer.size() * sizeof(float));
        std::memcpy(prevLatentVel.data(), latentVelBuffer.data(), latentVelBuffer.size() * sizeof(float));
        firstStep = false;
        
        // ASYNC: Training now happens in its own thread.
        
        gTotalSteps = ++localSteps; gEpisodes = mEpisodes; gAgent1Reward = allRewards[0]; gAgent2Reward = allRewards[1]; gAvgReward = (allRewards[0] + allRewards[1]) * 0.5f;
        auto& env0 = vecEnv->GetEnv(0); gAgent1HP = env0.GetRobot1().hp; gAgent2HP = env0.GetRobot2().hp;
        
        totalEnvStepsAccum += numEnvs; auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = now - lastSpsTime;
        if (elapsed.count() >= 1.0f) { gSPS = totalEnvStepsAccum / elapsed.count(); totalEnvStepsAccum = 0; lastSpsTime = now; }
    }
}

int main(int argc, char* argv[]) {
    int cmdNumEnvs = 0;
    int cmdRenderEnv = 0;  // OPTIMIZATION: Select which env to render (default 0)
    for (int i = 1; i < argc; i++) { 
        std::string arg = argv[i]; 
        if (arg == "--envs" && i + 1 < argc) cmdNumEnvs = std::stoi(argv[++i]); 
        else if (arg == "--render-env" && i + 1 < argc) cmdRenderEnv = std::stoi(argv[++i]);
    }
    const char* home = std::getenv("HOME"); std::string checkpointDir = (home ? std::string(home) : ".") + "/.joltrl/checkpoints"; EnsureDir(checkpointDir);
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "JOLTrl - Physics Mastery Suite", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window); glfwSwapInterval(0); glewInit();  // vsync OFF for less stutter
    glfwSetWindowSizeCallback(window, window_size_callback); glEnable(GL_DEPTH_TEST); glfwSetCursorPosCallback(window, mouse_callback);
    OverlayUIRefactored ui; ui.Init(window); ui.LoadSettings();
    int targetNumEnvs = (cmdNumEnvs > 0) ? cmdNumEnvs : ui.GetConfig().numEnvs;
    VectorizedEnv* vecEnv = new VectorizedEnv(targetNumEnvs, ui.GetStepsPerEpisode());
    std::string robotConfigPath = ui.GetConfig().robotConfigPath;
    vecEnv->Init(robotConfigPath);
    int stateDim = vecEnv->GetObservationDim(); int actionDim = vecEnv->GetActionDim();
    std::string robotName = vecEnv->GetEnv(0).GetRobot1().config.name;
    std::string robotCheckpointDir = checkpointDir + "/" + (robotName.empty() ? "unknown_bot" : robotName); EnsureDir(robotCheckpointDir);
    gRenderer = new Renderer(1280, 720);
    TD3Config config;
    auto trainer = std::make_unique<TD3Trainer>(stateDim, actionDim, config);
    auto opponentTrainer = std::make_unique<TD3Trainer>(stateDim, actionDim, config);
    auto buffer = std::make_unique<ReplayBuffer>(1000000, stateDim, actionDim, config.latentDim);
    
    std::string modelPath = robotCheckpointDir + "/model_final.bin";
    fprintf(stderr, "[main] Checking model path: %s\n", modelPath.c_str());
    fflush(stderr);
    
    if (fs::exists(modelPath)) {
        fprintf(stderr, "[main] Model exists, loading...\n");
        fflush(stderr);
        try { trainer->Load(modelPath); } catch(...) {}
        fprintf(stderr, "[main] Model loaded\n");
        fflush(stderr);
    }
    
    fprintf(stderr, "[main] Copying weights to opponent...\n");
    fflush(stderr);
    opponentTrainer->GetModel().GetActor().SetAllWeights(trainer->GetModel().GetActor().GetAllWeights());
    fprintf(stderr, "[main] Weights copied\n");
    fflush(stderr);

    fprintf(stderr, "[main] Creating threads...\n");
    fflush(stderr);
    std::thread ioThread(IOWorker, trainer.get());
    std::thread trainingThread(TrainingLoop, trainer.get(), buffer.get());
    std::thread simThread(SimulationLoop, vecEnv, trainer.get(), opponentTrainer.get(), buffer.get(), &ui, stateDim, actionDim);
    fprintf(stderr, "[main] Threads created\n");
    fflush(stderr);
    auto last_time = std::chrono::high_resolution_clock::now();
    auto last_diag_report = std::chrono::high_resolution_clock::now();
    int renderEnvIdx = cmdRenderEnv;  // Use command-line specified env

    while (!glfwWindowShouldClose(window)) {
        DIAGNOSE_SCOPE("UILoop: TotalFrame");
        glfwPollEvents(); auto now = std::chrono::high_resolution_clock::now(); float dt = std::chrono::duration<float>(now - last_time).count(); last_time = now;
        
        // Periodic Diagnostic Report (every 10s)
        if (std::chrono::duration<float>(now - last_diag_report).count() > 10.0f) {
            PerformanceDiagnoser::Get().PrintReport();
            last_diag_report = now;
        }

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) { if (!gCam.active) { gCam.active = true; glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); gFirstMouse = true; } }
        else if (gCam.active) { gCam.active = false; glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); }
        process_input(window, dt);
        gSimPaused = ui.IsPaused(); ui.UpdateStats((int)gTotalSteps, gEpisodes, gSPS, gAvgReward, renderEnvIdx, vecEnv->GetNumEnvs());
        vecEnv->SetDomainRandomization(ui.GetConfig().dr); // Apply UI changes to environments
        if (gNewFrameReady.exchange(false)) gReadBufferIdx.store(gIntermediateBufferIdx.exchange(gReadBufferIdx.load()));
        int rIdx = gReadBufferIdx.load();
        if (rIdx < 3 && renderEnvIdx < (int)targetNumEnvs) {
            const auto& v = gVisualBuffers[rIdx][renderEnvIdx]; ui.UpdateAgentHP(v.r1.hp, v.r2.hp); ui.UpdateAgentRewards(gAgent1Reward, gAgent2Reward);
            {
                DIAGNOSE_MUTEX_LOCK(gSimMutex, "UILoop: RenderDraw");
                gRenderer->Draw(&vecEnv->GetPhysicsCore()->GetPhysicsSystem(), gCam.position, renderEnvIdx, gCam.front, ui.GetGraphics().showCollisionShapes, ui.GetGraphics().showAABBs, false, ui.GetGraphics().showRobot1, ui.GetGraphics().showRobot2, &v);
            }
        }
        if (ui.ShouldRestartSim()) {
            gSimRunning = false; gTrainingRunning = false;
            if (simThread.joinable()) simThread.join();
            if (trainingThread.joinable()) trainingThread.join();
            { 
                DIAGNOSE_MUTEX_LOCK(gSimMutex, "MainLoop: SimRestart");
                delete vecEnv; vecEnv = new VectorizedEnv(ui.GetConfig().numEnvs, ui.GetStepsPerEpisode()); vecEnv->Init(ui.GetConfig().robotConfigPath); targetNumEnvs = ui.GetConfig().numEnvs; 
            }
            gSimRunning = true; gTrainingRunning = true;
            trainingThread = std::thread(TrainingLoop, trainer.get(), buffer.get());
            simThread = std::thread(SimulationLoop, vecEnv, trainer.get(), opponentTrainer.get(), buffer.get(), &ui, stateDim, actionDim); ui.ClearRestartRequest();
        }
        ui.NewFrame(); ImGui::Begin("JOLTrl Control Center"); ui.DrawAllTabs(); ImGui::End(); ui.Render(); glfwSwapBuffers(window);
    }
    gSimRunning = false; gTrainingRunning = false;
    if (simThread.joinable()) simThread.join();
    if (trainingThread.joinable()) trainingThread.join();
    gIORunning = false; gIOCV.notify_all(); if (ioThread.joinable()) ioThread.join();
    trainer->Save(robotCheckpointDir + "/model_final.bin", "robots/"+robotName+".json", vecEnv->GetEnv(0).GetRobot1().config.numSatellites, stateDim);
    delete gRenderer; delete vecEnv; ui.Shutdown(); glfwTerminate(); return 0;
}
