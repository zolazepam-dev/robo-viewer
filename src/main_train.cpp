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
#include <queue>
#include <condition_variable>

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

// IO THREAD STATE
struct IOTask {
    enum Type { SAVE_MODEL, SNAPSHOT_OPPONENT };
    Type type;
    std::string path;
    std::string robotPath;
    int numSatellites;
    int obsDim;
    std::vector<float> weights;
};
std::queue<IOTask> gIOQueue;
std::mutex gIOMutex;
std::condition_variable gIOCV;
std::atomic<bool> gIORunning{true};

void IOWorker(TD3Trainer* trainer) {
    while (gIORunning) {
        IOTask task;
        {
            std::unique_lock<std::mutex> lock(gIOMutex);
            gIOCV.wait(lock, []{ return !gIOQueue.empty() || !gIORunning; });
            if (!gIORunning && gIOQueue.empty()) break;
            task = std::move(gIOQueue.front());
            gIOQueue.pop();
        }
        if (task.type == IOTask::SAVE_MODEL) trainer->Save(task.path, task.robotPath, task.numSatellites, task.obsDim);
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
    if (!gCam.active) return;
    float vel = gCam.speed * dt;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) gCam.position += gCam.front * vel;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) gCam.position -= gCam.front * vel;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) gCam.position -= glm::normalize(glm::cross(gCam.front, gCam.up)) * vel;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) gCam.position += glm::normalize(glm::cross(gCam.front, gCam.up)) * vel;
}

void EnsureDir(const std::string& path) { if (!fs::exists(path)) fs::create_directories(path); }
void window_size_callback(GLFWwindow* window, int width, int height) { if (gRenderer) { glViewport(0, 0, width, height); gRenderer->Resize(width, height); } }

void SimulationLoop(VectorizedEnv* vecEnv, TD3Trainer* trainer, TD3Trainer* opponentTrainer, ReplayBuffer* buffer, OverlayUIRefactored* ui, int stateDim, int actionDim)
{
    long long localSteps = 0; int mEpisodes = 0;
    auto lastSpsTime = std::chrono::high_resolution_clock::now(); int totalEnvStepsAccum = 0;
    AlignedVector32<float> robotActions(vecEnv->GetNumEnvs() * 2 * actionDim);
    AlignedVector32<float> prevObs(vecEnv->GetNumEnvs() * stateDim * 2, 0.0f);
    bool firstStep = true;

    while (gSimRunning) {
        if (gSimPaused) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue; }
        int numEnvs = vecEnv->GetNumEnvs();
        const auto& obs = vecEnv->GetObservations();
        static AlignedVector32<float> obs1Batch, obs2Batch; static std::vector<int> indices1, indices2;
        obs1Batch.resize(numEnvs * stateDim); indices1.resize(numEnvs);
        obs2Batch.resize(numEnvs * stateDim); indices2.resize(numEnvs);
        for (int i = 0; i < numEnvs; ++i) {
            std::memcpy(obs1Batch.data() + i * stateDim, (float*)obs.data() + (i * 2 * stateDim), stateDim * sizeof(float));
            indices1[i] = i * 2;
            std::memcpy(obs2Batch.data() + i * stateDim, (float*)obs.data() + (i * 2 * stateDim + stateDim), stateDim * sizeof(float));
            indices2[i] = i * 2 + 1;
        }
        trainer->SelectActionBatchWithLatent(obs1Batch.data(), robotActions.data(), numEnvs, indices1);
        opponentTrainer->SelectActionBatchWithLatent(obs2Batch.data(), robotActions.data() + (numEnvs * actionDim), numEnvs, indices2);
        
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < numEnvs; ++i) vecEnv->GetEnv(i).QueueActions(robotActions.data() + (i * actionDim), robotActions.data() + (numEnvs * actionDim + i * actionDim));
        
        { std::lock_guard<std::mutex> lock(gSimMutex);
          vecEnv->GetPhysicsCore()->GetPhysicsSystem().Update(1.0f / 120.0f, 1, vecEnv->GetPhysicsCore()->GetTempAllocator(), vecEnv->GetPhysicsCore()->GetJobSystem());
          vecEnv->HarvestStates(); }

        int writeIdx = gWriteBufferIdx.load();
        if (gVisualBuffers[writeIdx].size() != (size_t)numEnvs) gVisualBuffers[writeIdx].resize(numEnvs);
        auto& bi = vecEnv->GetPhysicsCore()->GetPhysicsSystem().GetBodyInterface();
        for (int i = 0; i < numEnvs; ++i) {
            auto& env = vecEnv->GetEnv(i); auto& r1 = env.GetRobot1(); auto& r2 = env.GetRobot2();
            if (r1.IsValid()) { auto p = bi.GetPosition(r1.mainBodyId); auto q = bi.GetRotation(r1.mainBodyId); gVisualBuffers[writeIdx][i].r1 = {p.GetX(), p.GetY(), p.GetZ(), q.GetX(), q.GetY(), q.GetZ(), q.GetW(), r1.hp}; }
            if (r2.IsValid()) { auto p = bi.GetPosition(r2.mainBodyId); auto q = bi.GetRotation(r2.mainBodyId); gVisualBuffers[writeIdx][i].r2 = {p.GetX(), p.GetY(), p.GetZ(), q.GetX(), q.GetY(), q.GetZ(), q.GetW(), r2.hp}; }
        }
        gIntermediateBufferIdx.store(gWriteBufferIdx.exchange(gIntermediateBufferIdx.load()));
        gNewFrameReady = true;

        const auto& allObs = vecEnv->GetObservations(); const auto& allRewards = vecEnv->GetRewards(); const auto& allDones = vecEnv->GetDones();
        if (!firstStep) {
            #pragma omp parallel for num_threads(8)
            for (int i = 0; i < numEnvs; ++i) {
                buffer->Add(prevObs.data() + i * 2 * stateDim, robotActions.data() + i * actionDim, allRewards[i * 2], allObs.data() + i * 2 * stateDim, allDones[i]);
                buffer->Add(prevObs.data() + i * 2 * stateDim + stateDim, robotActions.data() + numEnvs * actionDim + i * actionDim, allRewards[i * 2 + 1], allObs.data() + i * 2 * stateDim + stateDim, allDones[i]);
                if (allDones[i]) { mEpisodes++; std::lock_guard<std::mutex> lock(gSimMutex); vecEnv->Reset(i); }
            }
        }
        std::memcpy(prevObs.data(), allObs.data(), allObs.size() * sizeof(float)); firstStep = false;
        if (localSteps % 16 == 0 && buffer->Size() >= 256) { for (int u = 0; u < 2; ++u) trainer->Train(*buffer); }
        
        gTotalSteps = ++localSteps; gEpisodes = mEpisodes; gAgent1Reward = allRewards[0]; gAgent2Reward = allRewards[1]; gAvgReward = (allRewards[0] + allRewards[1]) * 0.5f;
        auto& env0 = vecEnv->GetEnv(0); gAgent1HP = env0.GetRobot1().hp; gAgent2HP = env0.GetRobot2().hp;
        
        totalEnvStepsAccum += numEnvs; auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = now - lastSpsTime;
        if (elapsed.count() >= 1.0f) { gSPS = totalEnvStepsAccum / elapsed.count(); totalEnvStepsAccum = 0; lastSpsTime = now; }
    }
}

int main(int argc, char* argv[]) {
    int cmdNumEnvs = 0;
    for (int i = 1; i < argc; i++) { std::string arg = argv[i]; if (arg == "--envs" && i + 1 < argc) cmdNumEnvs = std::stoi(argv[++i]); }
    const char* home = std::getenv("HOME"); std::string checkpointDir = (home ? std::string(home) : ".") + "/.joltrl/checkpoints"; EnsureDir(checkpointDir);
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "JOLTrl - Pro Training Suite", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window); glfwSwapInterval(0); glewInit();
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
    TD3Trainer trainer(stateDim, actionDim); TD3Trainer opponentTrainer(stateDim, actionDim); ReplayBuffer buffer(1000000, stateDim, actionDim);
    std::string modelPath = robotCheckpointDir + "/model_final.bin";
    if (fs::exists(modelPath)) try { trainer.Load(modelPath); } catch(...) {}
    opponentTrainer.GetModel().GetActor().SetAllWeights(trainer.GetModel().GetActor().GetAllWeights());
    
    std::thread ioThread(IOWorker, &trainer);
    std::thread simThread(SimulationLoop, vecEnv, &trainer, &opponentTrainer, &buffer, &ui, stateDim, actionDim);
    auto last_time = std::chrono::high_resolution_clock::now();
    int renderEnvIdx = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents(); auto now = std::chrono::high_resolution_clock::now(); float dt = std::chrono::duration<float>(now - last_time).count(); last_time = now;
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) { if (!gCam.active) { gCam.active = true; glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); gFirstMouse = true; } }
        else if (gCam.active) { gCam.active = false; glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); }
        process_input(window, dt);
        gSimPaused = ui.IsPaused(); ui.UpdateStats((int)gTotalSteps, gEpisodes, gSPS, gAvgReward, renderEnvIdx, vecEnv->GetNumEnvs());
        if (gNewFrameReady.exchange(false)) gReadBufferIdx.store(gIntermediateBufferIdx.exchange(gReadBufferIdx.load()));
        int rIdx = gReadBufferIdx.load();
        if (rIdx < 3 && !gVisualBuffers[rIdx].empty() && renderEnvIdx < (int)gVisualBuffers[rIdx].size()) {
            const auto& v = gVisualBuffers[rIdx][renderEnvIdx]; ui.UpdateAgentHP(v.r1.hp, v.r2.hp); ui.UpdateAgentRewards(gAgent1Reward, gAgent2Reward);
            gRenderer->Draw(&vecEnv->GetPhysicsCore()->GetPhysicsSystem(), gCam.position, renderEnvIdx, gCam.front, ui.GetGraphics().showCollisionShapes, ui.GetGraphics().showAABBs, false, ui.GetGraphics().showRobot1, ui.GetGraphics().showRobot2, &v);
        }
        if (ui.ShouldRestartSim()) {
            gSimRunning = false; if (simThread.joinable()) simThread.join();
            { std::lock_guard<std::mutex> lock(gSimMutex); delete vecEnv; vecEnv = new VectorizedEnv(ui.GetConfig().numEnvs, ui.GetStepsPerEpisode()); vecEnv->Init(ui.GetConfig().robotConfigPath); }
            gSimRunning = true; simThread = std::thread(SimulationLoop, vecEnv, &trainer, &opponentTrainer, &buffer, &ui, stateDim, actionDim); ui.ClearRestartRequest();
        }
        ui.NewFrame(); ImGui::Begin("JOLTrl Control Center"); ui.DrawAllTabs(); ImGui::End(); ui.Render(); glfwSwapBuffers(window);
    }
    gSimRunning = false; if (simThread.joinable()) simThread.join();
    gIORunning = false; gIOCV.notify_all(); if (ioThread.joinable()) ioThread.join();
    trainer.Save(robotCheckpointDir + "/model_final.bin", "robots/"+robotName+".json", vecEnv->GetEnv(0).GetRobot1().config.numSatellites, stateDim);
    delete gRenderer; delete vecEnv; ui.Shutdown(); glfwTerminate(); return 0;
}
