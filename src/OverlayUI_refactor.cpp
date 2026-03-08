#include "OverlayUI_refactor.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#define TOOLTIP(text) if (ImGui::IsItemHovered()) { ImGui::SetTooltip(text); }

namespace {
    const ImVec4 mColorAccent = ImVec4(0.0f, 1.0f, 0.85f, 1.0f);
    const ImVec4 mColorAccent2 = ImVec4(1.0f, 0.0f, 0.5f, 1.0f);
    const ImVec4 mColorWarning = ImVec4(1.0f, 0.8f, 0.0f, 1.0f);
    const ImVec4 mColorText = ImVec4(0.9f, 0.95f, 0.9f, 1.0f);
    const ImVec4 mColorDim = ImVec4(0.4f, 0.5f, 0.45f, 1.0f);
}

OverlayUIRefactored::OverlayUIRefactored()
{
    for (int i = 0; i < 5; ++i) {
        mRewardHistory[i].reserve(HISTORY_MAX);
    }
    for (int i = 0; i < 4; ++i) {
        mPhysicsHistory[i].reserve(HISTORY_MAX);
    }
}

void OverlayUIRefactored::Init(GLFWwindow* window)
{
    IMGUI_CHECKVERSION();
    mContext = ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    DrawCyberpunkStyle();
    LoadSettings(); 
}

void OverlayUIRefactored::NewFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void OverlayUIRefactored::Render()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void OverlayUIRefactored::Shutdown()
{
    SaveSettings();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    if (mContext) ImGui::DestroyContext(mContext);
    mContext = nullptr;
}

void OverlayUIRefactored::UItoCentral(CentralConfig& cfg)
{
    cfg.physics.gravityY = mPhysics.gravityY;
    cfg.physics.timestep = mPhysics.timestep;
    cfg.physics.velocitySteps = mPhysics.velocitySteps;
    cfg.physics.positionSteps = mPhysics.positionSteps;
    cfg.physics.Baumgarte = mPhysics.Baumgarte;
    cfg.physics.penetrationSlop = mPhysics.penetrationSlop;
    cfg.physics.speculativeContactDistance = mPhysics.speculativeContactDistance;
    cfg.physics.allowSleep = mPhysics.allowSleep;
    cfg.physics.timeScale = mTimeScale;
    cfg.physics.stepsPerEpisode = mStepsPerEpisode;

    cfg.robot.enginePower = mRobotTune.enginePower;
    cfg.robot.reactionWheelPower = mRobotTune.reactionWheelPower;
    cfg.robot.shellRadius = mRobotTune.shellRadius;
    cfg.robot.shellThickness = mRobotTune.shellThickness;
    cfg.robot.shellMass = mRobotTune.shellMass;
    cfg.robot.motorSpeed = mRobotTune.motorSpeed;
    cfg.robot.motorTorque = mRobotTune.motorTorque;

    cfg.graphics.showCollisionShapes = mGraphics.showCollisionShapes;
    cfg.graphics.showAABBs = mGraphics.showAABBs;
    cfg.graphics.showContactPoints = mGraphics.showContactPoints;
    cfg.graphics.showRobot1 = mGraphics.showRobot1;
    cfg.graphics.showRobot2 = mGraphics.showRobot2;
    cfg.graphics.showInternalEngines = mGraphics.showInternalEngines;
    cfg.graphics.cameraDistance = mGraphics.cameraDistance;
    cfg.graphics.cameraAzimuth = mGraphics.cameraAzimuth;
    cfg.graphics.cameraElevation = mGraphics.cameraElevation;

    cfg.training.numEnvs = mConfig.numEnvs;
    cfg.training.checkpointInterval = mConfig.checkpointInterval;
    cfg.training.checkpointDir = mConfig.checkpointDir;
    cfg.training.robotConfigPath = mConfig.robotConfigPath;
}

void OverlayUIRefactored::CentraltoUI(const CentralConfig& cfg)
{
    mPhysics.gravityY = cfg.physics.gravityY;
    mPhysics.timestep = cfg.physics.timestep;
    mPhysics.velocitySteps = cfg.physics.velocitySteps;
    mPhysics.positionSteps = cfg.physics.positionSteps;
    mPhysics.Baumgarte = cfg.physics.Baumgarte;
    mPhysics.penetrationSlop = cfg.physics.penetrationSlop;
    mPhysics.speculativeContactDistance = cfg.physics.speculativeContactDistance;
    mPhysics.allowSleep = cfg.physics.allowSleep;
    mTimeScale = cfg.physics.timeScale;
    mStepsPerEpisode = cfg.physics.stepsPerEpisode;

    mRobotTune.enginePower = cfg.robot.enginePower;
    mRobotTune.reactionWheelPower = cfg.robot.reactionWheelPower;
    mRobotTune.shellRadius = cfg.robot.shellRadius;
    mRobotTune.shellThickness = cfg.robot.shellThickness;
    mRobotTune.shellMass = cfg.robot.shellMass;
    mRobotTune.motorSpeed = cfg.robot.motorSpeed;
    mRobotTune.motorTorque = cfg.robot.motorTorque;

    mGraphics.showCollisionShapes = cfg.graphics.showCollisionShapes;
    mGraphics.showAABBs = cfg.graphics.showAABBs;
    mGraphics.showContactPoints = cfg.graphics.showContactPoints;
    mGraphics.showRobot1 = cfg.graphics.showRobot1;
    mGraphics.showRobot2 = cfg.graphics.showRobot2;
    mGraphics.showInternalEngines = cfg.graphics.showInternalEngines;
    mGraphics.cameraDistance = cfg.graphics.cameraDistance;
    mGraphics.cameraAzimuth = cfg.graphics.cameraAzimuth;
    mGraphics.cameraElevation = cfg.graphics.cameraElevation;

    mConfig.numEnvs = cfg.training.numEnvs;
    mConfig.checkpointInterval = cfg.training.checkpointInterval;
    mConfig.checkpointDir = cfg.training.checkpointDir;
    mConfig.robotConfigPath = cfg.training.robotConfigPath;
    
    mRobotSelection.availableRobots.clear();
    for (const auto& def : cfg.robotDefinitions) {
        mRobotSelection.availableRobots.push_back({
            def.name, def.configPath, def.enginePower, def.reactionWheelPower,
            def.shellRadius, def.shellThickness, def.shellMass,
            def.motorSpeed, def.motorTorque
        });
    }
}

void OverlayUIRefactored::SaveSettings(const std::string& path)
{
    UItoCentral(ConfigManager::GetInstance().GetConfig());
    ConfigManager::GetInstance().SaveConfig(path);
}

void OverlayUIRefactored::LoadSettings(const std::string& path)
{
    if (ConfigManager::GetInstance().LoadConfig(path)) {
        CentraltoUI(ConfigManager::GetInstance().GetConfig());
    }
}

void OverlayUIRefactored::DrawAllTabs() { DrawTabBar(); }

void OverlayUIRefactored::UpdateStats(int totalSteps, int episodes, float sps, float avgReward, int currentEnv, int numEnvs)
{
    mTotalSteps = totalSteps; mEpisodes = episodes; mSPS = sps; mAvgReward = avgReward; mRenderEnvIdx = currentEnv; mNumEnvs = numEnvs;
}

void OverlayUIRefactored::PushRewardData(float damageDealt, float damageTaken, float airtime, float energy, float scalar)
{
    mRewardHistory[0].push_back(damageDealt); mRewardHistory[1].push_back(damageTaken); mRewardHistory[2].push_back(airtime);
    mRewardHistory[3].push_back(energy); mRewardHistory[4].push_back(scalar);
    for (int i = 0; i < 5; ++i) if (mRewardHistory[i].size() > HISTORY_MAX) mRewardHistory[i].erase(mRewardHistory[i].begin());
}

void OverlayUIRefactored::PushPhysicsMetrics(float s, float b, float c, float i)
{
    mPhysicsHistory[0].push_back(s); mPhysicsHistory[1].push_back(b); mPhysicsHistory[2].push_back(c); mPhysicsHistory[3].push_back(i);
    for (int j = 0; j < 4; ++j) if (mPhysicsHistory[j].size() > HISTORY_MAX) mPhysicsHistory[j].erase(mPhysicsHistory[j].begin());
}

void OverlayUIRefactored::DrawCyberpunkStyle()
{
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(8, 8); style.FramePadding = ImVec2(6, 4);
    style.WindowRounding = 4.0f; style.FrameRounding = 3.0f;
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text] = mColorText;
    colors[ImGuiCol_WindowBg] = ImVec4(0.03f, 0.03f, 0.07f, 0.95f);
    colors[ImGuiCol_Border] = ImVec4(0.0f, 1.0f, 0.85f, 0.3f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.0f, 1.0f, 0.85f, 1.0f);
    colors[ImGuiCol_CheckMark] = mColorAccent;
    colors[ImGuiCol_SliderGrab] = mColorAccent;
    colors[ImGuiCol_Button] = ImVec4(0.0f, 1.0f, 0.85f, 0.2f);
}

void OverlayUIRefactored::PlotLine(const char* label, const std::vector<float>& data, float scale_min, float scale_max)
{
    if (!data.empty()) ImGui::PlotLines(label, data.data(), (int)data.size(), 0, nullptr, scale_min, scale_max, ImVec2(0, 50));
}

void OverlayUIRefactored::DrawTabBar()
{
    if (ImGui::BeginTabBar("MainTabBar")) {
        if (ImGui::BeginTabItem("Training")) { DrawTrainingTab(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Physics")) { DrawPhysicsTab(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Robots")) { DrawRobotsTab(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Graphics")) { DrawGraphicsTab(); ImGui::EndTabItem(); }
        ImGui::EndTabBar();
    }
}

void OverlayUIRefactored::DrawTrainingTab()
{
    ImGui::TextColored(mColorAccent, "TRAINING CONTROLS");
    ImGui::SliderInt("Num Envs", &mConfig.numEnvs, 1, 256);
    if (ImGui::Button(mPaused ? "RESUME" : "PAUSE", ImVec2(120, 30))) mPaused = !mPaused;
    ImGui::SameLine();
    if (ImGui::Button("SAVE CONFIG", ImVec2(120, 30))) SaveSettings();
    
    ImGui::Separator();
    ImGui::SliderInt("Watch Env", &mRenderEnvIdx, 0, std::max(0, mNumEnvs - 1));
    ImGui::SliderFloat("Time Scale", &mTimeScale, 0.1f, 4.0f, "%.2f");
    ImGui::Checkbox("Restart Sim", &mRestartRequested);

    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "STATISTICS");
    ImGui::Text("Steps: %d", mTotalSteps);
    ImGui::Text("Episodes: %d", mEpisodes);
    ImGui::TextColored(mColorWarning, "SPS: %.0f", mSPS);
    
    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "REWARDS");
    ImGui::Text("Avg: %.3f", mAvgReward);
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "A1: %.3f", mAgent1Reward);
    ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "A2: %.3f", mAgent2Reward);
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "HEALTH");
    ImGui::Text("A1: %.1f | A2: %.1f", mAgent1HP, mAgent2HP);
}

void OverlayUIRefactored::DrawPhysicsTab()
{
    ImGui::TextColored(mColorAccent, "PHYSICS");
    ImGui::SliderFloat("Gravity", &mPhysics.gravityY, -20.0f, 0.0f);
    ImGui::SliderFloat("Timestep", &mPhysics.timestep, 0.001f, 0.02f, "%.4f");
}

void OverlayUIRefactored::DrawRobotsTab()
{
    ImGui::TextColored(mColorAccent, "MASTER CONFIGURATION");
    ImGui::Separator();
    ImGui::TextWrapped("Robot settings are managed via viewer_config.json.");
    if (ImGui::Button("RELOAD CONFIG", ImVec2(200, 30))) {
        LoadSettings();
        mRestartRequested = true;
    }
}

void OverlayUIRefactored::DrawGraphicsTab()
{
    ImGui::TextColored(mColorAccent, "VISUALS");
    ImGui::Checkbox("Collision Shapes", &mGraphics.showCollisionShapes);
    ImGui::SliderFloat("Cam Dist", &mGraphics.cameraDistance, 5.0f, 100.0f);
    ImGui::SliderFloat("Azimuth", &mGraphics.cameraAzimuth, 0.0f, 360.0f);
}

bool OverlayUIRefactored::GetAndClearSaveRequest(std::string& n) { if (mConfig.saveRequested) { n = mConfig.policySaveName; mConfig.saveRequested = false; return true; } return false; }
bool OverlayUIRefactored::GetAndClearLoadRequest(std::string& n) { if (mConfig.loadRequested) { n = mConfig.checkpointLoadName; mConfig.loadRequested = false; return true; } return false; }
bool OverlayUIRefactored::GetAndClearGraphRequest() { bool r = mLaunchGraphRequested; mLaunchGraphRequested = false; return r; }
bool OverlayUIRefactored::GetSpawnRequest(SpawnRequest& r) { if (mSpawnRequest.valid) { r = mSpawnRequest; mSpawnRequest.valid = false; return true; } return false; }
void OverlayUIRefactored::SetSpawnClickPosition(const JPH::Vec3& p) { mPendingSpawnPos = p; }
bool OverlayUIRefactored::GetAndClearLoadConfigRequest() { bool r = mRobotSelection.loadConfigRequested; mRobotSelection.loadConfigRequested = false; return r; }
bool OverlayUIRefactored::GetAndClearCreateCheckpointFolderRequest(std::string& n) { if (mRobotSelection.createCheckpointFolderRequested) { n = mRobotSelection.newCheckpointFolderName; mRobotSelection.createCheckpointFolderRequested = false; return true; } return false; }
const std::string& OverlayUIRefactored::GetSelectedRobotType() const { return mRobotSelection.availableRobots[mRobotSelection.selectedRobotIndex].name; }
// Inline methods moved to header

