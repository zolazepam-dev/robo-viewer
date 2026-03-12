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

    // Robot tuning
    cfg.robot.enginePower = mRobotTune.enginePower;
    cfg.robot.reactionWheelPower = mRobotTune.reactionWheelPower;
    cfg.robot.shellRadius = mRobotTune.shellRadius;
    cfg.robot.shellThickness = mRobotTune.shellThickness;
    cfg.robot.shellMass = mRobotTune.shellMass;
    cfg.robot.motorSpeed = mRobotTune.motorSpeed;
    cfg.robot.motorTorque = mRobotTune.motorTorque;
    
    cfg.physics.friction = mPhysics.friction;
    cfg.physics.restitution = mPhysics.restitution;
    cfg.physics.linearDamping = mPhysics.linearDamping;
    cfg.physics.angularDamping = mPhysics.angularDamping;
    cfg.physics.maxPenetrationVelocity = mPhysics.maxPenetrationVelocity;
    cfg.physics.numSubSteps = mPhysics.numSubSteps;
    cfg.physics.warmStarting = mPhysics.warmStarting;

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

    // Robot tuning
    mRobotTune.enginePower = cfg.robot.enginePower;
    mRobotTune.reactionWheelPower = cfg.robot.reactionWheelPower;
    mRobotTune.shellRadius = cfg.robot.shellRadius;
    mRobotTune.shellThickness = cfg.robot.shellThickness;
    mRobotTune.shellMass = cfg.robot.shellMass;
    mRobotTune.motorSpeed = cfg.robot.motorSpeed;
    mRobotTune.motorTorque = cfg.robot.motorTorque;
    
    mPhysics.friction = cfg.physics.friction;
    mPhysics.restitution = cfg.physics.restitution;
    mPhysics.linearDamping = cfg.physics.linearDamping;
    mPhysics.angularDamping = cfg.physics.angularDamping;
    mPhysics.maxPenetrationVelocity = cfg.physics.maxPenetrationVelocity;
    mPhysics.numSubSteps = cfg.physics.numSubSteps;
    mPhysics.warmStarting = cfg.physics.warmStarting;

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
        mRobotSelection.availableRobots.push_back({ def.name, def.configPath });
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
    ImGui::SliderInt("Num Envs", &mConfig.numEnvs, 1, 2048);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of parallel physics environments. Increase to push CPU usage.");
    
    if (ImGui::Button(mPaused ? "RESUME" : "PAUSE", ImVec2(120, 30))) mPaused = !mPaused;
    ImGui::SameLine();
    if (ImGui::Button("SAVE CONFIG", ImVec2(120, 30))) SaveSettings();
    
    ImGui::Separator();
    ImGui::SliderInt("Watch Env", &mRenderEnvIdx, 0, std::max(0, mNumEnvs - 1));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Select which environment to visualize.");
    
    ImGui::SliderFloat("Time Scale", &mTimeScale, 0.1f, 10.0f, "%.2f");
    ImGui::SliderInt("Steps/Episode", &mStepsPerEpisode, 100, 20000);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max steps before environment reset.");
    
    ImGui::Checkbox("Restart Sim", &mRestartRequested);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Apply 'Num Envs' or 'Steps/Episode' changes by restarting simulation.");

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
    ImGui::TextColored(mColorAccent, "SOLVER SETTINGS");
    ImGui::SliderFloat("Gravity", &mPhysics.gravityY, -20.0f, 0.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Global gravity force (m/s²).\nStandard Earth gravity: -9.81\nLower values simulate moon/mars gravity.\nHigher values increase falling speed and impact forces.");

    ImGui::SliderFloat("Timestep", &mPhysics.timestep, 0.001f, 0.033f, "%.4f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Time delta for each physics simulation step.\n0.0083s = 120Hz (recommended)\n0.0167s = 60Hz (standard)\nSmaller values = more accurate but slower simulation.");

    ImGui::SliderInt("Sub Steps", &mPhysics.numSubSteps, 1, 16);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of sub-steps per timestep.\nHigher values improve stability for fast-moving objects\nbut increase CPU cost. Recommended: 1-4.");

    ImGui::SliderInt("Vel Steps", &mPhysics.velocitySteps, 1, 32);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Velocity constraint solver iterations.\nHigher values = more accurate velocity resolution\nbut diminishing returns after 8-16 iterations.");

    ImGui::SliderInt("Pos Steps", &mPhysics.positionSteps, 1, 16);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Position constraint solver iterations.\nCorrects positional drift and penetration.\nHigher values = less jitter but more CPU usage.");

    ImGui::SliderFloat("Baumgarte", &mPhysics.Baumgarte, 0.01f, 1.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Baumgarte stabilization factor.\nControls rate of constraint error correction.\n0.1-0.3 = smooth, 0.5+ = aggressive correction.\nToo high can cause instability.");

    ImGui::Checkbox("Warm Starting", &mPhysics.warmStarting);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reuse constraint impulses from previous frame.\nSignificantly improves solver convergence speed.\nKeep enabled for better performance.");

    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "COLLISION & DYNAMICS");
    ImGui::SliderFloat("Friction", &mPhysics.friction, 0.0f, 2.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Coefficient of friction for all surfaces.\n0.0 = ice (no friction)\n0.5 = wood/standard\n1.0+ = high grip (rubber)\nAffects sliding, traction, and energy loss.");

    ImGui::SliderFloat("Restitution", &mPhysics.restitution, 0.0f, 1.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Coefficient of restitution (bounciness).\n0.0 = no bounce (clay)\n0.3 = moderate bounce (plastic)\n1.0 = perfectly elastic (superball)\nHigher values = more energy retained on impact.");

    ImGui::SliderFloat("Lin Damping", &mPhysics.linearDamping, 0.0f, 1.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Linear velocity damping (air resistance).\n0.0 = no drag (vacuum)\n0.05 = light air resistance\n1.0 = heavy drag (underwater)\nSimulates energy loss from air friction.");

    ImGui::SliderFloat("Ang Damping", &mPhysics.angularDamping, 0.0f, 1.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Angular velocity damping (rotational drag).\n0.0 = no rotational drag\n0.05 = light air resistance\n1.0 = heavy rotational drag\nReduces spinning over time.");

    ImGui::SliderFloat("Max Pen Vel", &mPhysics.maxPenetrationVelocity, 0.1f, 20.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Maximum velocity for penetration resolution.\nPrevents explosive separation of interpenetrating objects.\nLower = gentler correction, Higher = faster separation.");

    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "ADVANCED SOLVER");
    ImGui::SliderFloat("Penetration Slop", &mPhysics.penetrationSlop, 0.0f, 0.1f, "%.4f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Allowed penetration depth before correction.\nSmall values (0.001-0.01) = tight constraints\nLarger values = more stable but visible penetration.\nDefault: 0.005m (5mm).");

    ImGui::SliderFloat("Speculative Contact", &mPhysics.speculativeContactDistance, 0.0f, 0.1f, "%.4f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Distance for predictive contact detection.\nPredicts future collisions to prevent tunneling.\nHigher values = better prevention but more false positives.\nDefault: 0.01m (10mm).");

    ImGui::Checkbox("Allow Sleep", &mPhysics.allowSleep);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Allow bodies to enter sleep state when idle.\nReduces CPU usage for stationary objects.\nDisable for RL training to ensure consistent behavior.");

    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "DOMAIN RANDOMIZATION");
    ImGui::Checkbox("Enable DR", &mConfig.dr.enabled);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Domain Randomization: randomizes physics parameters\non each environment reset. Improves policy\ngeneralization to real-world variations.");

    if (mConfig.dr.enabled) {
        ImGui::SliderFloat("Grav Range", &mConfig.dr.gravityRange, 0.0f, 5.0f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Random gravity variation: g ± range.\nExample: 2.0 means gravity varies between\n-11.81 and -7.81 m/s² on each reset.");

        ImGui::SliderFloat("Frict Range", &mConfig.dr.frictionRange, 0.0f, 0.5f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Random friction variation: μ ± range.\nExample: 0.2 means friction varies by ±0.2\nfrom the base value on each reset.");

        ImGui::SliderFloat("Rest Range", &mConfig.dr.restitutionRange, 0.0f, 0.5f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Random restitution variation: e ± range.\nExample: 0.1 means bounciness varies by ±0.1\nfrom the base value on each reset.");
    }
}

void OverlayUIRefactored::DrawRobotsTab()
{
    ImGui::TextColored(mColorAccent, "ROBOT SELECTION");
    if (ImGui::BeginCombo("Select Robot", mRobotSelection.availableRobots[mRobotSelection.selectedRobotIndex].name.c_str())) {
        for (int i = 0; i < (int)mRobotSelection.availableRobots.size(); i++) {
            bool selected = (i == mRobotSelection.selectedRobotIndex);
            if (ImGui::Selectable(mRobotSelection.availableRobots[i].name.c_str(), selected)) {
                mRobotSelection.selectedRobotIndex = i;
                mConfig.robotConfigPath = mRobotSelection.availableRobots[i].configFile;
            }
        }
        ImGui::EndCombo();
    }
    
    ImGui::Separator();
    ImGui::TextWrapped("Robot-specific parameters (mass, power, sensors) are defined in the robot's JSON configuration file.");
    ImGui::TextColored(mColorDim, "Path: %s", mConfig.robotConfigPath.c_str());

    ImGui::Separator();
    if (ImGui::Button("APPLY & RESTART", ImVec2(200, 30))) {
        SaveSettings();
        mRestartRequested = true;
    }
}

void OverlayUIRefactored::DrawGraphicsTab()
{
    ImGui::TextColored(mColorAccent, "VISUALIZATION");
    ImGui::Checkbox("Collision Shapes", &mGraphics.showCollisionShapes);
    ImGui::Checkbox("AABBs", &mGraphics.showAABBs);
    ImGui::Checkbox("Contact Points", &mGraphics.showContactPoints);
    ImGui::Checkbox("Internal Engines", &mGraphics.showInternalEngines);
    
    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "VISIBILITY");
    ImGui::Checkbox("Show Robot 1", &mGraphics.showRobot1);
    ImGui::Checkbox("Show Robot 2", &mGraphics.showRobot2);
    
    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "CAMERA");
    ImGui::SliderFloat("Dist", &mGraphics.cameraDistance, 5.0f, 200.0f);
    ImGui::SliderFloat("Yaw", &mGraphics.cameraAzimuth, 0.0f, 360.0f);
    ImGui::SliderFloat("Pitch", &mGraphics.cameraElevation, -89.0f, 89.0f);
}

bool OverlayUIRefactored::GetAndClearSaveRequest(std::string& n) { if (mConfig.saveRequested) { n = mConfig.policySaveName; mConfig.saveRequested = false; return true; } return false; }
bool OverlayUIRefactored::GetAndClearLoadRequest(std::string& n) { if (mConfig.loadRequested) { n = mConfig.checkpointLoadName; mConfig.loadRequested = false; return true; } return false; }
bool OverlayUIRefactored::GetAndClearGraphRequest() { bool r = mLaunchGraphRequested; mLaunchGraphRequested = false; return r; }
