#include "OverlayUI_refactor.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <fstream>
#include <iostream>
#include <cmath>
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
    
    // Disable loading/saving of imgui.ini - always use defaults from config
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // Disable .ini file
    io.WantSaveIniSettings = false;
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    DrawCyberpunkStyle();
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

void OverlayUIRefactored::DrawAllTabs()
{
    DrawTabBar();
}

void OverlayUIRefactored::Shutdown()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(mContext);
    mContext = nullptr;
}

void OverlayUIRefactored::UpdateStats(int totalSteps, int episodes, float sps, float avgReward,
                                       int currentEnv, int numEnvs)
{
    mTotalSteps = totalSteps;
    mEpisodes = episodes;
    mSPS = sps;
    mAvgReward = avgReward;
    mRenderEnvIdx = currentEnv;
    mNumEnvs = numEnvs;
}

void OverlayUIRefactored::PushRewardData(float damageDealt, float damageTaken, 
                                         float airtime, float energy, float scalar)
{
    mRewardHistory[0].push_back(damageDealt);
    mRewardHistory[1].push_back(damageTaken);
    mRewardHistory[2].push_back(airtime);
    mRewardHistory[3].push_back(energy);
    mRewardHistory[4].push_back(scalar);
    
    for (int i = 0; i < 5; ++i) {
        if (mRewardHistory[i].size() > HISTORY_MAX) {
            mRewardHistory[i].erase(mRewardHistory[i].begin());
        }
    }
}

void OverlayUIRefactored::PushPhysicsMetrics(float solverTime, float broadphaseTime, 
                                               float collisionTime, float integrateTime)
{
    mPhysicsHistory[0].push_back(solverTime);
    mPhysicsHistory[1].push_back(broadphaseTime);
    mPhysicsHistory[2].push_back(collisionTime);
    mPhysicsHistory[3].push_back(integrateTime);
    
    for (int i = 0; i < 4; ++i) {
        if (mPhysicsHistory[i].size() > HISTORY_MAX) {
            mPhysicsHistory[i].erase(mPhysicsHistory[i].begin());
        }
    }
}

void OverlayUIRefactored::UpdateBatteryHistory(const BatteryState& r1, const BatteryState& r2, 
                                                const JPH::Vec3& cog1, float mass1,
                                                const JPH::Vec3& cog2, float mass2)
{
    mBatteryHistoryR1.Push(r1.currentEnergy, r1.temperature, r1.currentCharge, r1.currentDraw,
                           cog1.GetX(), cog1.GetY(), cog1.GetZ(), mass1);
    mBatteryHistoryR2.Push(r2.currentEnergy, r2.temperature, r2.currentCharge, r2.currentDraw,
                           cog2.GetX(), cog2.GetY(), cog2.GetZ(), mass2);
}

void OverlayUIRefactored::DrawBatteryTab()
{
    const ImVec4 mColorAccent = ImVec4(0.0f, 1.0f, 0.85f, 1.0f);
    const ImVec4 mColorAccent2 = ImVec4(1.0f, 0.0f, 0.5f, 1.0f);
    
    if (ImGui::BeginTabBar("BatteryTabs")) {
        // Robot 1 Tab
        if (ImGui::BeginTabItem("Robot 1")) {
            DrawBatteryPanel(mBatteryHistoryR1, mColorAccent);
            ImGui::EndTabItem();
        }
        
        // Robot 2 Tab
        if (ImGui::BeginTabItem("Robot 2")) {
            DrawBatteryPanel(mBatteryHistoryR2, mColorAccent2);
            ImGui::EndTabItem();
        }
        
        // Comparison Tab
        if (ImGui::BeginTabItem("Comparison")) {
            DrawBatteryComparison();
            ImGui::EndTabItem();
        }
        
        ImGui::EndTabBar();
    }
}

void OverlayUIRefactored::DrawBatteryPanel(const BatteryHistory& history, ImVec4 color)
{
    float barWidth = ImGui::GetContentRegionAvail().x;
    
    // Get latest values from history
    int idx = (history.writeIdx > 0) ? history.writeIdx - 1 : history.HISTORY_SIZE - 1;
    float energy = history.energyHistory[idx];
    float temp = history.tempHistory[idx];
    float chargeRate = history.chargeRateHistory[idx];
    float dischargeRate = history.dischargeRateHistory[idx];
    
    // Battery Level
    ImGui::TextColored(color, "Energy Storage");
    float energyPct = energy / BATTERY_DEFAULT_CAPACITY;
    ImVec4 energyColor = energyPct > 0.6f ? ImVec4(0,1,0.5,1) : energyPct > 0.3f ? ImVec4(1,0.85,0,1) : ImVec4(1,0,0,1);
    
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, energyColor);
    ImGui::ProgressBar(energyPct, ImVec2(barWidth, 40), "");
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::Text("%.0f/%.0f J (%.1f%%)", energy, BATTERY_DEFAULT_CAPACITY, energyPct * 100.0f);
    
    // Temperature
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1,0.3,0,1), "Thermal Management");
    float tempPct = (temp - BATTERY_AMBIENT_TEMP) / (BATTERY_OVERHEAT_TEMP - BATTERY_AMBIENT_TEMP);
    tempPct = fmaxf(0.0f, fminf(1.0f, tempPct));
    ImVec4 tempColor = tempPct > 0.8f ? ImVec4(1,0,0,1) : tempPct > 0.5f ? ImVec4(1,0.85,0,1) : ImVec4(0,1,0.5,1);
    
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, tempColor);
    ImGui::ProgressBar(tempPct, ImVec2(barWidth, 25), "");
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::Text("%.1f°C / %.0f°C", temp, BATTERY_OVERHEAT_TEMP);
    
    // Charge/Discharge Rates
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("⚡ Power Flow");
    ImGui::TextColored(ImVec4(0,1,0.5,1), "  Charging: %.1f J/s", chargeRate);
    ImGui::TextColored(ImVec4(1,0.3,0,1), "  Discharging: %.1f J/s", dischargeRate);
    
    // Center of Gravity & Mass
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("⚖️ Mass & Balance");
    ImGui::Text("  Total Mass: %.1f kg", history.massHistory[idx]);
    ImGui::Text("  CoG Offset: (%.3f, %.3f, %.3f)m", 
                history.cogXHistory[idx], history.cogYHistory[idx], history.cogZHistory[idx]);
    
    float cogDist = std::sqrt(history.cogXHistory[idx] * history.cogXHistory[idx] +
                              history.cogYHistory[idx] * history.cogYHistory[idx] +
                              history.cogZHistory[idx] * history.cogZHistory[idx]);
    ImGui::Text("  CoG Displacement: %.3f m", cogDist);
    
    float cogStability = fmaxf(0.0f, 1.0f - cogDist / 0.5f);
    ImVec4 cogColor = cogStability > 0.7f ? ImVec4(0,1,0.5,1) : cogStability > 0.3f ? ImVec4(1,0.85,0,1) : ImVec4(1,0,0,1);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, cogColor);
    ImGui::ProgressBar(cogStability, ImVec2(barWidth, 20), "");
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::Text("Stability");
    
    // Real-time Graphs
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("📊 Real-time Graphs");
    
    ImGui::Text("Energy Level");
    ImGui::PlotLines("##EnergyHist", history.energyHistory.data(), 300, 0, nullptr, 0.0f, BATTERY_DEFAULT_CAPACITY, ImVec2(barWidth, 80));
    
    ImGui::Text("Temperature");
    ImGui::PlotLines("##TempHist", history.tempHistory.data(), 300, 0, nullptr, BATTERY_AMBIENT_TEMP, BATTERY_OVERHEAT_TEMP, ImVec2(barWidth, 60));
    
    ImGui::Text("Charge/Discharge Rate");
    ImGui::PlotLines("##ChargeHist", history.chargeRateHistory.data(), 300, 0, nullptr, 0.0f, 100.0f, ImVec2(barWidth, 60));
}
void OverlayUIRefactored::DrawBatteryComparison()
{
    float barWidth = ImGui::GetContentRegionAvail().x / 2.0f - 10.0f;
    
    ImGui::Columns(2, "BatteryCompare", false);
    
    // Robot 1
    ImGui::PushStyleColor(ImGuiCol_Text, mColorAccent);
    ImGui::Text("Robot 1");
    ImGui::PopStyleColor();
    ImGui::Separator();
    
    int idx1 = (mBatteryHistoryR1.writeIdx > 0) ? mBatteryHistoryR1.writeIdx - 1 : mBatteryHistoryR1.HISTORY_SIZE - 1;
    float energy1 = mBatteryHistoryR1.energyHistory[idx1];
    float temp1 = mBatteryHistoryR1.tempHistory[idx1];
    float charge1 = mBatteryHistoryR1.chargeRateHistory[idx1];
    float discharge1 = mBatteryHistoryR1.dischargeRateHistory[idx1];
    
    ImGui::Text("Energy: %.0f J", energy1);
    ImGui::Text("Temp: %.1f°C", temp1);
    ImGui::Text("Charging: %.1f J/s", charge1);
    ImGui::Text("Discharging: %.1f J/s", discharge1);
    
    ImGui::NextColumn();
    
    // Robot 2
    ImGui::PushStyleColor(ImGuiCol_Text, mColorAccent2);
    ImGui::Text("Robot 2");
    ImGui::PopStyleColor();
    ImGui::Separator();
    
    int idx2 = (mBatteryHistoryR2.writeIdx > 0) ? mBatteryHistoryR2.writeIdx - 1 : mBatteryHistoryR2.HISTORY_SIZE - 1;
    float energy2 = mBatteryHistoryR2.energyHistory[idx2];
    float temp2 = mBatteryHistoryR2.tempHistory[idx2];
    float charge2 = mBatteryHistoryR2.chargeRateHistory[idx2];
    float discharge2 = mBatteryHistoryR2.dischargeRateHistory[idx2];
    
    ImGui::Text("Energy: %.0f J", energy2);
    ImGui::Text("Temp: %.1f°C", temp2);
    ImGui::Text("Charging: %.1f J/s", charge2);
    ImGui::Text("Discharging: %.1f J/s", discharge2);
    
    ImGui::Columns(1);
    
    // Side-by-side energy graph
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Energy Comparison");
    
    ImGui::PlotLines("##R1Energy", mBatteryHistoryR1.energyHistory.data(), 300, 0, "R1", 0.0f, BATTERY_DEFAULT_CAPACITY, ImVec2(barWidth, 100));
    ImGui::SameLine();
    ImGui::PlotLines("##R2Energy", mBatteryHistoryR2.energyHistory.data(), 300, 0, "R2", 0.0f, BATTERY_DEFAULT_CAPACITY, ImVec2(barWidth, 100));
}

void OverlayUIRefactored::DrawCyberpunkStyle()
{
    ImGuiStyle& style = ImGui::GetStyle();
    
    style.WindowPadding = ImVec2(8, 8);
    style.FramePadding = ImVec2(6, 4);
    style.CellPadding = ImVec2(4, 2);
    style.ItemSpacing = ImVec2(8, 4);
    style.ItemInnerSpacing = ImVec2(6, 4);
    style.TouchExtraPadding = ImVec2(0, 0);
    style.IndentSpacing = 20.0f;
    style.ScrollbarSize = 14.0f;
    style.GrabMinSize = 10.0f;
    
    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.TabBorderSize = 0.0f;
    
    style.WindowRounding = 4.0f;
    style.ChildRounding = 4.0f;
    style.FrameRounding = 3.0f;
    style.PopupRounding = 3.0f;
    style.ScrollbarRounding = 9.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 3.0f;
    
    style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
    style.WindowMenuButtonPosition = ImGuiDir_Left;
    style.ColorButtonPosition = ImGuiDir_Right;
    style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
    style.SelectableTextAlign = ImVec2(0.0f, 0.0f);
    
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text] = mColorText;
    colors[ImGuiCol_TextDisabled] = mColorDim;
    colors[ImGuiCol_WindowBg] = ImVec4(0.03f, 0.03f, 0.07f, 0.95f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.02f, 0.02f, 0.05f, 1.0f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.04f, 0.04f, 0.09f, 0.98f);
    colors[ImGuiCol_Border] = ImVec4(0.0f, 1.0f, 0.85f, 0.3f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.05f, 0.05f, 0.1f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.0f, 1.0f, 0.85f, 0.15f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.0f, 1.0f, 0.85f, 0.25f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.0f, 0.5f, 0.4f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.0f, 1.0f, 0.85f, 1.0f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.0f, 0.3f, 0.25f, 0.8f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.04f, 0.04f, 0.08f, 1.0f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.05f, 0.6f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.0f, 1.0f, 0.85f, 0.4f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.0f, 1.0f, 0.85f, 0.6f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.0f, 1.0f, 0.85f, 0.8f);
    colors[ImGuiCol_CheckMark] = mColorAccent;
    colors[ImGuiCol_SliderGrab] = mColorAccent;
    colors[ImGuiCol_SliderGrabActive] = mColorAccent2;
    colors[ImGuiCol_Button] = ImVec4(0.0f, 1.0f, 0.85f, 0.2f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.0f, 1.0f, 0.85f, 0.4f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.0f, 1.0f, 0.85f, 0.6f);
    colors[ImGuiCol_Header] = ImVec4(0.0f, 1.0f, 0.85f, 0.2f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.0f, 1.0f, 0.85f, 0.35f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.0f, 1.0f, 0.85f, 0.5f);
    colors[ImGuiCol_Separator] = ImVec4(0.0f, 1.0f, 0.85f, 0.25f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.0f, 1.0f, 0.85f, 0.5f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.0f, 1.0f, 0.85f, 0.7f);
    colors[ImGuiCol_ResizeGrip] = mColorAccent;
    colors[ImGuiCol_ResizeGripHovered] = mColorAccent;
    colors[ImGuiCol_ResizeGripActive] = mColorAccent2;
    colors[ImGuiCol_Tab] = ImVec4(0.0f, 0.5f, 0.4f, 0.8f);
    colors[ImGuiCol_TabHovered] = mColorAccent;
    colors[ImGuiCol_TabActive] = ImVec4(0.0f, 0.8f, 0.7f, 1.0f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.0f, 0.3f, 0.25f, 0.8f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.0f, 0.5f, 0.4f, 1.0f);
    colors[ImGuiCol_PlotLines] = mColorAccent;
    colors[ImGuiCol_PlotLinesHovered] = mColorAccent2;
    colors[ImGuiCol_PlotHistogram] = mColorAccent;
    colors[ImGuiCol_PlotHistogramHovered] = mColorAccent2;
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.0f, 1.0f, 0.85f, 0.35f);
    colors[ImGuiCol_DragDropTarget] = mColorWarning;
    colors[ImGuiCol_NavHighlight] = mColorAccent;
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.0f, 1.0f, 0.85f, 0.6f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.8f, 0.8f, 0.8f, 0.2f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.6f);
}

void OverlayUIRefactored::PlotLine(const char* label, const std::vector<float>& data, float scale_min, float scale_max)
{
    if (data.empty()) return;
    
    int count = static_cast<int>(data.size());
    ImGui::PlotLines(label, data.data(), count, 0, nullptr, scale_min, scale_max, ImVec2(0, 50));
}

void OverlayUIRefactored::DrawTabBar()
{
    const char* tabs[] = {"Training", "Physics", "Robots", "Graphics", "Policy", "Spawn", "Episodes", "Battery", "Graph"};
    static int selectedTab = 0;
    
    if (ImGui::BeginTabBar("MainTabBar", ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < 9; ++i) {
            if (ImGui::BeginTabItem(tabs[i])) {
                selectedTab = i;
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
        
        switch (selectedTab) {
            case 0: DrawTrainingTab(); break;
            case 1: DrawPhysicsTab(); break;
            case 2: DrawRobotsTab(); break;
            case 3: DrawGraphicsTab(); break;
            case 4: DrawPolicyTab(); break;
            case 5: DrawSpawnTab(); break;
            case 6: DrawEpisodesTab(); break;
            case 7: DrawBatteryTab(); break;
            case 8: DrawGraphSelector(); break;
        }
    }
}

void OverlayUIRefactored::DrawTrainingTab()
{
    ImGui::TextColored(mColorAccent, "TRAINING CONTROLS");
    ImGui::Separator();
    
    ImGui::SliderInt("Num Envs", &mConfig.numEnvs, 1, 1024);
    TOOLTIP("Number of parallel training environments")
    ImGui::SliderInt("Steps/Episode", &mStepsPerEpisode, 100, 10000);
    TOOLTIP("Maximum steps before episode reset")
    ImGui::SliderInt("Checkpoint Interval", &mConfig.checkpointInterval, 1000, 500000);
    TOOLTIP("Save model every N steps")
    
    if (mLoadedSettings.isLoaded) {
        ImGui::Separator();
        ImGui::TextColored(mColorWarning, "Settings file loaded - click Apply to use");
        if (ImGui::Button("APPLY LOADED SETTINGS", ImVec2(200, 30))) {
            // Copy loaded settings to active settings
            mPhysics = mLoadedSettings.physics;
            mRobotTune = mLoadedSettings.robot;
            mConfig.numEnvs = mLoadedSettings.training.numEnvs;
            mConfig.checkpointInterval = mLoadedSettings.training.checkpointInterval;
            mConfig.checkpointDir = mLoadedSettings.training.checkpointDir;
            mGraphics = mLoadedSettings.graphics;
            
            // Update local variables
            mTimeScale = mPhysics.timeScale;
            mStepsPerEpisode = mPhysics.stepsPerEpisode;
            
            mLoadedSettings.isLoaded = false;
            std::cout << "[OverlayUI] Loaded settings applied to active config" << std::endl;
        }
        TOOLTIP("Apply the settings loaded from file")
    }
    
    ImGui::Separator();
    ImGui::TextColored(mColorDim, "RUNTIME CONTROL");
    ImGui::Separator();
    
    if (ImGui::Button(mPaused ? "RESUME" : "PAUSE", ImVec2(120, 30))) {
        mPaused = !mPaused;
    }
    TOOLTIP("Pause/resume training")
    ImGui::SameLine();
    if (ImGui::Button("STEP ONE", ImVec2(120, 30))) {
        mStepOne = true;
    }
    TOOLTIP("Execute single training step when paused")
    ImGui::SameLine();
    if (ImGui::Button("RESET", ImVec2(120, 30))) {
        mResetRequested = true;
    }
    TOOLTIP("Reset all environments")
    
    ImGui::Separator();
    ImGui::TextColored(mColorDim, "ENVIRONMENT");
    ImGui::Separator();
    
    ImGui::SliderInt("Watch Env", &mRenderEnvIdx, 0, std::max(0, mNumEnvs - 1));
    TOOLTIP("Select environment to visualize")
    ImGui::SliderFloat("Time Scale", &mTimeScale, 0.1f, 4.0f, "%.2f");
    TOOLTIP("Physics time scale (1.0 = realtime)")
    ImGui::Checkbox("Restart with New Envs", &mRestartRequested);
    TOOLTIP("Restart simulation with new environment count")
    
    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "STATISTICS");
    ImGui::Separator();
    ImGui::Text("Total Steps: %d", mTotalSteps);
    ImGui::Text("Episodes: %d", mEpisodes);
    ImGui::TextColored(mColorWarning, "SPS: %.0f", mSPS);
    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "AGENT REWARDS");
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Agent 1: %.3f", mAgent1Reward);
    ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "Agent 2: %.3f", mAgent2Reward);
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "HEALTH");
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Agent 1: %.1f / 100", mAgent1HP);
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Agent 2: %.1f / 100", mAgent2HP);
    ImGui::Text("HP Δ: A1 %.3f | A2 %.3f", mAgent1HPDelta, mAgent2HPDelta);
    ImGui::Text("Impulse: %.3f", mLastImpulse);
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "LEAGUE PLAY");
    ImGui::Text("Opponent ID: %d", mCurrentOpponentIdx);
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "DEBUG VALUES");
    ImGui::Separator();
    ImGui::Text("DEBUG: mConfig.numEnvs = %d", mConfig.numEnvs);
    ImGui::Text("DEBUG: mStepsPerEpisode = %d", mStepsPerEpisode);
    ImGui::Text("DEBUG: mConfig.checkpointInterval = %d", mConfig.checkpointInterval);
    ImGui::Text("DEBUG: mPaused = %s", mPaused ? "true" : "false");
    ImGui::Text("DEBUG: mStepOne = %s", mStepOne ? "true" : "false");
    ImGui::Text("DEBUG: mResetRequested = %s", mResetRequested ? "true" : "false");
    ImGui::Text("DEBUG: mRenderEnvIdx = %d", mRenderEnvIdx);
    ImGui::Text("DEBUG: mTimeScale = %.2f", mTimeScale);
    ImGui::Text("DEBUG: mRestartRequested = %s", mRestartRequested ? "true" : "false");
    ImGui::Text("DEBUG: mAgent1Reward = %.3f", mAgent1Reward);
    ImGui::Text("DEBUG: mAgent2Reward = %.3f", mAgent2Reward);
    ImGui::Text("DEBUG: mAgent1HP = %.1f", mAgent1HP);
    ImGui::Text("DEBUG: mAgent2HP = %.1f", mAgent2HP);
    ImGui::Text("DEBUG: mCurrentOpponentIdx = %d", mCurrentOpponentIdx);
}

void OverlayUIRefactored::DrawPhysicsTab()
{
    ImGui::TextColored(mColorAccent, "PHYSICS TUNABLES");
    ImGui::Separator();
    
    ImGui::SliderFloat("Gravity Y", &mPhysics.gravityY, -20.0f, 0.0f, "%.2f");
    TOOLTIP("Gravity acceleration in Y direction (negative = downward)")
    ImGui::SliderFloat("Timestep", &mPhysics.timestep, 1.0f/240.0f, 1.0f/30.0f, "%.4f");
    TOOLTIP("Physics simulation timestep (smaller = more accurate)")
    ImGui::SliderInt("Velocity Steps", &mPhysics.velocitySteps, 1, 10);
    TOOLTIP("Number of velocity solver iterations per step")
    ImGui::SliderInt("Position Steps", &mPhysics.positionSteps, 1, 10);
    TOOLTIP("Number of position solver iterations per step")
    
    ImGui::Separator();
    ImGui::TextColored(mColorDim, "SOLVER");
    ImGui::Separator();
    ImGui::SliderFloat("Baumgarte", &mPhysics.Baumgarte, 0.0f, 1.0f, "%.3f");
    TOOLTIP("Baumgarte stabilization factor for contact constraints")
    ImGui::SliderFloat("Penetration Slop", &mPhysics.penetrationSlop, 0.001f, 0.1f, "%.3f");
    TOOLTIP("Allowed penetration depth before correction")
    ImGui::SliderFloat("Speculative Contact Dist", &mPhysics.speculativeContactDistance, 0.001f, 0.1f, "%.3f");
    TOOLTIP("Distance for speculative contact generation")
    
    ImGui::Separator();
    ImGui::TextColored(mColorDim, "BEHAVIOR");
    ImGui::Separator();
    
    ImGui::Checkbox("Allow Sleep", &mPhysics.allowSleep);
    TOOLTIP("Allow bodies to sleep when not moving")
    ImGui::SliderFloat("Time Scale", &mPhysics.timeScale, 0.1f, 4.0f, "%.2f");
    TOOLTIP("Physics time scale multiplier")
    ImGui::SliderInt("Steps Per Episode", &mPhysics.stepsPerEpisode, 100, 10000);
    TOOLTIP("Maximum physics steps per episode")
    
    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "CURRENT VALUES");
    ImGui::Separator();
    ImGui::Text("Gravity: %.2f", mPhysics.gravityY);
    ImGui::Text("Timestep: %.6f", mPhysics.timestep);
    ImGui::Text("Solver: %d/%d", mPhysics.velocitySteps, mPhysics.positionSteps);
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "DEBUG VALUES");
    ImGui::Separator();
    ImGui::Text("DEBUG: mPhysics.gravityY = %.2f", mPhysics.gravityY);
    ImGui::Text("DEBUG: mPhysics.timestep = %.6f", mPhysics.timestep);
    ImGui::Text("DEBUG: mPhysics.velocitySteps = %d", mPhysics.velocitySteps);
    ImGui::Text("DEBUG: mPhysics.positionSteps = %d", mPhysics.positionSteps);
    ImGui::Text("DEBUG: mPhysics.Baumgarte = %.3f", mPhysics.Baumgarte);
    ImGui::Text("DEBUG: mPhysics.penetrationSlop = %.3f", mPhysics.penetrationSlop);
    ImGui::Text("DEBUG: mPhysics.speculativeContactDistance = %.3f", mPhysics.speculativeContactDistance);
    ImGui::Text("DEBUG: mPhysics.allowSleep = %s", mPhysics.allowSleep ? "true" : "false");
    ImGui::Text("DEBUG: mPhysics.timeScale = %.2f", mPhysics.timeScale);
    ImGui::Text("DEBUG: mPhysics.stepsPerEpisode = %d", mPhysics.stepsPerEpisode);
}

void OverlayUIRefactored::DrawRobotsTab()
{
    ImGui::TextColored(mColorAccent, "ROBOT PARAMETERS");
    ImGui::Separator();
    
    ImGui::SliderFloat("Engine Power", &mRobotTune.enginePower, 10.0f, 500.0f, "%.1f");
    TOOLTIP("Thrust force generated by internal engines (Newtons)")
    ImGui::SliderFloat("Reaction Wheel Power", &mRobotTune.reactionWheelPower, 100.0f, 20000.0f, "%.1f");
    TOOLTIP("Torque from reaction wheels for rotation control (Nm)")
    
    ImGui::Separator();
    ImGui::TextColored(mColorDim, "SHELL GEOMETRY");
    ImGui::Separator();
    
    ImGui::SliderFloat("Shell Radius", &mRobotTune.shellRadius, 0.5f, 5.0f, "%.2f");
    TOOLTIP("Radius of the robot shell (meters)")
    ImGui::SliderFloat("Shell Thickness", &mRobotTune.shellThickness, 0.1f, 1.0f, "%.2f");
    TOOLTIP("Thickness of the robot shell (meters)")
    ImGui::SliderFloat("Shell Mass", &mRobotTune.shellMass, 5.0f, 100.0f, "%.1f");
    TOOLTIP("Mass of the robot shell (kg)")
    
    ImGui::Separator();
    ImGui::TextColored(mColorDim, "MOTOR SETTINGS");
    ImGui::Separator();
    
    ImGui::SliderFloat("Motor Speed", &mRobotTune.motorSpeed, 1.0f, 50.0f, "%.1f");
    TOOLTIP("Maximum angular velocity of motors (rad/s)")
    ImGui::SliderFloat("Motor Torque", &mRobotTune.motorTorque, 50.0f, 2000.0f, "%.1f");
    TOOLTIP("Maximum torque output from motors (Nm)")
    
    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "APPLIED VALUES");
    ImGui::Separator();
    ImGui::Text("Engine: %.1f N", mRobotTune.enginePower);
    ImGui::Text("Reaction Wheel: %.1f Nm", mRobotTune.reactionWheelPower);
    ImGui::Text("Shell: %.2fm x %.2fm, %.1fkg", mRobotTune.shellRadius, mRobotTune.shellThickness, mRobotTune.shellMass);
    ImGui::Text("Motor: %.1f rad/s, %.1f Nm", mRobotTune.motorSpeed, mRobotTune.motorTorque);
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "DEBUG VALUES");
    ImGui::Separator();
    ImGui::Text("DEBUG: mRobotTune.enginePower = %.1f", mRobotTune.enginePower);
    ImGui::Text("DEBUG: mRobotTune.reactionWheelPower = %.1f", mRobotTune.reactionWheelPower);
    ImGui::Text("DEBUG: mRobotTune.shellRadius = %.2f", mRobotTune.shellRadius);
    ImGui::Text("DEBUG: mRobotTune.shellThickness = %.2f", mRobotTune.shellThickness);
    ImGui::Text("DEBUG: mRobotTune.shellMass = %.1f", mRobotTune.shellMass);
    ImGui::Text("DEBUG: mRobotTune.motorSpeed = %.1f", mRobotTune.motorSpeed);
    ImGui::Text("DEBUG: mRobotTune.motorTorque = %.1f", mRobotTune.motorTorque);
}

void OverlayUIRefactored::DrawGraphicsTab()
{
    ImGui::TextColored(mColorAccent, "DEBUG VISUALIZATION");
    ImGui::Separator();
    
    ImGui::Checkbox("Show Collision Shapes", &mGraphics.showCollisionShapes);
    TOOLTIP("Display wireframe collision shapes for all bodies")
    ImGui::Checkbox("Show AABBs", &mGraphics.showAABBs);
    TOOLTIP("Display axis-aligned bounding boxes")
    ImGui::Checkbox("Show Contact Points", &mGraphics.showContactPoints);
    TOOLTIP("Display contact points during collision detection")
    
    ImGui::Separator();
    ImGui::TextColored(mColorDim, "ROBOT VISIBILITY");
    ImGui::Separator();
    
    ImGui::Checkbox("Show Robot 1", &mGraphics.showRobot1);
    TOOLTIP("Toggle visibility of robot 1")
    ImGui::Checkbox("Show Robot 2", &mGraphics.showRobot2);
    TOOLTIP("Toggle visibility of robot 2")
    ImGui::Checkbox("Show Internal Engines", &mGraphics.showInternalEngines);
    TOOLTIP("Toggle visibility of internal engine components")
    
    ImGui::Separator();
    ImGui::TextColored(mColorDim, "CAMERA");
    ImGui::Separator();
    
    ImGui::SliderFloat("Distance", &mGraphics.cameraDistance, 5.0f, 50.0f, "%.1f");
    TOOLTIP("Camera distance from target (meters)")
    ImGui::SliderFloat("Azimuth", &mGraphics.cameraAzimuth, 0.0f, 360.0f, "%.1f");
    TOOLTIP("Camera azimuth angle (degrees)")
    ImGui::SliderFloat("Elevation", &mGraphics.cameraElevation, -89.0f, 89.0f, "%.1f");
    TOOLTIP("Camera elevation angle (degrees)")
    
    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "CURRENT VIEW");
    ImGui::Separator();
    ImGui::Text("Distance: %.1f", mGraphics.cameraDistance);
    ImGui::Text("Azimuth: %.1f", mGraphics.cameraAzimuth);
    ImGui::Text("Elevation: %.1f", mGraphics.cameraElevation);
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "DEBUG VALUES");
    ImGui::Separator();
    ImGui::Text("DEBUG: mGraphics.showCollisionShapes = %s", mGraphics.showCollisionShapes ? "true" : "false");
    ImGui::Text("DEBUG: mGraphics.showAABBs = %s", mGraphics.showAABBs ? "true" : "false");
    ImGui::Text("DEBUG: mGraphics.showContactPoints = %s", mGraphics.showContactPoints ? "true" : "false");
    ImGui::Text("DEBUG: mGraphics.showRobot1 = %s", mGraphics.showRobot1 ? "true" : "false");
    ImGui::Text("DEBUG: mGraphics.showRobot2 = %s", mGraphics.showRobot2 ? "true" : "false");
    ImGui::Text("DEBUG: mGraphics.showInternalEngines = %s", mGraphics.showInternalEngines ? "true" : "false");
    ImGui::Text("DEBUG: mGraphics.cameraDistance = %.1f", mGraphics.cameraDistance);
    ImGui::Text("DEBUG: mGraphics.cameraAzimuth = %.1f", mGraphics.cameraAzimuth);
    ImGui::Text("DEBUG: mGraphics.cameraElevation = %.1f", mGraphics.cameraElevation);
}

void OverlayUIRefactored::DrawPolicyTab()
{
    ImGui::TextColored(mColorAccent, "POLICY MANAGEMENT");
    ImGui::Separator();
    
    static char saveNameBuf[256] = "";
    if (!mConfig.policySaveName.empty()) {
        strncpy(saveNameBuf, mConfig.policySaveName.c_str(), sizeof(saveNameBuf) - 1);
    }
    if (ImGui::InputText("Save Name", saveNameBuf, sizeof(saveNameBuf))) {
        mConfig.policySaveName = saveNameBuf;
    }
    TOOLTIP("Name for saving the policy checkpoint")
    ImGui::SameLine();
    if (ImGui::Button("SAVE", ImVec2(80, 25))) {
        mConfig.saveRequested = true;
    }
    TOOLTIP("Save current policy to checkpoint")
    
    static char loadNameBuf[256] = "";
    if (!mConfig.checkpointLoadName.empty()) {
        strncpy(loadNameBuf, mConfig.checkpointLoadName.c_str(), sizeof(loadNameBuf) - 1);
    }
    if (ImGui::InputText("Load Name", loadNameBuf, sizeof(loadNameBuf))) {
        mConfig.checkpointLoadName = loadNameBuf;
    }
    TOOLTIP("Name of checkpoint to load")
    ImGui::SameLine();
    if (ImGui::Button("LOAD", ImVec2(80, 25))) {
        mConfig.loadRequested = true;
    }
    TOOLTIP("Load policy from checkpoint")
    
    ImGui::Separator();
    ImGui::TextColored(mColorDim, "DIRECTORY");
    ImGui::Separator();
    
    static char checkpointDirBuf[512] = "";
    if (!mConfig.checkpointDir.empty()) {
        strncpy(checkpointDirBuf, mConfig.checkpointDir.c_str(), sizeof(checkpointDirBuf) - 1);
    }
    if (ImGui::InputText("Checkpoint Dir", checkpointDirBuf, sizeof(checkpointDirBuf))) {
        mConfig.checkpointDir = checkpointDirBuf;
    }
    TOOLTIP("Directory for saving/loading checkpoints")
    
    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "MANUAL CONTROL");
    ImGui::Separator();
    
    ImGui::Checkbox("Manual Torque Override", &mConfig.manualTorqueOverride);
    TOOLTIP("Override policy with manual torque control")
    mManualOverride = mConfig.manualTorqueOverride;
    
    if (mConfig.manualTorqueOverride) {
        ImGui::TextColored(mColorWarning, "WARNING: Policy disabled!");
    }
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "DEBUG VALUES");
    ImGui::Separator();
    ImGui::Text("DEBUG: mConfig.policySaveName = %s", mConfig.policySaveName.c_str());
    ImGui::Text("DEBUG: mConfig.checkpointLoadName = %s", mConfig.checkpointLoadName.c_str());
    ImGui::Text("DEBUG: mConfig.checkpointDir = %s", mConfig.checkpointDir.c_str());
    ImGui::Text("DEBUG: mConfig.manualTorqueOverride = %s", mConfig.manualTorqueOverride ? "true" : "false");
    ImGui::Text("DEBUG: mConfig.saveRequested = %s", mConfig.saveRequested ? "true" : "false");
    ImGui::Text("DEBUG: mConfig.loadRequested = %s", mConfig.loadRequested ? "true" : "false");
}

void OverlayUIRefactored::DrawSpawnTab()
{
    ImGui::TextColored(mColorAccent, "SPAWN CONTROLS");
    ImGui::Separator();
    
    const char* robotTypes[] = {"internal_engine", "shell_bot", "wedge_bot", "drum_bot"};
    static int selectedType = 0;
    
    ImGui::TextColored(mColorDim, "ROBOT TYPE");
    ImGui::Separator();
    ImGui::Combo("Type", &selectedType, robotTypes, 4);
    TOOLTIP("Select the type of robot to spawn")
    mPendingSpawnType = robotTypes[selectedType];
    
    ImGui::Separator();
    ImGui::TextColored(mColorDim, "SPAWN POSITION");
    ImGui::Separator();
    
    float pos[3] = {mPendingSpawnPos.GetX(), mPendingSpawnPos.GetY(), mPendingSpawnPos.GetZ()};
    ImGui::InputFloat3("Position", pos);
    TOOLTIP("XYZ position to spawn robot at")
    mPendingSpawnPos = JPH::Vec3(pos[0], pos[1], pos[2]);
    
    ImGui::Separator();
    ImGui::TextColored(mColorDim, "QUICK SPAWN");
    ImGui::Separator();
    
    if (ImGui::Button("SPAWN AT POSITION", ImVec2(200, 30))) {
        mSpawnRequest.valid = true;
        mSpawnRequest.type = mPendingSpawnType;
        mSpawnRequest.position = mPendingSpawnPos;
        mSpawnRequest.params = mRobotTune;
    }
    TOOLTIP("Spawn the selected robot at the specified position")
    
    ImGui::Separator();
    ImGui::TextColored(mColorWarning, "Click in 3D view to set spawn position");
    ImGui::Separator();
    
    ImGui::Text("Pending Position: (%.1f, %.1f, %.1f)", 
                 mPendingSpawnPos.GetX(), mPendingSpawnPos.GetY(), mPendingSpawnPos.GetZ());
    ImGui::Text("Pending Type: %s", mPendingSpawnType.c_str());
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "DEBUG VALUES");
    ImGui::Separator();
    ImGui::Text("DEBUG: mPendingSpawnType = %s", mPendingSpawnType.c_str());
    ImGui::Text("DEBUG: mPendingSpawnPos = (%.1f, %.1f, %.1f)", mPendingSpawnPos.GetX(), mPendingSpawnPos.GetY(), mPendingSpawnPos.GetZ());
    ImGui::Text("DEBUG: mSpawnRequest.valid = %s", mSpawnRequest.valid ? "true" : "false");
}

void OverlayUIRefactored::DrawEpisodesTab()
{
    ImGui::TextColored(mColorAccent, "EPISODE HISTORY");
    ImGui::Separator();
    
    int historyCount = static_cast<int>(mRewardHistory[0].size());
    ImGui::Text("Episode Data Points: %d", historyCount);
    ImGui::Text("Total Episodes: %d", mEpisodes);
    
    if (historyCount > 0) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "DAMAGE DEALT");
        TOOLTIP("Reward from damaging opponent")
        PlotLine("##dd", mRewardHistory[0], -1.0f, 1.0f);
        
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "DAMAGE TAKEN");
        TOOLTIP("Penalty for receiving damage")
        PlotLine("##dt", mRewardHistory[1], -1.0f, 1.0f);
        
        ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "AIRTIME");
        TOOLTIP("Reward for staying airborne")
        PlotLine("##air", mRewardHistory[2], -1.0f, 1.0f);
        
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "ENERGY USED");
        TOOLTIP("Penalty for high energy consumption")
        PlotLine("##ene", mRewardHistory[3], -1.0f, 1.0f);
        
        ImGui::TextColored(mColorAccent, "TOTAL REWARD");
        TOOLTIP("Combined reward from all components")
        PlotLine("##tot", mRewardHistory[4], -2.0f, 2.0f);
    } else {
        ImGui::TextColored(mColorDim, "No episode data yet...");
    }
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "DEBUG VALUES");
    ImGui::Separator();
    ImGui::Text("DEBUG: mRewardHistory[0].size() = %zu", mRewardHistory[0].size());
    ImGui::Text("DEBUG: mEpisodes = %d", mEpisodes);
}

void OverlayUIRefactored::DrawGraphSelector()
{
    ImGui::TextColored(mColorAccent, "GRAPH SELECTOR");
    ImGui::Separator();
    
    const char* graphTypes[] = {"Reward Components", "Loss Curves", "Physics Metrics", "Action Distributions", "Value Functions"};
    int currentGraph = static_cast<int>(mCurrentGraph);
    
    ImGui::RadioButton("Reward Components", &currentGraph, 0);
    TOOLTIP("View reward component breakdowns")
    ImGui::RadioButton("Loss Curves", &currentGraph, 1);
    TOOLTIP("View training loss curves")
    ImGui::RadioButton("Physics Metrics", &currentGraph, 2);
    TOOLTIP("View physics solver performance metrics")
    ImGui::RadioButton("Action Distributions", &currentGraph, 3);
    TOOLTIP("View action distribution histograms")
    ImGui::RadioButton("Value Functions", &currentGraph, 4);
    TOOLTIP("View value function estimates")
    
    mCurrentGraph = static_cast<GraphSelect>(currentGraph);
    
    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "EXTERNAL VISUALIZER");
    if (ImGui::Button("LAUNCH 3D GRAPH (MICROBOARD)", ImVec2(250, 30))) {
        mLaunchGraphRequested = true;
    }
    TOOLTIP("Launch external 3D graph visualizer")
    
    ImGui::Separator();
    ImGui::TextColored(mColorAccent, "CURRENT GRAPH");
    ImGui::Separator();
    ImGui::Text("%s", graphTypes[currentGraph]);
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "DEBUG VALUES");
    ImGui::Separator();
    ImGui::Text("DEBUG: mCurrentGraph = %d (%s)", currentGraph, graphTypes[currentGraph]);
    ImGui::Text("DEBUG: mLaunchGraphRequested = %s", mLaunchGraphRequested ? "true" : "false");
    
    switch (mCurrentGraph) {
        case GraphSelect::REWARD_COMPONENTS:
            DrawEpisodesTab();
            break;
        case GraphSelect::PHYSICS_METRICS:
            if (!mPhysicsHistory[0].empty()) {
                ImGui::TextColored(mColorAccent, "SOLVER TIME");
                PlotLine("##solver", mPhysicsHistory[0], 0.0f, 10.0f);
                ImGui::TextColored(mColorWarning, "BROADPHASE");
                PlotLine("##broad", mPhysicsHistory[1], 0.0f, 5.0f);
                ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "COLLISION");
                PlotLine("##col", mPhysicsHistory[2], 0.0f, 5.0f);
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "INTEGRATE");
                PlotLine("##int", mPhysicsHistory[3], 0.0f, 2.0f);
            } else {
                ImGui::TextColored(mColorDim, "No physics metrics yet...");
            }
            break;
        case GraphSelect::LOSS_CURVES:
            ImGui::TextColored(mColorDim, "Loss curves not yet implemented");
            break;
        case GraphSelect::ACTION_DISTRIBUTIONS:
            ImGui::TextColored(mColorDim, "Action distributions not yet implemented");
            break;
        case GraphSelect::VALUE_FUNCTIONS:
            ImGui::TextColored(mColorDim, "Value functions not yet implemented");
            break;
    }
}

bool OverlayUIRefactored::GetAndClearSaveRequest(std::string& outName)
{
    if (mConfig.saveRequested) {
        outName = mConfig.policySaveName;
        mConfig.saveRequested = false;
        return true;
    }
    return false;
}

bool OverlayUIRefactored::GetAndClearLoadRequest(std::string& outName)
{
    if (mConfig.loadRequested) {
        outName = mConfig.checkpointLoadName;
        mConfig.loadRequested = false;
        return true;
    }
    return false;
}

bool OverlayUIRefactored::GetAndClearGraphRequest()
{
    if (mLaunchGraphRequested) {
        mLaunchGraphRequested = false;
        return true;
    }
    return false;
}

bool OverlayUIRefactored::GetSpawnRequest(SpawnRequest& outRequest)
{
    if (mSpawnRequest.valid) {
        outRequest = mSpawnRequest;
        mSpawnRequest.valid = false;
        return true;
    }
    return false;
}

void OverlayUIRefactored::SetSpawnClickPosition(const JPH::Vec3& pos)
{
    mPendingSpawnPos = pos;
}

void OverlayUIRefactored::SaveAllSettings(const std::string& path)
{
    json j;
    
    j["physics"]["gravityY"] = mPhysics.gravityY;
    j["physics"]["timestep"] = mPhysics.timestep;
    j["physics"]["velocitySteps"] = mPhysics.velocitySteps;
    j["physics"]["positionSteps"] = mPhysics.positionSteps;
    j["physics"]["Baumgarte"] = mPhysics.Baumgarte;
    j["physics"]["penetrationSlop"] = mPhysics.penetrationSlop;
    j["physics"]["speculativeContactDistance"] = mPhysics.speculativeContactDistance;
    j["physics"]["allowSleep"] = mPhysics.allowSleep;
    j["physics"]["timeScale"] = mPhysics.timeScale;
    j["physics"]["stepsPerEpisode"] = mPhysics.stepsPerEpisode;
    
    j["robot"]["enginePower"] = mRobotTune.enginePower;
    j["robot"]["reactionWheelPower"] = mRobotTune.reactionWheelPower;
    j["robot"]["shellRadius"] = mRobotTune.shellRadius;
    j["robot"]["shellThickness"] = mRobotTune.shellThickness;
    j["robot"]["shellMass"] = mRobotTune.shellMass;
    j["robot"]["motorSpeed"] = mRobotTune.motorSpeed;
    j["robot"]["motorTorque"] = mRobotTune.motorTorque;
    
    j["training"]["numEnvs"] = mConfig.numEnvs;
    j["training"]["checkpointInterval"] = mConfig.checkpointInterval;
    j["training"]["checkpointDir"] = mConfig.checkpointDir;
    
    j["graphics"]["showCollisionShapes"] = mGraphics.showCollisionShapes;
    j["graphics"]["showAABBs"] = mGraphics.showAABBs;
    j["graphics"]["showContactPoints"] = mGraphics.showContactPoints;
    j["graphics"]["showRobot1"] = mGraphics.showRobot1;
    j["graphics"]["showRobot2"] = mGraphics.showRobot2;
    j["graphics"]["showInternalEngines"] = mGraphics.showInternalEngines;
    j["graphics"]["cameraDistance"] = mGraphics.cameraDistance;
    j["graphics"]["cameraAzimuth"] = mGraphics.cameraAzimuth;
    j["graphics"]["cameraElevation"] = mGraphics.cameraElevation;
    
    std::ofstream ofs(path);
    ofs << j.dump(4);
}

void OverlayUIRefactored::LoadAllSettings(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) return;
    
    json j;
    ifs >> j;
    
    // Load into pending/loaded structure (NOT active settings)
    if (j.contains("physics")) {
        mLoadedSettings.physics.gravityY = j["physics"].value("gravityY", -9.81f);
        mLoadedSettings.physics.timestep = j["physics"].value("timestep", 1.0f/60.0f);
        mLoadedSettings.physics.velocitySteps = j["physics"].value("velocitySteps", 4);
        mLoadedSettings.physics.positionSteps = j["physics"].value("positionSteps", 2);
        mLoadedSettings.physics.Baumgarte = j["physics"].value("Baumgarte", 0.2f);
        mLoadedSettings.physics.penetrationSlop = j["physics"].value("penetrationSlop", 0.01f);
        mLoadedSettings.physics.speculativeContactDistance = j["physics"].value("speculativeContactDistance", 0.01f);
        mLoadedSettings.physics.allowSleep = j["physics"].value("allowSleep", false);
        mLoadedSettings.physics.timeScale = j["physics"].value("timeScale", 1.0f);
        mLoadedSettings.physics.stepsPerEpisode = j["physics"].value("stepsPerEpisode", 1000);
    }
    
    if (j.contains("robot")) {
        mLoadedSettings.robot.enginePower = j["robot"].value("enginePower", 100.0f);
        mLoadedSettings.robot.reactionWheelPower = j["robot"].value("reactionWheelPower", 6000.0f);
        mLoadedSettings.robot.shellRadius = j["robot"].value("shellRadius", 2.0f);
        mLoadedSettings.robot.shellThickness = j["robot"].value("shellThickness", 0.3f);
        mLoadedSettings.robot.shellMass = j["robot"].value("shellMass", 25.0f);
        mLoadedSettings.robot.motorSpeed = j["robot"].value("motorSpeed", 10.0f);
        mLoadedSettings.robot.motorTorque = j["robot"].value("motorTorque", 450.0f);
    }
    
    if (j.contains("training")) {
        mLoadedSettings.training.numEnvs = j["training"].value("numEnvs", 8);
        mLoadedSettings.training.checkpointInterval = j["training"].value("checkpointInterval", 50000);
        mLoadedSettings.training.checkpointDir = j["training"].value("checkpointDir", "checkpoints");
    }
    
    if (j.contains("graphics")) {
        mLoadedSettings.graphics.showCollisionShapes = j["graphics"].value("showCollisionShapes", false);
        mLoadedSettings.graphics.showAABBs = j["graphics"].value("showAABBs", false);
        mLoadedSettings.graphics.showContactPoints = j["graphics"].value("showContactPoints", false);
        mLoadedSettings.graphics.showRobot1 = j["graphics"].value("showRobot1", true);
        mLoadedSettings.graphics.showRobot2 = j["graphics"].value("showRobot2", true);
        mLoadedSettings.graphics.showInternalEngines = j["graphics"].value("showInternalEngines", true);
        mLoadedSettings.graphics.cameraDistance = j["graphics"].value("cameraDistance", 20.0f);
        mLoadedSettings.graphics.cameraAzimuth = j["graphics"].value("cameraAzimuth", 45.0f);
        mLoadedSettings.graphics.cameraElevation = j["graphics"].value("cameraElevation", 30.0f);
    }
    
    mLoadedSettings.isLoaded = true;
    
    std::cout << "[OverlayUI] Settings loaded from file - click Apply to use them" << std::endl;
}
