#pragma once

#include <Jolt/Jolt.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include <vector>
#include <array>
#include <cmath>
#include <cstring>


struct BattleNotification
{
    std::string message;
    ImVec4 color;
    float displayTime = 0.0f;
    float maxDisplayTime = 3.0f;
};

struct RewardHistory
{
    static constexpr int HISTORY_SIZE = 200;
    
    std::array<float, HISTORY_SIZE> damageDealtHistory{};
    std::array<float, HISTORY_SIZE> damageTakenHistory{};
    std::array<float, HISTORY_SIZE> airtimeHistory{};
    std::array<float, HISTORY_SIZE> energyHistory{};
    std::array<float, HISTORY_SIZE> scalarHistory{};
    
    int writeIdx = 0;
    int count = 0;
    
    void Push(float damageDealt, float damageTaken, float airtime, float energy, float scalar)
    {
        damageDealtHistory[writeIdx] = damageDealt;
        damageTakenHistory[writeIdx] = damageTaken;
        airtimeHistory[writeIdx] = airtime;
        energyHistory[writeIdx] = energy;
        scalarHistory[writeIdx] = scalar;
        
        writeIdx = (writeIdx + 1) % HISTORY_SIZE;
        count = std::min(count + 1, HISTORY_SIZE);
    }
    
    void Clear()
    {
        damageDealtHistory.fill(0.0f);
        damageTakenHistory.fill(0.0f);
        airtimeHistory.fill(0.0f);
        energyHistory.fill(0.0f);
        scalarHistory.fill(0.0f);
        writeIdx = 0;
        count = 0;
    }
};

class CyberpunkUI
{
public:
    CyberpunkUI();
    ~CyberpunkUI() = default;
    
    void Init(GLFWwindow* window);
    void BeginFrame();
    void EndFrame();
    void Shutdown();
    
    void SetTrainingStats(int totalSteps, int episodes, float sps, float avgReward);
    void SetEnvironmentStats(int numEnvs, int selectedEnv);
    void PushRewardHistory(float damageDealt, float damageTaken, float airtime, float energy, float scalar);
    void AddBattleNotification(const std::string& msg, ImVec4 color);
    void DrawBattleNotifications();
    void UpdateBattleNotifications(float dt);
    
    bool IsPaused() const { return mPaused; }
    bool IsRenderingEnabled() const { return mRenderEnabled; }
    int GetSelectedEnv() const { return mSelectedEnv; }
    
    void DrawMainWindow();
    void DrawRewardGraphs();
    void DrawLidarHUD(const float* lidarDistances, int numRays);

private:
    void ApplyCyberpunkStyle();
    
    bool mPaused = false;
    bool mRenderEnabled = true;
    int mSelectedEnv = 0;
    
    int mTotalSteps = 0;
    int mEpisodes = 0;
    float mSPS = 0.0f;
    float mAvgReward = 0.0f;
    int mNumEnvs = 0;
    
    RewardHistory mRewardHistory;
    std::vector<BattleNotification> mNotifications;
    
    // Public colors for notifications
    ImVec4 mColorAccent{0.0f, 1.0f, 0.85f, 1.0f};
    ImVec4 mColorAccent2{1.0f, 0.0f, 0.5f, 1.0f};
    ImVec4 mColorAccent3{0.0f, 0.5f, 1.0f, 1.0f};
    ImVec4 mColorWarning{1.0f, 0.85f, 0.0f, 1.0f};
    ImVec4 mColorText{0.9f, 0.9f, 0.95f, 1.0f};
    ImVec4 mColorDim{0.5f, 0.5f, 0.6f, 0.8f};
    
private:
    ImVec4 mColorBg{0.02f, 0.02f, 0.05f, 1.0f};
};
