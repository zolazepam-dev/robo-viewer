#pragma once

#include <Jolt/Jolt.h>
#include "imgui.h"
#include <string>
#include <vector>
#include <GLFW/glfw3.h>
#include "ConfigManager.h"
#include "CombatEnv.h"

// Forward declarations
class VectorizedEnv;
class TD3Trainer;
class PhysicsCore;

// ================================
// COMPREHENSIVE UI REFACTOR
// ================================

struct PhysicsTunables {
    float gravityY = -9.81f;
    float timestep = 1.0f / 120.0f;
    int velocitySteps = 8;
    int positionSteps = 3;
    float Baumgarte = 0.3f;
    float penetrationSlop = 0.005f;
    float speculativeContactDistance = 0.01f;
    bool allowSleep = false;
    float timeScale = 1.0f;
    int stepsPerEpisode = 10000;
    
    // NEW EXHAUSTIVE PHYSICS TUNABLES
    float friction = 0.5f;
    float restitution = 0.0f;
    float linearDamping = 0.05f;
    float angularDamping = 0.05f;
    float maxPenetrationVelocity = 1.0f;
    int numSubSteps = 1;
    bool warmStarting = true;
};

struct TrainingConfigUI {
    int numEnvs = 64;
    int checkpointInterval = 50000;
    std::string checkpointDir = "checkpoints";
    std::string robotConfigPath = "robots/bouncy_orbiter.json";
    std::string checkpointLoadName = "";
    std::string policySaveName = "";
    bool saveRequested = false;
    bool loadRequested = false;
    bool manualTorqueOverride = false;
    DomainRandomization dr;
};

struct RobotTunables {
    float enginePower = 100.0f;
    float reactionWheelPower = 5000.0f;
    float shellRadius = 1.0f;
    float shellThickness = 0.2f;
    float shellMass = 20.0f;
    float motorSpeed = 10.0f;
    float motorTorque = 200.0f;
};

struct RobotConfiguration {
    std::string name;
    std::string configFile;
};

struct RobotSelectionUI {
    int selectedRobotIndex = 0;
    std::vector<RobotConfiguration> availableRobots;
    bool loadConfigRequested = false;
};

struct GraphicsSettings {
    bool showCollisionShapes = false;
    bool showAABBs = false;
    bool showContactPoints = false;
    bool showRobot1 = true;
    bool showRobot2 = true;
    bool showInternalEngines = true;
    float cameraDistance = 20.0f;
    float cameraAzimuth = 45.0f;
    float cameraElevation = 30.0f;
};

enum class GraphSelect {
    REWARD_COMPONENTS,
    LOSS_CURVES,
    PHYSICS_METRICS,
    ACTION_DISTRIBUTIONS,
    VALUE_FUNCTIONS
};

class OverlayUIRefactored {
public:
    OverlayUIRefactored();
    ~OverlayUIRefactored() = default;
    
    void Init(GLFWwindow* window);
    void NewFrame();
    void Render();
    void Shutdown();
    
    void UpdateStats(int totalSteps, int episodes, float sps, float avgReward,
                     int currentEnv, int numEnvs);
    void UpdateAgentRewards(float agent1, float agent2) {
        mAgent1Reward = agent1;
        mAgent2Reward = agent2;
    }
    void UpdateAgentHP(float hp1, float hp2) {
        mAgent1HP = hp1;
        mAgent2HP = hp2;
    }
    void SetOpponentIndex(int idx) { mCurrentOpponentIdx = idx; }
    void PushRewardData(float damageDealt, float damageTaken, 
                        float airtime, float energy, float scalar);
    void PushPhysicsMetrics(float solverTime, float broadphaseTime, 
                            float collisionTime, float integrateTime);
    
    // Getters
    bool IsPaused() const { return mPaused; }
    bool ShouldStepOne() const { return mStepOne; }
    bool ShouldReset() const { return mResetRequested; }
    bool ShouldRestartSim() const { return mRestartRequested; }
    void ClearRestartRequest() { mRestartRequested = false; }
    void ClearResetRequest() { mResetRequested = false; }
    void ClearStepOne() { mStepOne = false; }
    float GetTimeScale() const { return mTimeScale; }
    int GetRenderEnvIdx() const { return mRenderEnvIdx; }
    int GetStepsPerEpisode() const { return mStepsPerEpisode; }
    
    const PhysicsTunables& GetPhysics() const { return mPhysics; }
    const GraphicsSettings& GetGraphics() const { return mGraphics; }
    const TrainingConfigUI& GetConfig() const { return mConfig; }
    const RobotTunables& GetRobots() const { return mRobotTune; }
    
    // Policy management
    bool GetAndClearSaveRequest(std::string& outName);
    bool GetAndClearLoadRequest(std::string& outName);
    bool GetAndClearGraphRequest();
    bool GetManualOverride() const { return mManualOverride; }
    
    // Settings save/load
    void SaveSettings(const std::string& path = "viewer_config.json");
    void LoadSettings(const std::string& path = "viewer_config.json");

    void UItoCentral(CentralConfig& cfg);
    void CentraltoUI(const CentralConfig& cfg);

    void DrawAllTabs();
    
private:
    void DrawTabBar();
    void DrawTrainingTab();
    void DrawPhysicsTab();
    void DrawRobotsTab();
    void DrawGraphicsTab();
    
    void DrawCyberpunkStyle();
    
    // State
    bool mPaused = false;
    bool mStepOne = false;
    bool mResetRequested = false;
    bool mRestartRequested = false;
    bool mLaunchGraphRequested = false;
    bool mManualOverride = false;
    float mTimeScale = 1.0f;
    int mRenderEnvIdx = 0;
    int mStepsPerEpisode = 1000;
    
    PhysicsTunables mPhysics;
    GraphicsSettings mGraphics;
    TrainingConfigUI mConfig;
    RobotTunables mRobotTune;
    RobotSelectionUI mRobotSelection;
    
    std::vector<float> mRewardHistory[5];
    std::vector<float> mPhysicsHistory[4];
    static constexpr int HISTORY_MAX = 500;
    
    int mTotalSteps = 0;
    int mEpisodes = 0;
    float mSPS = 0.0f;
    float mAvgReward = 0.0f;
    float mAgent1Reward = 0.0f;
    float mAgent2Reward = 0.0f;
    float mAgent1HP = 100.0f;
    float mAgent2HP = 100.0f;
    int mCurrentOpponentIdx = 0;
    int mNumEnvs = 0;
    
    ImGuiContext* mContext = nullptr;
};
