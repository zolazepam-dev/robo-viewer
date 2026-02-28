#pragma once

#include <Jolt/Jolt.h>
#include <imgui.h>
#include <string>
#include <vector>

// Forward declarations
class VectorizedEnv;
class TD3Trainer;
class PhysicsCore;

// ================================
// COMPREHENSIVE UI REFACTOR
// ================================

struct PhysicsTunables {
    float gravityY = -9.81f;
    float timestep = 1.0f / 60.0f;
    int velocitySteps = 4;
    int positionSteps = 2;
    float Baumgarte = 0.2f;
    float penetrationSlop = 0.01f;
    float speculativeContactDistance = 0.01f;
    bool allowSleep = false;
    float timeScale = 1.0f;
    int stepsPerEpisode = 1000;
};

struct RobotTunables {
    float enginePower = 100.0f;
    float reactionWheelPower = 6000.0f;
    float shellRadius = 2.0f;
    float shellThickness = 0.3f;
    float shellMass = 25.0f;
    float motorSpeed = 10.0f;
    float motorTorque = 450.0f;
};

struct TrainingConfigUI {
    int numEnvs = 8;
    int checkpointInterval = 50000;
    std::string checkpointDir = "checkpoints";
    std::string checkpointLoadName = "";
    std::string policySaveName = "";
    bool saveRequested = false;
    bool loadRequested = false;
    bool manualTorqueOverride = false;
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
    void PushRewardData(float damageDealt, float damageTaken, 
                        float airtime, float energy, float scalar);
    void PushPhysicsMetrics(float solverTime, float broadphaseTime, 
                            float collisionTime, float integrateTime);
    
    // Getters
    bool IsPaused() const { return mPaused; }
    bool ShouldStepOne() const { return mStepOne; }
    bool ShouldReset() const { return mResetRequested; }
    bool ShouldRestartSim() const { return mRestartRequested; }
    float GetTimeScale() const { return mTimeScale; }
    int GetRenderEnvIdx() const { return mRenderEnvIdx; }
    int GetStepsPerEpisode() const { return mStepsPerEpisode; }
    
    const PhysicsTunables& GetPhysics() const { return mPhysics; }
    const RobotTunables& GetRobots() const { return mRobotTune; }
    const GraphicsSettings& GetGraphics() const { return mGraphics; }
    GraphSelect GetGraphSelect() const { return mCurrentGraph; }
    
    // Policy management
    bool GetAndClearSaveRequest(std::string& outName);
    bool GetAndClearLoadRequest(std::string& outName);
    bool GetManualOverride() const { return mManualOverride; }
    
    // Spawn system
    struct SpawnRequest {
        bool valid = false;
        std::string type = "internal_engine";
        JPH::Vec3 position{0,5,0};
        RobotTunables params;
    };
    bool GetSpawnRequest(SpawnRequest& outRequest);
    void SetSpawnClickPosition(const JPH::Vec3& pos);
    
    // Settings persistence
    void SaveAllSettings(const std::string& path);
    void LoadAllSettings(const std::string& path);
    
    /*
    ========================================
    MAIN CODE INTEGRATION CHECKLIST
    ========================================
    
    1. PHYSICS INTEGRATION (PhysicsCore.cpp / VectorizedEnv.cpp):
       - Apply mPhysics.timestep * mTimeScale as physics timestep
       - Set gravity: mPhysics.gravityY on Y axis
       - Apply solver: mPhysics.velocitySteps, mPhysics.positionSteps
       - Update JPH::PhysicsSettings when values change (not every frame)
       - Handle mStepsPerEpisode in episode termination logic
    
    2. ROBOT PARAMETER INTEGRATION (CombatEnv.cpp / InternalRobot.cpp):
       - Use mRobotTune.enginePower for max engine force
       - Use mRobotTune.reactionWheelPower for reaction torque
       - Use mRobotTune.shellRadius, thickness, mass for body creation
       - Support runtime updates via BodyInterface::SetMassProperties()
    
    3. TRAINING LOOP (main_train.cpp or similar):
       - Check ShouldReset() to reset environments
       - Check ShouldRestartSim() to restart with new numEnvs (requires reconstructing vecEnv)
       - Check GetStepsPerEpisode() for episode termination condition
       - Apply timeScale before stepping physics
       - Call UpdateStats() every frame with current metrics
       - Call PushRewardData() when episode ends or per-step
       - Call PushPhysicsMetrics() with timing data from profiling
    
    4. POLICY MANAGEMENT (TD3Trainer.cpp):
       - When saveRequested: trainer.Save(checkpointDir + "/" + policySaveName + ".bin")
       - When loadRequested: trainer.Load(checkpointDir + "/" + policyLoadName + ".bin")
       - Use manualTorqueOverride to bypass policy network and use zero/random actions
    
    5. SPAWN SYSTEM (CombatEnv.cpp):
       - In mouse callback, convert screen to world, call SetSpawnClickPosition()
       - In main loop, check GetSpawnRequest() and create robot at position
       - New robot should use current RobotTunables parameters
    
    6. RENDERING (Renderer.cpp):
       - Filter robot rendering by GraphicsSettings show flags
       - Add debug shape rendering for collision shapes, AABBs, contact points
       - Apply camera settings from GraphicsSettings
    
    7. MICROBOARD INTEGRATION:
       - Use GetGraphSelect() to determine which metrics to plot
       - Feed data from PushRewardData() and PushPhysicsMetrics()
       - Save/load UI settings with SaveAllSettings() / LoadAllSettings()
    */
    
private:
    void DrawTabBar();
    void DrawTrainingTab();
    void DrawPhysicsTab();
    void DrawRobotsTab();
    void DrawGraphicsTab();
    void DrawPolicyTab();
    void DrawSpawnTab();
    void DrawEpisodesTab();
    void DrawGraphSelector();
    
    void DrawCyberpunkStyle();
    void PlotLine(const char* label, const std::vector<float>& data, float scale_min, float scale_max);
    
    // State
    bool mPaused = false;
    bool mStepOne = false;
    bool mResetRequested = false;
    bool mRestartRequested = false;
    float mTimeScale = 1.0f;
    int mRenderEnvIdx = 0;
    int mStepsPerEpisode = 1000;
    
    PhysicsTunables mPhysics;
    RobotTunables mRobotTune;
    GraphicsSettings mGraphics;
    TrainingConfigUI mConfig;
    
    SpawnRequest mSpawnRequest;
    JPH::Vec3 mPendingSpawnPos;
    std::string mPendingSpawnType;
    
    GraphSelect mCurrentGraph = GraphSelect::REWARD_COMPONENTS;
    
    std::vector<float> mRewardHistory[5];
    std::vector<float> mPhysicsHistory[4];
    int mHistoryWritePos = 0;
    static constexpr int HISTORY_MAX = 500;
    
    int mTotalSteps = 0;
    int mEpisodes = 0;
    float mSPS = 0.0f;
    float mAvgReward = 0.0f;
    int mNumEnvs = 0;
    
    ImGuiContext* mContext = nullptr;
};
