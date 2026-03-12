#pragma once

#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <queue>
#include <string>

// Forward declarations
class VectorizedEnv;
class TD3Trainer;
class ReplayBuffer;
class Renderer;
class OverlayUIRefactored;
struct IOTask;

// Camera state (for mouse callbacks)
struct FreeCamera {
    glm::vec3 position{0.0f, 15.0f, 40.0f};
    glm::vec3 front{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float yaw = -90.0f;
    float pitch = -20.0f;
    float speed = 30.0f;
    float sensitivity = 0.1f;
    bool active = false;
};

// Visual triple buffering
struct VisualTripleBuffering {
    std::atomic<int> readBufferIdx{0};
    std::atomic<int> writeBufferIdx{1};
    std::atomic<int> intermediateBufferIdx{2};
    std::atomic<bool> newFrameReady{false};
    EnvVisualState visualBuffers[3][NUM_PARALLEL_ENVS];
};

// ============================================================================
// Application State - Encapsulates all global state
// ============================================================================
struct ApplicationState {
    // Simulation state
    std::atomic<bool> simRunning{true};
    std::atomic<bool> simPaused{false};
    std::atomic<bool> trainingRunning{true};
    std::atomic<bool> ioRunning{true};
    
    // Statistics
    std::atomic<long long> totalSteps{0};
    std::atomic<int> episodes{0};
    std::atomic<float> avgReward{0.0f};
    std::atomic<float> sps{0.0f};
    std::atomic<float> agent1HP{100.0f};
    std::atomic<float> agent2HP{100.0f};
    std::atomic<float> agent1Reward{0.0f};
    std::atomic<float> agent2Reward{0.0f};
    
    // Thread synchronization
    std::mutex simMutex;
    
    // Camera state
    FreeCamera camera;
    double lastX = 0.0;
    double lastY = 0.0;
    bool firstMouse = true;
    
    // Visual triple buffering
    VisualTripleBuffering visualBuffer;
};

// ============================================================================
// Application Class - Owns all subsystems
// ============================================================================
class Application {
public:
    Application();
    ~Application();
    
    // Initialize all subsystems
    bool Init(int argc, char* argv[]);
    
    // Main run loop
    void Run();
    
    // Shutdown
    void Shutdown();
    
    // Getters for state
    ApplicationState& GetState() { return mState; }
    VectorizedEnv* GetEnv() { return mVecEnv.get(); }
    TD3Trainer* GetTrainer() { return mTrainer.get(); }
    TD3Trainer* GetOpponentTrainer() { return mOpponentTrainer.get(); }
    ReplayBuffer* GetReplayBuffer() { return mBuffer.get(); }
    Renderer* GetRenderer() { return mRenderer.get(); }
    OverlayUIRefactored* GetUI() { return mUI.get(); }
    
    // Window handle for callbacks
    void* GetWindow() { return mWindow; }
    
private:
    // Subsystems
    std::unique_ptr<VectorizedEnv> mVecEnv;
    std::unique_ptr<TD3Trainer> mTrainer;
    std::unique_ptr<TD3Trainer> mOpponentTrainer;
    std::unique_ptr<ReplayBuffer> mBuffer;
    std::unique_ptr<Renderer> mRenderer;
    std::unique_ptr<OverlayUIRefactored> mUI;
    
    // Application state
    ApplicationState mState;
    
    // Camera state
    glm::vec3 mCameraPosition{0.0f, 15.0f, 40.0f};
    glm::vec3 mCameraFront{0.0f, 0.0f, -1.0f};
    glm::vec3 mCameraUp{0.0f, 1.0f, 0.0f};
    float mCameraYaw = -90.0f;
    float mCameraPitch = -20.0f;
    float mCameraSpeed = 30.0f;
    float mCameraSensitivity = 0.1f;
    bool mCameraActive = false;
    
    // Window
    void* mWindow = nullptr;
    int mWindowWidth = 1280;
    int mWindowHeight = 720;
    
    // Threads
    std::thread mSimThread;
    std::thread mTrainingThread;
    std::thread mIOThread;
    
    // Configuration
    int mTargetNumEnvs = 0;
    int mRenderEnvIdx = 0;
    std::string mCheckpointDir;
    std::string mRobotCheckpointDir;
    std::string mRobotName;
    int mStateDim = 0;
    int mActionDim = 0;
    
    // Visual triple buffering
    VisualTripleBuffering* mVisualBuffer = nullptr;
    
    // Internal methods
    void InitWindow(int argc, char* argv[]);
    void InitEnvironment();
    void InitTrainer();
    void CreateThreads();
    void JoinThreads();
    void SaveCheckpoint();
    void LoadCheckpoint();
};

// Global application instance (for callbacks)
extern Application* gApp;

// Thread functions
void IOWorker(Application* app);
void TrainingLoop(Application* app);
void SimulationLoop(Application* app);
