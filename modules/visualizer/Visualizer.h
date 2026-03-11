/**
 * @file Visualizer.h
 * @brief OpenGL rendering and ImGui UI system
 */

#pragma once

#include <glm/glm.hpp>
#include "../common/Types.h"
#include <GLFW/glfw3.h>
#include <string>
#include <memory>

// Forward declarations
class PhysicsWorld;

/**
 * @brief Camera state
 */
struct Camera {
    glm::vec3 position{0.0f, 15.0f, 40.0f};
    glm::vec3 front{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float yaw = -90.0f;
    float pitch = -20.0f;
    float speed = 30.0f;
    float sensitivity = 0.1f;
    bool active = false;
    
    void UpdateFront();
    glm::vec3 GetRight() const;
};

/** Graphics settings */
struct GraphicsSettings {
    bool showCollisionShapes = false;
    bool showAABBs = false;
    bool showContactPoints = false;
    bool showRobot1 = true;
    bool showRobot2 = true;
    int renderEnvIdx = 0;
    float cameraDistance = 20.0f;
    float cameraAzimuth = 45.0f;
    float cameraElevation = 30.0f;
};

/**
 * @brief Visualizer class
 */
class Visualizer {
public:
    Visualizer(int width = 1280, int height = 720);
    ~Visualizer();
    
    Visualizer(const Visualizer&) = delete;
    Visualizer& operator=(const Visualizer&) = delete;
    
    bool Init();
    void Shutdown();
    bool ShouldClose() const;
    void PollEvents();
    void SwapBuffers();
    void Render(PhysicsWorld* physicsWorld, UserCommands& userCommands);
    void BeginFrame(int totalSteps, int episodes, float sps, 
                    float avgReward, int currentEnv, int numEnvs);
    
    const UserCommands& GetCommands() const { return mCommands; }
    Camera& GetCamera() { return mCamera; }
    const Camera& GetCamera() const { return mCamera; }
    const GraphicsSettings& GetGraphicsSettings() const { return mGraphics; }
    GraphicsSettings& GetGraphicsSettings() { return mGraphics; }
    GLFWwindow* GetWindow() { return mWindow; }
    void GetFramebufferSize(int* width, int* height) const;
    
    void SetTimeScale(float scale) { mCommands.timeScale = scale; }
    void SetStepsPerEpisode(int steps) { mCommands.stepsPerEpisode = steps; }
    void UpdateAgentRewards(float agent1, float agent2);
    void UpdateAgentHP(float hp1, float hp2);
    void SetOpponentIndex(int idx);
    
    bool IsPaused() const { return mCommands.pause; }
    bool ShouldStepOne() const { return mCommands.stepOne; }
    void ClearStepOne() { mCommands.stepOne = false; }
    bool ShouldReset() const { return mCommands.reset; }
    void ClearReset() { mCommands.reset = false; }
    int GetRenderEnvIdx() const { return mCommands.renderEnvIdx; }
    float GetTimeScale() const { return mCommands.timeScale; }
    int GetStepsPerEpisode() const { return mCommands.stepsPerEpisode; }
    
    bool GetAndClearSaveRequest(std::string& outName);
    bool GetAndClearLoadRequest(std::string& outName);

private:
    void SetupCallbacks();
    void ProcessKeyboard(float dt);
    void ProcessMouse(double xpos, double ypos);
    void OnMouseButton(int button, int action, int mods);
    void OnScroll(double xoffset, double yoffset);
    void OnWindowSize(int width, int height);
    
    GLFWwindow* mWindow = nullptr;
    void* mRenderer = nullptr;  // Legacy Renderer*
    void* mUI = nullptr;        // Legacy OverlayUIRefactored*
    
    Camera mCamera;
    GraphicsSettings mGraphics;
    UserCommands mCommands;
    
    int mWidth, mHeight;
    double mLastX, mLastY;
    bool mFirstMouse;
    
    float mAgent1Reward = 0.0f, mAgent2Reward = 0.0f;
    float mAgent1HP = 100.0f, mAgent2HP = 100.0f;
    int mCurrentOpponentIdx = 0;
};
