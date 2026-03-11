/**
 * @file Visualizer.cpp
 * @brief Implementation of Visualizer class - Stub for legacy integration
 */

#include <GL/glew.h>
#include "Visualizer.h"
#include "../physics/PhysicsWorld.h"
#include <iostream>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void Camera::UpdateFront() {
    glm::vec3 dir;
    dir.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    dir.y = sin(glm::radians(pitch));
    dir.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(dir);
}

glm::vec3 Camera::GetRight() const {
    return glm::normalize(glm::cross(front, up));
}

Visualizer::Visualizer(int width, int height)
    : mWidth(width), mHeight(height), mLastX(width / 2.0), mLastY(height / 2.0), mFirstMouse(true) {
    mCamera.UpdateFront();
}

Visualizer::~Visualizer() { Shutdown(); }

bool Visualizer::Init() {
    if (!glfwInit()) { std::cerr << "[Visualizer] Failed to initialize GLFW\n"; return false; }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    mWindow = glfwCreateWindow(mWidth, mHeight, "JOLTrl - Pro Training Suite", nullptr, nullptr);
    if (!mWindow) { std::cerr << "[Visualizer] Failed to create GLFW window\n"; glfwTerminate(); return false; }
    
    glfwMakeContextCurrent(mWindow);
    glfwSwapInterval(0);
    glewInit();
    glEnable(GL_DEPTH_TEST);
    
    SetupCallbacks();
    std::cout << "[Visualizer] Initialized (legacy Renderer/UI disabled for modular migration)\n";
    return true;
}

void Visualizer::Shutdown() {
    if (mWindow) { glfwDestroyWindow(mWindow); mWindow = nullptr; }
    glfwTerminate();
}

bool Visualizer::ShouldClose() const { return mWindow && glfwWindowShouldClose(mWindow); }
void Visualizer::PollEvents() { glfwPollEvents(); }
void Visualizer::SwapBuffers() { if (mWindow) glfwSwapBuffers(mWindow); }

void Visualizer::Render(PhysicsWorld* physicsWorld, UserCommands& userCommands) {
    if (!mWindow) return;
    mCommands = UserCommands();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    userCommands = mCommands;
}

void Visualizer::BeginFrame(int totalSteps, int episodes, float sps, float avgReward, int currentEnv, int numEnvs) {}
void Visualizer::GetFramebufferSize(int* width, int* height) const {
    if (mWindow) glfwGetFramebufferSize(mWindow, width, height);
    else { if (width) *width = mWidth; if (height) *height = mHeight; }
}

void Visualizer::UpdateAgentRewards(float agent1, float agent2) { mAgent1Reward = agent1; mAgent2Reward = agent2; }
void Visualizer::UpdateAgentHP(float hp1, float hp2) { mAgent1HP = hp1; mAgent2HP = hp2; }
void Visualizer::SetOpponentIndex(int idx) { mCurrentOpponentIdx = idx; }
bool Visualizer::GetAndClearSaveRequest(std::string& outName) { return false; }
bool Visualizer::GetAndClearLoadRequest(std::string& outName) { return false; }

void Visualizer::SetupCallbacks() {
    glfwSetWindowUserPointer(mWindow, this);
    glfwSetCursorPosCallback(mWindow, [](GLFWwindow* w, double x, double y) {
        Visualizer* self = (Visualizer*)glfwGetWindowUserPointer(w);
        if (self) self->ProcessMouse(x, y);
    });
    glfwSetMouseButtonCallback(mWindow, [](GLFWwindow* w, int b, int a, int m) {
        Visualizer* self = (Visualizer*)glfwGetWindowUserPointer(w);
        if (self) self->OnMouseButton(b, a, m);
    });
    glfwSetScrollCallback(mWindow, [](GLFWwindow* w, double x, double y) {
        Visualizer* self = (Visualizer*)glfwGetWindowUserPointer(w);
        if (self) self->OnScroll(x, y);
    });
    glfwSetWindowSizeCallback(mWindow, [](GLFWwindow* w, int width, int height) {
        Visualizer* self = (Visualizer*)glfwGetWindowUserPointer(w);
        if (self) self->OnWindowSize(width, height);
    });
}

void Visualizer::ProcessKeyboard(float dt) {
    if (!mCamera.active || !mWindow) return;
    float vel = mCamera.speed * dt;
    if (glfwGetKey(mWindow, GLFW_KEY_W) == GLFW_PRESS) mCamera.position += mCamera.front * vel;
    if (glfwGetKey(mWindow, GLFW_KEY_S) == GLFW_PRESS) mCamera.position -= mCamera.front * vel;
    if (glfwGetKey(mWindow, GLFW_KEY_A) == GLFW_PRESS) mCamera.position -= mCamera.GetRight() * vel;
    if (glfwGetKey(mWindow, GLFW_KEY_D) == GLFW_PRESS) mCamera.position += mCamera.GetRight() * vel;
    if (glfwGetKey(mWindow, GLFW_KEY_E) == GLFW_PRESS) mCamera.position += mCamera.up * vel;
    if (glfwGetKey(mWindow, GLFW_KEY_Q) == GLFW_PRESS) mCamera.position -= mCamera.up * vel;
}

void Visualizer::ProcessMouse(double xpos, double ypos) {
    if (!mCamera.active) { mLastX = xpos; mLastY = ypos; return; }
    if (mFirstMouse) { mLastX = xpos; mLastY = ypos; mFirstMouse = false; }
    float xoff = (float)(xpos - mLastX) * mCamera.sensitivity;
    float yoff = (float)(mLastY - ypos) * mCamera.sensitivity;
    mLastX = xpos; mLastY = ypos;
    mCamera.yaw += xoff; mCamera.pitch += yoff;
    mCamera.pitch = std::clamp(mCamera.pitch, -89.0f, 89.0f);
    mCamera.UpdateFront();
}

void Visualizer::OnMouseButton(int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS && !mCamera.active) {
        mCamera.active = true;
        glfwSetInputMode(mWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        mFirstMouse = true;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE && mCamera.active) {
        mCamera.active = false;
        glfwSetInputMode(mWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}

void Visualizer::OnScroll(double xoffset, double yoffset) {}

void Visualizer::OnWindowSize(int width, int height) {
    mWidth = width; mHeight = height;
    glViewport(0, 0, width, height);
}
