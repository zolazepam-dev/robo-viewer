#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "src/Renderer.h"
#include "src/PhysicsCore.h"
#include "src/CombatRobot.h"
#include "src/OverlayUI.h"

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

// Camera State
struct FreeCamera {
    glm::vec3 position{0.0f, 15.0f, 40.0f};
    glm::vec3 front{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float yaw = -90.0f;
    float pitch = -20.0f;
    float speed = 20.0f;
    float sensitivity = 0.1f;
    bool active = false;
};

FreeCamera gCam;
double lastX, lastY;
bool firstMouse = true;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!gCam.active) return;
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    
    float xoffset = (float)(xpos - lastX) * gCam.sensitivity;
    float yoffset = (float)(lastY - ypos) * gCam.sensitivity;
    lastX = xpos; lastY = ypos;

    gCam.yaw += xoffset;
    gCam.pitch += yoffset;
    gCam.pitch = std::clamp(gCam.pitch, -89.0f, 89.0f);

    glm::vec3 direction;
    direction.x = cos(glm::radians(gCam.yaw)) * cos(glm::radians(gCam.pitch));
    direction.y = sin(glm::radians(gCam.pitch));
    direction.z = sin(glm::radians(gCam.yaw)) * cos(glm::radians(gCam.pitch));
    gCam.front = glm::normalize(direction);
}

void process_input(GLFWwindow* window, float dt) {
    if (!gCam.active) return;
    float velocity = gCam.speed * dt;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) gCam.position += gCam.front * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) gCam.position -= gCam.front * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) gCam.position -= glm::normalize(glm::cross(gCam.front, gCam.up)) * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) gCam.position += glm::normalize(glm::cross(gCam.front, gCam.up)) * velocity;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) gCam.position += gCam.up * velocity;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) gCam.position -= gCam.up * velocity;
}

int main() {
    if (!glfwInit()) return 1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "JOLTrl - Pro Viewer", nullptr, nullptr);
    if (!window) return 1;
    glfwMakeContextCurrent(window);
    glewInit();
    glEnable(GL_DEPTH_TEST);
    glfwSetCursorPosCallback(window, mouse_callback);

    CyberpunkUI ui;
    ui.Init(window);

    PhysicsCore physicsCore;
    physicsCore.Init(1);
    auto& bodyInterface = physicsCore.GetPhysicsSystem().GetBodyInterface();

    // Large Room
    float roomSize = 50.0f;
    auto createWall = [&](JPH::Vec3 ext, JPH::RVec3 pos) {
        auto shape = JPH::BoxShapeSettings(ext).Create().Get();
        bodyInterface.CreateAndAddBody(JPH::BodyCreationSettings(shape, pos, JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    };
    createWall({roomSize, 1.0f, roomSize}, {0, -1, 0});
    createWall({roomSize, 1.0f, roomSize}, {0, roomSize*2, 0});
    createWall({1, roomSize, roomSize}, {-roomSize, roomSize, 0});
    createWall({1, roomSize, roomSize}, {roomSize, roomSize, 0});
    createWall({roomSize, roomSize, 1}, {0, roomSize, -roomSize});
    createWall({roomSize, roomSize, 1}, {0, roomSize, roomSize});

    CombatRobotLoader loader;
    JPH::RVec3 p1(-15, 5, 0), p2(15, 5, 0);
    CombatRobotData r1 = loader.LoadRobot("robots/combat_bot.json", &physicsCore.GetPhysicsSystem(), p1, 0, 0);
    CombatRobotData r2 = loader.LoadRobot("robots/combat_bot.json", &physicsCore.GetPhysicsSystem(), p2, 0, 1);

    Renderer renderer(1280, 720);
    float timeScale = 1.0f;
    float damping = 0.5f;
    JPH::PhysicsSettings settings = physicsCore.GetSettings();

    auto lastTime = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!gCam.active) { gCam.active = true; glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); firstMouse = true; }
        } else {
            if (gCam.active) { gCam.active = false; glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); }
        }

        process_input(window, dt);
        physicsCore.Step((1.0f/60.0f) * timeScale);

        renderer.Draw(&physicsCore.GetPhysicsSystem(), gCam.position, 0, gCam.front);

        ui.BeginFrame();
        
        ImGui::Begin("Pro Controller");
        ImGui::Text("Hold Right Click to Fly (WASD/QE)");
        ImGui::SliderFloat("Time Scale", &timeScale, 0.0f, 5.0f);
        
        if (ImGui::CollapsingHeader("Physics Solver")) {
            bool changed = false;
            int v = settings.mNumVelocitySteps;
            int p = settings.mNumPositionSteps;
            if (ImGui::SliderInt("Vel Steps", &v, 1, 20)) { settings.mNumVelocitySteps = v; changed = true; }
            if (ImGui::SliderInt("Pos Steps", &p, 1, 20)) { settings.mNumPositionSteps = p; changed = true; }
            if (ImGui::SliderFloat("Baumgarte", &settings.mBaumgarte, 0.0f, 1.0f)) changed = true;
            if (changed) physicsCore.SetSettings(settings);
        }

        if (ImGui::Button("Reset Scene")) {
            loader.ResetRobot(r1, &physicsCore.GetPhysicsSystem(), p1);
            loader.ResetRobot(r2, &physicsCore.GetPhysicsSystem(), p2);
        }
        ImGui::End();
        
        ui.EndFrame();
        glfwSwapBuffers(window);
    }

    physicsCore.Shutdown();
    ui.Shutdown();
    glfwTerminate();
    return 0;
}
