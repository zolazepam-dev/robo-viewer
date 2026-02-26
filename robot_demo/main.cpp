#include <iostream>
#include <chrono>
#include <vector>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "src/Renderer.h"
#include "src/PhysicsCore.h"
#include "src/CombatRobot.h"
#include "src/OverlayUI.h"

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

int main() {
    std::cout << "[Robot Demo] Initializing Robot Viewer..." << std::endl;

    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "JOLTrl - Robot Demo", nullptr, nullptr);
    if (!window) return 1;
    glfwMakeContextCurrent(window);
    glewInit();
    
    glEnable(GL_DEPTH_TEST);
    glfwSwapInterval(1);

    CyberpunkUI ui;
    ui.Init(window);

    PhysicsCore physicsCore;
    if (!physicsCore.Init(1)) return 1;

    auto& bodyInterface = physicsCore.GetPhysicsSystem().GetBodyInterface();

    // Create Floor
    JPH::BoxShapeSettings floorShapeSettings(JPH::Vec3(100.0f, 1.0f, 100.0f));
    auto floorShapeResult = floorShapeSettings.Create();
    bodyInterface.CreateAndAddBody(JPH::BodyCreationSettings(floorShapeResult.Get(), JPH::RVec3(0, -1, 0), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);

    // Load Robot
    CombatRobotLoader loader;
    JPH::RVec3 spawnPos(0, 5, 0);
    CombatRobotData robot = loader.LoadRobot("robots/combat_bot.json", &physicsCore.GetPhysicsSystem(), spawnPos, 0, 0);

    Renderer renderer(1280, 720);
    Camera camera;
    camera.distance = 20.0f;
    camera.pitch = 0.4f;
    camera.yaw = 0.0f;

    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

        auto now = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;

        physicsCore.Step(1.0f / 60.0f);
        camera.yaw += 0.1f * deltaTime;

        float camX = camera.distance * std::cos(camera.pitch) * std::sin(camera.yaw);
        float camY = camera.distance * std::sin(camera.pitch) + 2.0f;
        float camZ = camera.distance * std::cos(camera.pitch) * std::cos(camera.yaw);
        
        glm::vec3 pos(camX, camY, camZ);
        glm::vec3 target(0.0f, 2.0f, 0.0f);
        glm::vec3 front = glm::normalize(target - pos);
        
        renderer.Draw(&physicsCore.GetPhysicsSystem(), pos, 0, front);

        ui.BeginFrame();
        ImGui::Begin("Robot Inspector");
        ImGui::Text("Orbiting Combat Robot");
        JPH::RVec3 rbPos = bodyInterface.GetCenterOfMassPosition(robot.mainBodyId);
        ImGui::Text("Main Body Y: %.2f", (float)rbPos.GetY());
        
        if (ImGui::Button("Reset Robot")) {
            loader.ResetRobot(robot, &physicsCore.GetPhysicsSystem(), spawnPos);
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
