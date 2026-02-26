#include <iostream>
#include <chrono>
#include <vector>
#include <random>

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
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "JOLTrl - Random Actions", nullptr, nullptr);
    if (!window) return 1;
    glfwMakeContextCurrent(window);
    glewInit();
    
    glEnable(GL_DEPTH_TEST);
    glfwSwapInterval(1);

    CyberpunkUI ui;
    ui.Init(window);

    PhysicsCore physicsCore;
    physicsCore.Init(1);
    auto& bodyInterface = physicsCore.GetPhysicsSystem().GetBodyInterface();

    // Floor
    auto floorShape = JPH::BoxShapeSettings(JPH::Vec3(100.0f, 1.0f, 100.0f)).Create().Get();
    bodyInterface.CreateAndAddBody(JPH::BodyCreationSettings(floorShape, JPH::RVec3(0, -1, 0), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);

    // Load Robot
    CombatRobotLoader loader;
    JPH::RVec3 spawnPos(0, 5, 0);
    CombatRobotData robot = loader.LoadRobot("robots/combat_bot.json", &physicsCore.GetPhysicsSystem(), spawnPos, 0, 0);

    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> actions(ACTIONS_PER_ROBOT, 0.0f);

    Renderer renderer(1280, 720);
    Camera camera;
    camera.distance = 25.0f;
    auto lastTime = std::chrono::high_resolution_clock::now();
    bool paused = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;

        if (!paused) {
            for (int i = 0; i < ACTIONS_PER_ROBOT; ++i) actions[i] = dis(gen);
            loader.ApplyActions(robot, actions.data(), &physicsCore.GetPhysicsSystem());
            physicsCore.Step(1.0f / 60.0f);
        }
        
        camera.yaw += 0.1f * dt;
        float camX = camera.distance * std::cos(camera.pitch) * std::sin(camera.yaw);
        float camY = camera.distance * std::sin(camera.pitch) + 2.0f;
        float camZ = camera.distance * std::cos(camera.pitch) * std::cos(camera.yaw);
        
        glm::vec3 pos(camX, camY, camZ);
        glm::vec3 front = glm::normalize(glm::vec3(0, 2, 0) - pos);
        renderer.Draw(&physicsCore.GetPhysicsSystem(), pos, 0, front);

        ui.BeginFrame();
        ImGui::Begin("Random Actions Demo");
        if (ImGui::Button(paused ? "Resume" : "Pause")) paused = !paused;
        if (ImGui::Button("Reset Robot")) loader.ResetRobot(robot, &physicsCore.GetPhysicsSystem(), spawnPos);
        ImGui::Text("Energy Used: %.3f", robot.totalEnergyUsed);
        ImGui::End();
        ui.EndFrame();
        
        glfwSwapBuffers(window);
    }

    physicsCore.Shutdown();
    ui.Shutdown();
    glfwTerminate();
    return 0;
}
