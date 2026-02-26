#include <iostream>
#include <chrono>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "src/Renderer.h"
#include "src/PhysicsCore.h"
#include "src/OverlayUI.h"

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

int main() {
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "JOLTrl - Sphere Demo", nullptr, nullptr);
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

    // Sphere
    auto sphereShape = JPH::SphereShapeSettings(1.0f).Create().Get();
    JPH::BodyID sphereID = bodyInterface.CreateAndAddBody(JPH::BodyCreationSettings(sphereShape, JPH::RVec3(0, 15, 0), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING_BASE), JPH::EActivation::Activate);

    Renderer renderer(1280, 720);
    Camera camera;
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;

        physicsCore.Step(1.0f / 60.0f);
        camera.yaw += 0.2f * dt;

        float camX = camera.distance * std::cos(camera.pitch) * std::sin(camera.yaw);
        float camY = camera.distance * std::sin(camera.pitch);
        float camZ = camera.distance * std::cos(camera.pitch) * std::cos(camera.yaw);
        
        glm::vec3 pos(camX, camY, camZ);
        glm::vec3 front = glm::normalize(glm::vec3(0, 0, 0) - pos);
        renderer.Draw(&physicsCore.GetPhysicsSystem(), pos, 0, front);

        ui.BeginFrame();
        ImGui::Begin("Sphere Stats");
        JPH::RVec3 p = bodyInterface.GetCenterOfMassPosition(sphereID);
        ImGui::Text("Height: %.2f", (float)p.GetY());
        if (ImGui::Button("Reset")) {
            bodyInterface.SetPosition(sphereID, JPH::RVec3(0, 15, 0), JPH::EActivation::Activate);
            bodyInterface.SetLinearAndAngularVelocity(sphereID, JPH::Vec3::sZero(), JPH::Vec3::sZero());
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
