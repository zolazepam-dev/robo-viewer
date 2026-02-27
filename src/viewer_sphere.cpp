#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <thread>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "Renderer.h"
#include "PhysicsCore.h"

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

Camera gCamera;

int main() {
    std::cout << "[JOLTrl] Starting Viewer Sphere Simulation..." << std::endl;

    // 1. Init GLFW & OpenGL
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Sphere Viewer", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { glfwTerminate(); return 1; }
    
    glEnable(GL_DEPTH_TEST);
    glfwSwapInterval(1); // V-Sync

    // 2. Init ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // 3. Init Physics
    PhysicsCore physicsCore;
    if (!physicsCore.Init(1)) {
        std::cerr << "Failed to initialize PhysicsCore" << std::endl;
        return 1;
    }

    auto& bodyInterface = physicsCore.GetPhysicsSystem().GetBodyInterface();

    // Create Ground
    JPH::BoxShapeSettings groundShapeSettings(JPH::Vec3(50.0f, 1.0f, 50.0f));
    JPH::ShapeSettings::ShapeResult groundShapeResult = groundShapeSettings.Create();
    JPH::BodyCreationSettings groundSettings(groundShapeResult.Get(), JPH::RVec3(0, -1, 0), JPH::Quat::Identity(), JPH::EMotionType::Static, Layers::STATIC);
    bodyInterface.CreateAndAddBody(groundSettings, JPH::EActivation::DontActivate);

    // Create Sphere
    JPH::SphereShapeSettings sphereShapeSettings(1.0f);
    JPH::ShapeSettings::ShapeResult sphereShapeResult = sphereShapeSettings.Create();
    JPH::BodyCreationSettings sphereSettings(sphereShapeResult.Get(), JPH::RVec3(0, 5, 0), JPH::Quat::Identity(), JPH::EMotionType::Dynamic, Layers::MOVING_BASE);
    JPH::BodyID sphereID = bodyInterface.CreateAndAddBody(sphereSettings, JPH::EActivation::Activate);

    // 4. Init Renderer
    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    Renderer renderer(fbW, fbH);
    
    gCamera.distance = 25.0f;
    gCamera.pitch = 0.4f;
    gCamera.yaw = 0.0f;

    // 5. Simulation Loop
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-15.0f, 15.0f);
    bool paused = false;

    auto last_time = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = now - last_time;
        last_time = now;

        if (!paused) {
            // Random Action: Apply impulse every few frames or just every frame for chaos
            JPH::Vec3 randomImpulse(dist(gen), 0, dist(gen));
            bodyInterface.AddImpulse(sphereID, randomImpulse);

            // Step physics
            physicsCore.Step(1.0f / 60.0f);
        }

        // Camera Orbit (optional)
        gCamera.yaw += 0.005f;

        // Render
        float camX = gCamera.distance * std::cos(gCamera.pitch) * std::sin(gCamera.yaw);
        float camY = gCamera.distance * std::sin(gCamera.pitch);
        float camZ = gCamera.distance * std::cos(gCamera.pitch) * std::cos(gCamera.yaw);
        
        renderer.Draw(&physicsCore.GetPhysicsSystem(), glm::vec3(camX, camY, camZ), 0);

        // ImGui UI
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        ImGui::Begin("Sphere Controller");
        ImGui::Text("Sphere Position: %.2f, %.2f, %.2f", 
            (float)bodyInterface.GetCenterOfMassPosition(sphereID).GetX(),
            (float)bodyInterface.GetCenterOfMassPosition(sphereID).GetY(),
            (float)bodyInterface.GetCenterOfMassPosition(sphereID).GetZ());
        
        if (ImGui::Button(paused ? "Resume" : "Pause")) paused = !paused;
        if (ImGui::Button("Reset Sphere")) {
            bodyInterface.SetPosition(sphereID, JPH::RVec3(0, 5, 0), JPH::EActivation::Activate);
            bodyInterface.SetLinearAndAngularVelocity(sphereID, JPH::Vec3::sZero(), JPH::Vec3::sZero());
        }
        ImGui::SliderFloat("Cam Distance", &gCamera.distance, 5.0f, 100.0f);
        ImGui::End();
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }

    physicsCore.Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();

    return 0;
}
