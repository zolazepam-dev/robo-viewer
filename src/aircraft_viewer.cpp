#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <iomanip>
#include "PhysicsCore.h"
#include "Renderer.h"
#include "Aircraft.h"

// ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

struct FreeCamera {
    glm::vec3 position{0.0f, 15.0f, 40.0f};
    glm::vec3 front{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float yaw = -90.0f;
    float pitch = -20.0f;
    float speed = 50.0f;
    float sensitivity = 0.1f;
    bool active = false;
};

FreeCamera gCam;
Renderer* gRendererPtr = nullptr;
double gLastX, gLastY;
bool gFirstMouse = true;
bool gPaused = true;

void window_size_callback(GLFWwindow* window, int width, int height) {
    if (gRendererPtr) {
        glViewport(0, 0, width, height);
        gRendererPtr->Resize(width, height);
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    if (!gCam.active) {
        gLastX = xpos; gLastY = ypos;
        return;
    }
    if (gFirstMouse) { gLastX = xpos; gLastY = ypos; gFirstMouse = false; }
    float xoff = (float)(xpos - gLastX) * gCam.sensitivity;
    float yoff = (float)(gLastY - ypos) * gCam.sensitivity;
    gLastX = xpos; gLastY = ypos;
    gCam.yaw += xoff; gCam.pitch += yoff;
    gCam.pitch = std::clamp(gCam.pitch, -89.0f, 89.0f);
    glm::vec3 dir;
    dir.x = cos(glm::radians(gCam.yaw)) * cos(glm::radians(gCam.pitch));
    dir.y = sin(glm::radians(gCam.pitch));
    dir.z = sin(glm::radians(gCam.yaw)) * cos(glm::radians(gCam.pitch));
    gCam.front = glm::normalize(dir);
}

void process_input(GLFWwindow* window, float dt) {
    if (!gCam.active) return;
    float vel = gCam.speed * dt;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) gCam.position += gCam.front * vel;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) gCam.position -= gCam.front * vel;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) gCam.position -= glm::normalize(glm::cross(gCam.front, gCam.up)) * vel;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) gCam.position += glm::normalize(glm::cross(gCam.front, gCam.up)) * vel;
}

int main() {
    if (!glfwInit()) return -1;
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "F-22 Raptor Jolt Simulation", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    
    glewInit();
    
    // ImGui Setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    ImGui::StyleColorsDark();

    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);

    PhysicsCore physics;
    physics.Init(1);
    
    JPH::BodyInterface& body_interface = physics.GetPhysicsSystem().GetBodyInterface();
    
    std::cout << "[F-22 Viewer] Creating Floor..." << std::endl;
    // Create Massive Floor: 20km x 20km
    JPH::BoxShapeSettings floor_shape(JPH::Vec3(10000.0f, 1.0f, 10000.0f));
    auto floor_result = floor_shape.Create();
    if (floor_result.HasError()) {
        std::cerr << "[F-22 Viewer] Floor creation failed: " << floor_result.GetError() << std::endl;
        return -1;
    }
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(floor_result.Get(), JPH::RVec3(0.0f, -1.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);

    std::cout << "[F-22 Viewer] Initializing Renderer..." << std::endl;
    Renderer renderer(1280, 720);
    gRendererPtr = &renderer;
    
    std::cout << "[F-22 Viewer] Creating Aircraft..." << std::endl;
    Aircraft f22;
    f22.Create(&physics.GetPhysicsSystem(), JPH::RVec3(0, 100, 0), Layers::MOVING_BASE);

    if (f22.GetMainBodyId().IsInvalid()) {
        std::cerr << "[F-22 Viewer] ERROR: F-22 Main Body ID is invalid!" << std::endl;
        return -1;
    }

    // Initialize camera position to be behind the aircraft
    gCam.position = glm::vec3(0, 105, -20);

    auto last_time = std::chrono::high_resolution_clock::now();
    
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;

        // Camera handling
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            gCam.active = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        } else {
            gCam.active = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            gFirstMouse = true;
        }
        process_input(window, dt);

        // Physics & Aero
        float actions[4] = {0, 0, 0, 0};
        if (!gPaused) {
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) actions[0] = 1.0f; 
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS && !gCam.active) actions[1] = 1.0f; // Pitch down
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS && !gCam.active) actions[1] = -1.0f; // Pitch up
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS && !gCam.active) actions[2] = -1.0f; // Roll left
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS && !gCam.active) actions[2] = 1.0f;  // Roll right
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) actions[3] = -1.0f; // Yaw left
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) actions[3] = 1.0f;  // Yaw right

            f22.ApplyAerodynamics(&physics.GetPhysicsSystem(), actions, dt);
            physics.Step(std::min(dt, 0.02f));
        }

        JPH::BodyID acBodyId = f22.GetMainBodyId();
        if (acBodyId.IsInvalid()) {
            std::cerr << "[F-22 Viewer] ERROR: Aircraft body lost!" << std::endl;
            break;
        }
        JPH::RVec3 acPos = body_interface.GetPosition(acBodyId);
        JPH::Vec3 acVel = body_interface.GetLinearVelocity(acBodyId);

        // Follow Camera (Rigid Chase Cam)
        if (!gCam.active) {
            JPH::RMat44 worldTransform = body_interface.GetWorldTransform(f22.GetMainBodyId());
            JPH::Quat rot = worldTransform.GetRotation().GetQuaternion();
            JPH::Vec3 up = rot * JPH::Vec3(0, 1, 0);
            JPH::Vec3 forward = rot * JPH::Vec3(0, 0, 1);
            
            // Standard chase position: 15m back, 3m up relative to plane
            JPH::Vec3 localOffset(0, 3.0f, -15.0f);
            JPH::RVec3 targetCamPos = worldTransform * localOffset;
            
            // Hard lock to avoid "flying past"
            gCam.position = glm::vec3(targetCamPos.GetX(), targetCamPos.GetY(), targetCamPos.GetZ());
            gCam.front = glm::vec3(forward.GetX(), forward.GetY(), forward.GetZ());
            gCam.up = glm::vec3(up.GetX(), up.GetY(), up.GetZ());
        } else {
            gCam.up = glm::vec3(0, 1, 0);
        }

        renderer.Draw(&physics, gCam.position, 0, gCam.front, gCam.up);

        // ImGui UI
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin("F-22 Control Panel", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        
        ImGui::Text("Status: %s", gPaused ? "PAUSED" : "FLYING");
        ImGui::Text("Speed: %.1f m/s (%.0f km/h)", acVel.Length(), acVel.Length() * 3.6f);
        ImGui::Text("Altitude: %.1f m", (float)acPos.GetY());
        
        if (ImGui::Button(gPaused ? "Resume" : "Pause")) {
            gPaused = !gPaused;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Aircraft")) {
            body_interface.SetPositionAndRotation(f22.GetMainBodyId(), JPH::RVec3(0, 100, 0), JPH::Quat::sIdentity(), JPH::EActivation::Activate);
            body_interface.SetLinearAndAngularVelocity(f22.GetMainBodyId(), JPH::Vec3::sZero(), JPH::Vec3::sZero());
            std::cout << "[F-22 Viewer] Aircraft Reset!" << std::endl;
        }

        ImGui::Separator();
        ImGui::Text("Controls:");
        ImGui::BulletText("SPACE: Thrust");
        ImGui::BulletText("W/S: Pitch");
        ImGui::BulletText("A/D: Roll");
        ImGui::BulletText("Q/E: Yaw");
        ImGui::BulletText("Right Click: Freelook");

        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}
