#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "src/PhysicsCore.h"
#include "src/Renderer.h"
#include "src/Aircraft.h"

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

FreeCamera gCam;
Renderer* gRendererPtr = nullptr;
double gLastX, gLastY;
bool gFirstMouse = true;

void window_size_callback(GLFWwindow* window, int width, int height) {
    if (gRendererPtr) {
        glViewport(0, 0, width, height);
        gRendererPtr->Resize(width, height);
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
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
    std::cout << "[F-22 Viewer] Initializing GLFW..." << std::endl;
    if (!glfwInit()) return -1;
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "F-22 Jolt Viewer", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    
    std::cout << "[F-22 Viewer] Initializing GLEW..." << std::endl;
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "GLEW Init Error: " << glewGetErrorString(err) << std::endl;
        return -1;
    }
    
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);

    std::cout << "[F-22 Viewer] Initializing Physics..." << std::endl;
    PhysicsCore physics;
    physics.Init(1);
    
    JPH::BodyInterface& body_interface = physics.GetPhysicsSystem().GetBodyInterface();
    
    // Create Floor: 10km x 10km
    JPH::BoxShapeSettings floor_shape(JPH::Vec3(5000.0f, 1.0f, 5000.0f));
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(floor_shape.Create().Get(), JPH::RVec3(0.0f, -1.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);

    std::cout << "[F-22 Viewer] Creating F-22..." << std::endl;
    Renderer renderer(1280, 720);
    gRendererPtr = &renderer;
    
    Aircraft f22;
    f22.Create(&physics.GetPhysicsSystem(), JPH::RVec3(0, 50, 0), Layers::MOVING_BASE); // Start higher

    std::cout << "[F-22 Viewer] Physics bodies: " << physics.GetPhysicsSystem().GetNumBodies() << std::endl;

    std::cout << "[F-22 Viewer] Entering Main Loop..." << std::endl;
    auto last_time = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;

        JPH::RVec3 acPos = body_interface.GetPosition(f22.GetMainBodyId());
        
        // Auto-reset if out of bounds
        if (acPos.GetY() < -100.0f || acPos.LengthSq() > 10000.0f * 10000.0f || glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            body_interface.SetPositionAndRotation(f22.GetMainBodyId(), JPH::RVec3(0, 50, 0), JPH::Quat::sIdentity(), JPH::EActivation::Activate);
            body_interface.SetLinearAndAngularVelocity(f22.GetMainBodyId(), JPH::Vec3::sZero(), JPH::Vec3::sZero());
            std::cout << "[F-22 Viewer] Aircraft Reset!" << std::endl;
        }

        if (frameCount++ % 60 == 0) {
            JPH::RVec3 pos = body_interface.GetPosition(f22.GetMainBodyId());
            std::cout << "[F-22] Pos: " << pos.GetX() << ", " << pos.GetY() << ", " << pos.GetZ() << std::endl;
        }

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            gCam.active = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        } else {
            gCam.active = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            gFirstMouse = true;
        }
        process_input(window, dt);

        float actions[4] = {0, 0, 0, 0};
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) actions[0] = 1.0f; // Thrust
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) actions[1] = 1.0f;    // Pitch Up
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) actions[1] = -1.0f; // Pitch Down
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) actions[2] = -1.0f; // Roll Left
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) actions[2] = 1.0f; // Roll Right

        f22.ApplyAerodynamics(&physics.GetPhysicsSystem(), actions);
        physics.Step(std::min(dt, 0.033f)); // Clamp dt to avoid huge steps

        // Make camera follow aircraft if not active
        if (!gCam.active) {
            gCam.position = glm::vec3(acPos.GetX(), acPos.GetY() + 5.0f, acPos.GetZ() + 20.0f);
            gCam.front = glm::normalize(glm::vec3(acPos.GetX(), acPos.GetY(), acPos.GetZ()) - gCam.position);
        }

        renderer.Draw(&physics, gCam.position, 0, gCam.front);
        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
