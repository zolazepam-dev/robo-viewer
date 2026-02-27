// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "src/VectorizedEnv.h"
#include "src/Renderer.h"
#include "src/OverlayUI.h"
#include <iostream>
#include <algorithm>

struct FreeCamera {
    glm::vec3 position{0.0f, 6.0f, 20.0f};
    glm::vec3 front{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float yaw = -90.0f;
    float pitch = 0.0f;
    float speed = 15.0f;
    float sensitivity = 0.1f;
    bool active = false;
};

FreeCamera gCam;
double lastX, lastY;
bool firstMouse = true;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!gCam.active) return;
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoff = (float)(xpos - lastX) * gCam.sensitivity;
    float yoff = (float)(lastY - ypos) * gCam.sensitivity;
    lastX = xpos; lastY = ypos;
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
    
    // Safe camera movement with bounds checking
    glm::vec3 right = glm::normalize(glm::cross(gCam.front, gCam.up));
    // Ensure right vector is valid (not zero or NaN)
    if (std::isnan(right.x) || std::isnan(right.y) || std::isnan(right.z) ||
        std::isinf(right.x) || std::isinf(right.y) || std::isinf(right.z) ||
        glm::length(right) < 0.1f) {
        right = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) gCam.position += gCam.front * vel;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) gCam.position -= gCam.front * vel;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) gCam.position -= right * vel;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) gCam.position += right * vel;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) gCam.position += gCam.up * vel;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) gCam.position -= gCam.up * vel;
    
    // Limit camera position to reasonable bounds to prevent crashes
    gCam.position.x = std::clamp(gCam.position.x, -100.0f, 100.0f);
    gCam.position.y = std::clamp(gCam.position.y, -50.0f, 100.0f);
    gCam.position.z = std::clamp(gCam.position.z, -100.0f, 100.0f);
}

int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "JOLTrl - Training Room Debugger", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glewInit();
    glEnable(GL_DEPTH_TEST);
    glfwSetCursorPosCallback(window, mouse_callback);

    CyberpunkUI ui;
    ui.Init(window);

    // This creates the room geometry exactly as it exists in VectorizedEnv::Init()
    VectorizedEnv vecEnv(1);
    vecEnv.Init(false);
    
    Renderer renderer(1280, 720);

    auto last_time = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] Room Geometry Loaded. Use WASD to fly inside." << std::endl;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!gCam.active) { gCam.active = true; glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); firstMouse = true; }
        } else {
            if (gCam.active) { gCam.active = false; glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); }
        }
        process_input(window, dt);

        // Render the physics world (The Room)
        renderer.Draw(vecEnv.GetGlobalPhysics(), gCam.position, 0, gCam.front);

        ui.BeginFrame();
        ImGui::Begin("Room Geometry Debugger");
        ImGui::Text("Inspecting Training Room (12x12x12)");
        ImGui::Text("Pos: %.1f, %.1f, %.1f", gCam.position.x, gCam.position.y, gCam.position.z);
        ImGui::Separator();
        ImGui::Text("Hold Right-Click to fly");
        ImGui::End();
        ui.EndFrame();
        
        glfwSwapBuffers(window);
    }

    ui.Shutdown();
    glfwTerminate();
    return 0;
}
