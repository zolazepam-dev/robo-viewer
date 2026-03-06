#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
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
double gLastX, gLastY;
bool gFirstMouse = true;

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
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "F-22 Jolt Viewer", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glewInit();
    glfwSetCursorPosCallback(window, mouse_callback);

    PhysicsCore physics;
    physics.Init(1);
    
    Renderer renderer(1280, 720);
    Aircraft f22;
    f22.Create(&physics.GetPhysicsSystem(), JPH::RVec3(0, 10, 0), Layers::MOVING_BASE);

    auto last_time = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;

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
        physics.Step(1.0f / 60.0f);

        renderer.Draw(&physics, gCam.position, 0, gCam.front);
        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
