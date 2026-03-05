#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Geometry/AABox.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <random>
#include <string>
#include <filesystem>

#include "Renderer.h"
#include "PhysicsCore.h"
#include "RobotLoader.h"

namespace fs = std::filesystem;

// Camera state
static glm::vec3 cameraPos(0.0f, 5.0f, 15.0f);
static glm::vec3 cameraFront(0.0f, 0.0f, -1.0f);
static glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);
static glm::vec3 cameraRight(1.0f, 0.0f, 0.0f);
static float lastX = 640.0f;
static float lastY = 360.0f;
static bool firstMouse = true;
static float cameraSpeed = 0.1f;
static float mouseSensitivity = 0.1f;
static float yaw = -90.0f;
static float pitch = 0.0f;
static bool paused = true;

static void Usage()
{
    std::cout << "Usage: mjcf_viewer [--mjcf <path>] [--random]\n";
    std::cout << "Controls:\n";
    std::cout << "  WASD - Move camera horizontally\n";
    std::cout << "  QE - Move camera up/down\n";
    std::cout << "  Mouse - Look around\n";
    std::cout << "  Shift - Move faster\n";
    std::cout << "  Space - Toggle mouse capture\n";
    std::cout << "  P - Toggle pause (starts paused)\n";
    std::cout << "  ESC - Exit\n";
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
    cameraRight = glm::normalize(glm::cross(cameraFront, cameraUp));
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Toggle mouse capture
    static bool spacePressed = false;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        if (!spacePressed) {
            static bool mouseCaptured = true;
            mouseCaptured = !mouseCaptured;
            if (mouseCaptured) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                firstMouse = true;
            } else {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
            spacePressed = true;
        }
    } else {
        spacePressed = false;
    }

    // Toggle pause with P key
    static bool pPressed = false;
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
        if (!pPressed) {
            paused = !paused;
            pPressed = true;
        }
    } else {
        pPressed = false;
    }

    float velocity = cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        velocity *= 3.0f;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += velocity * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= velocity * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= cameraRight * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += cameraRight * velocity;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        cameraPos += cameraUp * velocity;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        cameraPos -= cameraUp * velocity;
}

int main(int argc, char* argv[])
{
    std::string mjcfPath = "mujoco_robot_combat.xml";
    bool randomForces = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mjcf" && i + 1 < argc) {
            mjcfPath = argv[++i];
        } else if (arg == "--random") {
            randomForces = true;
        } else if (arg == "--help") {
            Usage();
            return 0;
        }
    }

    if (!fs::exists(mjcfPath)) {
        std::cerr << "MJCF file not found: " << mjcfPath << std::endl;
        Usage();
        return -1;
    }

    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW" << std::endl;
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(1280, 720, "MJCF Viewer (Free Cam)", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    
    // Set mouse callback and disable cursor initially
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to init GLEW" << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);

    Renderer renderer(1280, 720);

    PhysicsCore core;
    if (!core.Init(1)) {
        std::cerr << "PhysicsCore init failed" << std::endl;
        return -1;
    }

    // Create ground plane at y=0 (MuJoCo z=0 becomes Jolt y=0)
    {
        auto& bodyInterface = core.GetPhysicsSystem().GetBodyInterface();
        JPH::BoxShapeSettings groundShape(JPH::Vec3(50.0f, 0.1f, 50.0f));
        auto result = groundShape.Create();
        if (!result.HasError()) {
            JPH::BodyCreationSettings settings(
                result.Get(),
                JPH::RVec3(0.0f, 0.0f, 0.0f),
                JPH::Quat::sIdentity(),
                JPH::EMotionType::Static,
                Layers::STATIC
            );
            JPH::Body* ground = bodyInterface.CreateBody(settings);
            if (ground) {
                bodyInterface.AddBody(ground->GetID(), JPH::EActivation::DontActivate);
            }
        }
    }

    RobotLoader loader;
    RobotData data = loader.LoadRobot(mjcfPath, &core.GetPhysicsSystem());
    std::cout << "Loaded " << data.bodies.size() << " bodies, " << data.constraints.size() << " constraints." << std::endl;
    if (data.bodies.empty()) {
        std::cerr << "Failed to load MJCF: " << mjcfPath << std::endl;
        return -1;
    }
    
    // Debug: compute bounding box of robot
    JPH::AABox aabb;
    bool hasBounds = false;
    auto& bodyInterface = core.GetPhysicsSystem().GetBodyInterface();
    for (JPH::BodyID id : data.bodies) {
        JPH::RVec3 pos = bodyInterface.GetPosition(id);
        JPH::Quat rot = bodyInterface.GetRotation(id);
        JPH::AABox bodyBounds = bodyInterface.GetShape(id)->GetLocalBounds().Transformed(JPH::RMat44::sRotationTranslation(rot, pos));
        if (!hasBounds) {
            aabb = bodyBounds;
            hasBounds = true;
        } else {
            aabb.Encapsulate(bodyBounds);
        }
    }
    glm::vec3 robotCenter(0.0f, 0.0f, 0.0f);
    glm::vec3 robotExtent(0.0f, 0.0f, 0.0f);
    if (hasBounds) {
        JPH::Vec3 center = aabb.GetCenter();
        JPH::Vec3 extent = aabb.GetExtent();
        robotCenter = glm::vec3(center.GetX(), center.GetY(), center.GetZ());
        robotExtent = glm::vec3(extent.GetX(), extent.GetY(), extent.GetZ());
        std::cout << "Robot bounds: center (" << center.GetX() << ", " << center.GetY() << ", " << center.GetZ() 
                  << "), extent (" << extent.GetX() << ", " << extent.GetY() << ", " << extent.GetZ() << ")" << std::endl;
        
        // Set initial camera position based on robot bounds
        cameraPos = robotCenter + glm::vec3(0.0f, robotExtent.y * 3.0f, robotExtent.z * 5.0f);
        cameraFront = glm::normalize(robotCenter - cameraPos);
        cameraRight = glm::normalize(glm::cross(cameraFront, cameraUp));
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);

    std::cout << "Viewer started PAUSED. Press P to resume physics, ESC to exit." << std::endl;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        processInput(window);

        if (randomForces && !paused) {
            for (const auto& id : data.bodies) {
                if (bodyInterface.GetMotionType(id) == JPH::EMotionType::Dynamic) {
                    JPH::Vec3 f(dist(rng), dist(rng), dist(rng));
                    bodyInterface.AddForce(id, f);
                }
            }
        }

        if (!paused) {
            core.GetPhysicsSystem().Update(1.0f / 120.0f, 1, core.GetTempAllocator(), core.GetJobSystem());
        }

        // Camera looking at robot center
        glm::vec3 target = robotCenter;
        glm::vec3 lookAt = target + cameraFront;
        renderer.Draw(&core, cameraPos, 0, cameraFront, false, false, false, true, true);
        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
