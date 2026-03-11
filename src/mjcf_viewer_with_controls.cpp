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

static void Usage()
{
    std::cout << "Usage: mjcf_viewer [--mjcf <path>] [--random]\n";
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
    GLFWwindow* window = glfwCreateWindow(1280, 720, "MJCF Viewer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
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

    // Create ground plane
    {
        auto& bodyInterface = core.GetPhysicsSystem().GetBodyInterface();
        JPH::BoxShapeSettings groundShape(JPH::Vec3(50.0f, 1.0f, 50.0f));
        auto result = groundShape.Create();
        if (!result.HasError()) {
            JPH::BodyCreationSettings settings(
                result.Get(),
                JPH::RVec3(0.0f, -1.0f, 0.0f),
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
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (randomForces) {
            for (const auto& id : data.bodies) {
                if (bodyInterface.GetMotionType(id) == JPH::EMotionType::Dynamic) {
                    JPH::Vec3 f(dist(rng), dist(rng), dist(rng));
                    bodyInterface.AddForce(id, f);
                }
            }
        }

        core.GetPhysicsSystem().Update(1.0f / 120.0f, 1, core.GetTempAllocator(), core.GetJobSystem());

        // Camera looking at robot center
        glm::vec3 target = robotCenter;
        glm::vec3 cameraPos = target + glm::vec3(0.0f, robotExtent.y * 1.5f, robotExtent.z * 2.0f);
        if (!hasBounds) {
            // fallback
            target = glm::vec3(0.0f, 0.0f, 0.0f);
            cameraPos = glm::vec3(0.0f, 5.0f, 10.0f);
        }
        glm::vec3 cameraFront = glm::normalize(target - cameraPos);
        renderer.Draw(&core, cameraPos, 0, cameraFront, false, false, false, true, true);
        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
