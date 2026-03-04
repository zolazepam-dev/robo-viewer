#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <random>
#include <string>

#include "Renderer.h"
#include "PhysicsCore.h"
#include "RobotLoader.h"

static void Usage()
{
    std::cout << "Usage: mjcf_viewer [--mjcf <path>] [--random]\n";
}

int main(int argc, char* argv[])
{
    std::string mjcfPath = "third_party/mujoco_menagerie/cartpole/cartpole.xml";
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

    RobotLoader loader;
    RobotData data = loader.LoadRobot(mjcfPath, &core.GetPhysicsSystem());
    if (data.bodies.empty()) {
        std::cerr << "Failed to load MJCF: " << mjcfPath << std::endl;
        return -1;
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);
    auto& bodyInterface = core.GetPhysicsSystem().GetBodyInterface();

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

        renderer.Draw(&core, glm::vec3(0.0f, 10.0f, 25.0f), 0, glm::vec3(0.0f, -0.2f, -1.0f), false, false, false, true, true);
        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
