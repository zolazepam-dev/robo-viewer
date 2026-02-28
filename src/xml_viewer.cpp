#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <regex>
#include <chrono>

// Jolt Physics
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

// Project Renderer & Physics
#include "Renderer.h"
#include "PhysicsCore.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

Camera gCamera; // Global dependency for renderer

struct SimpleCamera {
    glm::vec3 pos{0.0f, 0.0f, 50.0f}; // Pulled way back for spikes
    glm::vec3 front{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
};

SimpleCamera gCam;

int main(int argc, char** argv) {
    if (argc < 2) return 1;

    JPH::RegisterDefaultAllocator();
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();

    std::vector<JPH::Vec3> vertices;
    std::ifstream file(argv[1]);
    std::string line;
    std::regex v_regex("<Vertex x=\"([^\"]+)\" y=\"([^\"]+)\" z=\"([^\"]+)\"");
    std::smatch match;
    while (std::getline(file, line)) {
        if (std::regex_search(line, match, v_regex)) {
            vertices.push_back(JPH::Vec3(std::stof(match[1]), std::stof(match[2]), std::stof(match[3])));
        }
    }

    // Build the Spiky Hull
    JPH::ConvexHullShapeSettings hullSettings(vertices.data(), (int)vertices.size());
    JPH::Ref<JPH::Shape> shape = hullSettings.Create().Get();

    PhysicsCore core;
    core.Init(1);
    core.GetPhysicsSystem().SetGravity(JPH::Vec3::sZero());

    JPH::BodyInterface &bi = core.GetPhysicsSystem().GetBodyInterface();
    
    // The Spiky Object - Dynamic so it spins
    JPH::BodyCreationSettings bcs(shape, JPH::RVec3(0, 0, 0), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING_BASE);
    bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    bcs.mMassPropertiesOverride.mMass = 10.0f;
    JPH::BodyID bodyId = bi.CreateAndAddBody(bcs, JPH::EActivation::Activate);
    
    // Set a slow, majestic spin
    bi.SetAngularVelocity(bodyId, JPH::Vec3(0.1f, 0.2f, 0.05f)); 

    if (!glfwInit()) return 1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "STELATED MORNINGSTAR - NO FLOOR", nullptr, nullptr);
    if (!window) return 1;
    glfwMakeContextCurrent(window);
    glewInit();

    Renderer renderer(1280, 720);
    
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        core.GetPhysicsSystem().Update(1.0f/60.0f, 1, core.GetTempAllocator(), core.GetJobSystem());

        glClearColor(0.0f, 0.0f, 0.02f, 1.0f); // Void Blue
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Draw the system
        renderer.Draw(&(core.GetPhysicsSystem()), gCam.pos, 0, gCam.front);
        
        glfwSwapBuffers(window);
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
    }

    glfwTerminate();
    return 0;
}
