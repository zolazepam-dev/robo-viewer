// MUST BE FIRST
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "src/PhysicsCore.h"
#include "src/Renderer.h"
#include "src/RobotTemplates.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Robot Template Viewer - Arm & Snake", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    std::cout << "=== Robot Template Viewer ===" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  1-3: View Arm (3DOF/5DOF/7DOF)" << std::endl;
    std::cout << "  4-6: View Snake (4/8/12 segment)" << std::endl;
    std::cout << "  7: View Arm with Gripper" << std::endl;
    std::cout << "  8: View Snake with Head" << std::endl;
    std::cout << "  R: Reset robot" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;

    PhysicsCore physicsCore;
    if (!physicsCore.Init(1)) {
        std::cerr << "Failed to initialize physics" << std::endl;
        return -1;
    }

    // Create larger floor
    JPH::BoxShapeSettings floorShape(JPH::Vec3(30.0f, 1.0f, 30.0f));
    JPH::RefConst<JPH::Shape> floor = floorShape.Create().Get();
    JPH::BodyInterface& bodyInterface = physicsCore.GetPhysicsSystem().GetBodyInterface();
    bodyInterface.CreateAndAddBody(
        JPH::BodyCreationSettings(floor, JPH::RVec3(0.0f, 0.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC),
        JPH::EActivation::DontActivate
    );
    std::cout << "Floor created at Y=0" << std::endl;

    Renderer renderer(1280, 720);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 3.0f;
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.03f, 0.03f, 0.07f, 0.95f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.0f, 1.0f, 0.85f, 1.0f);

    int currentRobot = 0;
    CombatRobotData currentRobotData;
    std::vector<std::string> robotNames = {
        "Arm 3DOF", "Arm 5DOF", "Arm 7DOF",
        "Snake 4-Seg", "Snake 8-Seg", "Snake 12-Seg",
        "Arm with Gripper", "Snake with Head"
    };
    
    // Camera controls
    float camDistance = 25.0f;
    float camYaw = 0.0f;
    float camPitch = 0.35f;
    glm::vec3 camTarget(0.0f, 5.0f, 0.0f);
    float camPanX = 0.0f, camPanZ = 0.0f;
    
    // Robot joint control
    std::vector<float> jointVelocities; // Current velocity for each joint
    const float maxJointSpeed = 10.0f;
    const float jointAccel = 50.0f;

    auto loadRobot = [&](int index) {
        // Spawn robots HIGH above the floor so they fall and we can see them
        JPH::RVec3 spawnPos(0.0f, 8.0f, 0.0f);
        
        std::cout << "\n=== Creating Robot: " << robotNames[index] << " ===" << std::endl;
        std::cout << "Spawn position: (" << spawnPos.GetX() << ", " << spawnPos.GetY() << ", " << spawnPos.GetZ() << ")" << std::endl;
        
        switch (index) {
            case 0: currentRobotData = RobotTemplates::CreateArm3DOF(&physicsCore.GetPhysicsSystem(), spawnPos, 0); break;
            case 1: currentRobotData = RobotTemplates::CreateArm5DOF(&physicsCore.GetPhysicsSystem(), spawnPos, 0); break;
            case 2: currentRobotData = RobotTemplates::CreateArm7DOF(&physicsCore.GetPhysicsSystem(), spawnPos, 0); break;
            case 3: currentRobotData = RobotTemplates::CreateSnake4Segment(&physicsCore.GetPhysicsSystem(), spawnPos, 0); break;
            case 4: currentRobotData = RobotTemplates::CreateSnake8Segment(&physicsCore.GetPhysicsSystem(), spawnPos, 0); break;
            case 5: currentRobotData = RobotTemplates::CreateSnake12Segment(&physicsCore.GetPhysicsSystem(), spawnPos, 0); break;
            case 6: currentRobotData = RobotTemplates::CreateArmWithGripper(&physicsCore.GetPhysicsSystem(), spawnPos, 0); break;
            case 7: currentRobotData = RobotTemplates::CreateSnakeWithHead(&physicsCore.GetPhysicsSystem(), spawnPos, 0); break;
        }
        
        std::cout << "Result:" << std::endl;
        std::cout << "  Main Body ID: " << (currentRobotData.mainBodyId.IsInvalid() ? "INVALID" : "VALID") << std::endl;
        std::cout << "  Bodies: " << currentRobotData.bodies.size() << std::endl;
        std::cout << "  Hinge Joints: " << currentRobotData.hingeJoints.size() << std::endl;
        std::cout << "  6DOF Joints: " << currentRobotData.sixDofJoints.size() << std::endl;
        
        // Print body positions
        if (!currentRobotData.bodies.empty()) {
            for (size_t i = 0; i < currentRobotData.bodies.size() && i < 5; i++) {
                if (!currentRobotData.bodies[i].IsInvalid()) {
                    JPH::RVec3 pos = bodyInterface.GetPosition(currentRobotData.bodies[i]);
                    std::cout << "  Body " << i << " at: (" << pos.GetX() << ", " << pos.GetY() << ", " << pos.GetZ() << ")" << std::endl;
                }
            }
        }
        std::cout << "==========================================\n" << std::endl;
    };

    loadRobot(0);
    glfwSwapInterval(1);

    double lastKeyTime = 0.0;
    double lastFrameTime = glfwGetTime();
    int frameCount = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        double currentTime = glfwGetTime();
        bool reload = false;
        
        float dt = static_cast<float>(currentTime - lastFrameTime);
        lastFrameTime = currentTime;
        
        // Camera controls (hold keys)
        float camSpeed = 10.0f * dt;
        float rotSpeed = 1.5f * dt;
        
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            camTarget.x -= sin(camYaw) * camSpeed;
            camTarget.z -= cos(camYaw) * camSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            camTarget.x += sin(camYaw) * camSpeed;
            camTarget.z += cos(camYaw) * camSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            camTarget.x -= cos(camYaw) * camSpeed;
            camTarget.z += sin(camYaw) * camSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            camTarget.x += cos(camYaw) * camSpeed;
            camTarget.z -= sin(camYaw) * camSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camYaw -= rotSpeed;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camYaw += rotSpeed;
        if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) camPitch -= rotSpeed;
        if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) camPitch += rotSpeed;
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) camDistance += camSpeed;
        if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) camDistance = std::max(5.0f, camDistance - camSpeed);
        
        // Clamp pitch
        camPitch = std::max(-1.5f, std::min(1.5f, camPitch));
        
        // Robot switching with key release detection
        if (currentTime - lastKeyTime > 0.3) {
            int newRobot = -1;
            if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) newRobot = 0;
            else if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) newRobot = 1;
            else if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) newRobot = 2;
            else if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) newRobot = 3;
            else if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) newRobot = 4;
            else if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) newRobot = 5;
            else if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS) newRobot = 6;
            else if (glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS) newRobot = 7;
            else if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) newRobot = currentRobot; // Reset current
            
            if (newRobot >= 0 && newRobot != currentRobot) {
                reload = true;
                currentRobot = newRobot;
                lastKeyTime = currentTime;
                std::cout << "\n>>> Switching to: " << robotNames[currentRobot] << " <<<\n" << std::endl;
            }
            // R key disabled for now - causes crash
        }

        if (reload) {
            std::cout << "Cleaning up old robot..." << std::endl;
            
            // Remove constraints first
            for (auto* joint : currentRobotData.hingeJoints) {
                if (joint) {
                    physicsCore.GetPhysicsSystem().RemoveConstraint(joint);
                }
            }
            for (auto* joint : currentRobotData.sixDofJoints) {
                if (joint) {
                    physicsCore.GetPhysicsSystem().RemoveConstraint(joint);
                }
            }
            
            // Then remove bodies
            for (auto& bodyId : currentRobotData.bodies) {
                if (!bodyId.IsInvalid()) {
                    bodyInterface.RemoveBody(bodyId);
                    bodyInterface.DestroyBody(bodyId);
                }
            }
            
            currentRobotData.bodies.clear();
            currentRobotData.hingeJoints.clear();
            currentRobotData.sixDofJoints.clear();
            currentRobotData.mainBodyId = JPH::BodyID();
            jointVelocities.clear();
            
            std::cout << "Loading new robot..." << std::endl;
            loadRobot(currentRobot);
            
            // Initialize joint velocities
            jointVelocities.resize(currentRobotData.hingeJoints.size(), 0.0f);
            
            // Reset camera target to new robot position
            if (!currentRobotData.bodies.empty() && !currentRobotData.bodies[0].IsInvalid()) {
                JPH::RVec3 pos = bodyInterface.GetPosition(currentRobotData.bodies[0]);
                camTarget = glm::vec3(static_cast<float>(pos.GetX()), 
                                      static_cast<float>(pos.GetY()) + 3.0f,
                                      static_cast<float>(pos.GetZ()));
            }
            
            reload = false;
            std::cout << "Robot switch complete!" << std::endl;
        }
        
        // === ROBOT JOINT CONTROL ===
        if (!currentRobotData.hingeJoints.empty()) {
            // Control first few joints with keyboard
            // I/K = Joint 0 (base rotation)
            // O/L = Joint 1 (shoulder)
            // ,/. = Joint 2 (elbow)
            // [/] = Gripper fingers
            
            float dt = 1.0f / 60.0f;
            
            // Joint 0 - Base rotation (I/K)
            if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
                jointVelocities[0] = std::min(maxJointSpeed, jointVelocities[0] + jointAccel * dt);
            } else if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
                jointVelocities[0] = std::max(-maxJointSpeed, jointVelocities[0] - jointAccel * dt);
            } else {
                jointVelocities[0] *= 0.9f; // Damping
            }
            
            if (currentRobotData.hingeJoints.size() > 0) {
                currentRobotData.hingeJoints[0]->SetMotorState(JPH::EMotorState::Velocity);
                currentRobotData.hingeJoints[0]->SetTargetAngularVelocity(jointVelocities[0]);
            }
            
            // Joint 1 - Shoulder (O/L)
            if (currentRobotData.hingeJoints.size() > 1) {
                if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
                    jointVelocities[1] = std::min(maxJointSpeed, jointVelocities[1] + jointAccel * dt);
                } else if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
                    jointVelocities[1] = std::max(-maxJointSpeed, jointVelocities[1] - jointAccel * dt);
                } else {
                    jointVelocities[1] *= 0.9f;
                }
                currentRobotData.hingeJoints[1]->SetMotorState(JPH::EMotorState::Velocity);
                currentRobotData.hingeJoints[1]->SetTargetAngularVelocity(jointVelocities[1]);
            }
            
            // Joint 2 - Elbow (, / .)
            if (currentRobotData.hingeJoints.size() > 2) {
                if (glfwGetKey(window, GLFW_KEY_COMMA) == GLFW_PRESS) {
                    jointVelocities[2] = std::min(maxJointSpeed, jointVelocities[2] + jointAccel * dt);
                } else if (glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS) {
                    jointVelocities[2] = std::max(-maxJointSpeed, jointVelocities[2] - jointAccel * dt);
                } else {
                    jointVelocities[2] *= 0.9f;
                }
                currentRobotData.hingeJoints[2]->SetMotorState(JPH::EMotorState::Velocity);
                currentRobotData.hingeJoints[2]->SetTargetAngularVelocity(jointVelocities[2]);
            }
            
            // Gripper fingers ([ / ])
            if (currentRobotData.hingeJoints.size() > 5) {
                float gripperSpeed = 5.0f;
                if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) {
                    // Close gripper
                    currentRobotData.hingeJoints[5]->SetMotorState(JPH::EMotorState::Velocity);
                    currentRobotData.hingeJoints[5]->SetTargetAngularVelocity(gripperSpeed);
                    if (currentRobotData.hingeJoints.size() > 6) {
                        currentRobotData.hingeJoints[6]->SetMotorState(JPH::EMotorState::Velocity);
                        currentRobotData.hingeJoints[6]->SetTargetAngularVelocity(-gripperSpeed);
                    }
                } else if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) {
                    // Open gripper
                    currentRobotData.hingeJoints[5]->SetMotorState(JPH::EMotorState::Velocity);
                    currentRobotData.hingeJoints[5]->SetTargetAngularVelocity(-gripperSpeed);
                    if (currentRobotData.hingeJoints.size() > 6) {
                        currentRobotData.hingeJoints[6]->SetMotorState(JPH::EMotorState::Velocity);
                        currentRobotData.hingeJoints[6]->SetTargetAngularVelocity(gripperSpeed);
                    }
                }
            }
        }

        // Physics step
        physicsCore.Step(1.0f / 60.0f);

        // Get first body position for camera target
        JPH::RVec3 robotPos(0.0f, 5.0f, 0.0f);
        if (!currentRobotData.bodies.empty() && !currentRobotData.bodies[0].IsInvalid()) {
            robotPos = bodyInterface.GetPosition(currentRobotData.bodies[0]);
            camTarget = glm::vec3(static_cast<float>(robotPos.GetX()), 
                                  static_cast<float>(robotPos.GetY()), 
                                  static_cast<float>(robotPos.GetZ()));
        }

        // Clear screen
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        // Calculate camera position from spherical coordinates
        glm::vec3 cameraPos(
            camTarget.x + camDistance * cos(camPitch) * sin(camYaw),
            camTarget.y + camDistance * sin(camPitch),
            camTarget.z + camDistance * cos(camPitch) * cos(camYaw)
        );
        glm::vec3 cameraFront = camTarget - cameraPos;

        // Draw physics bodies
        renderer.Draw(&physicsCore.GetPhysicsSystem(), cameraPos, 0, cameraFront);

        // Draw UI
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
        ImGui::Begin("Robot Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Current Robot: %s", robotNames[currentRobot].c_str());
        ImGui::Separator();
        ImGui::Text("Bodies: %d", (int)currentRobotData.bodies.size());
        ImGui::Text("Hinge Joints: %d", (int)currentRobotData.hingeJoints.size());
        ImGui::Text("6DOF Joints: %d", (int)currentRobotData.sixDofJoints.size());
        ImGui::Separator();
        if (!currentRobotData.bodies.empty() && !currentRobotData.bodies[0].IsInvalid()) {
            JPH::RVec3 pos = bodyInterface.GetPosition(currentRobotData.bodies[0]);
            ImGui::Text("Robot Pos: (%.2f, %.2f, %.2f)", pos.GetX(), pos.GetY(), pos.GetZ());
        }
        ImGui::Separator();
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("Camera: Dist=%.1f Yaw=%.0f Pitch=%.0f", camDistance, camYaw * 180.0f / 3.14159f, camPitch * 180.0f / 3.14159f);
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0, 1, 0.85f, 1), "ROBOT CONTROLS:");
        ImGui::Text("I/K: Base rotate | O/L: Shoulder");
        ImGui::Text(",/. : Elbow | []: Gripper open/close");
        ImGui::Separator();
        ImGui::Text("1-8: Switch robots");
        ImGui::Text("WASD: Pan | QE: Rotate | ZX: Pitch | CV: Zoom");
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        
        frameCount++;
        if (frameCount % 60 == 0) {
            std::cout << "Running... Frame " << frameCount << std::endl;
        }
    }

    // Cleanup
    for (auto& bodyId : currentRobotData.bodies) {
        if (!bodyId.IsInvalid()) {
            bodyInterface.RemoveBody(bodyId);
            bodyInterface.DestroyBody(bodyId);
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    physicsCore.Shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "Viewer exited cleanly" << std::endl;
    return 0;
}
