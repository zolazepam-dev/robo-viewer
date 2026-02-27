#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>

int main() {
    std::cout << "[TEST] Starting minimal room test..." << std::endl;
    
    // Initialize Jolt
    JPH::RegisterTypes();
    
    // Create physics system
    JPH::PhysicsSystem physicsSystem;
    JPH::TempAllocatorImpl temp_allocator(10 * 1024 * 1024); // 10 MB
    JPH::JobSystemThreadPool job_system(2, temp_allocator); // 2 threads
    
    JPH::PhysicsSettings settings;
    physicsSystem.Init(1024, 64, 1024, 1024, JPH::BroadPhaseLayerInterface::sDefault, JPH::ObjectVsBroadPhaseLayerFilter::sDefault, JPH::ObjectLayerPairFilter::sDefault);
    
    // Create body interface
    JPH::BodyInterface& body_interface = physicsSystem.GetBodyInterface();
    
    // --- Create single 12x12x12 room ---
    // Floor (13x13, 0.5m thick) at y=0
    JPH::BoxShapeSettings floor_shape(JPH::Vec3(6.5f, 0.5f, 6.5f));
    JPH::RefConst<JPH::Shape> floor = floor_shape.Create().Get();
    JPH::BodyCreationSettings floor_settings(floor, JPH::RVec3(0.0f, -0.5f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, 0);
    JPH::BodyID floor_id = body_interface.CreateAndAddBody(floor_settings, JPH::EActivation::DontActivate);
    
    // Ceiling at y=12
    JPH::BodyCreationSettings ceil_settings(floor, JPH::RVec3(0.0f, 12.5f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, 0);
    JPH::BodyID ceil_id = body_interface.CreateAndAddBody(ceil_settings, JPH::EActivation::DontActivate);
    
    // Walls
    JPH::BoxShapeSettings wall_x_shape(JPH::Vec3(0.5f, 6.0f, 6.0f)); // X walls
    JPH::RefConst<JPH::Shape> wall_x = wall_x_shape.Create().Get();
    
    JPH::BoxShapeSettings wall_z_shape(JPH::Vec3(6.0f, 6.0f, 0.5f)); // Z walls
    JPH::RefConst<JPH::Shape> wall_z = wall_z_shape.Create().Get();
    
    // East wall (x=6)
    JPH::BodyCreationSettings east_wall(wall_x, JPH::RVec3(6.0f, 6.0f, 0.0f), JPH::Quat::sRotation(JPH::Vec3::sAxisY(), JPH::DegreesToRadians(90.0f)), JPH::EMotionType::Static, 0);
    JPH::BodyID east_id = body_interface.CreateAndAddBody(east_wall, JPH::EActivation::DontActivate);
    
    // West wall (x=-6)
    JPH::BodyCreationSettings west_wall(wall_x, JPH::RVec3(-6.0f, 6.0f, 0.0f), JPH::Quat::sRotation(JPH::Vec3::sAxisY(), JPH::DegreesToRadians(90.0f)), JPH::EMotionType::Static, 0);
    JPH::BodyID west_id = body_interface.CreateAndAddBody(west_wall, JPH::EActivation::DontActivate);
    
    // North wall (z=-6)
    JPH::BodyCreationSettings north_wall(wall_z, JPH::RVec3(0.0f, 6.0f, -6.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, 0);
    JPH::BodyID north_id = body_interface.CreateAndAddBody(north_wall, JPH::EActivation::DontActivate);
    
    // South wall (z=6)
    JPH::BodyCreationSettings south_wall(wall_z, JPH::RVec3(0.0f, 6.0f, 6.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, 0);
    JPH::BodyID south_id = body_interface.CreateAndAddBody(south_wall, JPH::EActivation::DontActivate);
    
    std::cout << "[TEST] Room created: floor=" << floor_id.GetIndex() 
              << ", ceiling=" << ceil_id.GetIndex()
              << ", east=" << east_id.GetIndex()
              << ", west=" << west_id.GetIndex()
              << ", north=" << north_id.GetIndex()
              << ", south=" << south_id.GetIndex() << std::endl;
    
    // Add two spheres (robots) at (-2, 2.5, 0) and (2, 2.5, 0)
    JPH::SphereShapeSettings sphere_shape(0.5f);
    JPH::RefConst<JPH::Shape> sphere = sphere_shape.Create().Get();
    
    JPH::BodyCreationSettings robot1(sphere, JPH::RVec3(-2.0f, 2.5f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, 1);
    JPH::BodyID robot1_id = body_interface.CreateAndAddBody(robot1, JPH::EActivation::Activate);
    
    JPH::BodyCreationSettings robot2(sphere, JPH::RVec3(2.0f, 2.5f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, 2);
    JPH::BodyID robot2_id = body_interface.CreateAndAddBody(robot2, JPH::EActivation::Activate);
    
    std::cout << "[TEST] Robots created: robot1=" << robot1_id.GetIndex() 
              << ", robot2=" << robot2_id.GetIndex() << std::endl;
    
    // Step physics once
    physicsSystem.Update(1.0f / 60.0f, 1, temp_allocator, job_system);
    
    // Get positions
    JPH::RVec3 pos1 = body_interface.GetPosition(robot1_id);
    JPH::RVec3 pos2 = body_interface.GetPosition(robot2_id);
    
    std::cout << "[TEST] Robot positions: robot1=(" << pos1.GetX() << "," << pos1.GetY() << "," << pos1.GetZ() << ")"
              << ", robot2=(" << pos2.GetX() << "," << pos2.GetY() << "," << pos2.GetZ() << ")" << std::endl;
    
    // Check if they're still at spawn positions (no explosion)
    bool robot1_ok = std::abs(pos1.GetX() + 2.0f) < 0.01f && 
                     std::abs(pos1.GetY() - 2.5f) < 0.01f && 
                     std::abs(pos1.GetZ() - 0.0f) < 0.01f;
    
    bool robot2_ok = std::abs(pos2.GetX() - 2.0f) < 0.01f && 
                     std::abs(pos2.GetY() - 2.5f) < 0.01f && 
                     std::abs(pos2.GetZ() - 0.0f) < 0.01f;
    
    std::cout << "[TEST] Robot stability: robot1=" << (robot1_ok ? "OK" : "EXPLODED") 
              << ", robot2=" << (robot2_ok ? "OK" : "EXPLODED") << std::endl;
    
    if (robot1_ok && robot2_ok) {
        std::cout << "[TEST] SUCCESS: Room and robots are stable!" << std::endl;
        return 0;
    } else {
        std::cout << "[TEST] FAILURE: Robots exploded or moved unexpectedly" << std::endl;
        return 1;
    }
}