#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>

#include <emscripten.h>
#include <emscripten/bind.h>
#include <memory>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include <functional>
#include <condition_variable>

// Jolt Physics state with Dimensional Ghosting support
struct PhysicsWorld {
    JPH::PhysicsSystem* physicsSystem;
    JPH::TempAllocator* tempAllocator;
    JPH::JobSystem* jobSystem;
    
    // Custom interfaces for Dimensional Ghosting
    JPH::BroadPhaseLayerInterface* bpInterface;
    JPH::ObjectVsBroadPhaseLayerFilter* objVsBpFilter;
    JPH::ObjectLayerPairFilter* objPairFilter;
    
    uint32_t numEnvs;
};

// Robot state structure for Wasm transport
struct RobotState {
    float x, y, z;
    float vx, vy, vz;
    float rotation[4]; // quaternion
    float jointAngles[20];
    float jointVelocities[20];
    int32_t health;
    int32_t active;
};

// Combat environment state structure for Wasm transport
struct CombatEnvState {
    RobotState robots[2];
    JPH::BodyID physicsBodyIds[2]; // Jolt body IDs for each robot
    PhysicsWorld* physicsWorld;    // Pointer to the physics world
    float arenaBounds[6]; // minX, maxX, minY, maxY, minZ, maxZ
    int32_t stepCount;
    int32_t done;
    float reward[2];
    int32_t id;
    int32_t render;
};
