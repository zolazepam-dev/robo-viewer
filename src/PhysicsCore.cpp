#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include "PhysicsCore.h"

#include <cstdint>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

PhysicsCore::~PhysicsCore() { Shutdown(); }

bool PhysicsCore::Init(uint32_t numParallelEnvs) {
    if (mInitialized) return true;
    mNumEnvs = numParallelEnvs;
    JPH::RegisterDefaultAllocator();
    
    uint32_t tempAllocSize = 512 * 1024 * 1024;  // Increased temp memory for more envs
    std::cout << "[PhysicsCore] Allocating Temp Memory..." << "\n";
    mTempAllocator = new JPH::TempAllocatorImpl(tempAllocSize);

    // Use more worker threads for better CPU utilization
    // Detect available cores and use most of them (leave 1 for OS and main thread)
    uint32_t hardwareThreads = std::thread::hardware_concurrency();
    uint32_t joltWorkerThreads = std::max(1u, hardwareThreads - 2);  // Leave 2 cores for OS/main
    if (joltWorkerThreads > 16) joltWorkerThreads = 16;  // Cap at 16 for stability

    std::cout << "[PhysicsCore] Hardware threads: " << hardwareThreads << ", Using: " << joltWorkerThreads << "\n";
    
    // Scale temp memory with number of environments
    // Base: 512MB + 2MB per environment for large env counts
    tempAllocSize = 512 * 1024 * 1024 + (mNumEnvs * 2 * 1024 * 1024);
    std::cout << "[PhysicsCore] Allocating Temp Memory: " << (tempAllocSize / 1024 / 1024) << " MB..." << "\n";
    mTempAllocator = new JPH::TempAllocatorImpl(tempAllocSize);

    std::cout << "[PhysicsCore] Creating Job System..." << "\n";
    mJobSystem = new JPH::JobSystemThreadPool(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, joltWorkerThreads);
    
    // Pin threads to available cores (skip core 0 for OS)
    for (uint32_t i = 0; i < joltWorkerThreads; ++i) {
        int targetCore = (i + 1) % hardwareThreads;  // Distribute across cores, skip 0
        pthread_t threadHandle = mJobSystem->mThreads[i].native_handle();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(targetCore, &cpuset);
        std::cout << "[PhysicsCore] Pinning thread " << i << " to core " << targetCore << "\n";
        pthread_setaffinity_np(threadHandle, sizeof(cpu_set_t), &cpuset);
    }
    std::cout << "[PhysicsCore] Checking Jolt Factory..." << "\n";
    if (JPH::Factory::sInstance == nullptr) {
        std::cout << "[PhysicsCore] Registering Jolt Allocator..." << "\n";
        JPH::RegisterDefaultAllocator();
        std::cout << "[PhysicsCore] Creating Jolt Factory..." << "\n";
        JPH::Factory::sInstance = new JPH::Factory();
        std::cout << "[PhysicsCore] Registering Jolt Types..." << "\n";
        JPH::RegisterTypes();
    }
    std::cout << "[PhysicsCore] Creating BP Layers..." << "\n";
    mBroadPhaseLayerInterface = new BPLayerInterfaceImpl(mNumEnvs);
    mObjectVsBroadPhaseLayerFilter = new ObjectVsBroadPhaseLayerFilterImpl();
    mObjectLayerPairFilter = new ObjectLayerPairFilterImpl();
    
    std::cout << "[PhysicsCore] Creating PhysicsSystem..." << "\n";
    mPhysicsSystem = new JPH::PhysicsSystem();

    // Create group filter table for self-collision management
    mGroupFilter = new JPH::GroupFilterTable(2); // 2 sub-groups
    mGroupFilter->DisableCollision(0, 0); // Parts in sub-group 0 don't collide with each other

    // Initialize physics system
    // Scale body limits based on number of environments
    // Each env has 2 robots with ~10 bodies each + arena objects + KOTH markers
    const uint32_t bodiesPerEnv = 25;  // 2 robots * ~10 bodies + KOTH marker + buffer
    const uint32_t maxBodies = std::max(10240u, mNumEnvs * bodiesPerEnv);
    const uint32_t numBodyMutexes = 0;
    const uint32_t maxBodyPairs = std::max(10240u, mNumEnvs * 100);  // Scale with envs
    const uint32_t maxContactConstraints = std::max(10240u, mNumEnvs * 50);  // Scale with envs
    
    std::cout << "[PhysicsCore] Body limits: " << maxBodies << " bodies, " 
              << maxBodyPairs << " pairs, " << maxContactConstraints << " constraints" << "\n";
    
    mPhysicsSystem->Init(
        maxBodies,  // maxBodies - scaled with environments
        numBodyMutexes,  // numBodyMutexes
        maxBodyPairs,  // maxBodyPairs - scaled
        maxContactConstraints,  // maxContactConstraints - scaled
        *mBroadPhaseLayerInterface, 
        *mObjectVsBroadPhaseLayerFilter, 
        *mObjectLayerPairFilter
    );
    
    std::cout << "[PhysicsCore] PhysicsSystem initialized successfully" << "\n";
    mPhysicsSystem->SetGravity(JPH::Vec3(0.0f, -9.81f, 0.0f));
    mInitialized = true;
    return true;
}

void PhysicsCore::Shutdown() {
    std::cout << "[PhysicsCore] Shutdown starting..." << "\n";
    if (!mInitialized) {
        std::cout << "[PhysicsCore] Not initialized, skipping." << "\n";
        return;
    }
    
    if (mPhysicsSystem) {
        std::cout << "[PhysicsCore] Deleting PhysicsSystem..." << "\n";
        delete mPhysicsSystem; 
        mPhysicsSystem = nullptr;
    }
    
    if (mObjectLayerPairFilter) { delete mObjectLayerPairFilter; mObjectLayerPairFilter = nullptr; }
    if (mObjectVsBroadPhaseLayerFilter) { delete mObjectVsBroadPhaseLayerFilter; mObjectVsBroadPhaseLayerFilter = nullptr; }
    if (mBroadPhaseLayerInterface) { delete mBroadPhaseLayerInterface; mBroadPhaseLayerInterface = nullptr; }
    
    if (mJobSystem) {
        std::cout << "[PhysicsCore] Deleting JobSystem..." << "\n";
        delete mJobSystem; 
        mJobSystem = nullptr;
    }
    
    if (mTempAllocator) {
        std::cout << "[PhysicsCore] Deleting TempAllocator..." << "\n";
        delete mTempAllocator; 
        mTempAllocator = nullptr;
    }
    
    mInitialized = false;
    std::cout << "[PhysicsCore] Shutdown complete." << "\n";
}

void PhysicsCore::GetBodiesByLayers(JPH::BodyIDVector& outBodies, const std::vector<JPH::ObjectLayer>& layers) const {
    outBodies.clear();
    if (!mInitialized || !mPhysicsSystem) return;
    JPH::BodyIDVector allBodies;
    mPhysicsSystem->GetBodies(allBodies);
    const JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    for (const JPH::BodyID& bodyId : allBodies) {
        if (bodyId.IsInvalid()) continue;
        JPH::ObjectLayer layer = bodyInterface.GetObjectLayer(bodyId);
        if (std::find(layers.begin(), layers.end(), layer) != layers.end()) outBodies.push_back(bodyId);
    }
}

void PhysicsCore::GetBodiesByLayer(JPH::BodyIDVector& outBodies, JPH::ObjectLayer layer) const {
    outBodies.clear();
    if (!mInitialized || !mPhysicsSystem) return;
    JPH::BodyIDVector allBodies;
    mPhysicsSystem->GetBodies(allBodies);
    const JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    for (const JPH::BodyID& bodyId : allBodies) {
        if (bodyId.IsInvalid()) continue;
        if (bodyInterface.GetObjectLayer(bodyId) == layer) outBodies.push_back(bodyId);
    }
}

void PhysicsCore::Step(float deltaTime) {
    if (!mInitialized) return;
    mPhysicsSystem->Update(deltaTime, 1, mTempAllocator, mJobSystem);
}
