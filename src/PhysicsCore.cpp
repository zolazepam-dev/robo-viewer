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
    std::cout << "[PhysicsCore] Allocating Temp Memory..." << std::endl;
    mTempAllocator = new JPH::TempAllocatorImpl(tempAllocSize);
    
    // Use more worker threads for better CPU utilization
    // Detect available cores and use most of them (leave 1 for OS and main thread)
    uint32_t hardwareThreads = std::thread::hardware_concurrency();
    uint32_t joltWorkerThreads = std::max(1u, hardwareThreads - 2);  // Leave 2 cores for OS/main
    if (joltWorkerThreads > 16) joltWorkerThreads = 16;  // Cap at 16 for stability
    
    std::cout << "[PhysicsCore] Hardware threads: " << hardwareThreads << ", Using: " << joltWorkerThreads << std::endl;
    std::cout << "[PhysicsCore] Creating Job System..." << std::endl;
    mJobSystem = new JPH::JobSystemThreadPool(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, joltWorkerThreads);
    
    // Pin threads to available cores (skip core 0 for OS)
    for (uint32_t i = 0; i < joltWorkerThreads; ++i) {
        int targetCore = (i + 1) % hardwareThreads;  // Distribute across cores, skip 0
        pthread_t threadHandle = mJobSystem->mThreads[i].native_handle();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(targetCore, &cpuset);
        std::cout << "[PhysicsCore] Pinning thread " << i << " to core " << targetCore << std::endl;
        pthread_setaffinity_np(threadHandle, sizeof(cpu_set_t), &cpuset);
    }
    std::cout << "[PhysicsCore] Checking Jolt Factory..." << std::endl;
    if (JPH::Factory::sInstance == nullptr) {
        std::cout << "[PhysicsCore] Registering Jolt Allocator..." << std::endl;
        JPH::RegisterDefaultAllocator();
        std::cout << "[PhysicsCore] Creating Jolt Factory..." << std::endl;
        JPH::Factory::sInstance = new JPH::Factory();
        std::cout << "[PhysicsCore] Registering Jolt Types..." << std::endl;
        JPH::RegisterTypes();
    }
    std::cout << "[PhysicsCore] Creating BP Layers..." << std::endl;
    mBroadPhaseLayerInterface = new BPLayerInterfaceImpl(mNumEnvs);
    mObjectVsBroadPhaseLayerFilter = new ObjectVsBroadPhaseLayerFilterImpl();
    mObjectLayerPairFilter = new ObjectLayerPairFilterImpl();
    const uint32_t maxBodies = 10240;
    const uint32_t numBodyMutexes = 0;
    const uint32_t maxBodyPairs = 10240;
    const uint32_t maxContactConstraints = 10240;
    std::cout << "[PhysicsCore] Creating PhysicsSystem..." << std::endl;
    mPhysicsSystem = new JPH::PhysicsSystem();
    
    // Create group filter table for self-collision management
    mGroupFilter = new JPH::GroupFilterTable(2); // 2 sub-groups
    mGroupFilter->DisableCollision(0, 0); // Parts in sub-group 0 don't collide with each other
    // By default, 0 and 1 WILL collide if they are in the same GroupID
    
    mPhysicsSystem->Init(maxBodies, numBodyMutexes, maxBodyPairs, maxContactConstraints, *mBroadPhaseLayerInterface, *mObjectVsBroadPhaseLayerFilter, *mObjectLayerPairFilter);
    mPhysicsSystem->SetGravity(JPH::Vec3(0.0f, -9.81f, 0.0f));
    mInitialized = true;
    return true;
}

void PhysicsCore::Shutdown() {
    std::cout << "[PhysicsCore] Shutdown starting..." << std::endl;
    if (!mInitialized) {
        std::cout << "[PhysicsCore] Not initialized, skipping." << std::endl;
        return;
    }
    
    if (mPhysicsSystem) {
        std::cout << "[PhysicsCore] Deleting PhysicsSystem..." << std::endl;
        delete mPhysicsSystem; 
        mPhysicsSystem = nullptr;
    }
    
    if (mObjectLayerPairFilter) { delete mObjectLayerPairFilter; mObjectLayerPairFilter = nullptr; }
    if (mObjectVsBroadPhaseLayerFilter) { delete mObjectVsBroadPhaseLayerFilter; mObjectVsBroadPhaseLayerFilter = nullptr; }
    if (mBroadPhaseLayerInterface) { delete mBroadPhaseLayerInterface; mBroadPhaseLayerInterface = nullptr; }
    
    if (mJobSystem) {
        std::cout << "[PhysicsCore] Deleting JobSystem..." << std::endl;
        delete mJobSystem; 
        mJobSystem = nullptr;
    }
    
    if (mTempAllocator) {
        std::cout << "[PhysicsCore] Deleting TempAllocator..." << std::endl;
        delete mTempAllocator; 
        mTempAllocator = nullptr;
    }
    
    mInitialized = false;
    std::cout << "[PhysicsCore] Shutdown complete." << std::endl;
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
