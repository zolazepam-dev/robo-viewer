/**
 * @file PhysicsWorld.cpp
 * @brief Implementation of PhysicsWorld
 */

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include "PhysicsWorld.h"

#include <cstdint>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <algorithm>

BroadPhaseLayerInterfaceImpl::BroadPhaseLayerInterfaceImpl(uint32_t numEnvs) : mNumEnvs(numEnvs) {}

uint BroadPhaseLayerInterfaceImpl::GetNumBroadPhaseLayers() const {
    return BroadPhaseLayers::NUM_LAYERS;
}

JPH::BroadPhaseLayer BroadPhaseLayerInterfaceImpl::GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const {
    if (inLayer == PhysicsLayers::STATIC) return JPH::BroadPhaseLayer(BroadPhaseLayers::STATIC);
    return JPH::BroadPhaseLayer(BroadPhaseLayers::DYNAMIC);
}

bool ObjectVsBroadPhaseLayerFilterImpl::ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const {
    if (inLayer1 == PhysicsLayers::STATIC) return inLayer2 == JPH::BroadPhaseLayer(BroadPhaseLayers::DYNAMIC);
    return true;
}

bool ObjectLayerPairFilterImpl::ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const {
    if (inObject1 >= PhysicsLayers::GHOST_BASE || inObject2 >= PhysicsLayers::GHOST_BASE) return false;
    if (inObject1 == PhysicsLayers::STATIC && inObject2 == PhysicsLayers::STATIC) return false;
    if (inObject1 == PhysicsLayers::STATIC || inObject2 == PhysicsLayers::STATIC) return true;
    return inObject1 == inObject2;
}

PhysicsWorld::~PhysicsWorld() { Shutdown(); }

bool PhysicsWorld::Init(uint32_t numParallelEnvs, uint32_t numWorkerThreads, const int* pinToCores) {
    if (mInitialized) return true;
    mNumEnvs = numParallelEnvs;

    JPH::RegisterDefaultAllocator();
    mTempAllocator = new JPH::TempAllocatorImpl(256 * 1024 * 1024);
    mJobSystem = new JPH::JobSystemThreadPool(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, numWorkerThreads);

    if (pinToCores != nullptr) {
        for (uint32_t i = 0; i < numWorkerThreads && i < 10; ++i) {
            pthread_t threadHandle = mJobSystem->mThreads[i].native_handle();
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(pinToCores[i], &cpuset);
            pthread_setaffinity_np(threadHandle, sizeof(cpu_set_t), &cpuset);
        }
    }

    static std::once_flag joltInitFlag;
    std::call_once(joltInitFlag, []() {
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    });

    mBroadPhaseLayerInterface = new BroadPhaseLayerInterfaceImpl(mNumEnvs);
    mObjectVsBroadPhaseLayerFilter = new ObjectVsBroadPhaseLayerFilterImpl();
    mObjectLayerPairFilter = new ObjectLayerPairFilterImpl();

    const uint32_t maxBodies = std::max<uint32_t>(2048, mNumEnvs * 60 + 256);
    const uint32_t numBodyMutexes = std::max<uint32_t>(1, mNumEnvs / 2);
    const uint32_t maxBodyPairs = std::min<uint32_t>(131072, maxBodies * 8);

    mPhysicsSystem = new JPH::PhysicsSystem();
    mPhysicsSystem->Init(maxBodies, numBodyMutexes, maxBodyPairs, maxBodyPairs,
        *mBroadPhaseLayerInterface, *mObjectVsBroadPhaseLayerFilter, *mObjectLayerPairFilter);
    mPhysicsSystem->SetGravity(JPH::Vec3(0.0f, -9.81f, 0.0f));

    JPH::PhysicsSettings settings;
    settings.mNumVelocitySteps = 2;
    settings.mNumPositionSteps = 1;
    settings.mBaumgarte = 0.2f;
    settings.mAllowSleeping = false;
    mPhysicsSystem->SetPhysicsSettings(settings);
    mContactListener = new ContactListener();
    mContactListener->Init(mNumEnvs, 14);

    mInitialized = true;
    std::cout << "[PhysicsWorld] Initialized for " << mNumEnvs << " environments." << std::endl;
    return true;
}

void PhysicsWorld::Shutdown() {
    if (!mInitialized) return;
    delete mContactListener; mContactListener = nullptr;
    delete mPhysicsSystem; mPhysicsSystem = nullptr;
    delete mObjectLayerPairFilter; mObjectLayerPairFilter = nullptr;
    delete mObjectVsBroadPhaseLayerFilter; mObjectVsBroadPhaseLayerFilter = nullptr;
    delete mBroadPhaseLayerInterface; mBroadPhaseLayerInterface = nullptr;
    if (JPH::Factory::sInstance) { delete JPH::Factory::sInstance; JPH::Factory::sInstance = nullptr; }
    delete mJobSystem; mJobSystem = nullptr;
    delete mTempAllocator; mTempAllocator = nullptr;
    mInitialized = false;
}

void PhysicsWorld::Step(float deltaTime) {
    if (!mInitialized) return;
    mPhysicsSystem->Update(deltaTime, 1, mTempAllocator, mJobSystem);
}

void PhysicsWorld::GetBodiesByLayers(JPH::BodyIDVector& outBodies, const std::vector<JPH::ObjectLayer>& layers) const {
    outBodies.clear();
    if (!mInitialized || !mPhysicsSystem) return;
    JPH::BodyIDVector allBodies;
    mPhysicsSystem->GetBodies(allBodies);
    const JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    for (const JPH::BodyID& bodyId : allBodies) {
        if (bodyId.IsInvalid()) continue;
        JPH::ObjectLayer layer = bodyInterface.GetObjectLayer(bodyId);
        if (std::find(layers.begin(), layers.end(), layer) != layers.end()) {
            outBodies.push_back(bodyId);
        }
    }
}

void PhysicsWorld::GetBodiesByLayer(JPH::BodyIDVector& outBodies, JPH::ObjectLayer layer) const {
    outBodies.clear();
    if (!mInitialized || !mPhysicsSystem) return;
    JPH::BodyIDVector allBodies;
    mPhysicsSystem->GetBodies(allBodies);
    const JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    for (const JPH::BodyID& bodyId : allBodies) {
        if (bodyId.IsInvalid()) continue;
        if (bodyInterface.GetObjectLayer(bodyId) == layer) {
            outBodies.push_back(bodyId);
        }
    }
}

void ContactListener::Init(uint32_t numEnvs, int numSatellites) {
    mForceReadingsPerEnv.resize(numEnvs);
    for (auto& readings : mForceReadingsPerEnv) {
        readings[0].Reset(numSatellites);
        readings[1].Reset(numSatellites);
    }
}

ForceSensorReading& ContactListener::GetForceReading(uint32_t envIdx, int robotIdx) {
    return mForceReadingsPerEnv[envIdx][robotIdx];
}

void ContactListener::ResetForceReadings(uint32_t envIdx, int numSatellites) {
    mForceReadingsPerEnv[envIdx][0].Reset(numSatellites);
    mForceReadingsPerEnv[envIdx][1].Reset(numSatellites);
}
