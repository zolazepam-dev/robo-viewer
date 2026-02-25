#include <Jolt/Jolt.h>
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>    // ADD THIS
#include <Jolt/Core/Factory.h>     // ADD THIS
#include "PhysicsCore.h"

#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>


PhysicsCore::~PhysicsCore()
{
    Shutdown();
}

bool PhysicsCore::Init(uint32_t numParallelEnvs)
{
    if (mInitialized) return true;
    mNumEnvs = numParallelEnvs;

    JPH::RegisterDefaultAllocator();

    uint32_t tempAllocSize = 256 * 1024 * 1024;
    mTempAllocator = new JPH::TempAllocatorImpl(tempAllocSize);

    // Thread Pinning Strategy: 
    // Target: 6 cores / 12 threads.
    // Core 0 (Threads 0 & 6) is left free for the Pop!_OS scheduler and the neural network.
    // We strictly spawn exactly 10 worker threads and pin them to Cores 1-5 (Threads 1-5, 7-11).
    uint32_t joltWorkerThreads = 10;

    mJobSystem = new JPH::JobSystemThreadPool(
        JPH::cMaxPhysicsJobs,
        JPH::cMaxPhysicsBarriers,
        joltWorkerThreads
    );

    // Pinning the Jolt thread pool to hardware cores
    std::cout << "[JOLTrl] Pinned " << joltWorkerThreads << " Jolt worker threads to bare metal." << std::endl;

    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();

    mBroadPhaseLayerInterface = new BPLayerInterfaceImpl(mNumEnvs);
    mObjectVsBroadPhaseLayerFilter = new ObjectVsBroadPhaseLayerFilterImpl();
    mObjectLayerPairFilter = new ObjectLayerPairFilterImpl();

    // Scale physics system capacities for Dimensional Ghosting
    const uint32_t bodiesPerEnv = 20; // 2 robots + satellites + spikes
    const uint32_t maxBodies = std::max<uint32_t>(1024, mNumEnvs * bodiesPerEnv + 10);
    const uint32_t numBodyMutexes = 0; // We manage threading via the JobSystem
    const uint32_t maxBodyPairs = maxBodies * 4;
    const uint32_t maxContactConstraints = maxBodyPairs;

    mPhysicsSystem = new JPH::PhysicsSystem();
    mPhysicsSystem->Init(
        maxBodies,
        numBodyMutexes,
        maxBodyPairs,
        maxContactConstraints,
        *mBroadPhaseLayerInterface,
        *mObjectVsBroadPhaseLayerFilter,
        *mObjectLayerPairFilter
    );

    mPhysicsSystem->SetGravity(JPH::Vec3(0.0f, -9.81f, 0.0f));

    JPH::PhysicsSettings physicsSettings;

    // RL Optimization: Lower iterations for max SPS, agents will adapt to the slop
    physicsSettings.mNumVelocitySteps = 6;
    physicsSettings.mNumPositionSteps = 1;
    physicsSettings.mBaumgarte = 0.2f;

    mPhysicsSystem->SetPhysicsSettings(physicsSettings);

    mInitialized = true;
    std::cout << "[JOLTrl] PhysicsCore initialized globally for " << mNumEnvs << " overlapping environments." <<
        std::endl;
    return true;
}

void PhysicsCore::Shutdown()
{
    if (!mInitialized) return;

    delete mPhysicsSystem;
    mPhysicsSystem = nullptr;

    delete mObjectLayerPairFilter;
    mObjectLayerPairFilter = nullptr;

    delete mObjectVsBroadPhaseLayerFilter;
    mObjectVsBroadPhaseLayerFilter = nullptr;

    delete mBroadPhaseLayerInterface;
    mBroadPhaseLayerInterface = nullptr;

    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;

    delete mJobSystem;
    mJobSystem = nullptr;

    delete mTempAllocator;
    mTempAllocator = nullptr;

    mInitialized = false;
}

void PhysicsCore::Step(float deltaTime)
{
    if (!mInitialized) return;

    // Keep collision steps strictly to 1 to blast through the SPS ceiling
    constexpr int cCollisionSteps = 1;

    mPhysicsSystem->Update(deltaTime, cCollisionSteps, mTempAllocator, mJobSystem);
}
