// MUST BE FIRST
#include <Jolt/Jolt.h>
#include "VectorizedEnv.h"

#include <iostream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>

#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/PhysicsSystem.h>

// Global single instance of CombatContactListener
CombatContactListener* gCombatContactListener = nullptr;

VectorizedEnv::VectorizedEnv(int numEnvs)
    : mNumEnvs(numEnvs)
{
}

void VectorizedEnv::Init(bool initRobots)
{
    std::cout << "[VectorizedEnv::Init] Start" << std::endl;
    
    if (!mPhysicsCore.Init(mNumEnvs))
    {
        std::cerr << "[JOLTrl] FATAL: Global PhysicsCore failed to initialize!" << std::endl;
        return;
    }

    std::cout << "[VectorizedEnv::Init] PhysicsCore initialized" << std::endl;

    // Create and register the global CombatContactListener
    gCombatContactListener = &CombatContactListener::Get();
    mPhysicsCore.GetPhysicsSystem().SetContactListener(gCombatContactListener);

    std::cout << "[VectorizedEnv::Init] Contact listener registered" << std::endl;

    if (initRobots) {
        std::cout << "[VectorizedEnv::Init] Initializing " << mNumEnvs << " environments" << std::endl;
        mEnvs.resize(mNumEnvs);
        for (int i = 0; i < mNumEnvs; ++i)
        {
            if (i % 10 == 0) {
                std::cout << "[VectorizedEnv::Init] Initializing environment " << i << std::endl;
            }
            mEnvs[i].Init(i, &mPhysicsCore.GetPhysicsSystem(), &mRobotLoader);
            std::cout << "[VectorizedEnv] Env " << i << " robot1 mainBodyId: " << mEnvs[i].GetRobot1().mainBodyId.GetIndex() 
                      << " robot2 mainBodyId: " << mEnvs[i].GetRobot2().mainBodyId.GetIndex() << std::endl;
        }

        std::cout << "[VectorizedEnv::Init] All environments initialized" << std::endl;

        mObservationDim = mEnvs[0].GetObservationDim();
        mAllObservations.resize(mNumEnvs * mObservationDim * 2, 0.0f);
        mAllRewards.resize(mNumEnvs * 2, 0.0f);
        mAllDones.resize(mNumEnvs, false);
        mAllVectorRewards.resize(mNumEnvs);  // PRE-ALLOCATE: Zero-allocation mandate
    }

    std::cout << "[VectorizedEnv::Init] Complete" << std::endl;
}

void VectorizedEnv::Step(const AlignedVector32<float>& actions)
{
    const int actionDim = ACTIONS_PER_ROBOT;
    for (int i = 0; i < mNumEnvs; ++i)
    {
        if (mAllDones[i]) continue;
        mEnvs[i].QueueActions(actions.data() + (i * 2 * actionDim), actions.data() + (i * 2 * actionDim + actionDim));
    }

    mPhysicsCore.Step(1.0f / 120.0f);

    for (int i = 0; i < mNumEnvs; ++i)
    {
        if (mAllDones[i]) continue;
        
        int obsOffset = i * mObservationDim * 2;
        float* obs1 = mAllObservations.data() + obsOffset;
        float* obs2 = mAllObservations.data() + obsOffset + mObservationDim;
        float* reward1 = mAllRewards.data() + (i * 2);
        float* reward2 = mAllRewards.data() + (i * 2 + 1);
        bool done = false;

        mEnvs[i].HarvestState(obs1, obs2, reward1, reward2, done);
        mAllDones[i] = done;
    }

    // NO RESIZE: Already pre-allocated in Init()
    for (int i = 0; i < mNumEnvs; ++i) {
        if (mAllDones[i]) continue;
        mAllVectorRewards[i] = mEnvs[i].GetRobot1Reward();
    }
}


void VectorizedEnv::HarvestAfterPhysics()
{
    // Harvest state from all environments after physics was updated externally
    // Simple inline loop - thread pool overhead was negating benefits
    
    for (int i = 0; i < mNumEnvs; ++i) {
        if (mAllDones[i]) continue;
        int obsOffset = i * mObservationDim * 2;
        float* obs1 = mAllObservations.data() + obsOffset;
        float* obs2 = mAllObservations.data() + obsOffset + mObservationDim;
        float* reward1 = mAllRewards.data() + (i * 2);
        float* reward2 = mAllRewards.data() + (i * 2 + 1);
        bool done = false;
        mEnvs[i].HarvestState(obs1, obs2, reward1, reward2, done);
        mAllDones[i] = done;
        mAllVectorRewards[i] = mEnvs[i].GetRobot1Reward();
    }
}

void VectorizedEnv::Reset(int envIndex)
{
    if (envIndex < 0)
    {
        for (auto& env : mEnvs) env.Reset();
        std::fill(mAllDones.begin(), mAllDones.end(), false);
    }
    else
    {
        mEnvs[envIndex].Reset();
        mAllDones[envIndex] = false;
    }
}

void VectorizedEnv::ResetDoneEnvs()
{
    for (int i = 0; i < mNumEnvs; ++i)
    {
        if (mAllDones[i])
        {
            mEnvs[i].Reset();
            mAllDones[i] = false;
        }
    }
}

VectorizedEnv::~VectorizedEnv()
{
    Shutdown();
}

void VectorizedEnv::Shutdown()
{
    if (gCombatContactListener)
    {
        mPhysicsCore.GetPhysicsSystem().SetContactListener(nullptr);
        gCombatContactListener = nullptr;
    }
    mPhysicsCore.Shutdown();
    mEnvs.clear();
    mAllObservations.clear();
    mAllRewards.clear();
    mAllDones.clear();
}
