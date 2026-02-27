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

    // --- BUILD THE SINGLE SOURCE OF TRUTH ARENA (36x36x36) ---
    JPH::BodyInterface& body_interface = mPhysicsCore.GetPhysicsSystem().GetBodyInterface();
    
    // Floor: 36x36 meters, 2.0m thick
    JPH::BoxShapeSettings floor_shape(JPH::Vec3(18.0f, 1.0f, 18.0f));
    JPH::RefConst<JPH::Shape> floor = floor_shape.Create().Get();
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(floor, JPH::RVec3(0.0f, 1.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    
    // Ceiling
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(floor, JPH::RVec3(0.0f, 36.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    
    // Walls: 2.0m thick
    JPH::BoxShapeSettings wall_shape(JPH::Vec3(18.0f, 18.0f, 1.0f));
    JPH::RefConst<JPH::Shape> wall = wall_shape.Create().Get();
    
    // North/South (z = +/- 18.0 + offset)
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(0.0f, 18.0f, -19.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(0.0f, 18.0f, 19.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    
    // East/West (x = +/- 18.0 + offset, rotated)
    JPH::Quat rot90 = JPH::Quat::sRotation(JPH::Vec3::sAxisY(), JPH::DegreesToRadians(90.0f));
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(19.0f, 18.0f, 0.0f), rot90, JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(-19.0f, 18.0f, 0.0f), rot90, JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    // -------------------------------------------------

    std::cout << "[VectorizedEnv::Init] Arena built" << std::endl;

    if (initRobots) {
        std::cout << "[VectorizedEnv::Init] Initializing " << mNumEnvs << " environments" << std::endl;
        mEnvs.resize(mNumEnvs);
        for (int i = 0; i < mNumEnvs; ++i)
        {
            std::cout << "[VectorizedEnv::Init] Initializing environment " << i << std::endl;
            mEnvs[i].Init(i, &mPhysicsCore.GetPhysicsSystem(), &mRobotLoader);
            std::cout << "[VectorizedEnv::Init] Environment " << i << " initialized" << std::endl;
        }

        std::cout << "[VectorizedEnv::Init] All environments initialized" << std::endl;

        mObservationDim = mEnvs[0].GetObservationDim();
        mAllObservations.resize(mNumEnvs * mObservationDim * 2, 0.0f);
        mAllRewards.resize(mNumEnvs * 2, 0.0f);
        mAllDones.resize(mNumEnvs, false);
    }

    std::cout << "[VectorizedEnv::Init] Optimizing broad phase" << std::endl;
    mPhysicsCore.GetPhysicsSystem().OptimizeBroadPhase();
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

    mPhysicsCore.Step(1.0f / 60.0f);

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
    if (gCombatContactListener)
    {
        mPhysicsCore.GetPhysicsSystem().SetContactListener(nullptr);
        gCombatContactListener = nullptr;
    }
}
