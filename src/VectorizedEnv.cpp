// MUST BE FIRST
#include <Jolt/Jolt.h>
#include "VectorizedEnv.h"

#include <iostream>
#include <algorithm>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Body/BodyInterface.h>

VectorizedEnv::VectorizedEnv(int numEnvs)
    : mNumEnvs(numEnvs)
{
}

void VectorizedEnv::Init()
{
    if (!mPhysicsCore.Init(mNumEnvs))
    {
        std::cerr << "[JOLTrl] FATAL: Global PhysicsCore failed to initialize!" << std::endl;
        return;
    }

    // --- THE GLOBAL ARENA FLOOR ---
    // A single, massive 200x200 static plane for all 128 dimensions to share
    JPH::BodyInterface& body_interface = mPhysicsCore.GetPhysicsSystem().GetBodyInterface();
    
    // 1. Create the shape settings safely
    JPH::BoxShapeSettings floor_shape_settings(JPH::Vec3(100.0f, 1.0f, 100.0f));
    
    // 2. Bake it into a reference-counted Shape BEFORE passing to the body settings
    JPH::RefConst<JPH::Shape> floor_shape = floor_shape_settings.Create().Get();
    
    // 3. Create the body
    JPH::BodyCreationSettings floor_settings(floor_shape, JPH::RVec3(0.0f, -1.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC);
    body_interface.CreateAndAddBody(floor_settings, JPH::EActivation::DontActivate);
    // ------------------------------

    mEnvs.resize(mNumEnvs);
    for (int i = 0; i < mNumEnvs; ++i)
    {
        mEnvs[i].Init(i, &mPhysicsCore.GetPhysicsSystem(), &mRobotLoader);
    }

    mObservationDim = mEnvs[0].GetObservationDim();

    mAllObservations.resize(mNumEnvs * mObservationDim * 2, 0.0f);
    mAllRewards.resize(mNumEnvs * 2, 0.0f);
    mAllDones.resize(mNumEnvs, false);

    Reset(-1);

    mPhysicsCore.GetPhysicsSystem().OptimizeBroadPhase();
    std::cout << "[JOLTrl] Global BroadPhase optimized. Engine ready." << std::endl;
}

void VectorizedEnv::Step(const std::vector<float>& actions)
{
    const int actionDim = ACTIONS_PER_ROBOT;

    for (int i = 0; i < mNumEnvs; ++i)
    {
        if (mAllDones[i]) continue;

        const float* actions1 = actions.data() + (i * 2 * actionDim);
        const float* actions2 = actions1 + actionDim;

        mEnvs[i].QueueActions(actions1, actions2);
    }

    // The Global Matrix Crunch (1/120f guarantees RL stability)
    mPhysicsCore.Step(1.0f / 120.0f);

    for (int i = 0; i < mNumEnvs; ++i)
    {
        if (mAllDones[i]) continue;

        StepResult result = mEnvs[i].HarvestState();
 int obsOffset = i * mObservationDim * 2;
        std::copy(result.obs_robot1.begin(), result.obs_robot1.end(), mAllObservations.begin() + obsOffset);
        std::copy(result.obs_robot2.begin(), result.obs_robot2.end(),
                  mAllObservations.begin() + obsOffset + mObservationDim);

        int rewardOffset = i * 2;
        mAllRewards[rewardOffset] = result.reward1.Scalar();
        mAllRewards[rewardOffset + 1] = result.reward2.Scalar();

        mAllDones[i] = result.done;
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