/**
 * @file VectorizedOctopodEnv.cpp
 * @brief Vectorized environment for parallel octopod training
 */

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

#include <iostream>
#include <cstring>
#include "VectorizedOctopodEnv.h"

VectorizedOctopodEnv::VectorizedOctopodEnv(int numEnvs, int stepsPerEpisode)
    : mNumEnvs(numEnvs), mStepsPerEpisode(stepsPerEpisode)
{
    mAllObservations.resize(mObservationDim * 2 * mNumEnvs);
    mAllRewards.resize(2 * mNumEnvs);
    mAllDones.resize(mNumEnvs, 0.0f);
}

VectorizedOctopodEnv::~VectorizedOctopodEnv()
{
    Shutdown();
}

void VectorizedOctopodEnv::Init(bool initRobots)
{
    std::cout << "[VectorizedOctopodEnv] Initializing with " << mNumEnvs << " environments..." << std::endl;
    
    // Initialize physics system with number of parallel environments
    mPhysicsCore.Init(mNumEnvs);
    
    std::cout << "[VectorizedOctopodEnv] Physics system initialized." << std::endl;
    
    // Create the arena (Floor, Walls)
    JPH::BodyInterface& body_interface = mPhysicsCore.GetPhysicsSystem().GetBodyInterface();
    
    // Floor: 100x100 meters, 2.0m thick (y=0 is the center of the 2m thick box, so it goes from -1 to 1)
    // Actually Jolt's BoxShape half extent means Vec3(50, 1, 50) is 100x2x100.
    JPH::BoxShapeSettings floor_shape(JPH::Vec3(50.0f, 1.0f, 50.0f));
    JPH::RefConst<JPH::Shape> floor = floor_shape.Create().Get();
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(floor, JPH::RVec3(0.0f, -1.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    
    // Walls: 100m long, 20m high, 2m thick
    JPH::BoxShapeSettings wall_shape(JPH::Vec3(50.0f, 10.0f, 1.0f));
    JPH::RefConst<JPH::Shape> wall = wall_shape.Create().Get();
    
    // North/South (z = +/- 50.0 + offset)
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(0.0f, 10.0f, -51.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(0.0f, 10.0f, 51.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    
    // East/West (x = +/- 50.0 + offset, rotated)
    JPH::Quat rot90 = JPH::Quat::sRotation(JPH::Vec3::sAxisY(), JPH::DegreesToRadians(90.0f));
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(51.0f, 10.0f, 0.0f), rot90, JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(-51.0f, 10.0f, 0.0f), rot90, JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);

    std::cout << "[VectorizedOctopodEnv] Arena built." << std::endl;
    
    if (initRobots) {
        // Pre-allocate environments
        mEnvs.reserve(mNumEnvs);
        for (int i = 0; i < mNumEnvs; i++) {
            // Create environments one by one
            mEnvs.emplace_back();
            mEnvs[i].Init(i, &mPhysicsCore, mStepsPerEpisode);
            std::cout << "[VectorizedOctopodEnv] Environment " << i << " initialized" << std::endl;
        }
    }
    
    std::cout << "[VectorizedOctopodEnv] All environments initialized." << std::endl;
}

void VectorizedOctopodEnv::Shutdown()
{
    mEnvs.clear();
    mPhysicsCore.Shutdown();
}

void VectorizedOctopodEnv::Step(const AlignedVector32<float>& actions)
{
    // Distribute actions to each environment
    for (int i = 0; i < mNumEnvs; i++) {
        const float* act1 = &actions[(i * 2 + 0) * mActionDim];
        const float* act2 = &actions[(i * 2 + 1) * mActionDim];
        mEnvs[i].QueueActions(act1, act2);
    }
    
    // Step physics
    mPhysicsCore.Step(1.0f / 60.0f);
}

void VectorizedOctopodEnv::HarvestStates()
{
    // Collect observations and rewards from all environments
    for (int i = 0; i < mNumEnvs; i++) {
        float* obs1 = &mAllObservations[(i * 2 + 0) * mObservationDim];
        float* obs2 = &mAllObservations[(i * 2 + 1) * mObservationDim];
        float* rew1 = &mAllRewards[(i * 2 + 0)];
        float* rew2 = &mAllRewards[(i * 2 + 1)];
        bool done = false;
        
        mEnvs[i].HarvestState(obs1, obs2, rew1, rew2, done);
        mAllDones[i] = done ? 1.0f : 0.0f;
    }
}

void VectorizedOctopodEnv::Reset(int envIndex)
{
    if (envIndex >= 0 && envIndex < mNumEnvs) {
        mEnvs[envIndex].Reset();
    } else {
        // Reset all
        for (int i = 0; i < mNumEnvs; i++) {
            mEnvs[i].Reset();
        }
    }
}

void VectorizedOctopodEnv::ResetDoneEnvs()
{
    for (int i = 0; i < mNumEnvs; i++) {
        if (mAllDones[i] > 0.5f) {
            mEnvs[i].Reset();
        }
    }
}
