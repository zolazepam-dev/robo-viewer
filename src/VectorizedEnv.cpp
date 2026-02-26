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

const float kRoomSize = 12.0f;        // Single room is 12x12x12 meters
const float kWallThickness = 1.0f;    // 1 meter thick walls

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

    // Create and register the global CombatContactListener
    gCombatContactListener = &CombatContactListener::Get();
    mPhysicsCore.GetPhysicsSystem().SetContactListener(gCombatContactListener);

    // --- SINGLE 12x12x12 METER ROOM ---
    JPH::BodyInterface& body_interface = mPhysicsCore.GetPhysicsSystem().GetBodyInterface();
    
    // Floor (12x12, at bottom)
    JPH::BoxShapeSettings floor_shape(JPH::Vec3(6.0f, 0.5f, 6.0f)); // half-extents: x=6, y=0.5, z=6
    JPH::RefConst<JPH::Shape> floor = floor_shape.Create().Get();
    JPH::BodyCreationSettings floor_settings(floor, JPH::RVec3(0.0f, 0.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC);
    body_interface.CreateAndAddBody(floor_settings, JPH::EActivation::DontActivate);
    
    // Ceiling (at y=12)
    JPH::BodyCreationSettings ceil_settings(floor, JPH::RVec3(0.0f, 12.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC);
    body_interface.CreateAndAddBody(ceil_settings, JPH::EActivation::DontActivate);
    
    // Walls (12m tall, 0.5m thick)
    JPH::BoxShapeSettings wall_x_shape(JPH::Vec3(0.5f, 6.0f, 6.0f)); // vertical walls on X sides (rotate them)
    JPH::RefConst<JPH::Shape> wall_x = wall_x_shape.Create().Get();
    
    JPH::BoxShapeSettings wall_z_shape(JPH::Vec3(6.0f, 6.0f, 0.5f)); // vertical walls on Z sides
    JPH::RefConst<JPH::Shape> wall_z = wall_z_shape.Create().Get();
    
    // East wall (x = 6)
    JPH::BodyCreationSettings east_wall(wall_x, JPH::RVec3(6.0f, 6.0f, 0.0f), JPH::Quat::sRotation(JPH::Vec3::sAxisY(), JPH::DegreesToRadians(90.0f)), JPH::EMotionType::Static, Layers::STATIC);
    body_interface.CreateAndAddBody(east_wall, JPH::EActivation::DontActivate);
    
    // West wall (x = -6)
    JPH::BodyCreationSettings west_wall(wall_x, JPH::RVec3(-6.0f, 6.0f, 0.0f), JPH::Quat::sRotation(JPH::Vec3::sAxisY(), JPH::DegreesToRadians(90.0f)), JPH::EMotionType::Static, Layers::STATIC);
    body_interface.CreateAndAddBody(west_wall, JPH::EActivation::DontActivate);
    
    // North wall (z = -6)
    JPH::BodyCreationSettings north_wall(wall_z, JPH::RVec3(0.0f, 6.0f, -6.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC);
    body_interface.CreateAndAddBody(north_wall, JPH::EActivation::DontActivate);
    
    // South wall (z = 6)
    JPH::BodyCreationSettings south_wall(wall_z, JPH::RVec3(0.0f, 6.0f, 6.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC);
    body_interface.CreateAndAddBody(south_wall, JPH::EActivation::DontActivate);
    // -------------------------------------------------

    mEnvs.resize(mNumEnvs);
    std::cout << "[JOLTrl] Initializing " << mNumEnvs << " environments..." << std::endl;
    for (int i = 0; i < mNumEnvs; ++i)
    {
        std::cout << "[JOLTrl] Init env " << i << "..." << std::flush;
        mEnvs[i].Init(i, &mPhysicsCore.GetPhysicsSystem(), &mRobotLoader);
        std::cout << " done" << std::endl;
    }
    std::cout << "[JOLTrl] All environments initialized." << std::endl;

    mObservationDim = mEnvs[0].GetObservationDim();

    std::cout << "[JOLTrl] Resizing observation and reward arrays..." << std::endl;
    mAllObservations.resize(mNumEnvs * mObservationDim * 2, 0.0f);
    mAllRewards.resize(mNumEnvs * 2, 0.0f);
    mAllDones.resize(mNumEnvs, false);
    
    AssertAligned32(mAllObservations.data());
    AssertAligned32(mAllRewards.data());

    std::cout << "[JOLTrl] Calling Reset(-1)..." << std::endl;
    Reset(-1);
    std::cout << "[JOLTrl] Reset(-1) completed." << std::endl;

    // Call broadphase optimization after a small delay to ensure all bodies are added
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    mPhysicsCore.GetPhysicsSystem().OptimizeBroadPhase();
    std::cout << "[JOLTrl] Global BroadPhase optimized. Engine ready." << std::endl;
}

void VectorizedEnv::Step(const AlignedVector32<float>& actions)
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

VectorizedEnv::~VectorizedEnv()
{
    if (gCombatContactListener)
    {
        mPhysicsCore.GetPhysicsSystem().SetContactListener(nullptr);
        gCombatContactListener = nullptr;
    }
}

