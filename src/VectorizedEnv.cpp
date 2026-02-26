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

const float kRoomSize = 12.0f;        // Each room is 12x12x12 meters
const float kWallThickness = 1.0f;    // 1 meter thick walls
const float kRoomSpacing = kRoomSize + kWallThickness;  // 13m total spacing between rooms
const int kNumRoomsPerDimension = 3;  // 3x3x3 grid
const int kTotalRooms = kNumRoomsPerDimension * kNumRoomsPerDimension * kNumRoomsPerDimension;

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

    // --- Create 3x3x3 room matrix with walls, floor, and ceiling ---
    JPH::BodyInterface& body_interface = mPhysicsCore.GetPhysicsSystem().GetBodyInterface();

    // Create shapes for room components (walls, floor, ceiling)
    JPH::BoxShapeSettings floor_shape_settings(JPH::Vec3(kRoomSize / 2, kWallThickness / 2, kRoomSize / 2));
    JPH::RefConst<JPH::Shape> floor_shape = floor_shape_settings.Create().Get();

    JPH::BoxShapeSettings ceiling_shape_settings(JPH::Vec3(kRoomSize / 2, kWallThickness / 2, kRoomSize / 2));
    JPH::RefConst<JPH::Shape> ceiling_shape = ceiling_shape_settings.Create().Get();

    JPH::BoxShapeSettings wall_x_shape_settings(JPH::Vec3(kWallThickness / 2, kRoomSize / 2, kRoomSize / 2));
    JPH::RefConst<JPH::Shape> wall_x_shape = wall_x_shape_settings.Create().Get();

    JPH::BoxShapeSettings wall_z_shape_settings(JPH::Vec3(kRoomSize / 2, kRoomSize / 2, kWallThickness / 2));
    JPH::RefConst<JPH::Shape> wall_z_shape = wall_z_shape_settings.Create().Get();

    JPH::BoxShapeSettings wall_y_shape_settings(JPH::Vec3(kRoomSize / 2, kRoomSize / 2, kWallThickness / 2));
    JPH::RefConst<JPH::Shape> wall_y_shape = wall_y_shape_settings.Create().Get();

    // Create all rooms in 3x3x3 grid
    for (int x = 0; x < kNumRoomsPerDimension; ++x)
    {
        for (int y = 0; y < kNumRoomsPerDimension; ++y)
        {
            for (int z = 0; z < kNumRoomsPerDimension; ++z)
            {
                // Calculate room origin position
                JPH::RVec3 room_origin(
                    x * kRoomSpacing,
                    y * kRoomSpacing,
                    z * kRoomSpacing
                );

                // --- Floor ---
                JPH::RVec3 floor_pos = room_origin + JPH::RVec3(0, kWallThickness / 2, 0);
                JPH::BodyCreationSettings floor_settings(
                    floor_shape,
                    floor_pos,
                    JPH::Quat::sIdentity(),
                    JPH::EMotionType::Static,
                    Layers::STATIC
                );
                body_interface.CreateAndAddBody(floor_settings, JPH::EActivation::DontActivate);

                // --- Ceiling ---
                JPH::RVec3 ceiling_pos = room_origin + JPH::RVec3(0, kRoomSize + kWallThickness / 2, 0);
                JPH::BodyCreationSettings ceiling_settings(
                    ceiling_shape,
                    ceiling_pos,
                    JPH::Quat::sIdentity(),
                    JPH::EMotionType::Static,
                    Layers::STATIC
                );
                body_interface.CreateAndAddBody(ceiling_settings, JPH::EActivation::DontActivate);

                // --- Walls ---
                // Positive X wall
                JPH::RVec3 wall_pos_x_pos = room_origin + JPH::RVec3(kRoomSize + kWallThickness / 2, kRoomSize / 2, 0);
                JPH::BodyCreationSettings wall_x_pos_settings(
                    wall_x_shape,
                    wall_pos_x_pos,
                    JPH::Quat::sIdentity(),
                    JPH::EMotionType::Static,
                    Layers::STATIC
                );
                body_interface.CreateAndAddBody(wall_x_pos_settings, JPH::EActivation::DontActivate);

                // Negative X wall
                JPH::RVec3 wall_pos_x_neg = room_origin + JPH::RVec3(-kWallThickness / 2, kRoomSize / 2, 0);
                JPH::BodyCreationSettings wall_x_neg_settings(
                    wall_x_shape,
                    wall_pos_x_neg,
                    JPH::Quat::sIdentity(),
                    JPH::EMotionType::Static,
                    Layers::STATIC
                );
                body_interface.CreateAndAddBody(wall_x_neg_settings, JPH::EActivation::DontActivate);

                // Positive Z wall
                JPH::RVec3 wall_pos_z_pos = room_origin + JPH::RVec3(0, kRoomSize / 2, kRoomSize + kWallThickness / 2);
                JPH::BodyCreationSettings wall_z_pos_settings(
                    wall_z_shape,
                    wall_pos_z_pos,
                    JPH::Quat::sIdentity(),
                    JPH::EMotionType::Static,
                    Layers::STATIC
                );
                body_interface.CreateAndAddBody(wall_z_pos_settings, JPH::EActivation::DontActivate);

                // Negative Z wall
                JPH::RVec3 wall_pos_z_neg = room_origin + JPH::RVec3(0, kRoomSize / 2, -kWallThickness / 2);
                JPH::BodyCreationSettings wall_z_neg_settings(
                    wall_z_shape,
                    wall_pos_z_neg,
                    JPH::Quat::sIdentity(),
                    JPH::EMotionType::Static,
                    Layers::STATIC
                );
                body_interface.CreateAndAddBody(wall_z_neg_settings, JPH::EActivation::DontActivate);
            }
        }
    }
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

