// MUST BE FIRST
#include <Jolt/Jolt.h>
#include "VectorizedEnv.h"

#include <algorithm>
#include <thread>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>

#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/PhysicsSystem.h>

// Global single instance of CombatContactListener
CombatContactListener* gCombatContactListener = nullptr;

VectorizedEnv::VectorizedEnv(int numEnvs, int stepsPerEpisode)
    : mNumEnvs(numEnvs), mStepsPerEpisode(stepsPerEpisode)
{
}

void VectorizedEnv::Init(const std::string& robotConfigPath, bool initRobots)
{
    std::cout << "[VectorizedEnv] Initializing..." << "\n";
    
    // OPTIMIZATION: Removed verbose logging to reduce stutter

    if (!mPhysicsCore.Init(mNumEnvs))
    {
        std::cerr << "[VectorizedEnv] PhysicsCore init failed" << "\n";
        return;
    }
    std::cout << "[VectorizedEnv] PhysicsCore initialized" << "\n";

    // Create and register the global CombatContactListener
    gCombatContactListener = &CombatContactListener::Get();
    std::cout << "[VectorizedEnv] Contact listener created" << "\n";
    
    mPhysicsCore.GetPhysicsSystem().SetContactListener(gCombatContactListener);
    std::cout << "[VectorizedEnv] Contact listener registered" << "\n";

    // --- BUILD THE SINGLE SOURCE OF TRUTH ARENA (60x60x60) ---
    JPH::BodyInterface& body_interface = mPhysicsCore.GetPhysicsSystem().GetBodyInterface();
    std::cout << "[VectorizedEnv] Creating arena..." << "\n";

    // Floor: 60x60 meters, 2.0m thick (increased from 36x36)
    JPH::BoxShapeSettings floor_shape(JPH::Vec3(30.0f, 1.0f, 30.0f));
    JPH::RefConst<JPH::Shape> floor = floor_shape.Create().Get();
    std::cout << "[VectorizedEnv] Floor shape created" << "\n";
    
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(floor, JPH::RVec3(0.0f, 1.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    std::cout << "[VectorizedEnv] Floor added" << "\n";

    // Ceiling
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(floor, JPH::RVec3(0.0f, 60.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    std::cout << "[VectorizedEnv] Ceiling added" << "\n";

    // Walls: 5.0m thick (increased from 2.0m for better collision prevention)
    JPH::BoxShapeSettings wall_shape(JPH::Vec3(30.0f, 30.0f, 5.0f));
    JPH::RefConst<JPH::Shape> wall = wall_shape.Create().Get();
    std::cout << "[VectorizedEnv] Wall shape created" << "\n";

    // North/South (z = +/- 30.0 + offset)
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(0.0f, 30.0f, -35.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    std::cout << "[VectorizedEnv] North wall added" << "\n";
    
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(0.0f, 30.0f, 35.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    std::cout << "[VectorizedEnv] South wall added" << "\n";

    // East/West (x = +/- 30.0 + offset, rotated)
    JPH::Quat rot90 = JPH::Quat::sRotation(JPH::Vec3::sAxisY(), JPH::DegreesToRadians(90.0f));
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(35.0f, 30.0f, 0.0f), rot90, JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    std::cout << "[VectorizedEnv] East wall added" << "\n";
    
    body_interface.CreateAndAddBody(JPH::BodyCreationSettings(wall, JPH::RVec3(-35.0f, 30.0f, 0.0f), rot90, JPH::EMotionType::Static, Layers::STATIC), JPH::EActivation::DontActivate);
    std::cout << "[VectorizedEnv] West wall added" << "\n";
    std::cout << "[VectorizedEnv] Arena complete" << "\n";
    // -------------------------------------------------

    if (initRobots) {
        // Ensure contact listener has enough capacity for all environments
        CombatContactListener::Get().EnsureCapacity(mNumEnvs);

        std::cout << "[VectorizedEnv] Initializing " << mNumEnvs << " robot environments..." << "\n";
        mEnvs.resize(mNumEnvs);
        for (int i = 0; i < mNumEnvs; ++i)
        {
            if (i % 10 == 0) std::cout << "[VectorizedEnv] Init env " << i << "/" << mNumEnvs << "\n";
            try {
                mEnvs[i].Init(i, &mPhysicsCore.GetPhysicsSystem(), &mRobotLoader, robotConfigPath, mStepsPerEpisode);
            } catch (const std::exception& e) {
                std::cerr << "[VectorizedEnv] Exception at env " << i << ": " << e.what() << "\n";
                return;
            }
        }
        std::cout << "[VectorizedEnv] All robots initialized" << "\n";

        mObservationDim = mEnvs[0].GetObservationDim();
        mActionDim = mEnvs[0].GetRobot1Ref().config.actionsPerRobot;
        
        // Safety fallback: ensure actionDim is never 0
        if (mActionDim <= 0) {
            mActionDim = 56; // Default fallback for combat robots
            std::cerr << "[VectorizedEnv] WARNING: actionsPerRobot was 0, using fallback: " << mActionDim << "\n";
        }
        
        std::cout << "[VectorizedEnv] Obs dim: " << mObservationDim << ", Action dim: " << mActionDim << "\n";
        
        mAllObservations.resize(mNumEnvs * mObservationDim * 2, 0.0f);
        mAllRewards.resize(mNumEnvs * 2, 0.0f);
        mAllDones.resize(mNumEnvs, false);
        mAllVectorRewards.resize(mNumEnvs);
        std::cout << "[VectorizedEnv] Buffers allocated" << "\n";
        
        // Initialize SoA batch for SIMD-optimized processing
        std::cout << "[VectorizedEnv] Initializing SoA batch..." << "\n";
        mSoAInitialized = mSoABatch.Initialize(mNumEnvs, mObservationDim, mActionDim, 13);
        if (mSoAInitialized) {
            std::cout << "[VectorizedEnv] SoA batch initialized: " << mNumEnvs << " envs, " 
                      << mObservationDim << " obs dim, " << mActionDim << " action dim" << "\n";
        } else {
            std::cerr << "[VectorizedEnv] WARNING: SoA batch initialization failed" << "\n";
        }
    }

    mPhysicsCore.GetPhysicsSystem().OptimizeBroadPhase();
    std::cout << "[VectorizedEnv] Initialization complete!" << "\n";
}

void VectorizedEnv::Step(const AlignedVector32<float>& actions)
{
    const int actionDim = mActionDim;
    const int numEnvs = mNumEnvs;
    
    // 1. Parallel Action Queuing
    #pragma omp parallel for num_threads(8) schedule(static)
    for (int i = 0; i < numEnvs; ++i)
    {
        if (mAllDones[i]) continue;
        mEnvs[i].QueueActions(
            actions.data() + (i * 2 * actionDim),
            actions.data() + (i * 2 * actionDim + actionDim)
        );
    }

    // 2. Physics Step
    mPhysicsCore.Step(1.0f / 60.0f);

    // 3. Parallel State Harvesting (Fused & Zero-copy)
    #pragma omp parallel for num_threads(8) schedule(static)
    for (int i = 0; i < numEnvs; ++i)
    {
        if (mAllDones[i]) continue;

        float* obs = mAllObservations.data() + (i * mObservationDim * 2);
        float* rew = mAllRewards.data() + (i * 2);
        bool done = false;

        mEnvs[i].HarvestStateZeroCopy(obs, obs + mObservationDim, rew, rew + 1, 
                                       &done, &mAllVectorRewards[i]);
        mAllDones[i] = done;
    }
}

void VectorizedEnv::HarvestStatesParallel()
{
    const int numEnvs = mNumEnvs;
    const int obsDim = mObservationDim;
    
    // OPTIMIZED: Parallel harvesting with OpenMP
    #pragma omp parallel for num_threads(8) schedule(static)
    for (int i = 0; i < numEnvs; ++i)
    {
        if (mAllDones[i]) continue;

        int obsOffset = i * obsDim * 2;
        float* obs1 = mAllObservations.data() + obsOffset;
        float* obs2 = mAllObservations.data() + obsOffset + obsDim;
        float* reward1 = mAllRewards.data() + (i * 2);
        float* reward2 = mAllRewards.data() + (i * 2 + 1);
        bool done = false;

        mEnvs[i].HarvestState(obs1, obs2, reward1, reward2, done);
        mAllDones[i] = done;
    }

    // Parallel vector reward harvesting
    #pragma omp parallel for num_threads(8) schedule(static)
    for (int i = 0; i < numEnvs; ++i) {
        if (mAllDones[i]) continue;
        mAllVectorRewards[i] = mEnvs[i].GetRobot1Reward();
    }
}

void VectorizedEnv::HarvestStates()
{
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

    for (int i = 0; i < mNumEnvs; ++i) {
        if (mAllDones[i]) continue;
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
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < mNumEnvs; ++i)
    {
        if (mAllDones[i])
        {
            mEnvs[i].Reset();
            mAllDones[i] = false;
        }
    }
}

void VectorizedEnv::SetDomainRandomization(const DomainRandomization& dr)
{
    for (int i = 0; i < mNumEnvs; ++i)
    {
        mEnvs[i].SetDomainRandomization(dr);
    }
}

VectorizedEnv::~VectorizedEnv()
{
    Shutdown();
}

void VectorizedEnv::Shutdown()
{
    // OPTIMIZATION: Removed verbose logging
    if (mPhysicsCore.IsInitialized()) {
        try {
            mPhysicsCore.GetPhysicsSystem().SetContactListener(nullptr);
        } catch (...) {}
    }

    mEnvs.clear();
    mPhysicsCore.Shutdown();
}

bool VectorizedEnv::GetRenderState(float* redPos, float* bluePos, float* redSatPos, float* blueSatPos, float* redHealth, float* blueHealth)
{
    if (mEnvs.empty()) return false;
    
    const auto& robot1 = mEnvs[0].GetRobot1();
    const auto& robot2 = mEnvs[0].GetRobot2();
    
    JPH::BodyInterface& bodyInterface = mPhysicsCore.GetPhysicsSystem().GetBodyInterface();
    
    if (robot1.mainBodyId.IsInvalid() || robot2.mainBodyId.IsInvalid()) return false;
    
    JPH::RVec3 p1 = bodyInterface.GetPosition(robot1.mainBodyId);
    JPH::RVec3 p2 = bodyInterface.GetPosition(robot2.mainBodyId);
    
    redPos[0] = p1.GetX(); redPos[1] = p1.GetY(); redPos[2] = p1.GetZ();
    bluePos[0] = p2.GetX(); bluePos[1] = p2.GetY(); bluePos[2] = p2.GetZ();
    
    if (!robot1.satellites.empty() && !robot1.satellites[0].coreBodyId.IsInvalid()) {
        JPH::RVec3 sp1 = bodyInterface.GetPosition(robot1.satellites[0].coreBodyId);
        redSatPos[0] = sp1.GetX(); redSatPos[1] = sp1.GetY(); redSatPos[2] = sp1.GetZ();
    }
    
    if (!robot2.satellites.empty() && !robot2.satellites[0].coreBodyId.IsInvalid()) {
        JPH::RVec3 sp2 = bodyInterface.GetPosition(robot2.satellites[0].coreBodyId);
        blueSatPos[0] = sp2.GetX(); blueSatPos[1] = sp2.GetY(); blueSatPos[2] = sp2.GetZ();
    }
    
    *redHealth = robot1.hp;
    *blueHealth = robot2.hp;
    
    return true;
}

// Zero-copy observation access - direct pointer to env memory
const float* VectorizedEnv::GetObservationPtr(int envIdx, int robotIdx) const {
    if (envIdx < 0 || envIdx >= mNumEnvs) return nullptr;
    return mEnvs[envIdx].GetObservationPtr(robotIdx);
}

float* VectorizedEnv::GetRewardPtr(int envIdx, int robotIdx) {
    if (envIdx < 0 || envIdx >= mNumEnvs) return nullptr;
    return const_cast<float*>(mEnvs[envIdx].GetRewardPtr(robotIdx));
}

// Lock-free parallel action queuing - all envs queue simultaneously
void VectorizedEnv::QueueActionsParallel(const float* robot1Actions, const float* robot2Actions, int numEnvs) {
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < numEnvs; ++i) {
        mEnvs[i].QueueActions(
            robot1Actions + i * mActionDim,
            robot2Actions + i * mActionDim
        );
    }
}

// Zero-copy state harvesting - envs write directly to pre-allocated buffers
void VectorizedEnv::HarvestStatesZeroCopy() {
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < mNumEnvs; ++i) {
        float* obs = reinterpret_cast<float*>(mAllObservations.data()) + i * 2 * mObservationDim;
        float* rew = reinterpret_cast<float*>(mAllRewards.data()) + i * 2;
        
        bool done = false;
        mEnvs[i].HarvestStateZeroCopy(obs, obs + mObservationDim, rew, rew + 1, 
                                       &done, &mAllVectorRewards[i]);
        mAllDones[i] = done;
    }
}
