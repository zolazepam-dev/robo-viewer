/**
 * @file VectorizedOctopodEnv.h
 * @brief Vectorized environment for parallel octopod training
 */

#pragma once

#include <Jolt/Jolt.h>
#include <vector>
#include "OctopodEnv.h"
#include "OctopodLoader.h"
#include "PhysicsCore.h"
#include "AlignedAllocator.h"

class VectorizedOctopodEnv
{
public:
    VectorizedOctopodEnv(int numEnvs, int stepsPerEpisode = 7200);
    ~VectorizedOctopodEnv();
    VectorizedOctopodEnv(const VectorizedOctopodEnv& other) = delete;
    VectorizedOctopodEnv& operator=(const VectorizedOctopodEnv& other) = delete;

    void Init(bool initRobots = true);
    void Shutdown();
    void Step(const AlignedVector32<float>& actions);
    void HarvestStates();
    void Reset(int envIndex = -1);
    void ResetDoneEnvs();

    const AlignedVector32<float>& GetObservations() const { return mAllObservations; }
    const AlignedVector32<float>& GetRewards() const { return mAllRewards; }
    const std::vector<float>& GetDones() const { return mAllDones; }

    OctopodEnv& GetEnv(int index) { return mEnvs[index]; }
    int GetNumEnvs() const { return mNumEnvs; }
    int GetObservationDim() const { return mObservationDim; }
    int GetActionDim() const { return mActionDim; }

    JPH::PhysicsSystem* GetGlobalPhysics() { return &mPhysicsCore.GetPhysicsSystem(); }
    PhysicsCore* GetPhysicsCore() { return &mPhysicsCore; }

private:
    PhysicsCore mPhysicsCore;
    OctopodLoader::LoadedOctopod mRobotLoader;
    std::vector<OctopodEnv> mEnvs;

    int mNumEnvs;
    int mStepsPerEpisode;
    int mObservationDim = OCTOPOD_OBS_DIM;
    int mActionDim = OCTOPOD_ACTION_DIM;
    
    AlignedVector32<float> mAllObservations;
    AlignedVector32<float> mAllRewards;
    std::vector<float> mAllDones;
};
