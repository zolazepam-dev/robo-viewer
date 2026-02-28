#pragma once

#include <Jolt/Jolt.h>
#include <vector>
#include "CombatEnv.h"
#include "PhysicsCore.h"
#include "AlignedAllocator.h"

class VectorizedEnv
{
public:
    VectorizedEnv(int numEnvs);
    ~VectorizedEnv();
    VectorizedEnv(const VectorizedEnv& other) = default;
    VectorizedEnv& operator=(const VectorizedEnv& other) = default;

    void Init(bool initRobots = true);
    void Shutdown();
    void Step(const AlignedVector32<float>& actions);
    void Reset(int envIndex = -1);
    void ResetDoneEnvs();

    const AlignedVector32<float>& GetObservations() const { return mAllObservations; }
    const AlignedVector32<float>& GetRewards() const { return mAllRewards; }
    const std::vector<VectorReward>& GetVectorRewards() const { return mAllVectorRewards; }
    const std::vector<bool>& GetDones() const { return mAllDones; }

    CombatEnv& GetEnv(int index) { return mEnvs[index]; }
    int GetNumEnvs() const { return mNumEnvs; }
    int GetObservationDim() const { return mObservationDim; }
    int GetActionDim() const { return mActionDim; }

    JPH::PhysicsSystem* GetGlobalPhysics() { return &mPhysicsCore.GetPhysicsSystem(); }
    PhysicsCore* GetPhysicsCore() { return &mPhysicsCore; }

private:
    PhysicsCore mPhysicsCore;
    CombatRobotLoader mRobotLoader;
    std::vector<CombatEnv> mEnvs;

    int mNumEnvs;
    int mObservationDim = OBSERVATION_DIM; 
    int mActionDim = 56;
    
    AlignedVector32<float> mAllObservations;
    AlignedVector32<float> mAllRewards;
    std::vector<bool> mAllDones;
    std::vector<VectorReward> mAllVectorRewards;
};