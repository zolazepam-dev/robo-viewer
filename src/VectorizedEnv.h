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
    ~VectorizedEnv() = default;

    void Init();

    void Step(const std::vector<float>& actions);
    void Reset(int envIndex = -1);
    void ResetDoneEnvs();

    const AlignedVector32<float>& GetObservations() const { return mAllObservations; }
    const AlignedVector32<float>& GetRewards() const { return mAllRewards; }
    const std::vector<bool>& GetDones() const { return mAllDones; }

    CombatEnv& GetEnv(int index) { return mEnvs[index]; }
    int GetNumEnvs() const { return mNumEnvs; }
    int GetObservationDim() const { return mObservationDim; }
    int GetActionDim() const { return ACTIONS_PER_ROBOT; }

    JPH::PhysicsSystem* GetGlobalPhysics() { return &mPhysicsCore.GetPhysicsSystem(); }

private:
    PhysicsCore mPhysicsCore;
    CombatRobotLoader mRobotLoader;
    std::vector<CombatEnv> mEnvs;

    int mNumEnvs;
    int mObservationDim;

    AlignedVector32<float> mAllObservations;
    AlignedVector32<float> mAllRewards;
    std::vector<bool> mAllDones;
};