#pragma once

// MUST BE FIRST
#include <Jolt/Jolt.h>
#include <vector>
#include "CombatEnv.h"
#include "PhysicsCore.h"

class VectorizedEnv
{
public:
    VectorizedEnv(int numEnvs);
    ~VectorizedEnv() = default;

    void Init();

    void Step(const std::vector<float>& actions);
    void Reset(int envIndex = -1);
    void ResetDoneEnvs();

    const std::vector<float>& GetObservations() const { return mAllObservations; }
    const std::vector<float>& GetRewards() const { return mAllRewards; }
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

    std::vector<float> mAllObservations;
    std::vector<float> mAllRewards;
    std::vector<bool> mAllDones;
};