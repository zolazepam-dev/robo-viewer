#pragma once

#include <Jolt/Jolt.h>
#include <vector>
#include "CombatEnv.h"
#include "PhysicsCore.h"
#include "AlignedAllocator.h"
#include "SoAEnvironment.h"

class VectorizedEnv
{
public:
    VectorizedEnv(int numEnvs, int stepsPerEpisode = 7200);
    ~VectorizedEnv();
    VectorizedEnv(const VectorizedEnv& other) = delete;
    VectorizedEnv& operator=(const VectorizedEnv& other) = delete;

    void Init(const std::string& robotConfigPath, bool initRobots = true);
    void SetDomainRandomization(const DomainRandomization& dr);
    void Shutdown();
    void Step(const AlignedVector32<float>& actions);
    void HarvestStates();
    void HarvestStatesParallel();  // OpenMP-parallelized version
    void HarvestStatesZeroCopy();  // Zero-copy version - returns pointers
    void Reset(int envIndex = -1);
    void ResetDoneEnvs();
    void QueueActionsParallel(const float* robot1Actions, const float* robot2Actions, int numEnvs);  // Lock-free parallel queuing

    const AlignedVector32<float>& GetObservations() const { return mAllObservations; }
    const AlignedVector32<float>& GetRewards() const { return mAllRewards; }
    const std::vector<VectorReward>& GetVectorRewards() const { return mAllVectorRewards; }
    const std::vector<bool>& GetDones() const { return mAllDones; }
    
    // Zero-copy access - direct pointers to env memory
    const float* GetObservationPtr(int envIdx, int robotIdx) const;
    float* GetRewardPtr(int envIdx, int robotIdx);

    CombatEnv& GetEnv(int index) { return mEnvs[index]; }
    int GetNumEnvs() const { return mNumEnvs; }
    int GetObservationDim() const { return mObservationDim; }
    int GetActionDim() const { return mActionDim; }

    JPH::PhysicsSystem* GetGlobalPhysics() { return &mPhysicsCore.GetPhysicsSystem(); }
    PhysicsCore* GetPhysicsCore() { return &mPhysicsCore; }
    
    bool GetRenderState(float* redPos, float* bluePos, float* redSatPos, float* blueSatPos, float* redHealth, float* blueHealth);

private:
    PhysicsCore mPhysicsCore;
    CombatRobotLoader mRobotLoader;
    std::vector<CombatEnv> mEnvs;

    // SoA batch for SIMD-optimized processing
    opt::SoAEnvironmentBatch mSoABatch;
    bool mSoAInitialized = false;

    int mNumEnvs;
    int mStepsPerEpisode;
    int mObservationDim = 256; 
    int mActionDim = 56;
    
    AlignedVector32<float> mAllObservations;
    AlignedVector32<float> mAllRewards;
    std::vector<bool> mAllDones;
    std::vector<VectorReward> mAllVectorRewards;
};