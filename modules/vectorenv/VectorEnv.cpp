/**
 * @file VectorEnv.cpp
 * @brief Implementation of VectorEnv class
 */

#include "VectorEnv.h"
#include "../environment/CombatEnv.h"
#include <cstring>
#include <algorithm>
#include <omp.h>

VectorEnv::VectorEnv(int numEnvs, int stepsPerEpisode)
    : mNumEnvs(numEnvs)
    , mStepsPerEpisode(stepsPerEpisode)
{
    AllocateBuffers();
}

VectorEnv::~VectorEnv() {
    Shutdown();
}

void VectorEnv::Init(PhysicsWorld* physicsWorld, bool initRobots) {
    mPhysicsWorld = physicsWorld;
    
    if (!mPhysicsWorld) {
        throw std::runtime_error("VectorEnv::Init called with null physicsWorld");
    }
    
    mEnvs.clear();
    mEnvs.reserve(mNumEnvs);
    
    for (int i = 0; i < mNumEnvs; ++i) {
        auto env = std::make_unique<CombatEnv>();
        env->Init(static_cast<uint32_t>(i), physicsWorld);
        
        if (i == 0) {
            mObservationDim = env->GetObservationDim();
            mActionDim = env->GetActionDim();
            AllocateBuffers();
        }
        
        mEnvs.push_back(std::move(env));
    }
    
    // Initial observation harvest
    UpdateBuffers();
}

void VectorEnv::Shutdown() {
    for (auto& env : mEnvs) {
        env.reset();
    }
    mEnvs.clear();
    mPhysicsWorld = nullptr;
}

void VectorEnv::Step(const float* actions) {
    if (!mPhysicsWorld || mEnvs.empty()) return;
    
    // OPTIMIZATION: Parallelize action queuing
    #pragma omp parallel for
    for (int i = 0; i < mNumEnvs; ++i) {
        if (mAllDones[i]) continue;  // Skip done environments
        
        const float* actions1 = actions + (i * 2 * mActionDim);
        const float* actions2 = actions1 + mActionDim;
        
        mEnvs[i]->QueueActions(actions1, actions2);
    }
    
    // Step physics once for all environments (Dimensional Ghosting)
    constexpr float DT = 1.0f / 120.0f;
    mPhysicsWorld->Step(DT);
    
    // Update output buffers
    UpdateBuffers();
}

void VectorEnv::Reset(int envIndex) {
    if (envIndex < 0) {
        // Reset all environments
        for (auto& env : mEnvs) {
            env->Reset();
        }
        std::fill(mAllDones.begin(), mAllDones.end(), false);
    } else if (envIndex < mNumEnvs) {
        // Reset specific environment
        mEnvs[envIndex]->Reset();
        mAllDones[envIndex] = false;
    }
    
    UpdateBuffers();
}

void VectorEnv::ResetDoneEnvs() {
    for (int i = 0; i < mNumEnvs; ++i) {
        if (mAllDones[i]) {
            mEnvs[i]->Reset();
            mAllDones[i] = false;
        }
    }
    UpdateBuffers();
}

IEnvironment* VectorEnv::GetEnv(int index) {
    if (index < 0 || index >= mNumEnvs) return nullptr;
    return mEnvs[index].get();
}

const IEnvironment* VectorEnv::GetEnv(int index) const {
    if (index < 0 || index >= mNumEnvs) return nullptr;
    return mEnvs[index].get();
}

bool VectorEnv::AnyDone() const {
    return std::any_of(mAllDones.begin(), mAllDones.end(), [](bool d) { return d; });
}

int VectorEnv::CountDone() const {
    return static_cast<int>(std::count(mAllDones.begin(), mAllDones.end(), true));
}

void VectorEnv::AllocateBuffers() {
    if (mNumEnvs <= 0 || mObservationDim <= 0) return;
    
    // 32-byte alignment for AVX2
    size_t obsSize = static_cast<size_t>(mNumEnvs) * 2 * mObservationDim;
    size_t rewardSize = static_cast<size_t>(mNumEnvs) * 2;
    size_t vecRewardSize = static_cast<size_t>(mNumEnvs) * 2 * VECTOR_REWARD_DIM;
    size_t doneSize = static_cast<size_t>(mNumEnvs);
    
    mAllObservations.resize(obsSize, 0.0f);
    mAllRewards.resize(rewardSize, 0.0f);
    mAllVectorRewards.resize(vecRewardSize, 0.0f);
    mAllDones.resize(doneSize, 0);
    
    // Verify alignment
}

void VectorEnv::UpdateBuffers() {
    // OPTIMIZATION: Parallelize buffer updates
    #pragma omp parallel for
    for (int i = 0; i < mNumEnvs; ++i) {
        auto* env = mEnvs[i].get();
        if (!env) continue;
        
        int obsOffset = i * 2 * mObservationDim;
        int rewardOffset = i * 2;
        int vecRewardOffset = i * 2 * VECTOR_REWARD_DIM;
        
        float* obs1 = mAllObservations.data() + obsOffset;
        float* obs2 = obs1 + mObservationDim;
        
        env->GetObs(obs1, obs2);
        
        mAllRewards[rewardOffset + 0] = env->GetReward(0);
        mAllRewards[rewardOffset + 1] = env->GetReward(1);
        
        const VectorReward& vr1 = env->GetVectorReward(0);
        const VectorReward& vr2 = env->GetVectorReward(1);
        
        for (int j = 0; j < VECTOR_REWARD_DIM; ++j) {
            mAllVectorRewards[vecRewardOffset + j] = vr1.components[j];
            mAllVectorRewards[vecRewardOffset + VECTOR_REWARD_DIM + j] = vr2.components[j];
        }
        
        mAllDones[i] = env->IsDone() ? 1 : 0;
    }
}
