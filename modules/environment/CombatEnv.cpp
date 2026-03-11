/**
 * @file CombatEnv.cpp
 * @brief Implementation of CombatEnv class
 */

#include "CombatEnv.h"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <random>
#include <random>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>

void CombatEnv::Init(uint32_t envIndex, void* physicsWorld) {
    mEnvIndex = envIndex;
    mPhysicsWorld = static_cast<PhysicsWorld*>(physicsWorld);
    mStepCount = 0;
    mDone = false;
    
    if (mPhysicsWorld && mRobot1.GetConfig().numSatellites > 0) {
        mObservationDim = mRobot1.GetConfig().observationDim;
    }
    
    Reset();
}

void CombatEnv::Reset() {
    if (!mPhysicsWorld) return;
    
    mStepCount = 0;
    mDone = false;
    mPrevHP1 = INITIAL_HP;
    mPrevHP2 = INITIAL_HP;
    mPrevEnergy1 = 0.0f;
    mPrevEnergy2 = 0.0f;
    mRobot1.SetHP(INITIAL_HP);
    mRobot2.SetHP(INITIAL_HP);
    mRobot1.ResetEpisodeStats();
    mRobot2.ResetEpisodeStats();
    mReward1.Reset();
    mReward2.Reset();
    
    // Randomize KOTH point
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distXZ(-15.0f, 15.0f);
    std::uniform_real_distribution<float> distY(2.0f, 12.0f);
    mKothPoint = JPH::RVec3(distXZ(rng), distY(rng), distXZ(rng));
    
    // Update or create KOTH visual
    auto& system = mPhysicsWorld->GetSystem();
    JPH::BodyInterface& bodyInterface = system.GetBodyInterface();
    
    if (!mKothVisualId.IsInvalid()) {
        bodyInterface.SetPositionAndRotation(mKothVisualId, mKothPoint, 
                                              JPH::Quat::sIdentity(), JPH::EActivation::DontActivate);
    } else {
        JPH::SphereShapeSettings kothShape(0.8f);
        JPH::BodyCreationSettings kothSettings(
            kothShape.Create().Get(), mKothPoint, JPH::Quat::sIdentity(),
            JPH::EMotionType::Static, PhysicsLayers::GetGhostLayer(mEnvIndex)
        );
        mKothVisualId = bodyInterface.CreateAndAddBody(kothSettings, JPH::EActivation::DontActivate);
    }
    
    // Spawn positions
    JPH::RVec3 pos1(-10.0f, 5.0f, 0.0f);
    JPH::RVec3 pos2(10.0f, 5.0f, 0.0f);
    
    // Load or reset robots
    if (mRobot1.GetMainBodyId().IsInvalid()) {
        RobotConfig config = RobotConfig::LoadFromJSON("robots/combat_bot.json");
        mRobot1 = mRobotFactory.CreateRobot(config, *mPhysicsWorld, pos1, mEnvIndex, 0);
        mRobot1.SetType(RobotType::SATELLITE);
        
        mRobot2 = mRobotFactory.CreateRobot(config, *mPhysicsWorld, pos2, mEnvIndex, 1);
        mRobot2.SetType(RobotType::SATELLITE);
    } else {
        mRobotFactory.ResetRobot(mRobot1, *mPhysicsWorld, pos1);
        mRobotFactory.ResetRobot(mRobot2, *mPhysicsWorld, pos2);
    }
    
    // Reset force sensors
    int numSatellites = mRobot1.GetConfig().numSatellites;
    mPhysicsWorld->GetContactListener().ResetForceReadings(mEnvIndex, numSatellites);
}

void CombatEnv::Step(const float* action1, const float* action2) {
    if (mDone || !mPhysicsWorld) return;
    
    constexpr float DT = 1.0f / 120.0f;
    
    // Use the optimized queuing interface
    QueueActions(action1, action2);
    
    // Step physics
    mPhysicsWorld->Step(DT);
    
    // Update stats
    mRobot1.IncrementEpisodeSteps();
    mRobot2.IncrementEpisodeSteps();
    mStepCount++;
    
    // Calculate rewards
    CalculateRewards();
    
    // Check done conditions
    if (mRobot1.GetHP() <= 0.0f || mRobot2.GetHP() <= 0.0f || mStepCount >= mStepsPerEpisode) {
        mDone = true;
    }
}

void CombatEnv::QueueActions(const float* action1, const float* action2) {
    if (mDone || !mPhysicsWorld) return;
    
    constexpr float DT = 1.0f / 120.0f;
    
    // Apply actions (Queuing for next physics step)
    mRobotController.ApplyResidualActions(mRobot1, action1, *mPhysicsWorld, DT);
    mRobotController.ApplyResidualActions(mRobot2, action2, *mPhysicsWorld, DT);
}

void CombatEnv::GetObs(float* obs1, float* obs2) {
    if (!mPhysicsWorld) return;
    
    const auto& forces1 = mPhysicsWorld->GetContactListener().GetForceReading(mEnvIndex, 0);
    const auto& forces2 = mPhysicsWorld->GetContactListener().GetForceReading(mEnvIndex, 1);
    
    mRobotController.GetObservations(mRobot1, mRobot2, forces1, *mPhysicsWorld, obs1);
    mRobotController.GetObservations(mRobot2, mRobot1, forces2, *mPhysicsWorld, obs2);
}

float CombatEnv::GetReward(int robotIdx) const {
    if (robotIdx == 0) {
        return mReward1.Dot(mPreferenceVector);
    } else {
        return mReward2.Dot(mPreferenceVector);
    }
}

const VectorReward& CombatEnv::GetVectorReward(int robotIdx) const {
    return (robotIdx == 0) ? mReward1 : mReward2;
}

bool CombatEnv::IsDone() const {
    return mDone;
}

void CombatEnv::CalculateRewards() {
    // Damage dealt/taken
    float hpDelta1 = mPrevHP1 - mRobot1.GetHP();
    float hpDelta2 = mPrevHP2 - mRobot2.GetHP();
    
    mReward1.SetDamageDealt(hpDelta2 * DAMAGE_MULTIPLIER);
    mReward1.SetDamageTaken(-hpDelta1 * DAMAGE_MULTIPLIER);
    mReward2.SetDamageDealt(hpDelta1 * DAMAGE_MULTIPLIER);
    mReward2.SetDamageTaken(-hpDelta2 * DAMAGE_MULTIPLIER);
    
    // Survival bonus
    mReward1.SetSurvival(mRobot1.GetHP() > 0.0f ? 1.0f : 0.0f);
    mReward2.SetSurvival(mRobot2.GetHP() > 0.0f ? 1.0f : 0.0f);
    
    // Airtime (simplified)
    float airtime = ComputeAirtime();
    mReward1.SetAirtime(airtime > 0.0f ? airtime : 0.0f);
    mReward2.SetAirtime(airtime > 0.0f ? airtime : 0.0f);
    
    mPrevHP1 = mRobot1.GetHP();
    mPrevHP2 = mRobot2.GetHP();
}

float CombatEnv::ComputeAirtime() const {
    // Simplified airtime calculation
    return 0.0f;
}

float CombatEnv::ComputeEnergyUsed(const float* actions, int actionDim) const {
    float energy = 0.0f;
    for (int i = 0; i < actionDim; ++i) {
        energy += actions[i] * actions[i];
    }
    return energy;
}

void CombatEnv::UpdateForceSensors() {
    // Force sensors updated via ContactListener
}

void CombatEnv::BuildObservationVector(float* obs, const Robot& robot,
                                        const Robot& opponent, const ForceSensorReading& forces) {
    mRobotController.GetObservations(robot, opponent, forces, *mPhysicsWorld, obs);
}
