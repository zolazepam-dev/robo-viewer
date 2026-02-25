#include <Jolt/Jolt.h>
#include "CombatEnv.h"

#include <cmath>
#include <iostream>
#include <Jolt/Physics/Body/BodyInterface.h>

void CombatEnv::Init(uint32_t envIndex, JPH::PhysicsSystem* globalPhysics, CombatRobotLoader* globalLoader)
{
    mEnvIndex = envIndex;
    mPhysicsSystem = globalPhysics;
    mRobotLoader = globalLoader;

    JPH::RVec3 pos1(-ROBOT_SPAWN_OFFSET, 1.0f, 0.0f);
    JPH::RVec3 pos2(ROBOT_SPAWN_OFFSET, 1.0f, 0.0f);

    mRobot1 = mRobotLoader->LoadRobot("robots/combat_bot.json", mPhysicsSystem, pos1, mEnvIndex, 0);
    mRobot2 = mRobotLoader->LoadRobot("robots/combat_bot.json", mPhysicsSystem, pos2, mEnvIndex, 1);

    mStepCount = 0;
    mDone = false;
    mPrevHp1 = INITIAL_HP;
    mPrevHp2 = INITIAL_HP;
}

void CombatEnv::Reset()
{
    mStepCount = 0;
    mDone = false;
    mPrevHp1 = INITIAL_HP;
    mPrevHp2 = INITIAL_HP;

    JPH::RVec3 pos1(-ROBOT_SPAWN_OFFSET, 1.0f, 0.0f);
    JPH::RVec3 pos2(ROBOT_SPAWN_OFFSET, 1.0f, 0.0f);

    mRobotLoader->ResetRobot(mRobot1, mPhysicsSystem, pos1);
    mRobotLoader->ResetRobot(mRobot2, mPhysicsSystem, pos2);
}

void CombatEnv::QueueActions(const float* actions1, const float* actions2)
{
    if (mDone) return;
    mRobotLoader->ApplyActions(mRobot1, actions1, mPhysicsSystem);
    mRobotLoader->ApplyActions(mRobot2, actions2, mPhysicsSystem);
}

StepResult CombatEnv::HarvestState()
{
    StepResult result;
    result.obs_robot1.resize(mObservationDim);
    result.obs_robot2.resize(mObservationDim);

    if (mDone) return result;

    mStepCount++;
    CheckCollisions();

    mRobotLoader->GetObservations(mRobot1, mRobot2, result.obs_robot1.data(), mPhysicsSystem);
    mRobotLoader->GetObservations(mRobot2, mRobot1, result.obs_robot2.data(), mPhysicsSystem);

    CalculateRewards(result);

    if (mRobot1.hp <= 0.0f || mRobot2.hp <= 0.0f)
    {
        mDone = true;
        result.done = true;

        if (mRobot1.hp <= 0.0f && mRobot2.hp <= 0.0f)
        {
            result.winner = 0;
        }
        else if (mRobot2.hp <= 0.0f)
        {
            result.winner = 1;
            result.reward1 += 10.0f;
            result.reward2 -= 10.0f;
        }
        else
        {
            result.winner = 2;
            result.reward2 += 10.0f;
            result.reward1 -= 10.0f;
        }
    }

    if (mStepCount >= MAX_EPISODE_STEPS)
    {
        mDone = true;
        result.done = true;
        result.reward1 -= 1.0f;
        result.reward2 -= 1.0f;
    }

    mPrevHp1 = mRobot1.hp;
    mPrevHp2 = mRobot2.hp;

    return result;
}

void CombatEnv::CheckCollisions()
{
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    const float collisionThreshold = 0.55f;

    JPH::RVec3 r2Pos = bodyInterface.GetPosition(mRobot2.mainBodyId);
    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        JPH::RVec3 spikePos = bodyInterface.GetPosition(mRobot1.satellites[i].spikeBodyId);
        float dist = static_cast<float>((spikePos - r2Pos).Length());

        if (dist < collisionThreshold)
        {
            JPH::Vec3 spikeVel = bodyInterface.GetLinearVelocity(mRobot1.satellites[i].spikeBodyId);
            float damage = spikeVel.Length() * DAMAGE_MULTIPLIER * 0.001f;
            mRobot2.hp -= damage;
            mRobot1.totalDamageDealt += damage;
            mRobot2.totalDamageTaken += damage;
        }
    }

    JPH::RVec3 r1Pos = bodyInterface.GetPosition(mRobot1.mainBodyId);
    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        JPH::RVec3 spikePos = bodyInterface.GetPosition(mRobot2.satellites[i].spikeBodyId);
        float dist = static_cast<float>((spikePos - r1Pos).Length());

        if (dist < collisionThreshold)
        {
            JPH::Vec3 spikeVel = bodyInterface.GetLinearVelocity(mRobot2.satellites[i].spikeBodyId);
            float damage = spikeVel.Length() * DAMAGE_MULTIPLIER * 0.001f;
            mRobot1.hp -= damage;
            mRobot2.totalDamageDealt += damage;
            mRobot1.totalDamageTaken += damage;
        }
    }
}

void CombatEnv::CalculateRewards(StepResult& result)
{
    float dmgDealt1 = mRobot1.totalDamageDealt;
    float dmgDealt2 = mRobot2.totalDamageDealt;
    float dmgTaken1 = mRobot1.totalDamageTaken;
    float dmgTaken2 = mRobot2.totalDamageTaken;

    result.reward1 = dmgDealt1 * 0.1f - dmgTaken1 * 0.05f;
    result.reward2 = dmgDealt2 * 0.1f - dmgTaken2 * 0.05f;

    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    JPH::RVec3 pos1 = bodyInterface.GetPosition(mRobot1.mainBodyId);
    JPH::RVec3 pos2 = bodyInterface.GetPosition(mRobot2.mainBodyId);
    float dist = static_cast<float>((pos2 - pos1).Length());

    float proximityReward = -0.001f * (dist - 3.0f);
    result.reward1 += proximityReward;
    result.reward2 += proximityReward;
}