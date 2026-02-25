#include <Jolt/Jolt.h>
#include "CombatEnv.h"

#include <cmath>
#include <iostream>
#include <Jolt/Physics/Body/BodyInterface.h>

void CombatContactListener::OnContactAdded(const JPH::Body& body1, const JPH::Body& body2,
                                            const JPH::ContactManifold& manifold, JPH::ContactSettings& settings)
{
    ExtractImpulseData(body1, body2, manifold);
}

void CombatContactListener::OnContactPersisted(const JPH::Body& body1, const JPH::Body& body2,
                                                const JPH::ContactManifold& manifold, JPH::ContactSettings& settings)
{
    ExtractImpulseData(body1, body2, manifold);
}

void CombatContactListener::OnContactRemoved(const JPH::SubShapeIDPair& subShapePair)
{
}

void CombatContactListener::ExtractImpulseData(const JPH::Body& body1, const JPH::Body& body2,
                                                const JPH::ContactManifold& manifold)
{
    JPH::ObjectLayer layer1 = body1.GetObjectLayer();
    JPH::ObjectLayer layer2 = body2.GetObjectLayer();

    uint32_t envIdx1 = layer1 - Layers::MOVING_BASE;
    uint32_t envIdx2 = layer2 - Layers::MOVING_BASE;

    if (envIdx1 != envIdx2) return;

    float impulseMag = manifold.mPenetrationDepth * manifold.mWorldSpaceNormal.Length();
    
    if (envIdx1 < NUM_PARALLEL_ENVS)
    {
        int robotIdx = 0;
        mForceReadings[envIdx1][robotIdx].impulseMagnitude[0] += impulseMag;
    }
}

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
    mPrevEnergy1 = 0.0f;
    mPrevEnergy2 = 0.0f;
    mAirAccumulator1 = 0.0f;
    mAirAccumulator2 = 0.0f;
    mForceReadings1.Reset();
    mForceReadings2.Reset();
}

void CombatEnv::Reset()
{
    mStepCount = 0;
    mDone = false;
    mPrevHp1 = INITIAL_HP;
    mPrevHp2 = INITIAL_HP;
    mPrevEnergy1 = 0.0f;
    mPrevEnergy2 = 0.0f;
    mAirAccumulator1 = 0.0f;
    mAirAccumulator2 = 0.0f;
    mForceReadings1.Reset();
    mForceReadings2.Reset();

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
    UpdateForceSensors();

    BuildObservationVector(result.obs_robot1, mRobot1, mRobot2, mForceReadings1);
    BuildObservationVector(result.obs_robot2, mRobot2, mRobot1, mForceReadings2);

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
            result.reward1.damage_dealt += 10.0f;
            result.reward2.damage_taken += 10.0f;
        }
        else
        {
            result.winner = 2;
            result.reward2.damage_dealt += 10.0f;
            result.reward1.damage_taken += 10.0f;
        }
    }

    if (mStepCount >= MAX_EPISODE_STEPS)
    {
        mDone = true;
        result.done = true;
        result.reward1.energy_used += 1.0f;
        result.reward2.energy_used += 1.0f;
    }

    mPrevHp1 = mRobot1.hp;
    mPrevHp2 = mRobot2.hp;
    mPrevEnergy1 = mRobot1.totalEnergyUsed;
    mPrevEnergy2 = mRobot2.totalEnergyUsed;

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

void CombatEnv::UpdateForceSensors()
{
    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        if (mRobot1.satellites[i].rotationJoint != nullptr)
        {
            JPH::Vec3 lagrange = mRobot1.satellites[i].rotationJoint->GetTotalLambdaPosition();
            mForceReadings1.jointStress[i] = lagrange.Length() * 0.001f;
        }

        if (mRobot2.satellites[i].rotationJoint != nullptr)
        {
            JPH::Vec3 lagrange = mRobot2.satellites[i].rotationJoint->GetTotalLambdaPosition();
            mForceReadings2.jointStress[i] = lagrange.Length() * 0.001f;
        }
    }
}

void CombatEnv::BuildObservationVector(std::vector<float>& obs, const CombatRobotData& robot,
                                        const CombatRobotData& opponent, const ForceSensorReading& forces)
{
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    int idx = 0;

    JPH::RVec3 myPos = bodyInterface.GetPosition(robot.mainBodyId);
    JPH::Vec3 myVel = bodyInterface.GetLinearVelocity(robot.mainBodyId);
    JPH::Vec3 myAngVel = bodyInterface.GetAngularVelocity(robot.mainBodyId);

    obs[idx++] = static_cast<float>(myPos.GetX());
    obs[idx++] = static_cast<float>(myPos.GetY());
    obs[idx++] = static_cast<float>(myPos.GetZ());
    obs[idx++] = myVel.GetX();
    obs[idx++] = myVel.GetY();
    obs[idx++] = myVel.GetZ();
    obs[idx++] = myAngVel.GetX();
    obs[idx++] = myAngVel.GetY();
    obs[idx++] = myAngVel.GetZ();

    JPH::RVec3 oppPos = bodyInterface.GetPosition(opponent.mainBodyId);
    JPH::Vec3 oppVel = bodyInterface.GetLinearVelocity(opponent.mainBodyId);
    JPH::RVec3 relPos = oppPos - myPos;

    obs[idx++] = static_cast<float>(relPos.GetX());
    obs[idx++] = static_cast<float>(relPos.GetY());
    obs[idx++] = static_cast<float>(relPos.GetZ());
    obs[idx++] = oppVel.GetX();
    obs[idx++] = oppVel.GetY();
    obs[idx++] = oppVel.GetZ();

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        JPH::RVec3 pos = bodyInterface.GetPosition(robot.satellites[i].coreBodyId);
        JPH::Vec3 vel = bodyInterface.GetLinearVelocity(robot.satellites[i].coreBodyId);
        obs[idx++] = static_cast<float>(pos.GetX());
        obs[idx++] = static_cast<float>(pos.GetY());
        obs[idx++] = static_cast<float>(pos.GetZ());
        obs[idx++] = vel.GetX();
        obs[idx++] = vel.GetY();
        obs[idx++] = vel.GetZ();
    }

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        JPH::RVec3 pos = bodyInterface.GetPosition(robot.satellites[i].spikeBodyId);
        obs[idx++] = static_cast<float>(pos.GetX());
        obs[idx++] = static_cast<float>(pos.GetY());
        obs[idx++] = static_cast<float>(pos.GetZ());
    }

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        JPH::RVec3 pos = bodyInterface.GetPosition(opponent.satellites[i].spikeBodyId);
        obs[idx++] = static_cast<float>(pos.GetX());
        obs[idx++] = static_cast<float>(pos.GetY());
        obs[idx++] = static_cast<float>(pos.GetZ());
    }

    for (int i = 0; i < NUM_SATELLITES; ++i)
    {
        obs[idx++] = forces.impulseMagnitude[i];
        obs[idx++] = forces.jointStress[i];
    }

    obs[idx++] = robot.hp;
    obs[idx++] = opponent.hp;
    obs[idx++] = static_cast<float>((oppPos - myPos).Length());
}

float CombatEnv::ComputeAirtime() const
{
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    
    float airtime = 0.0f;
    JPH::RVec3 pos1 = bodyInterface.GetPosition(mRobot1.mainBodyId);
    if (pos1.GetY() > 1.0f) airtime += 0.01f;
    
    JPH::RVec3 pos2 = bodyInterface.GetPosition(mRobot2.mainBodyId);
    if (pos2.GetY() > 1.0f) airtime += 0.01f;
    
    return airtime;
}

float CombatEnv::ComputeEnergyUsed(const float* actions, int actionDim) const
{
    float energy = 0.0f;
    for (int i = 0; i < actionDim; ++i)
    {
        energy += std::abs(actions[i]) * 0.001f;
    }
    return -energy;
}

void CombatEnv::CalculateRewards(StepResult& result)
{
    float dmgDealt1 = mRobot1.totalDamageDealt;
    float dmgDealt2 = mRobot2.totalDamageDealt;
    float dmgTaken1 = mRobot1.totalDamageTaken;
    float dmgTaken2 = mRobot2.totalDamageTaken;

    result.reward1.damage_dealt = dmgDealt1 * 0.1f;
    result.reward1.damage_taken = -dmgTaken1 * 0.05f;
    result.reward1.airtime = ComputeAirtime() * 0.01f;
    result.reward1.energy_used = -mRobot1.totalEnergyUsed * 0.001f;

    result.reward2.damage_dealt = dmgDealt2 * 0.1f;
    result.reward2.damage_taken = -dmgTaken2 * 0.05f;
    result.reward2.airtime = ComputeAirtime() * 0.01f;
    result.reward2.energy_used = -mRobot2.totalEnergyUsed * 0.001f;

    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    JPH::RVec3 pos1 = bodyInterface.GetPosition(mRobot1.mainBodyId);
    JPH::RVec3 pos2 = bodyInterface.GetPosition(mRobot2.mainBodyId);
    float dist = static_cast<float>((pos2 - pos1).Length());

    float proximityReward = -0.001f * (dist - 3.0f);
    result.reward1.damage_dealt += proximityReward;
    result.reward2.damage_dealt += proximityReward;
}
