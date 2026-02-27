#include <Jolt/Jolt.h>
#include "CombatEnv.h"
#include "InternalRobot.h"

#include <cmath>
#include <iostream>
#include <random>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>

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

    uint32_t envIdx1 = (layer1 == Layers::STATIC) ? static_cast<uint32_t>(-1) : (layer1 - Layers::MOVING_BASE);
    uint32_t envIdx2 = (layer2 == Layers::STATIC) ? static_cast<uint32_t>(-1) : (layer2 - Layers::MOVING_BASE);

    uint32_t envIdx;
    if (layer1 == Layers::STATIC && layer2 != Layers::STATIC) {
        envIdx = envIdx2;
    } else if (layer2 == Layers::STATIC && layer1 != Layers::STATIC) {
        envIdx = envIdx1;
    } else if (layer1 != Layers::STATIC && layer2 != Layers::STATIC) {
        if (envIdx1 != envIdx2) return;
        envIdx = envIdx1;
    } else {
        return;
    }

    if (envIdx >= NUM_PARALLEL_ENVS) return;

    float impulseMag = manifold.mPenetrationDepth * manifold.mWorldSpaceNormal.Length();
    mForceReadingsPerEnv[envIdx][0].impulseMagnitude[0] += impulseMag;
    mForceReadingsPerEnv[envIdx][1].impulseMagnitude[0] += impulseMag;
}

void CombatEnv::Init(uint32_t envIndex, JPH::PhysicsSystem* globalPhysics, CombatRobotLoader* globalLoader)
{
    mEnvIndex = envIndex;
    mPhysicsSystem = globalPhysics;
    mRobotLoader = globalLoader;

    Reset();
}

void CombatEnv::Reset()
{
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    
    mStepCount = 0;
    mDone = false;
    mPrevHp1 = INITIAL_HP;
    mPrevHp2 = INITIAL_HP;
    mPrevEnergy1 = 0.0f;
    mPrevEnergy2 = 0.0f;
    mRobot1.hp = INITIAL_HP;
    mRobot2.hp = INITIAL_HP;
    mRobot1.totalDamageDealt = 0.0f;
    mRobot1.totalDamageTaken = 0.0f;
    mRobot2.totalDamageDealt = 0.0f;
    mRobot2.totalDamageTaken = 0.0f;
    mRobot1.totalEnergyUsed = 0.0f;
    mRobot2.totalEnergyUsed = 0.0f;
    CombatContactListener::Get().ResetForceReadings(mEnvIndex);

    // Randomize KOTH point
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distXZ(-15.0f, 15.0f);
    std::uniform_real_distribution<float> distY(2.0f, 12.0f);
    mKothPoint = JPH::RVec3(distXZ(rng), distY(rng), distXZ(rng));

    // Update KOTH visual position if it exists, otherwise create it
    if (!mKothVisualId.IsInvalid()) {
        bodyInterface.SetPositionAndRotation(mKothVisualId, mKothPoint, JPH::Quat::sIdentity(), JPH::EActivation::DontActivate);
    } else {
        JPH::SphereShapeSettings kothShape(0.8f);
        JPH::BodyCreationSettings kothSettings(kothShape.Create().Get(), mKothPoint, JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::GHOST_BASE + mEnvIndex);
        mKothVisualId = bodyInterface.CreateAndAddBody(kothSettings, JPH::EActivation::DontActivate);
    }

    // 2. Reset existing robots instead of destroying/recreating
    JPH::RVec3 pos1(-10.0f, 5.0f, 0.0f);
    JPH::RVec3 pos2(10.0f, 5.0f, 0.0f);
    
    if (mRobot1.mainBodyId.IsInvalid() || mRobot2.mainBodyId.IsInvalid()) {
        // First time initialization: load robots
        mRobot1 = InternalRobotLoader::LoadInternalRobot("robots/internal_bot.json", mPhysicsSystem, pos1, mEnvIndex, 0);
        mRobot1.type = RobotType::INTERNAL_ENGINE;
        
        mRobot2 = InternalRobotLoader::LoadInternalRobot("robots/internal_bot.json", mPhysicsSystem, pos2, mEnvIndex, 1);
        mRobot2.type = RobotType::INTERNAL_ENGINE;
    } else {
        // Reset robot 1
        bodyInterface.SetPositionAndRotation(mRobot1.mainBodyId, pos1, JPH::Quat::sIdentity(), JPH::EActivation::Activate);
        bodyInterface.SetLinearVelocity(mRobot1.mainBodyId, JPH::Vec3::sZero());
        bodyInterface.SetAngularVelocity(mRobot1.mainBodyId, JPH::Vec3::sZero());
        for (int i = 0; i < NUM_SATELLITES; ++i) {
            if (!mRobot1.satellites[i].coreBodyId.IsInvalid()) {
                // For internal engine robots, reset engine positions to initial relative positions
                if (mRobot1.type == RobotType::INTERNAL_ENGINE && i < 3) {
                    JPH::RVec3 engPos = pos1 + JPH::RVec3((i - 1) * 0.1f, 0.0f, 0.0f);
                    bodyInterface.SetPositionAndRotation(mRobot1.satellites[i].coreBodyId, engPos, JPH::Quat::sIdentity(), JPH::EActivation::Activate);
                }
                bodyInterface.SetLinearVelocity(mRobot1.satellites[i].coreBodyId, JPH::Vec3::sZero());
                bodyInterface.SetAngularVelocity(mRobot1.satellites[i].coreBodyId, JPH::Vec3::sZero());
            }
            if (!mRobot1.satellites[i].spikeBodyId.IsInvalid()) {
                bodyInterface.SetLinearVelocity(mRobot1.satellites[i].spikeBodyId, JPH::Vec3::sZero());
                bodyInterface.SetAngularVelocity(mRobot1.satellites[i].spikeBodyId, JPH::Vec3::sZero());
            }
        }
        
        // Reset robot 2
        bodyInterface.SetPositionAndRotation(mRobot2.mainBodyId, pos2, JPH::Quat::sIdentity(), JPH::EActivation::Activate);
        bodyInterface.SetLinearVelocity(mRobot2.mainBodyId, JPH::Vec3::sZero());
        bodyInterface.SetAngularVelocity(mRobot2.mainBodyId, JPH::Vec3::sZero());
        for (int i = 0; i < NUM_SATELLITES; ++i) {
            if (!mRobot2.satellites[i].coreBodyId.IsInvalid()) {
                // For internal engine robots, reset engine positions to initial relative positions
                if (mRobot2.type == RobotType::INTERNAL_ENGINE && i < 3) {
                    JPH::RVec3 engPos = pos2 + JPH::RVec3((i - 1) * 0.1f, 0.0f, 0.0f);
                    bodyInterface.SetPositionAndRotation(mRobot2.satellites[i].coreBodyId, engPos, JPH::Quat::sIdentity(), JPH::EActivation::Activate);
                }
                bodyInterface.SetLinearVelocity(mRobot2.satellites[i].coreBodyId, JPH::Vec3::sZero());
                bodyInterface.SetAngularVelocity(mRobot2.satellites[i].coreBodyId, JPH::Vec3::sZero());
            }
            if (!mRobot2.satellites[i].spikeBodyId.IsInvalid()) {
                bodyInterface.SetLinearVelocity(mRobot2.satellites[i].spikeBodyId, JPH::Vec3::sZero());
                bodyInterface.SetAngularVelocity(mRobot2.satellites[i].spikeBodyId, JPH::Vec3::sZero());
            }
        }
    }
}

void CombatEnv::QueueActions(const float* actions1, const float* actions2)
{
    if (mDone) return;
    
    if (mRobot1.type == RobotType::SATELLITE)
        mRobotLoader->ApplyResidualActions(mRobot1, actions1, mPhysicsSystem);
    else
        InternalRobotLoader::ApplyInternalActions(mRobot1, actions1, mPhysicsSystem);

    if (mRobot2.type == RobotType::SATELLITE)
        mRobotLoader->ApplyResidualActions(mRobot2, actions2, mPhysicsSystem);
    else
        InternalRobotLoader::ApplyInternalActions(mRobot2, actions2, mPhysicsSystem);
}

StepResult CombatEnv::HarvestState()
{
    StepResult result;
    if (mDone) return result;

    mStepCount++;
    CheckCollisions();
    UpdateForceSensors();

    CombatContactListener& listener = CombatContactListener::Get();
    const ForceSensorReading& forces1 = listener.GetForceReading(mEnvIndex, 0);
    const ForceSensorReading& forces2 = listener.GetForceReading(mEnvIndex, 1);

    BuildObservationVector(result.obs_robot1, mRobot1, mRobot2, forces1);
    BuildObservationVector(result.obs_robot2, mRobot2, mRobot1, forces2);

    CalculateRewards(result);

    if (mRobot1.hp <= 0.0f || mRobot2.hp <= 0.0f || mStepCount >= MAX_EPISODE_STEPS) {
        mDone = true;
        result.done = true;
    }

    return result;
}

void CombatEnv::CheckCollisions()
{
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    const float spikeThreshold = 0.55f;
    const float engineThreshold = 1.0f; // Larger radius for engine slam

    auto applyDamage = [&](CombatRobotData& attacker, CombatRobotData& victim) {
        if (attacker.mainBodyId.IsInvalid() || victim.mainBodyId.IsInvalid()) return;
        
        JPH::RVec3 victimPos = bodyInterface.GetPosition(victim.mainBodyId);
        
        if (attacker.type == RobotType::SATELLITE) {
            for (int i = 0; i < NUM_SATELLITES; ++i) {
                if (attacker.satellites[i].spikeBodyId.IsInvalid()) continue;
                JPH::RVec3 spikePos = bodyInterface.GetPosition(attacker.satellites[i].spikeBodyId);
                if ((spikePos - victimPos).LengthSq() < spikeThreshold * spikeThreshold) {
                    JPH::Vec3 vel = bodyInterface.GetLinearVelocity(attacker.satellites[i].spikeBodyId);
                    float damage = vel.Length() * DAMAGE_MULTIPLIER * 0.001f;
                    victim.hp -= damage;
                    attacker.totalDamageDealt += damage;
                    victim.totalDamageTaken += damage;
                }
            }
        } else if (attacker.type == RobotType::INTERNAL_ENGINE) {
            // Internal engines deal damage when they are near the opponent 
            // (effectively slamming through their own shell into the opponent)
            for (int i = 0; i < 3; ++i) {
                if (attacker.satellites[i].coreBodyId.IsInvalid()) continue;
                JPH::RVec3 engPos = bodyInterface.GetPosition(attacker.satellites[i].coreBodyId);
                if ((engPos - victimPos).LengthSq() < engineThreshold * engineThreshold) {
                    JPH::Vec3 vel = bodyInterface.GetLinearVelocity(attacker.satellites[i].coreBodyId);
                    float damage = vel.Length() * DAMAGE_MULTIPLIER * 0.002f; // Heavier slam
                    victim.hp -= damage;
                    attacker.totalDamageDealt += damage;
                    victim.totalDamageTaken += damage;
                }
            }
        }
    };

    applyDamage(mRobot1, mRobot2);
    applyDamage(mRobot2, mRobot1);
}

void CombatEnv::UpdateForceSensors()
{
    CombatContactListener& listener = CombatContactListener::Get();
    auto updateStress = [&](CombatRobotData& r, int rIdx) {
        if (r.mainBodyId.IsInvalid()) return;
        for (int i = 0; i < NUM_SATELLITES; ++i) {
            if (r.satellites[i].rotationJoint) {
                listener.GetForceReading(mEnvIndex, rIdx).jointStress[i] = r.satellites[i].rotationJoint->GetTotalLambdaPosition().Length() * 0.001f;
            }
        }
    };
    updateStress(mRobot1, 0);
    updateStress(mRobot2, 1);
}

void CombatEnv::BuildObservationVector(AlignedVector32<float>& obs, const CombatRobotData& robot,
                                        const CombatRobotData& opponent, const ForceSensorReading& forces)
{
    if (robot.mainBodyId.IsInvalid() || opponent.mainBodyId.IsInvalid()) {
        std::fill(obs.begin(), obs.end(), 0.0f);
        return;
    }
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    int idx = 0;

    JPH::RVec3 myPos = bodyInterface.GetPosition(robot.mainBodyId);
    JPH::Vec3 myVel = bodyInterface.GetLinearVelocity(robot.mainBodyId);
    JPH::Vec3 myAngVel = bodyInterface.GetAngularVelocity(robot.mainBodyId);
    JPH::Quat myRot = bodyInterface.GetRotation(robot.mainBodyId);

    obs[idx++] = (float)myPos.GetX(); obs[idx++] = (float)myPos.GetY(); obs[idx++] = (float)myPos.GetZ();
    obs[idx++] = myVel.GetX(); obs[idx++] = myVel.GetY(); obs[idx++] = myVel.GetZ();
    obs[idx++] = myAngVel.GetX(); obs[idx++] = myAngVel.GetY(); obs[idx++] = myAngVel.GetZ();

    JPH::RVec3 oppPos = bodyInterface.GetPosition(opponent.mainBodyId);
    JPH::Vec3 oppVel = bodyInterface.GetLinearVelocity(opponent.mainBodyId);
    JPH::RVec3 relPos = oppPos - myPos;

    obs[idx++] = (float)relPos.GetX(); obs[idx++] = (float)relPos.GetY(); obs[idx++] = (float)relPos.GetZ();
    obs[idx++] = oppVel.GetX(); obs[idx++] = oppVel.GetY(); obs[idx++] = oppVel.GetZ();

    // KOTH Point (Relative)
    JPH::RVec3 relKoth = mKothPoint - myPos;
    obs[idx++] = (float)relKoth.GetX() / 20.0f;
    obs[idx++] = (float)relKoth.GetY() / 20.0f;
    obs[idx++] = (float)relKoth.GetZ() / 20.0f;

    // Satellite / Engine states
    for (int i = 0; i < NUM_SATELLITES; ++i) {
        if (!robot.satellites[i].coreBodyId.IsInvalid()) {
            JPH::RVec3 p = bodyInterface.GetPosition(robot.satellites[i].coreBodyId);
            JPH::Vec3 v = bodyInterface.GetLinearVelocity(robot.satellites[i].coreBodyId);
            obs[idx++] = (float)p.GetX(); obs[idx++] = (float)p.GetY(); obs[idx++] = (float)p.GetZ();
            obs[idx++] = v.GetX(); obs[idx++] = v.GetY(); obs[idx++] = v.GetZ();
        } else {
            for (int k = 0; k < 6; ++k) obs[idx++] = 0.0f;
        }
    }

    // My Spikes
    for (int i = 0; i < NUM_SATELLITES; ++i) {
        if (!robot.satellites[i].spikeBodyId.IsInvalid()) {
            JPH::RVec3 p = bodyInterface.GetPosition(robot.satellites[i].spikeBodyId);
            obs[idx++] = (float)p.GetX(); obs[idx++] = (float)p.GetY(); obs[idx++] = (float)p.GetZ();
        } else {
            for (int k = 0; k < 3; ++k) obs[idx++] = 0.0f;
        }
    }

    // Opponent Spikes
    for (int i = 0; i < NUM_SATELLITES; ++i) {
        if (!opponent.satellites[i].spikeBodyId.IsInvalid()) {
            JPH::RVec3 p = bodyInterface.GetPosition(opponent.satellites[i].spikeBodyId);
            obs[idx++] = (float)p.GetX(); obs[idx++] = (float)p.GetY(); obs[idx++] = (float)p.GetZ();
        } else {
            for (int k = 0; k < 3; ++k) obs[idx++] = 0.0f;
        }
    }

    mRobotLoader->PerformLidarScan(const_cast<CombatRobotData&>(robot), mPhysicsSystem);
    for (int i = 0; i < NUM_LIDAR_RAYS; ++i) obs[idx++] = robot.lidarDistances[i] / 20.0f;

    obs[idx++] = robot.hp / 100.0f;
    obs[idx++] = opponent.hp / 100.0f;
    obs[idx++] = (float)(oppPos - myPos).Length() / 20.0f;
    obs[idx++] = myRot.RotateAxisY().Dot((oppPos - myPos).Normalized());
    obs[idx++] = (robot.hp - opponent.hp) / 100.0f;
    
    // Forces
    for (int i = 0; i < NUM_SATELLITES; ++i) obs[idx++] = forces.impulseMagnitude[i];
    for (int i = 0; i < NUM_SATELLITES; ++i) obs[idx++] = forces.jointStress[i];

    // Current Mass Observations (for the new mass-shifting feature)
    obs[idx++] = bodyInterface.GetShape(robot.mainBodyId)->GetMassProperties().mMass / 50.0f;
    for (int i = 0; i < 3; ++i) {
        if (!robot.satellites[i].coreBodyId.IsInvalid()) {
            obs[idx++] = bodyInterface.GetShape(robot.satellites[i].coreBodyId)->GetMassProperties().mMass / 10.0f;
        } else {
            obs[idx++] = 0.0f;
        }
    }

    while (idx < 256) obs[idx++] = 0.0f;
}

void CombatEnv::CalculateRewards(StepResult& result)
{
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    
    JPH::RVec3 pos1 = bodyInterface.GetPosition(mRobot1.mainBodyId);
    JPH::RVec3 pos2 = bodyInterface.GetPosition(mRobot2.mainBodyId);
    JPH::Vec3 vel1 = bodyInterface.GetLinearVelocity(mRobot1.mainBodyId);
    JPH::Vec3 vel2 = bodyInterface.GetLinearVelocity(mRobot2.mainBodyId);

    // 1. Proximity & Engagement Reward
    float dist = static_cast<float>((pos2 - pos1).Length());
    
    float prox1 = 0.0f;
    if (dist < 15.0f) {
        prox1 = 0.1f * (1.0f - (dist / 15.0f)); 
    } else {
        prox1 = -0.05f * (dist - 15.0f); // Scaling penalty for being far
    }

    // Directional Attack Reward (moving toward opponent)
    JPH::Vec3 toOpponent1 = JPH::Vec3(pos2 - pos1).Normalized();
    JPH::Vec3 toOpponent2 = JPH::Vec3(pos1 - pos2).Normalized();
    float approach1 = vel1.Dot(toOpponent1) * 0.02f;
    float approach2 = vel2.Dot(toOpponent2) * 0.02f;

    // Wall Penalty (Arena size is 36, so walls are at +/- 18)
    auto calcWallPenalty = [](const JPH::RVec3& p) {
        float px = std::abs(p.GetX());
        float pz = std::abs(p.GetZ());
        float maxD = std::max(px, pz);
        if (maxD > 14.0f) return -0.1f * (maxD - 14.0f); // Gradual penalty starting 4m from wall
        return 0.0f;
    };
    float wallPenalty1 = calcWallPenalty(pos1);
    float wallPenalty2 = calcWallPenalty(pos2);

    // 2. Damage Rewards (The primary objective)
    // We compute the true delta using HP changes
    float deltaDmgTaken1 = std::max(0.0f, mPrevHp1 - mRobot1.hp);
    float deltaDmgTaken2 = std::max(0.0f, mPrevHp2 - mRobot2.hp);
    
    // In a 1v1, damage taken by 2 is damage dealt by 1
    float deltaDmgDealt1 = deltaDmgTaken2;
    float deltaDmgDealt2 = deltaDmgTaken1;

    mPrevHp1 = mRobot1.hp;
    mPrevHp2 = mRobot2.hp;

    result.reward1.damage_dealt = deltaDmgDealt1;
    result.reward1.damage_taken = -deltaDmgTaken1;
    
    result.reward2.damage_dealt = deltaDmgDealt2;
    result.reward2.damage_taken = -deltaDmgTaken2;

    // 3. Efficiency & Survival
    float deltaEnergy1 = std::max(0.0f, mRobot1.totalEnergyUsed - mPrevEnergy1);
    float deltaEnergy2 = std::max(0.0f, mRobot2.totalEnergyUsed - mPrevEnergy2);
    mPrevEnergy1 = mRobot1.totalEnergyUsed;
    mPrevEnergy2 = mRobot2.totalEnergyUsed;

    result.reward1.energy_used = -deltaEnergy1 * 0.01f;
    result.reward2.energy_used = -deltaEnergy2 * 0.01f;

    // 4. KOTH Reward (Whoever is closest to the random target point)
    float distToKoth1 = static_cast<float>((pos1 - mKothPoint).Length());
    float distToKoth2 = static_cast<float>((pos2 - mKothPoint).Length());

    if (distToKoth1 < distToKoth2) {
        result.reward1.koth = 0.1f;
        result.reward2.koth = 0.0f;
    } else {
        result.reward1.koth = 0.0f;
        result.reward2.koth = 0.1f;
    }

    // 5. Altitude
    result.reward1.altitude = std::max(0.0f, (float)pos1.GetY() * 0.05f);
    result.reward2.altitude = std::max(0.0f, (float)pos2.GetY() * 0.05f);

    // Shape the damage reward to encourage engagement
    result.reward1.damage_dealt += (prox1 + approach1 + wallPenalty1);
    result.reward2.damage_dealt += (prox1 + approach2 + wallPenalty2); 
}
CombatContactListener& CombatContactListener::Get() { static CombatContactListener instance; return instance; }
