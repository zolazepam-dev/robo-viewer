#include <Jolt/Jolt.h>
#include "CombatEnv.h"
#include "CombatRobot.h"
#include "RobotFactory.h"
#include "RobotController.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/EstimateCollisionResponse.h>

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

    if (envIdx >= mForceReadingsPerEnv.size()) return;

    // Calculate actual impulse magnitude from contact manifold using Jolt's EstimateCollisionResponse
    JPH::CollisionEstimationResult result;
    float combinedFriction = 0.5f; // Default friction value (can be adjusted)
    float combinedRestitution = 0.3f; // Default restitution value (can be adjusted)
    JPH::EstimateCollisionResponse(body1, body2, manifold, result, combinedFriction, combinedRestitution);

    // Sum up all impulse magnitudes from contact points
    float impulseMag = 0.0f;
    for (const auto& impulse : result.mImpulses) {
        impulseMag += impulse.mContactImpulse;
        impulseMag += impulse.mFrictionImpulse1;
        impulseMag += impulse.mFrictionImpulse2;
    }

    auto& reading1 = mForceReadingsPerEnv[envIdx][0];
    auto& reading2 = mForceReadingsPerEnv[envIdx][1];

    if (reading1.impulseMagnitude.size() > 0)
        reading1.impulseMagnitude[0] += impulseMag;
    if (reading2.impulseMagnitude.size() > 0)
        reading2.impulseMagnitude[0] += impulseMag;
}

void CombatEnv::Init(uint32_t envIndex, JPH::PhysicsSystem* globalPhysics, CombatRobotLoader* globalLoader, const std::string& robotConfigPath, int stepsPerEpisode)
{
    mEnvIndex = envIndex;
    mPhysicsSystem = globalPhysics;
    mRobotLoader = globalLoader;
    mRobotConfigPath = robotConfigPath;
    mStepsPerEpisode = stepsPerEpisode;

    // Load robots FIRST before Reset()
    JPH::RVec3 spawnPos1(-10.0f, 10.0f, 0.0f);
    JPH::RVec3 spawnPos2(10.0f, 10.0f, 0.0f);
    CombatRobotData loaded1 = globalLoader->LoadRobot(robotConfigPath, globalPhysics, spawnPos1, envIndex, 0);
    CombatRobotData loaded2 = globalLoader->LoadRobot(robotConfigPath, globalPhysics, spawnPos2, envIndex, 1);
    
    // Copy loaded data to Robot objects
    mRobot1.mainBodyId = loaded1.mainBodyId;
    mRobot1.satellites = loaded1.satellites;
    mRobot1.bodyIds = loaded1.bodies;
    mRobot1.hingeJoints = loaded1.hingeJoints;
    mRobot1.sixDofJoints = loaded1.sixDofJoints;
    mRobot1.config = loaded1.config;
    mRobot1.envIndex = loaded1.envIndex;
    mRobot1.robotIndex = loaded1.robotIndex;
    mRobot1.collisionGroup = loaded1.collisionGroup;
    
    mRobot2.mainBodyId = loaded2.mainBodyId;
    mRobot2.satellites = loaded2.satellites;
    mRobot2.bodyIds = loaded2.bodies;
    mRobot2.hingeJoints = loaded2.hingeJoints;
    mRobot2.sixDofJoints = loaded2.sixDofJoints;
    mRobot2.config = loaded2.config;
    mRobot2.envIndex = loaded2.envIndex;
    mRobot2.robotIndex = loaded2.robotIndex;
    mRobot2.collisionGroup = loaded2.collisionGroup;

    Reset();
    
    // Initialize Controllers (The Brains)
    mController1 = std::make_unique<RobotController>(mRobot1);
    mController2 = std::make_unique<RobotController>(mRobot2);

    // Set observation dimension from robot configuration
    mObservationDim = mRobot1.config.observationDim;
    
    // Ensure observation buffers are properly sized
    mObs1.resize(mObservationDim);
    mObs2.resize(mObservationDim);
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

    // Use a member RNG or a properly seeded local one for thread safety during parallel resets
    std::mt19937 rng(std::random_device{}() + mEnvIndex); 
    
    // APPLY DOMAIN RANDOMIZATION
    if (mDR.enabled) {
        std::uniform_real_distribution<float> gravDist(9.81f - mDR.gravityRange, 9.81f + mDR.gravityRange);
        std::uniform_real_distribution<float> frictDist(0.5f - mDR.frictionRange, 0.5f + mDR.frictionRange);
        std::uniform_real_distribution<float> restDist(0.0f, mDR.restitutionRange);
        
        float gravity = gravDist(rng);
        float friction = std::clamp(frictDist(rng), 0.0f, 1.0f);
        float restitution = std::clamp(restDist(rng), 0.0f, 1.0f);
        float gravityFactor = gravity / 9.81f;
        
        auto applyDR = [&](Robot& robot) {
            for (auto& bodyId : robot.bodyIds) {
                if (!bodyId.IsInvalid()) {
                    bodyInterface.SetFriction(bodyId, friction);
                    bodyInterface.SetRestitution(bodyId, restitution);
                    bodyInterface.SetGravityFactor(bodyId, gravityFactor);
                }
            }
        };
        applyDR(mRobot1);
        applyDR(mRobot2);
    }

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

    // 2. Random spawn locations for both agents
    // Spawn on opposite sides of arena with random positions
    std::uniform_real_distribution<float> spawnX(-20.0f, -5.0f);  // Robot 1: left side
    std::uniform_real_distribution<float> spawnX2(5.0f, 20.0f);   // Robot 2: right side
    std::uniform_real_distribution<float> spawnY(2.0f, 8.0f);     // Height variation
    std::uniform_real_distribution<float> spawnZ(-15.0f, 15.0f);  // Z-axis variation
    
    JPH::RVec3 pos1(spawnX(rng), spawnY(rng), spawnZ(rng));
    JPH::RVec3 pos2(spawnX2(rng), spawnY(rng), spawnZ(rng));
    
    // Ensure minimum spawn distance (prevent spawn killing)
    float minDistance = 15.0f;
    float maxAttempts = 10;
    float attempts = 0;
    while ((pos2 - pos1).Length() < minDistance && attempts < maxAttempts) {
        pos1 = JPH::RVec3(spawnX(rng), spawnY(rng), spawnZ(rng));
        pos2 = JPH::RVec3(spawnX2(rng), spawnY(rng), spawnZ(rng));
        attempts++;
    }

     if (mRobot1.mainBodyId.IsInvalid() || mRobot2.mainBodyId.IsInvalid()) {
         // First time initialization: load blue-prints then build
         std::ifstream f(mRobotConfigPath);
         if (!f.is_open()) {
             return;
         }
         nlohmann::json j;
         try {
             f >> j;
         } catch (const nlohmann::json::parse_error& e) {
             return;
         }
         auto config = RobotConfig::LoadFromJSON(j);

         mRobot1 = RobotFactory::CreateRobot(config, mPhysicsSystem, pos1, mEnvIndex, 0);
         mRobot2 = RobotFactory::CreateRobot(config, mPhysicsSystem, pos2, mEnvIndex, 1);
         if (mRobot1.mainBodyId.IsInvalid() || mRobot2.mainBodyId.IsInvalid()) { return; }
         
         mRobot1.type = config.type;
         mRobot2.type = config.type;

         // Update internal pointers for controllers
         mController1 = std::make_unique<RobotController>(mRobot1);
         mController2 = std::make_unique<RobotController>(mRobot2);
     } else {
         // Reset robots using the new factory with random spawn positions
         RobotFactory::ResetRobot(mRobot1, mPhysicsSystem, pos1);
         RobotFactory::ResetRobot(mRobot2, mPhysicsSystem, pos2);
     }

    // CRITICAL FIX: Reset force sensors AFTER robots are loaded
    int numSatellites = mRobot1.config.numSatellites;
    CombatContactListener::Get().ResetForceReadings(mEnvIndex, numSatellites);
}

void CombatEnv::QueueActions(const float* actions1, const float* actions2)
{
    if (mDone) return;
    
    mController1->ApplyResidualActions(actions1, mPhysicsSystem, 1.0f / 120.0f);
    mController2->ApplyResidualActions(actions2, mPhysicsSystem, 1.0f / 120.0f);
}

void CombatEnv::HarvestState(float* obs1, float* obs2, float* reward1, float* reward2, bool& done)
{
    if (mDone) {
        done = true;
        return;
    }

    mStepCount++;
    // [REMOVED] std::cout << "[CombatEnv] CheckCollisions..." << "\n";
    CheckCollisions();
    // [REMOVED] std::cout << "[CombatEnv] UpdateForceSensors..." << "\n";
    UpdateForceSensors();

    CombatContactListener& listener = CombatContactListener::Get();
    const ForceSensorReading& forces1 = listener.GetForceReading(mEnvIndex, 0);
    const ForceSensorReading& forces2 = listener.GetForceReading(mEnvIndex, 1);

    mController1->GetObservations(mRobot2, obs1, forces1, mPhysicsSystem);
    mController2->GetObservations(mRobot1, obs2, forces2, mPhysicsSystem);

    CalculateRewards(*reward1, *reward2);

    if (mRobot1.hp <= 0.0f || mRobot2.hp <= 0.0f || mStepCount >= mStepsPerEpisode) {
        mDone = true;
    }
    
    done = mDone;
}

// Zero-copy harvesting - write directly to provided pointers (no intermediate buffers)
void CombatEnv::HarvestStateZeroCopy(float* obs1, float* obs2, float* reward1, float* reward2, 
                                     bool* done, VectorReward* vectorReward) {
    if (mDone) {
        if (done) *done = true;
        return;
    }

    mStepCount++;
    CheckCollisions();
    UpdateForceSensors();

    CombatContactListener& listener = CombatContactListener::Get();
    const ForceSensorReading& forces1 = listener.GetForceReading(mEnvIndex, 0);
    const ForceSensorReading& forces2 = listener.GetForceReading(mEnvIndex, 1);

    // Write observations directly to provided pointers
    mController1->GetObservations(mRobot2, obs1, forces1, mPhysicsSystem);
    mController2->GetObservations(mRobot1, obs2, forces2, mPhysicsSystem);

    CalculateRewards(*reward1, *reward2);
    
    // Write vector reward if provided
    if (vectorReward) {
        *vectorReward = mReward1;  // Copy reward struct
    }

    if (mRobot1.hp <= 0.0f || mRobot2.hp <= 0.0f || mStepCount >= mStepsPerEpisode) {
        mDone = true;
    }
    
    if (done) *done = mDone;
}

// Zero-copy observation pointer access
const float* CombatEnv::GetObservationPtr(int robotIdx) const {
    // Return pointer to internal observation buffer (if it exists)
    // For now, return nullptr - observations are computed on-demand
    return nullptr;
}

// Zero-copy reward pointer access
const float* CombatEnv::GetRewardPtr(int robotIdx) const {
    // Return pointer to first field (damage_dealt) as proxy for reward
    if (robotIdx == 0) return &mReward1.damage_dealt;
    if (robotIdx == 1) return &mReward2.damage_dealt;
    return nullptr;
}

void CombatEnv::CheckCollisions()
{
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    const float spikeThreshold = 0.55f;
    const float engineThreshold = 1.0f; // Larger radius for engine slam

    auto applyDamage = [&](Robot& attacker, Robot& victim) {
        if (attacker.mainBodyId.IsInvalid() || victim.mainBodyId.IsInvalid()) return;
        
        JPH::RVec3 victimPos = bodyInterface.GetPosition(victim.mainBodyId);
        JPH::Vec3 victimVel = bodyInterface.GetLinearVelocity(victim.mainBodyId);

        // BASE DAMAGE (Main body slam)
        JPH::RVec3 attackerPos = bodyInterface.GetPosition(attacker.mainBodyId);
        JPH::Vec3 attackerVel = bodyInterface.GetLinearVelocity(attacker.mainBodyId);
        float distSq = (attackerPos - victimPos).LengthSq();
        if (distSq < 5.0f * 5.0f) {
            float relativeVel = (attackerVel - victimVel).Length();
            if (relativeVel > 2.0f) {
                float damage = relativeVel * DAMAGE_MULTIPLIER * 0.1f;
                victim.hp = std::max(0.0f, victim.hp - damage);
                attacker.totalDamageDealt += damage;
                victim.totalDamageTaken += damage;
                std::cout << "[DAMAGE] Main body slam: " << damage << " HP (vel: " << relativeVel << ")\n";
            }
        }
        
        // MULTI-BODY DAMAGE (for robots using config.bodies format like bouncy_orbiter)
        for (int i = 0; i < (int)attacker.bodyIds.size(); ++i) {
            JPH::BodyID bodyId = attacker.bodyIds[i];
            if (bodyId.IsInvalid() || bodyId == attacker.mainBodyId || bodyId == victim.mainBodyId) continue;
            
            if (!bodyInterface.IsAdded(bodyId)) continue;
            
            JPH::RVec3 bodyPos = bodyInterface.GetPosition(bodyId);
            JPH::Vec3 bodyVel = bodyInterface.GetLinearVelocity(bodyId);
            float bodyDistSq = static_cast<float>((bodyPos - victimPos).LengthSq());
            
            if (bodyDistSq < 2.0f * 2.0f) {
                float relativeVel = (bodyVel - victimVel).Length();
                if (relativeVel > 2.0f) {
                    float damage = relativeVel * DAMAGE_MULTIPLIER * 0.05f;
                    victim.hp = std::max(0.0f, victim.hp - damage);
                    attacker.totalDamageDealt += damage;
                    victim.totalDamageTaken += damage;
                    std::cout << "[DAMAGE] Multi-body hit: " << damage << " HP (vel: " << relativeVel << ")\n";
                }
            }
        }
        
        // SATELLITE DAMAGE (legacy satellite format)
        if (attacker.type == RobotType::SATELLITE && !attacker.satellites.empty()) {
            for (int i = 0; i < (int)attacker.satellites.size(); ++i) {
                if (!attacker.satellites[i].spikeBodyId.IsInvalid()) {
                    JPH::RVec3 spikePos = bodyInterface.GetPosition(attacker.satellites[i].spikeBodyId);
                    if ((spikePos - victimPos).LengthSq() < spikeThreshold * spikeThreshold) {
                        JPH::Vec3 vel = bodyInterface.GetLinearVelocity(attacker.satellites[i].spikeBodyId);
                        float relativeVel = (vel - victimVel).Length();
                        if (relativeVel > 2.0f) {
                            float damage = relativeVel * DAMAGE_MULTIPLIER * 0.08f;
                            victim.hp = std::max(0.0f, victim.hp - damage);
                            attacker.totalDamageDealt += damage;
                            victim.totalDamageTaken += damage;
                            std::cout << "[DAMAGE] Spike hit: " << damage << " HP (vel: " << relativeVel << ")\n";
                        }
                    }
                }
                
                if (!attacker.satellites[i].coreBodyId.IsInvalid()) {
                    JPH::RVec3 satPos = bodyInterface.GetPosition(attacker.satellites[i].coreBodyId);
                    float satRadius = attacker.config.satellites.size() > i ? attacker.config.satellites[i].radius : 0.1f;
                    float satThreshold = satRadius + 0.6f;
                    
                    if ((satPos - victimPos).LengthSq() < satThreshold * satThreshold) {
                        JPH::Vec3 vel = bodyInterface.GetLinearVelocity(attacker.satellites[i].coreBodyId);
                        float relativeVel = (vel - victimVel).Length();
                        if (relativeVel > 2.0f) {
                            float damage = relativeVel * DAMAGE_MULTIPLIER * 0.04f;
                            victim.hp = std::max(0.0f, victim.hp - damage);
                            attacker.totalDamageDealt += damage;
                            victim.totalDamageTaken += damage;
                            std::cout << "[DAMAGE] Satellite hit: " << damage << " HP (vel: " << relativeVel << ")\n";
                        }
                    }
                }
            }
        } else if (attacker.type == RobotType::INTERNAL_ENGINE && !attacker.satellites.empty()) {
            for (int i = 0; i < (int)attacker.satellites.size(); ++i) {
                if (attacker.satellites[i].coreBodyId.IsInvalid()) continue;
                JPH::RVec3 engPos = bodyInterface.GetPosition(attacker.satellites[i].coreBodyId);
                if ((engPos - victimPos).LengthSq() < engineThreshold * engineThreshold) {
                    JPH::Vec3 vel = bodyInterface.GetLinearVelocity(attacker.satellites[i].coreBodyId);
                    float relativeVel = (vel - victimVel).Length();
                    if (relativeVel > 2.0f) {
                        float damage = relativeVel * DAMAGE_MULTIPLIER * 0.15f;
                        victim.hp = std::max(0.0f, victim.hp - damage);
                        attacker.totalDamageDealt += damage;
                        victim.totalDamageTaken += damage;
                        std::cout << "[DAMAGE] Engine slam: " << damage << " HP (vel: " << relativeVel << ")\n";
                    }
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
    auto updateStress = [&](Robot& r, int rIdx) {
        if (r.mainBodyId.IsInvalid()) return;
        int numSats = static_cast<int>(r.satellites.size());
        for (int i = 0; i < numSats; ++i) {
            if (r.satellites[i].rotationJoint) {
                listener.GetForceReading(mEnvIndex, rIdx).jointStress[i] = r.satellites[i].rotationJoint->GetTotalLambdaPosition().Length() * 0.001f;
            }
        }
    };
    updateStress(mRobot1, 0);
    updateStress(mRobot2, 1);
}

void CombatEnv::CalculateRewards(float& r1, float& r2)
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

    // Velocity-based damage multiplier (quadratic scaling)
    float speed1 = vel1.Length();
    float speed2 = vel2.Length();
    float velocityMultiplier1 = 1.0f + (speed1 * speed1 / 400.0f);  // +100% at 20 m/s
    float velocityMultiplier2 = 1.0f + (speed2 * speed2 / 400.0f);
    
    // Momentum advantage reward
    float momentumBonus1 = (speed1 > speed2) ? 0.2f : 0.0f;
    float momentumBonus2 = (speed2 > speed1) ? 0.2f : 0.0f;

    VectorReward vr1, vr2;
    vr1.damage_dealt = deltaDmgDealt1 * velocityMultiplier1 + momentumBonus1;
    vr1.damage_taken = -deltaDmgTaken1;
    
    vr2.damage_dealt = deltaDmgDealt2 * velocityMultiplier2 + momentumBonus2;
    vr2.damage_taken = -deltaDmgTaken2;

    // 3. Efficiency & Survival
    float deltaEnergy1 = std::max(0.0f, mRobot1.totalEnergyUsed - mPrevEnergy1);
    float deltaEnergy2 = std::max(0.0f, mRobot2.totalEnergyUsed - mPrevEnergy2);
    mPrevEnergy1 = mRobot1.totalEnergyUsed;
    mPrevEnergy2 = mRobot2.totalEnergyUsed;

    vr1.energy_used = -deltaEnergy1 * 0.01f;
    vr2.energy_used = -deltaEnergy2 * 0.01f;

    // 4. KOTH Reward (Whoever is closest to the random target point)
    float distToKoth1 = static_cast<float>((pos1 - mKothPoint).Length());
    float distToKoth2 = static_cast<float>((pos2 - mKothPoint).Length());

    // Continuous KOTH reward based on distance advantage
    float kothAdvantage = std::max(0.0f, distToKoth2 - distToKoth1);
    vr1.koth = 0.5f * kothAdvantage;
    vr2.koth = 0.5f * std::max(0.0f, distToKoth1 - distToKoth2);

    // 5. Altitude
    vr1.altitude = std::max(0.0f, (float)pos1.GetY() * 0.05f);
    vr2.altitude = std::max(0.0f, (float)pos2.GetY() * 0.05f);

    // Shape the damage reward to encourage engagement
    vr1.damage_dealt += (prox1 + approach1 + wallPenalty1);
    vr2.damage_dealt += (prox1 + approach2 + wallPenalty2); 

    if (!std::isfinite(vr1.Scalar())) { std::cerr << "[CombatEnv] Non-finite reward!" << "\n"; vr1.damage_dealt = 0; }
    r1 = vr1.Scalar();
    r2 = vr2.Scalar();
    
    mReward1 = vr1;
    mReward2 = vr2;
}
CombatContactListener& CombatContactListener::Get() { static CombatContactListener instance; return instance; }
