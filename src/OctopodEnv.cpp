/**
 * @file OctopodEnv.cpp
 * @brief Combat environment for octopod robots
 */

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/EstimateCollisionResponse.h>

#include "OctopodEnv.h"
#include "OctopodLoader.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <mutex>

void OctopodContactListener::OnContactAdded(const JPH::Body& body1, const JPH::Body& body2,
                                             const JPH::ContactManifold& manifold, JPH::ContactSettings& settings)
{
    ExtractImpulseData(body1, body2, manifold);
}

void OctopodContactListener::OnContactPersisted(const JPH::Body& body1, const JPH::Body& body2,
                                                 const JPH::ContactManifold& manifold, JPH::ContactSettings& settings)
{
    ExtractImpulseData(body1, body2, manifold);
}

void OctopodContactListener::OnContactRemoved(const JPH::SubShapeIDPair& subShapePair)
{
}

void OctopodContactListener::ExtractImpulseData(const JPH::Body& body1, const JPH::Body& body2,
                                                 const JPH::ContactManifold& manifold)
{
    // Use atomic operations or per-env storage to avoid race conditions
    // For now, skip if bodies don't belong to tracked robots
    // TODO: Implement proper thread-safe force accumulation
}

OctopodContactListener& OctopodContactListener::Get() { 
    static OctopodContactListener instance; 
    return instance; 
}

void OctopodEnv::Init(uint32_t envIndex, PhysicsCore* core, int stepsPerEpisode)
{
    std::cerr << "[OctopodEnv] Init for env " << envIndex << "..." << std::endl;
    mEnvIndex = envIndex;
    mCore = core;
    mPhysicsSystem = &core->GetPhysicsSystem();
    mStepsPerEpisode = stepsPerEpisode;

    Reset();

    // Store initial body positions relative to central body
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    
    if (mRobot1.IsValid()) {
        JPH::RVec3 centralPos = bodyInterface.GetPosition(mRobot1.centralBody);
        for (size_t i = 0; i < mRobot1.bodies.size() && i < 25; i++) {
            if (!mRobot1.bodies[i].IsInvalid()) {
                mInitialPositions1[i] = bodyInterface.GetPosition(mRobot1.bodies[i]) - centralPos;
            }
        }
    }
    
    if (mRobot2.IsValid()) {
        JPH::RVec3 centralPos = bodyInterface.GetPosition(mRobot2.centralBody);
        for (size_t i = 0; i < mRobot2.bodies.size() && i < 25; i++) {
            if (!mRobot2.bodies[i].IsInvalid()) {
                mInitialPositions2[i] = bodyInterface.GetPosition(mRobot2.bodies[i]) - centralPos;
            }
        }
    }
}

void OctopodEnv::Reset()
{
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    
    mStepCount = 0;
    mDone = false;
    mPrevHp1 = OCTOPOD_INITIAL_HP;
    mPrevHp2 = OCTOPOD_INITIAL_HP;
    mPrevEnergy1 = 0.0f;
    mPrevEnergy2 = 0.0f;
    mRobot1.hp = OCTOPOD_INITIAL_HP;
    mRobot2.hp = OCTOPOD_INITIAL_HP;
    mRobot1.totalDamageDealt = 0.0f;
    mRobot1.totalDamageTaken = 0.0f;
    mRobot2.totalDamageDealt = 0.0f;
    mRobot2.totalDamageTaken = 0.0f;
    mRobot1.totalEnergyUsed = 0.0f;
    mRobot2.totalEnergyUsed = 0.0f;

    // Randomize KOTH point
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distXZ(-15.0f, 15.0f);
    std::uniform_real_distribution<float> distY(2.0f, 12.0f);
    mKothPoint = JPH::RVec3(distXZ(rng), distY(rng), distXZ(rng));

    // Update or create KOTH visual
    if (!mKothVisualId.IsInvalid()) {
        bodyInterface.SetPositionAndRotation(mKothVisualId, mKothPoint, JPH::Quat::sIdentity(), JPH::EActivation::DontActivate);
    } else {
        JPH::SphereShapeSettings kothShape(0.8f);
        JPH::BodyCreationSettings kothSettings(kothShape.Create().Get(), mKothPoint, JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::GHOST_BASE + mEnvIndex);
        mKothVisualId = bodyInterface.CreateAndAddBody(kothSettings, JPH::EActivation::DontActivate);
    }

    // Load octopods - spawn further apart to avoid overlap
    JPH::RVec3 pos1(-12.0f, 1.0f, 0.0f);
    JPH::RVec3 pos2(12.0f, 1.0f, 0.0f);

    // Remove existing bodies if any (only on first reset)
    static std::once_flag initFlag;
    std::call_once(initFlag, [&]() {
        // First time - no bodies to remove
        mRobot1.envIndex = mEnvIndex;
        mRobot1.robotIndex = 0;
        mRobot2.envIndex = mEnvIndex;
        mRobot2.robotIndex = 1;
    });

    if (mRobot1.bodies.empty()) {
        // Load octopod from JSON
        std::cerr << "[OctopodEnv] Loading octopod 1..." << std::endl;
        auto loaded1 = OctopodLoader::LoadOctopod("robots/octopod.json", mPhysicsSystem, pos1, mEnvIndex, mCore->GetGroupFilter());
        mRobot1.bodies = loaded1.bodies;
        mRobot1.constraints = loaded1.constraints;
        mRobot1.bodyMap = loaded1.bodyMap;
        mRobot1.centralBody = loaded1.centralBody;
        mRobot1.envIndex = mEnvIndex;
        mRobot1.robotIndex = 0;

        std::cerr << "[OctopodEnv] Loading octopod 2..." << std::endl;
        auto loaded2 = OctopodLoader::LoadOctopod("robots/octopod.json", mPhysicsSystem, pos2, mEnvIndex, mCore->GetGroupFilter());
        mRobot2.bodies = loaded2.bodies;
        mRobot2.constraints = loaded2.constraints;
        mRobot2.bodyMap = loaded2.bodyMap;
        mRobot2.centralBody = loaded2.centralBody;
        mRobot2.envIndex = mEnvIndex;
        mRobot2.robotIndex = 1;
    } else {
        // Reset existing bodies to initial positions
        std::cerr << "[OctopodEnv] Resetting octopod positions..." << std::endl;
        
        // Restore all body positions using stored relative offsets
        auto resetRobot = [&](const OctopodRobot& robot, const JPH::RVec3& basePos, 
                              const std::array<JPH::RVec3, 25>& initialOffsets) {
            bodyInterface.SetPosition(robot.centralBody, basePos, JPH::EActivation::Activate);
            bodyInterface.SetLinearVelocity(robot.centralBody, JPH::Vec3::sZero());
            bodyInterface.SetAngularVelocity(robot.centralBody, JPH::Vec3::sZero());
            
            for (size_t i = 0; i < robot.bodies.size(); i++) {
                if (!robot.bodies[i].IsInvalid() && robot.bodies[i] != robot.centralBody) {
                    JPH::RVec3 newPos = basePos + initialOffsets[i];
                    bodyInterface.SetPosition(robot.bodies[i], newPos, JPH::EActivation::Activate);
                    bodyInterface.SetLinearVelocity(robot.bodies[i], JPH::Vec3::sZero());
                    bodyInterface.SetAngularVelocity(robot.bodies[i], JPH::Vec3::sZero());
                }
            }
        };
        
        resetRobot(mRobot1, pos1, mInitialPositions1);
        resetRobot(mRobot2, pos2, mInitialPositions2);
    }

    std::cerr << "[OctopodEnv] Octopod 1: " << mRobot1.bodies.size() << " bodies, " 
              << mRobot1.constraints.size() << " constraints" << std::endl;
    std::cerr << "[OctopodEnv] Octopod 2: " << mRobot2.bodies.size() << " bodies, " 
              << mRobot2.constraints.size() << " constraints" << std::endl;
}

void OctopodEnv::QueueActions(const float* actions1, const float* actions2)
{
    if (mDone) return;
    
    // Apply motor actions to constraints
    // Actions are normalized [-1, 1], map to joint angle limits
    constexpr float OCTOPOD_ANGLE_SCALE = 1.0f;  // Radians per action unit
    
    for (size_t i = 0; i < mRobot1.constraints.size() && i < OCTOPOD_ACTION_DIM; i++) {
        if (mRobot1.constraints[i]) {
            float targetAngle = actions1[i] * OCTOPOD_ANGLE_SCALE;
            mRobot1.constraints[i]->SetMotorState(JPH::EMotorState::Position);
            mRobot1.constraints[i]->SetTargetAngle(targetAngle);
        }
    }
    
    for (size_t i = 0; i < mRobot2.constraints.size() && i < OCTOPOD_ACTION_DIM; i++) {
        if (mRobot2.constraints[i]) {
            float targetAngle = actions2[i] * OCTOPOD_ANGLE_SCALE;
            mRobot2.constraints[i]->SetMotorState(JPH::EMotorState::Position);
            mRobot2.constraints[i]->SetTargetAngle(targetAngle);
        }
    }
}

void OctopodEnv::HarvestState(float* obs1, float* obs2, float* reward1, float* reward2, bool& done)
{
    if (mDone) {
        done = true;
        return;
    }

    mStepCount++;
    CheckCollisions();

    // Force readings disabled - race condition in multi-threaded physics
    std::array<float, 25> forces1 = {};
    std::array<float, 25> forces2 = {};

    BuildObservationVector(obs1, mRobot1, mRobot2, forces1);
    BuildObservationVector(obs2, mRobot2, mRobot1, forces2);

    CalculateRewards(*reward1, *reward2);

    if (mRobot1.hp <= 0.0f || mRobot2.hp <= 0.0f || mStepCount >= mStepsPerEpisode) {
        mDone = true;
    }
    
    done = mDone;
}

void OctopodEnv::CheckCollisions()
{
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    
    // Simple distance-based damage detection
    // When leg segments hit opponent's central body at high velocity, deal damage
    auto applyDamage = [&](OctopodRobot& attacker, OctopodRobot& victim) {
        if (!attacker.IsValid() || !victim.IsValid()) return;
        
        JPH::RVec3 victimPos = bodyInterface.GetPosition(victim.centralBody);
        
        // Use actual collision data from contact listener instead of distance checks
        // TODO: Wire up contact listener to provide collision pairs with impulse data
        for (const auto& bodyId : attacker.bodies) {
            if (bodyId.IsInvalid() || bodyId == attacker.centralBody) continue;
            
            JPH::RVec3 segmentPos = bodyInterface.GetPosition(bodyId);
            JPH::Vec3 vel = bodyInterface.GetLinearVelocity(bodyId);
            float speed = vel.Length();
            
            // Damage threshold
            if (speed > 5.0f) {
                float distSq = (segmentPos - victimPos).LengthSq();
                if (distSq < 1.0f * 1.0f) {  // Reduced to 1 meter as suggested
                    float damage = speed * OCTOPOD_DAMAGE_MULTIPLIER * 0.01f;
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

void OctopodEnv::UpdateForceSensors()
{
    // Force sensors are updated via contact listener
    // This method can be used for additional sensor processing if needed
}

void OctopodEnv::BuildObservationVector(float* obs, const OctopodRobot& robot,
                                  const OctopodRobot& opponent, const std::array<float, 25>& forces)
{
    if (!robot.IsValid() || !opponent.IsValid()) {
        std::memset(obs, 0, OCTOPOD_OBS_DIM * sizeof(float));
        return;
    }
    
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    int idx = 0;

    // Central body state
    JPH::RVec3 myPos = bodyInterface.GetPosition(robot.centralBody);
    JPH::Vec3 myVel = bodyInterface.GetLinearVelocity(robot.centralBody);
    JPH::Vec3 myAngVel = bodyInterface.GetAngularVelocity(robot.centralBody);
    JPH::Quat myRot = bodyInterface.GetRotation(robot.centralBody);

    // Position (3) - Relative to center of arena or just normalize
    obs[idx++] = (float)myPos.GetX() / 50.0f;
    obs[idx++] = (float)myPos.GetY() / 10.0f;
    obs[idx++] = (float)myPos.GetZ() / 50.0f;
    
    // Linear velocity (3) - Normalize by expected max speed (e.g., 10 m/s)
    obs[idx++] = myVel.GetX() / 10.0f;
    obs[idx++] = myVel.GetY() / 10.0f;
    obs[idx++] = myVel.GetZ() / 10.0f;
    
    // Angular velocity (3) - Normalize by expected max (e.g., 5 rad/s)
    obs[idx++] = myAngVel.GetX() / 5.0f;
    obs[idx++] = myAngVel.GetY() / 5.0f;
    obs[idx++] = myAngVel.GetZ() / 5.0f;
    
    // Rotation as quaternion (4) - Already normalized [0, 1]
    obs[idx++] = myRot.GetX();
    obs[idx++] = myRot.GetY();
    obs[idx++] = myRot.GetZ();
    obs[idx++] = myRot.GetW();

    // Opponent relative state
    JPH::RVec3 oppPos = bodyInterface.GetPosition(opponent.centralBody);
    JPH::Vec3 oppVel = bodyInterface.GetLinearVelocity(opponent.centralBody);
    JPH::RVec3 relPos = oppPos - myPos;
    
    // Relative position (3) - Normalize by typical combat distance
    obs[idx++] = (float)relPos.GetX() / 20.0f;
    obs[idx++] = (float)relPos.GetY() / 10.0f;
    obs[idx++] = (float)relPos.GetZ() / 20.0f;
    
    // Opponent velocity (3)
    obs[idx++] = oppVel.GetX() / 10.0f;
    obs[idx++] = oppVel.GetY() / 10.0f;
    obs[idx++] = oppVel.GetZ() / 10.0f;

    // KOTH point relative (3)
    JPH::RVec3 relKoth = mKothPoint - myPos;
    obs[idx++] = (float)relKoth.GetX() / 30.0f;
    obs[idx++] = (float)relKoth.GetY() / 15.0f;
    obs[idx++] = (float)relKoth.GetZ() / 30.0f;

    // Leg segment states (8 legs × 3 segments = 24 segments)
    // For each leg: coxa, femur, tibia positions and velocities
    const char* legSegments[] = {"coxa", "femur", "tibia"};
    for (int leg = 0; leg < 8; leg++) {
        for (int seg = 0; seg < 3; seg++) {
            std::string segName = "leg" + std::to_string(leg) + "_" + legSegments[seg];
            auto it = robot.bodyMap.find(segName);
            if (it != robot.bodyMap.end() && !it->second.IsInvalid()) {
                JPH::RVec3 segPos = bodyInterface.GetPosition(it->second);
                JPH::Vec3 segVel = bodyInterface.GetLinearVelocity(it->second);
                // Relative to central body
                obs[idx++] = (float)(segPos.GetX() - myPos.GetX()) / 2.0f;
                obs[idx++] = (float)(segPos.GetY() - myPos.GetY()) / 2.0f;
                obs[idx++] = (float)(segPos.GetZ() - myPos.GetZ()) / 2.0f;
                obs[idx++] = (segVel.GetX() - myVel.GetX()) / 5.0f;
                obs[idx++] = (segVel.GetY() - myVel.GetY()) / 5.0f;
                obs[idx++] = (segVel.GetZ() - myVel.GetZ()) / 5.0f;
            } else {
                for (int k = 0; k < 6; k++) obs[idx++] = 0.0f;
            }
        }
    }

    // Health and game state
    obs[idx++] = robot.hp / OCTOPOD_INITIAL_HP;
    obs[idx++] = opponent.hp / OCTOPOD_INITIAL_HP;
    obs[idx++] = (float)(oppPos - myPos).Length() / 20.0f;
    obs[idx++] = myRot.RotateAxisY().Dot((oppPos - myPos).Normalized());
    obs[idx++] = (robot.hp - opponent.hp) / OCTOPOD_INITIAL_HP;

    // Force readings (simplified - just total contact force)
    if (idx < OCTOPOD_OBS_DIM) {
        obs[idx++] = forces.empty() ? 0.0f : forces[0] / 100.0f;
    }

    // Fill remaining with zeros
    while (idx < OCTOPOD_OBS_DIM) {
        obs[idx++] = 0.0f;
    }
}

void OctopodEnv::CalculateRewards(float& r1, float& r2)
{
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    
    JPH::RVec3 pos1 = bodyInterface.GetPosition(mRobot1.centralBody);
    JPH::RVec3 pos2 = bodyInterface.GetPosition(mRobot2.centralBody);
    JPH::Vec3 vel1 = bodyInterface.GetLinearVelocity(mRobot1.centralBody);
    JPH::Vec3 vel2 = bodyInterface.GetLinearVelocity(mRobot2.centralBody);
    JPH::Quat rot1 = bodyInterface.GetRotation(mRobot1.centralBody);
    JPH::Quat rot2 = bodyInterface.GetRotation(mRobot2.centralBody);

    // --- SHARED REWARDS (Combat/KOTH) ---
    float dist = static_cast<float>((pos2 - pos1).Length());
    
    // Proximity reward (encourages getting close)
    float prox1 = 0.0f;
    if (dist < 20.0f) {
        prox1 = 0.2f * (1.0f - (dist / 20.0f));
    } else {
        prox1 = -0.01f * (dist - 20.0f); // Gentle penalty for being too far
    }

    // Directional approach reward (encourages moving toward opponent)
    JPH::Vec3 toOpponent1 = JPH::Vec3(pos2 - pos1).Normalized();
    JPH::Vec3 toOpponent2 = JPH::Vec3(pos1 - pos2).Normalized();
    float approach1 = vel1.Dot(toOpponent1) * 0.1f;
    float approach2 = vel2.Dot(toOpponent2) * 0.1f;

    // --- STABILITY & WALKING REWARDS (Critical for learning to walk) ---
    auto calcStabilityReward = [&](const JPH::RVec3& pos, const JPH::Quat& rot, const JPH::Vec3& vel, const OctopodRobot& robot) {
        float reward = 0.0f;
        
        // 1. Upright reward (keep body_central up vector aligned with Y)
        JPH::Vec3 up = rot.RotateAxisY();
        float upright = up.Dot(JPH::Vec3::sAxisY()); // 1.0 is perfect, < 0 is upside down
        reward += std::max(0.0f, upright) * 0.5f;
        
        // 2. Height reward (optimal height is ~0.8 - 1.2m)
        float height = (float)pos.GetY();
        if (height > 0.5f && height < 1.5f) {
            reward += 0.3f * (1.0f - std::abs(height - 1.0f) / 0.5f);
        } else if (height <= 0.5f) {
            reward -= 0.2f; // Penalty for touching ground with central body
        }
        
        // 3. Orientation reward (face opponent)
        JPH::Vec3 forward = rot.RotateAxisZ(); // Assuming Z is forward in local space
        JPH::Vec3 toOpp = (robot.robotIndex == 0) ? toOpponent1 : toOpponent2;
        float facing = forward.Dot(toOpp);
        reward += std::max(0.0f, facing) * 0.2f;

        // 4. Forward velocity reward (moving is good)
        float speed = vel.Length();
        reward += std::min(speed, 5.0f) * 0.1f;

        return reward;
    };

    float stability1 = calcStabilityReward(pos1, rot1, vel1, mRobot1);
    float stability2 = calcStabilityReward(pos2, rot2, vel2, mRobot2);

    // Wall penalty
    auto calcWallPenalty = [](const JPH::RVec3& p) {
        float px = std::abs(p.GetX());
        float pz = std::abs(p.GetZ());
        float maxD = std::max(px, pz);
        if (maxD > 45.0f) return -0.5f * (maxD - 45.0f); // Stiffer penalty for edge of 100x100 arena
        return 0.0f;
    };
    float wallPenalty1 = calcWallPenalty(pos1);
    float wallPenalty2 = calcWallPenalty(pos2);

    // Damage rewards
    float deltaDmgTaken1 = std::max(0.0f, mPrevHp1 - mRobot1.hp);
    float deltaDmgTaken2 = std::max(0.0f, mPrevHp2 - mRobot2.hp);
    float deltaDmgDealt1 = deltaDmgTaken2;
    float deltaDmgDealt2 = deltaDmgTaken1;

    mPrevHp1 = mRobot1.hp;
    mPrevHp2 = mRobot2.hp;

    VectorReward vr1, vr2;
    vr1.damage_dealt = deltaDmgDealt1 * OCTOPOD_REWARD_DAMAGE_DEALT;
    vr1.damage_taken = deltaDmgTaken1 * OCTOPOD_REWARD_DAMAGE_TAKEN;
    vr2.damage_dealt = deltaDmgDealt2 * OCTOPOD_REWARD_DAMAGE_DEALT;
    vr2.damage_taken = deltaDmgTaken2 * OCTOPOD_REWARD_DAMAGE_TAKEN;

    // Energy efficiency (slight penalty for movement)
    float deltaEnergy1 = std::max(0.0f, mRobot1.totalEnergyUsed - mPrevEnergy1);
    float deltaEnergy2 = std::max(0.0f, mRobot2.totalEnergyUsed - mPrevEnergy2);
    mPrevEnergy1 = mRobot1.totalEnergyUsed;
    mPrevEnergy2 = mRobot2.totalEnergyUsed;

    vr1.energy_used = deltaEnergy1 * OCTOPOD_REWARD_ENERGY;
    vr2.energy_used = deltaEnergy2 * OCTOPOD_REWARD_ENERGY;

    // KOTH reward
    float distToKoth1 = static_cast<float>((pos1 - mKothPoint).Length());
    float distToKoth2 = static_cast<float>((pos2 - mKothPoint).Length());
    if (distToKoth1 < 10.0f || distToKoth2 < 10.0f) {
        if (distToKoth1 < distToKoth2) {
            vr1.koth = 0.2f * (1.0f - distToKoth1 / 10.0f);
            vr2.koth = 0.0f;
        } else {
            vr1.koth = 0.0f;
            vr2.koth = 0.2f * (1.0f - distToKoth2 / 10.0f);
        }
    }

    // Altitude reward (legacy, keep for VectorReward dim)
    vr1.altitude = 0.0f; 
    vr2.altitude = 0.0f;

    // Combine shaping rewards into scalar output
    r1 = vr1.Scalar() + prox1 + approach1 + stability1 + wallPenalty1;
    r2 = vr2.Scalar() + prox1 + approach2 + stability2 + wallPenalty2;
    
    // Normalize final reward to reasonable range
    static int clampCount = 0;
    if (r1 > 1.0f || r1 < -1.0f) {
        if (++clampCount % 1000 == 0) {
            LOG_WARN("Reward clamping detected: r1=%.2f (clamped to %.2f)", r1, std::clamp(r1, -1.0f, 1.0f));
        }
    }
    r1 = std::clamp(r1, -1.0f, 1.0f);
    
    if (r2 > 1.0f || r2 < -1.0f) {
        if (++clampCount % 1000 == 0) {
            LOG_WARN("Reward clamping detected: r2=%.2f (clamped to %.2f)", r2, std::clamp(r2, -1.0f, 1.0f));
        }
    }
    r2 = std::clamp(r2, -1.0f, 1.0f);

    if (!std::isfinite(r1)) { r1 = 0.0f; }
    if (!std::isfinite(r2)) { r2 = 0.0f; }
    
    mReward1 = vr1;
    mReward2 = vr2;
}
