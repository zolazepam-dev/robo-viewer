#include "RobotController.h"
#include "RobotConfig.h"
#include "SIMDObservations.h"
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <iostream>

RobotController::RobotController(Robot& robot) : mRobot(robot) {
    int n = robot.config.numSatellites;
    mPidX.resize(n, {200.0f, 5.0f, 50.0f});
    mPidY.resize(n, {200.0f, 5.0f, 50.0f});
    mPidZ.resize(n, {200.0f, 5.0f, 50.0f});
    
    int actionDim = robot.config.actionsPerRobot;
    if (robot.type == RobotType::INTERNAL_ENGINE) actionDim = 56;
    if (actionDim <= 0) actionDim = 56;

    mBaseActions.resize(actionDim, 0.0f);
    mFinalActions.resize(actionDim, 0.0f);
    mResidualActions.resize(actionDim, 0.0f);
    mLidarDistances.resize(robot.config.numLidarRays, 0.0f);
}

void RobotController::ComputeBasePIDActions(JPH::PhysicsSystem* physicsSystem, float dt) {
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    for (int i = 0; i < mRobot.config.numSatellites; ++i) {
        if (i < (int)mRobot.satellites.size() && !mRobot.satellites[i].coreBodyId.IsInvalid()) {
            JPH::Vec3 angVel = bodyInterface.GetAngularVelocity(mRobot.satellites[i].coreBodyId);
            mBaseActions[i * mRobot.config.actionsPerSatellite + 0] = mPidX[i].Compute(0.0f, angVel.GetX(), dt);
            mBaseActions[i * mRobot.config.actionsPerSatellite + 1] = mPidY[i].Compute(0.0f, angVel.GetY(), dt);
            mBaseActions[i * mRobot.config.actionsPerSatellite + 2] = mPidZ[i].Compute(0.0f, angVel.GetZ(), dt);
        }
    }
}

void RobotController::ApplyResidualActions(const float* residualActions, JPH::PhysicsSystem* physicsSystem, float dt) {
    if (mRobot.mainBodyId.IsInvalid()) return;

    int actionDim = (int)mResidualActions.size();
    for (int i = 0; i < actionDim; ++i) {
        if (!std::isfinite(residualActions[i])) {
            mResidualActions[i] = 0.0f;
        } else {
            mResidualActions[i] = std::clamp(residualActions[i], -10.0f, 10.0f);
        }
    }

    if (mRobot.type == RobotType::AIRCRAFT) {
        ApplyAircraftAerodynamics(mResidualActions.data(), physicsSystem, dt);
    } else if (mRobot.type == RobotType::INTERNAL_ENGINE) {
        ApplyShardPhysics(mResidualActions.data(), physicsSystem, dt);
    } else {
        if (!mRobot.config.satellites.empty()) {
            ComputeBasePIDActions(physicsSystem, dt);
            for (int i = 0; i < mRobot.config.numSatellites; ++i) {
                int base = i * mRobot.config.actionsPerSatellite;
                mFinalActions[base + 0] = mBaseActions[base + 0] + mResidualActions[base + 0] * mRobot.config.rotationScale;
                mFinalActions[base + 1] = mBaseActions[base + 1] + mResidualActions[base + 1] * mRobot.config.rotationScale;
                mFinalActions[base + 2] = mBaseActions[base + 2] + mResidualActions[base + 2] * mRobot.config.rotationScale;
                mFinalActions[base + 3] = mResidualActions[base + 3] * mRobot.config.slideScale;
            }
            int satActions = mRobot.config.numSatellites * mRobot.config.actionsPerSatellite;
            for (int i = satActions; i < mRobot.config.actionsPerRobot; ++i) mFinalActions[i] = mResidualActions[i];
        } else {
            for (int i = 0; i < actionDim; ++i) mFinalActions[i] = mResidualActions[i];
        }
        ApplyPhysicsActions(mFinalActions.data(), physicsSystem);
    }
}

void RobotController::ApplyPhysicsActions(const float* actions, JPH::PhysicsSystem* physicsSystem) {
    if (mRobot.mainBodyId.IsInvalid()) return;
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    float energy = 0.0f;
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> jitter(-0.05f, 0.05f);
if (!mRobot.hingeJoints.empty() || !mRobot.sixDofJoints.empty()) {
    int numHinge = static_cast<int>(mRobot.hingeJoints.size());
    for (int i = 0; i < numHinge; ++i) {
        if (mRobot.hingeJoints[i]) {
            float targetVel = actions[i] * 15.0f + jitter(rng);
            mRobot.hingeJoints[i]->SetTargetAngularVelocity(targetVel);
        }
        energy += std::abs(actions[i]);
    }

    int actionIdx = numHinge;
    for (auto* joint : mRobot.sixDofJoints) {
        if (joint) {
            JPH::Vec3 targetAngVel(actions[actionIdx] * 10.0f, actions[actionIdx+1] * 10.0f, actions[actionIdx+2] * 10.0f);
            joint->SetTargetAngularVelocityCS(targetAngVel + JPH::Vec3(jitter(rng), jitter(rng), jitter(rng)));
            energy += targetAngVel.Length();
        }
        actionIdx += 3;
    }

    int base = actionIdx;
    if (mRobot.config.reactionWheelDim >= 3 && base + 2 < mRobot.config.actionsPerRobot) {
        JPH::Vec3 torque(actions[base] * mRobot.config.reactionTorqueScale, 
                         actions[base+1] * mRobot.config.reactionTorqueScale, 
                         actions[base+2] * mRobot.config.reactionTorqueScale);
        bodyInterface.AddTorque(mRobot.mainBodyId, torque);
    }
} else {
        for (int i = 0; i < mRobot.config.numSatellites; ++i) {
            if (i < (int)mRobot.satellites.size()) {
                if (mRobot.satellites[i].rotationJoint) {
                    JPH::Vec3 vel(actions[i*4], actions[i*4+1], actions[i*4+2]);
                    vel += JPH::Vec3(jitter(rng), jitter(rng), jitter(rng));
                    mRobot.satellites[i].rotationJoint->SetTargetVelocityCS(vel);
                }
                if (mRobot.satellites[i].slideJoint) {
                    mRobot.satellites[i].slideJoint->SetTargetVelocity(actions[i*4+3] + jitter(rng));
                }
            }
            energy += std::abs(actions[i*4]) + std::abs(actions[i*4+1]) + std::abs(actions[i*4+2]) + std::abs(actions[i*4+3]);
        }
        int satActions = mRobot.config.numSatellites * mRobot.config.actionsPerSatellite;
        if (mRobot.config.reactionWheelDim >= 3 && satActions + 2 < mRobot.config.actionsPerRobot) {
            JPH::Vec3 torque(actions[satActions] * mRobot.config.reactionTorqueScale, 
                             actions[satActions+1] * mRobot.config.reactionTorqueScale, 
                             actions[satActions+2] * mRobot.config.reactionTorqueScale);
            bodyInterface.AddTorque(mRobot.mainBodyId, torque);
        }
    }
    mRobot.totalEnergyUsed += energy * 0.001f;
}

void RobotController::ApplyAircraftAerodynamics(const float* actions, JPH::PhysicsSystem* physicsSystem, float dt) {
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    JPH::RMat44 worldTransform = bodyInterface.GetWorldTransform(mRobot.mainBodyId);
    JPH::Quat rot = worldTransform.GetRotation().GetQuaternion();
    float thrust = std::clamp(actions[0], 0.0f, 1.0f) * mRobot.config.thrustMax;
    bodyInterface.AddForce(mRobot.mainBodyId, rot * JPH::Vec3(0, 0, thrust));
}

void RobotController::ApplyShardPhysics(const float* actions, JPH::PhysicsSystem* physicsSystem, float dt) {
    if (mRobot.mainBodyId.IsInvalid()) return;
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    
    JPH::Vec3 torque(actions[0], actions[1], actions[2]);
    bodyInterface.AddTorque(mRobot.mainBodyId, torque * mRobot.config.reactionTorqueScale);
    
    JPH::Vec3 force(actions[3], actions[4], actions[5]);
    JPH::Quat rot = bodyInterface.GetRotation(mRobot.mainBodyId);
    bodyInterface.AddForce(mRobot.mainBodyId, rot * force * mRobot.config.rotationScale * 100.0f);

    for (int i = 0; i < (int)mRobot.satellites.size() && i < 2; ++i) {
        if (mRobot.satellites[i].rotationJoint) {
            float targetVel = actions[6 + i] * 10.0f;
            mRobot.satellites[i].rotationJoint->SetTargetVelocityCS(JPH::Vec3(0, 0, targetVel));
        }
    }
}

void RobotController::PerformLidarScan(JPH::PhysicsSystem* physicsSystem) {
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    JPH::RVec3 pos = bodyInterface.GetPosition(mRobot.mainBodyId);
    JPH::Quat rot = bodyInterface.GetRotation(mRobot.mainBodyId);
    const JPH::NarrowPhaseQuery& query = physicsSystem->GetNarrowPhaseQuery();
    JPH::IgnoreMultipleBodiesFilter filter;
    filter.IgnoreBody(mRobot.mainBodyId);
    for (const auto& s : mRobot.satellites) {
        if (!s.coreBodyId.IsInvalid()) filter.IgnoreBody(s.coreBodyId);
        if (!s.spikeBodyId.IsInvalid()) filter.IgnoreBody(s.spikeBodyId);
    }
    for (const auto& bid : mRobot.bodyIds) {
        if (!bid.IsInvalid()) filter.IgnoreBody(bid);
    }

    auto directions = CreateLidarDirections(mRobot.config.numLidarRays);
    for (int i = 0; i < mRobot.config.numLidarRays; ++i) {
        JPH::RRayCast ray{pos, rot * directions[i] * mRobot.config.lidarMaxDistance};
        JPH::RayCastResult res;
        if (query.CastRay(ray, res, {}, {}, filter)) {
            mLidarDistances[i] = res.mFraction * mRobot.config.lidarMaxDistance;
        } else {
            mLidarDistances[i] = mRobot.config.lidarMaxDistance;
        }
    }
}

std::vector<JPH::Vec3> RobotController::CreateLidarDirections(int numRays) {
    std::vector<JPH::Vec3> dirs;
    for (int i = 0; i < numRays; ++i) {
        float angle = 2.0f * 3.14159f * i / numRays;
        dirs.emplace_back(std::cos(angle), 0.0f, std::sin(angle));
    }
    return dirs;
}

void RobotController::GetObservations(const Robot& opponent, float* obs, const ForceSensorReading& forces, JPH::PhysicsSystem* physicsSystem) {
    if (mRobot.mainBodyId.IsInvalid()) return;
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    int idx = 0;
    const int maxObs = mRobot.config.observationDim;

    // Safety check - prevent buffer overflow
    if (maxObs <= 0 || obs == nullptr) {
        std::cerr << "[RobotController] ERROR: Invalid observation buffer (maxObs=" << maxObs << ")" << std::endl;
        return;
    }
    
    // === BASE OBSERVATIONS ===
    
    // 1-2. Self + Opponent body state (18 floats) - SIMD PACKED
    JPH::RVec3 myPos = bodyInterface.GetPosition(mRobot.mainBodyId);
    JPH::Vec3 myVel = bodyInterface.GetLinearVelocity(mRobot.mainBodyId);
    JPH::Vec3 myAngVel = bodyInterface.GetAngularVelocity(mRobot.mainBodyId);
    JPH::RVec3 oppPos = bodyInterface.GetPosition(opponent.mainBodyId);
    JPH::Vec3 oppVel = bodyInterface.GetLinearVelocity(opponent.mainBodyId);
    JPH::Vec3 oppAngVel = bodyInterface.GetAngularVelocity(opponent.mainBodyId);
    
    if (idx + 18 <= maxObs) {
        // Pack 18 floats using AVX2 (two 9-float groups)
        simd::Pack9Floats(obs, idx,
                         (float)myPos.GetX(), (float)myPos.GetY(), (float)myPos.GetZ(),
                         myVel.GetX(), myVel.GetY(), myVel.GetZ(),
                         myAngVel.GetX(), myAngVel.GetY(), myAngVel.GetZ());
        idx += 9;
        
        simd::Pack9Floats(obs, idx,
                         (float)oppPos.GetX(), (float)oppPos.GetY(), (float)oppPos.GetZ(),
                         oppVel.GetX(), oppVel.GetY(), oppVel.GetZ(),
                         oppAngVel.GetX(), oppAngVel.GetY(), oppAngVel.GetZ());
        idx += 9;
    }
    
    // 3. Relative position and velocity (compound: opponent - self)
    JPH::Vec3 relPos = JPH::Vec3(oppPos - myPos);
    JPH::Vec3 relVel = oppVel - myVel;
    float distance = relPos.Length();
    
    if (idx + 6 <= maxObs) {
        obs[idx++] = relPos.GetX(); obs[idx++] = relPos.GetY(); obs[idx++] = relPos.GetZ();
        obs[idx++] = relVel.GetX(); obs[idx++] = relVel.GetY(); obs[idx++] = relVel.GetZ();
    }
    
    // 4. Distance metrics (compound: normalized)
    const float ARENA_RADIUS = 30.0f;
    if (idx + 4 <= maxObs) {
        obs[idx++] = distance / ARENA_RADIUS;  // Normalized distance to opponent
        obs[idx++] = 1.0f / (distance + 0.1f);  // Inverse distance (attention-like)
        obs[idx++] = distance * distance;  // Squared distance (for energy-based rewards)
        obs[idx++] = std::sqrt(distance);  // Square root distance (diminishing returns)
    }
    
    // 5. Direction vectors (compound: normalized)
    JPH::Vec3 dirToOpponent = (distance > 0.01f) ? (relPos / distance) : JPH::Vec3(0, 0, 0);
    JPH::Vec3 myForward = JPH::Vec3(0, 0, 1);  // Assuming Z is forward
    JPH::Vec3 oppForward = JPH::Vec3(0, 0, 1);
    
    if (idx + 6 <= maxObs) {
        obs[idx++] = dirToOpponent.GetX(); obs[idx++] = dirToOpponent.GetY(); obs[idx++] = dirToOpponent.GetZ();
        obs[idx++] = myForward.GetX(); obs[idx++] = myForward.GetY(); obs[idx++] = myForward.GetZ();
    }
    
    // 6. Facing angles (compound: dot products)
    float facingDot = myForward.Dot(dirToOpponent);
    float oppFacingDot = oppForward.Dot(-dirToOpponent);
    float relativeFacing = myForward.Dot(oppForward);
    
    if (idx + 6 <= maxObs) {
        obs[idx++] = facingDot;  // Am I facing opponent? [-1, 1]
        obs[idx++] = oppFacingDot;  // Is opponent facing me? [-1, 1]
        obs[idx++] = relativeFacing;  // Are we facing same direction? [-1, 1]
        obs[idx++] = std::abs(facingDot);  // Absolute facing (0 = perpendicular, 1 = aligned)
        obs[idx++] = (facingDot + 1.0f) * 0.5f;  // Scaled to [0, 1]
        obs[idx++] = std::max(0.0f, facingDot);  // ReLU facing (only positive)
    }
    
    // 7. Satellite observations
    for (int i = 0; i < (int)mRobot.satellites.size() && idx + 12 <= maxObs; ++i) {
        auto& sat = mRobot.satellites[i];
        if (sat.coreBodyId.IsInvalid()) continue;
        
        JPH::RVec3 satPos = bodyInterface.GetPosition(sat.coreBodyId);
        JPH::Vec3 satVel = bodyInterface.GetLinearVelocity(sat.coreBodyId);
        JPH::Vec3 satRelPos = JPH::Vec3(satPos - myPos);
        JPH::Vec3 satRelToOpp = JPH::Vec3(satPos - oppPos);
        
        obs[idx++] = satRelPos.GetX(); obs[idx++] = satRelPos.GetY(); obs[idx++] = satRelPos.GetZ();
        obs[idx++] = satVel.GetX(); obs[idx++] = satVel.GetY(); obs[idx++] = satVel.GetZ();
        obs[idx++] = satRelToOpp.Length() / ARENA_RADIUS;  // Distance to opponent
        obs[idx++] = satRelPos.Length() / ARENA_RADIUS;  // Distance from center
        obs[idx++] = satRelPos.Dot(myVel) / (satRelPos.Length() + 0.1f);  // Radial velocity
        obs[idx++] = (float)i / mRobot.satellites.size();  // Satellite index (normalized)
    }
    
    // 8. LIDAR readings
    PerformLidarScan(physicsSystem);
    for (int i = 0; i < mRobot.config.numLidarRays && idx < maxObs; ++i) {
        obs[idx++] = mLidarDistances[i] / mRobot.config.lidarMaxDistance;
    }
    
    // 9. Compound LIDAR features
    if (idx + 8 <= maxObs && mRobot.config.numLidarRays > 0) {
        float minLidar = mLidarDistances[0];
        float maxLidar = mLidarDistances[0];
        float avgLidar = 0.0f;
        for (int i = 0; i < mRobot.config.numLidarRays; ++i) {
            if (mLidarDistances[i] < minLidar) minLidar = mLidarDistances[i];
            if (mLidarDistances[i] > maxLidar) maxLidar = mLidarDistances[i];
            avgLidar += mLidarDistances[i];
        }
        avgLidar /= mRobot.config.numLidarRays;
        
        obs[idx++] = minLidar / mRobot.config.lidarMaxDistance;  // Closest obstacle
        obs[idx++] = maxLidar / mRobot.config.lidarMaxDistance;  // Furthest obstacle
        obs[idx++] = avgLidar / mRobot.config.lidarMaxDistance;  // Average distance
        obs[idx++] = (maxLidar - minLidar) / mRobot.config.lidarMaxDistance;  // Variance proxy
        obs[idx++] = (minLidar < 2.0f) ? 1.0f : 0.0f;  // Binary: very close obstacle
        obs[idx++] = (avgLidar < 5.0f) ? 1.0f : 0.0f;  // Binary: crowded space
    }
    
    // 10. HP and energy
    if (idx + 8 <= maxObs) {
        float myHPNorm = mRobot.hp / 100.0f;
        float oppHPNorm = opponent.hp / 100.0f;
        float hpDiff = myHPNorm - oppHPNorm;
        float hpRatio = (oppHPNorm > 0.01f) ? (myHPNorm / oppHPNorm) : 3.0f;
        
        obs[idx++] = myHPNorm;
        obs[idx++] = oppHPNorm;
        obs[idx++] = hpDiff;  // HP advantage
        obs[idx++] = std::clamp(hpRatio, 0.0f, 3.0f) / 3.0f;  // Normalized HP ratio
        obs[idx++] = (hpDiff > 0.0f) ? 1.0f : 0.0f;  // Binary: HP advantage
        obs[idx++] = mRobot.totalEnergyUsed / 1000.0f;  // Normalized energy used
        obs[idx++] = opponent.totalEnergyUsed / 1000.0f;
        obs[idx++] = (mRobot.totalEnergyUsed - opponent.totalEnergyUsed) / 1000.0f;
    }
    
    // 11. Damage statistics (compound)
    if (idx + 6 <= maxObs) {
        float dmgDealtNorm = mRobot.totalDamageDealt / 100.0f;
        float dmgTakenNorm = mRobot.totalDamageTaken / 100.0f;
        float dmgDiff = dmgDealtNorm - dmgTakenNorm;
        
        obs[idx++] = dmgDealtNorm;
        obs[idx++] = dmgTakenNorm;
        obs[idx++] = dmgDiff;  // Damage trade advantage
        obs[idx++] = (dmgTakenNorm > 0.01f) ? (dmgDealtNorm / dmgTakenNorm) : 1.0f;
        obs[idx++] = std::clamp(dmgDealtNorm - dmgTakenNorm, -1.0f, 1.0f);  // Clamped trade
        obs[idx++] = (dmgDiff > 0.0f) ? 1.0f : 0.0f;  // Binary: winning trades
    }
    
    // 12. Velocity-based features (compound)
    if (idx + 8 <= maxObs) {
        float mySpeed = myVel.Length();
        float oppSpeed = oppVel.Length();
        float speedDiff = mySpeed - oppSpeed;
        float myKE = 0.5f * 30.0f * mySpeed * mySpeed;  // Kinetic energy proxy (assume 30kg base mass)
        
        obs[idx++] = mySpeed / 20.0f;  // Normalized speed
        obs[idx++] = oppSpeed / 20.0f;
        obs[idx++] = speedDiff / 20.0f;  // Speed advantage
        obs[idx++] = (mySpeed > oppSpeed) ? 1.0f : 0.0f;  // Binary: faster
        obs[idx++] = myKE / 1000.0f;  // Normalized kinetic energy
        obs[idx++] = (myKE > 100.0f) ? 1.0f : 0.0f;  // Binary: high energy
        obs[idx++] = mySpeed * distance;  // Momentum proxy
        obs[idx++] = (mySpeed > 5.0f && facingDot > 0.5f) ? 1.0f : 0.0f;  // Attack charge
    }
    
    // 13. Arena positioning (compound)
    if (idx + 6 <= maxObs) {
        float myHeight = (float)myPos.GetY();
        float oppHeight = (float)oppPos.GetY();
        float heightDiff = myHeight - oppHeight;
        float myDistFromCenter = std::sqrt(myPos.GetX() * myPos.GetX() + myPos.GetZ() * myPos.GetZ());
        
        obs[idx++] = myHeight / 20.0f;  // Normalized height
        obs[idx++] = oppHeight / 20.0f;
        obs[idx++] = heightDiff / 20.0f;  // Height advantage
        obs[idx++] = (myHeight > oppHeight) ? 1.0f : 0.0f;  // Binary: higher
        obs[idx++] = myDistFromCenter / ARENA_RADIUS;  // Distance from center
        obs[idx++] = (myDistFromCenter > ARENA_RADIUS * 0.8f) ? 1.0f : 0.0f;  // Near wall
    }
    
    // 14. Time and episode info
    static int episodeStep = 0;
    episodeStep++;
    if (idx + 3 <= maxObs) {
        obs[idx++] = episodeStep / 500.0f;  // Normalized episode progress
        obs[idx++] = std::sin(episodeStep * 0.01f);  // Sinusoidal time encoding
        obs[idx++] = std::cos(episodeStep * 0.01f);  // Cosine time encoding
    }
    
    // 15. Angular features (compound)
    if (idx + 6 <= maxObs) {
        float myAngSpeed = myAngVel.Length();
        float oppAngSpeed = oppAngVel.Length();
        
        obs[idx++] = myAngSpeed / 10.0f;  // Normalized angular speed
        obs[idx++] = oppAngSpeed / 10.0f;
        obs[idx++] = (myAngSpeed > 5.0f) ? 1.0f : 0.0f;  // Binary: spinning
        obs[idx++] = myAngVel.GetY() / 10.0f;  // Yaw rate
        obs[idx++] = (myAngVel.GetY() > 2.0f) ? 1.0f : 0.0f;  // Binary: turning fast
    }
    
    // 16. Advanced compound features
    if (idx + 10 <= maxObs) {
        float mySpeed = myVel.Length();
        float approachSpeed = -relVel.Dot(dirToOpponent);  // Positive = closing in
        
        obs[idx++] = approachSpeed / 20.0f;  // Closing speed
        obs[idx++] = (approachSpeed > 5.0f) ? 1.0f : 0.0f;  // Binary: charging
        obs[idx++] = facingDot * (1.0f / (distance + 0.1f));  // Aim * proximity
        obs[idx++] = mySpeed * facingDot;  // Speed * aim (charge metric)
        obs[idx++] = (mySpeed > 5.0f && distance < 10.0f) ? 1.0f : 0.0f;  // Attack ready
        obs[idx++] = (distance < 5.0f && facingDot > 0.5f) ? 1.0f : 0.0f;  // In range & aimed
        obs[idx++] = (distance < 3.0f) ? 1.0f : 0.0f;  // Binary: melee range
        obs[idx++] = (distance < 10.0f) ? 1.0f : 0.0f;  // Binary: mid range
        obs[idx++] = std::exp(-distance / 10.0f);  // Exponential decay attention
        obs[idx++] = 1.0f / (1.0f + distance / 10.0f);  // Sigmoid-like attention
    }
    
    // CRITICAL: Ensure we never write beyond buffer
    // Pad with zeros up to dimension, but NEVER exceed maxObs
    while (idx < maxObs && idx < mRobot.config.observationDim) {
        obs[idx++] = 0.0f;
    }
    
    // Final safety check
    if (idx > mRobot.config.observationDim) {
        std::cerr << "[RobotController] FATAL: Observation overflow! idx=" << idx 
                  << ", maxObs=" << mRobot.config.observationDim << std::endl;
        std::abort();
    }
}
