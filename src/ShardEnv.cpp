/**
 * @file ShardEnv.cpp
 * @brief Environment for orbital shard robot training
 */

#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include "ShardEnv.h"
#include "RobotLoader.h"

#include <iostream>
#include <cmath>
#include <random>

using namespace JPH;

void ShardEnv::Init(uint32_t envIndex, JPH::PhysicsSystem* physicsSystem, int stepsPerEpisode) {
    mEnvIndex = envIndex;
    mPhysicsSystem = physicsSystem;
    mStepsPerEpisode = stepsPerEpisode;
    
    Reset();
}

void ShardEnv::Reset() {
    mStepCount = 0;
    mDone = false;
    mRobot1.hp = SHARD_INITIAL_HP;
    mRobot2.hp = SHARD_INITIAL_HP;
    
    BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    
    // Remove existing bodies
    auto removeRobot = [&](ShardRobot& robot) {
        for (auto* constraint : robot.constraints) {
            if (constraint) {
                mPhysicsSystem->RemoveConstraint(constraint);
            }
        }
        for (const auto& bodyId : robot.bodies) {
            if (!bodyId.IsInvalid()) {
                bodyInterface.RemoveBody(bodyId);
                bodyInterface.DestroyBody(bodyId);
            }
        }
        robot.bodies.clear();
        robot.constraints.clear();
        robot.chassis = BodyID();
    };
    
    removeRobot(mRobot1);
    removeRobot(mRobot2);
    
    // Load shards from JSON
    RobotLoader loader;
    auto loaded1 = loader.LoadRobot("/home/cammyz/robo-viewer/robots/orbital_shard.json", mPhysicsSystem);
    mRobot1.bodies = loaded1.bodies;
    mRobot1.constraints = loaded1.constraints;
    mRobot1.chassis = loaded1.bodies.empty() ? JPH::BodyID() : loaded1.bodies[0];
    mRobot1.envIndex = mEnvIndex;
    mRobot1.robotIndex = 0;
    
    auto loaded2 = loader.LoadRobot("/home/cammyz/robo-viewer/robots/orbital_shard.json", mPhysicsSystem);
    mRobot2.bodies = loaded2.bodies;
    mRobot2.constraints = loaded2.constraints;
    mRobot2.chassis = loaded2.bodies.empty() ? JPH::BodyID() : loaded2.bodies[0];
    mRobot2.envIndex = mEnvIndex;
    mRobot2.robotIndex = 1;
    
    // Reposition robots to opposite sides
    JPH::RVec3 pos1(-10.0f, 10.0f, 0.0f);
    JPH::RVec3 pos2(10.0f, 10.0f, 0.0f);
    
    if (!mRobot1.chassis.IsInvalid()) {
        bodyInterface.SetPosition(mRobot1.chassis, pos1, JPH::EActivation::Activate);
        bodyInterface.SetLinearVelocity(mRobot1.chassis, JPH::Vec3::sZero());
        bodyInterface.SetAngularVelocity(mRobot1.chassis, JPH::Vec3::sZero());
    }
    if (!mRobot2.chassis.IsInvalid()) {
        bodyInterface.SetPosition(mRobot2.chassis, pos2, JPH::EActivation::Activate);
        bodyInterface.SetLinearVelocity(mRobot2.chassis, JPH::Vec3::sZero());
        bodyInterface.SetAngularVelocity(mRobot2.chassis, JPH::Vec3::sZero());
    }
}

void ShardEnv::QueueActions(const float* actions1, const float* actions2) {
    if (mDone) return;
    
    // Apply motor actions to constraints (velocity control)
    float velocityScale = 5.0f;  // radians per second
    
    for (size_t i = 0; i < mRobot1.constraints.size() && i < SHARD_ACTION_DIM; i++) {
        if (mRobot1.constraints[i]) {
            // Cast to HingeConstraint if possible
            JPH::HingeConstraint* hinge = dynamic_cast<JPH::HingeConstraint*>(mRobot1.constraints[i]);
            if (hinge) {
                float targetVelocity = actions1[i] * velocityScale;
                hinge->SetMotorState(EMotorState::Velocity);
                hinge->SetTargetAngularVelocity(targetVelocity);
            }
        }
    }
    
    for (size_t i = 0; i < mRobot2.constraints.size() && i < SHARD_ACTION_DIM; i++) {
        if (mRobot2.constraints[i]) {
            JPH::HingeConstraint* hinge = dynamic_cast<JPH::HingeConstraint*>(mRobot2.constraints[i]);
            if (hinge) {
                float targetVelocity = actions2[i] * velocityScale;
                hinge->SetMotorState(EMotorState::Velocity);
                hinge->SetTargetAngularVelocity(targetVelocity);
            }
        }
    }
}

void ShardEnv::HarvestState(float* obs1, float* obs2, float* reward1, float* reward2, bool& done) {
    if (mDone) {
        done = true;
        return;
    }
    
    mStepCount++;
    
    BuildObservationVector(obs1, mRobot1, mRobot2);
    BuildObservationVector(obs2, mRobot2, mRobot1);
    
    CalculateRewards(*reward1, *reward2);
    
    if (mRobot1.hp <= 0.0f || mRobot2.hp <= 0.0f || mStepCount >= mStepsPerEpisode) {
        mDone = true;
    }
    
    done = mDone;
}

void ShardEnv::BuildObservationVector(float* obs, const ShardRobot& robot, const ShardRobot& opponent) {
    BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    
    int idx = 0;
    
    // Chassis state (position, velocity, angular velocity)
    if (!robot.chassis.IsInvalid()) {
        RVec3 pos = bodyInterface.GetPosition(robot.chassis);
        Vec3 linVel = bodyInterface.GetLinearVelocity(robot.chassis);
        Vec3 angVel = bodyInterface.GetAngularVelocity(robot.chassis);
        Quat rot = bodyInterface.GetRotation(robot.chassis);
        
        // Normalized position (-1 to 1)
        obs[idx++] = pos.GetX() / 50.0f;
        obs[idx++] = (pos.GetY() - 10.0f) / 20.0f;  // Center around spawn height
        obs[idx++] = pos.GetZ() / 50.0f;
        
        // Velocity (normalized)
        obs[idx++] = linVel.GetX() / 20.0f;
        obs[idx++] = linVel.GetY() / 20.0f;
        obs[idx++] = linVel.GetZ() / 20.0f;
        
        // Angular velocity (normalized)
        obs[idx++] = angVel.GetX() / 10.0f;
        obs[idx++] = angVel.GetY() / 10.0f;
        obs[idx++] = angVel.GetZ() / 10.0f;
        
        // Rotation (quaternion)
        obs[idx++] = rot.GetX();
        obs[idx++] = rot.GetY();
        obs[idx++] = rot.GetZ();
        obs[idx++] = rot.GetW();
    } else {
        for (int i = 0; i < 14; i++) obs[idx++] = 0.0f;
    }
    
    // Wing states (simplified - just zeros for now)
    for (int i = 0; i < 4; i++) {
        obs[idx++] = 0.0f;
        obs[idx++] = 0.0f;
    }
    
    // Opponent relative position
    if (!robot.chassis.IsInvalid() && !opponent.chassis.IsInvalid()) {
        RVec3 myPos = bodyInterface.GetPosition(robot.chassis);
        RVec3 oppPos = bodyInterface.GetPosition(opponent.chassis);
        Vec3 relPos = oppPos - myPos;
        obs[idx++] = relPos.GetX() / 50.0f;
        obs[idx++] = relPos.GetY() / 20.0f;
        obs[idx++] = relPos.GetZ() / 50.0f;
    } else {
        obs[idx++] = 0.0f;
        obs[idx++] = 0.0f;
        obs[idx++] = 0.0f;
    }
    
    // HP
    obs[idx++] = robot.hp / SHARD_INITIAL_HP;
    obs[idx++] = opponent.hp / SHARD_INITIAL_HP;
    
    // Fill remaining with zeros
    while (idx < SHARD_OBS_DIM) {
        obs[idx++] = 0.0f;
    }
}

void ShardEnv::CalculateRewards(float& r1, float& r2) {
    r1 = 0.0f;
    r2 = 0.0f;
    
    // Damage reward
    float damage1 = SHARD_INITIAL_HP - mRobot1.hp;
    float damage2 = SHARD_INITIAL_HP - mRobot2.hp;
    
    r1 -= damage1 * 0.01f;  // Penalty for taking damage
    r2 -= damage2 * 0.01f;
    
    // Survival reward
    r1 += 0.01f;
    r2 += 0.01f;
    
    // Wing movement reward (encourage active control) - simplified
    r1 += 0.01f;
    r2 += 0.01f;
    
    // Win bonus
    if (mRobot2.hp <= 0.0f && mRobot1.hp > 0.0f) {
        r1 += 1.0f;
    }
    if (mRobot1.hp <= 0.0f && mRobot2.hp > 0.0f) {
        r2 += 1.0f;
    }
}
