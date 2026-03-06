#include <Jolt/Jolt.h>
#include "AircraftEnv.h"
#include <Jolt/Physics/Body/BodyInterface.h>
#include <cstring>

void AircraftEnv::Init(uint32_t envIndex, JPH::PhysicsSystem* physicsSystem) {
    mEnvIndex = envIndex;
    mPhysicsSystem = physicsSystem;
    Reset();
}

void AircraftEnv::Reset() {
    mAircraft1.Create(mPhysicsSystem, JPH::RVec3(-50.0f, 100.0f, 0.0f), 1); // Layer 1
    mAircraft2.Create(mPhysicsSystem, JPH::RVec3(50.0f, 100.0f, 0.0f), 1);
    mStepCount = 0;
}

void AircraftEnv::QueueActions(const float* actions1, const float* actions2) {
    mAircraft1.ApplyAerodynamics(mPhysicsSystem, actions1);
    mAircraft2.ApplyAerodynamics(mPhysicsSystem, actions2);
}

void AircraftEnv::HarvestState(float* obs1, float* obs2, float* reward1, float* reward2, bool& done) {
    mStepCount++;
    
    JPH::BodyInterface& bodyInterface = mPhysicsSystem->GetBodyInterface();
    
    auto fillObs = [&](float* obs, const Aircraft& ac, const Aircraft& opp) {
        JPH::RVec3 pos = bodyInterface.GetPosition(ac.GetMainBodyId());
        JPH::Vec3 vel = bodyInterface.GetLinearVelocity(ac.GetMainBodyId());
        JPH::Quat rot = bodyInterface.GetRotation(ac.GetMainBodyId());
        
        int idx = 0;
        obs[idx++] = (float)pos.GetX(); obs[idx++] = (float)pos.GetY(); obs[idx++] = (float)pos.GetZ();
        obs[idx++] = vel.GetX(); obs[idx++] = vel.GetY(); obs[idx++] = vel.GetZ();
        // Simplified obs for now
        while (idx < 256) obs[idx++] = 0.0f;
    };

    fillObs(obs1, mAircraft1, mAircraft2);
    fillObs(obs2, mAircraft2, mAircraft1);

    *reward1 = 0.0f;
    *reward2 = 0.0f;
    done = (mStepCount >= mMaxSteps);
}
