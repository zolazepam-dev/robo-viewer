#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include "Aircraft.h"
#include <vector>

class AircraftEnv {
public:
    AircraftEnv() = default;
    ~AircraftEnv() = default;

    void Init(uint32_t envIndex, JPH::PhysicsSystem* physicsSystem);
    void Reset();
    void QueueActions(const float* actions1, const float* actions2);
    void HarvestState(float* obs1, float* obs2, float* reward1, float* reward2, bool& done);

    const Aircraft& GetAircraft1() const { return mAircraft1; }
    const Aircraft& GetAircraft2() const { return mAircraft2; }

private:
    uint32_t mEnvIndex;
    JPH::PhysicsSystem* mPhysicsSystem;
    Aircraft mAircraft1;
    Aircraft mAircraft2;
    int mStepCount = 0;
    const int mMaxSteps = 7200;
};
