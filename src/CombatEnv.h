#pragma once

#include <Jolt/Jolt.h>
#include <vector>
#include <array>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/ContactListener.h>

#include "PhysicsCore.h"
#include "NeuralMath.h"
#include "NeuralNetwork.h"
#include "CombatRobot.h"

constexpr float ARENA_SIZE = 12.0f;
constexpr float ARENA_HALF = ARENA_SIZE * 0.5f;
constexpr float ROBOT_SPAWN_OFFSET = 3.0f;
constexpr float INITIAL_HP = 100.0f;
constexpr float DAMAGE_MULTIPLIER = 5.0f;
constexpr int MAX_EPISODE_STEPS = 120000;

constexpr int FORCE_SENSOR_DIM = NUM_SATELLITES * 2;
constexpr int OBSERVATION_BASE_DIM = 18 + (NUM_SATELLITES * 6) + (NUM_SATELLITES * 3) + (NUM_SATELLITES * 3);

struct ForceSensorReading
{
    float impulseMagnitude[NUM_SATELLITES] = {0.0f};
    float jointStress[NUM_SATELLITES] = {0.0f};

    void Reset()
    {
        for (int i = 0; i < NUM_SATELLITES; ++i)
        {
            impulseMagnitude[i] = 0.0f;
            jointStress[i] = 0.0f;
        }
    }
};

struct StepResult
{
    std::vector<float> obs_robot1;
    std::vector<float> obs_robot2;
    VectorReward reward1;
    VectorReward reward2;
    ForceSensorReading forces1;
    ForceSensorReading forces2;
    bool done = false;
    int winner = 0;
};

class CombatContactListener : public JPH::ContactListener
{
public:
    CombatContactListener() = default;

    void OnContactAdded(const JPH::Body& body1, const JPH::Body& body2,
                        const JPH::ContactManifold& manifold, JPH::ContactSettings& settings) override;
    void OnContactPersisted(const JPH::Body& body1, const JPH::Body& body2,
                            const JPH::ContactManifold& manifold, JPH::ContactSettings& settings) override;
    void OnContactRemoved(const JPH::SubShapeIDPair& subShapePair) override;

    void SetActiveEnv(uint32_t envIndex) { mActiveEnv = envIndex; }
    const ForceSensorReading& GetForceReadings(uint32_t envIndex, int robotIdx) const
    {
        return mForceReadings[envIndex][robotIdx];
    }

    void ResetForceReadings(uint32_t envIndex)
    {
        mForceReadings[envIndex][0].Reset();
        mForceReadings[envIndex][1].Reset();
    }

private:
    std::array<std::array<ForceSensorReading, 2>, NUM_PARALLEL_ENVS> mForceReadings;
    uint32_t mActiveEnv = 0;

    void ExtractImpulseData(const JPH::Body& body1, const JPH::Body& body2,
                            const JPH::ContactManifold& manifold);
};

class CombatEnv
{
public:
    CombatEnv() = default;
    ~CombatEnv() = default;

    CombatEnv(const CombatEnv&) = delete;
    CombatEnv& operator=(const CombatEnv&) = delete;
    CombatEnv(CombatEnv&&) = default;
    CombatEnv& operator=(CombatEnv&&) = default;

    void Init(uint32_t envIndex, JPH::PhysicsSystem* globalPhysics, CombatRobotLoader* globalLoader);
    void Reset();

    void QueueActions(const float* actions1, const float* actions2);
    StepResult HarvestState();

    const CombatRobotData& GetRobot1() const { return mRobot1; }
    const CombatRobotData& GetRobot2() const { return mRobot2; }
    int GetStepCount() const { return mStepCount; }
    bool IsDone() const { return mDone; }
    int GetObservationDim() const { return mObservationDim; }
    const ForceSensorReading& GetForceReadings1() const { return mForceReadings1; }
    const ForceSensorReading& GetForceReadings2() const { return mForceReadings2; }

private:
    void CheckCollisions();
    void CalculateRewards(StepResult& result);
    float ComputeAirtime() const;
    float ComputeEnergyUsed(const float* actions, int actionDim) const;
    void UpdateForceSensors();
    void BuildObservationVector(std::vector<float>& obs, const CombatRobotData& robot,
                                 const CombatRobotData& opponent, const ForceSensorReading& forces);

    JPH::PhysicsSystem* mPhysicsSystem = nullptr;
    CombatRobotLoader* mRobotLoader = nullptr;

    CombatRobotData mRobot1;
    CombatRobotData mRobot2;
    uint32_t mEnvIndex = 0;
    int mStepCount = 0;
    bool mDone = false;
    int mObservationDim = OBSERVATION_BASE_DIM + FORCE_SENSOR_DIM;

    float mPrevHp1 = INITIAL_HP;
    float mPrevHp2 = INITIAL_HP;
    float mPrevEnergy1 = 0.0f;
    float mPrevEnergy2 = 0.0f;

    ForceSensorReading mForceReadings1;
    ForceSensorReading mForceReadings2;

    float mAirAccumulator1 = 0.0f;
    float mAirAccumulator2 = 0.0f;
};
