#pragma once

// MUST BE FIRST
#include <Jolt/Jolt.h>
#include <vector>
#include <Jolt/Physics/PhysicsSystem.h>

#include "CombatRobot.h"

constexpr float ARENA_SIZE = 12.0f;
constexpr float ARENA_HALF = ARENA_SIZE * 0.5f;
constexpr float ROBOT_SPAWN_OFFSET = 3.0f;
constexpr float INITIAL_HP = 100.0f;
constexpr float DAMAGE_MULTIPLIER = 5.0f;
constexpr int MAX_EPISODE_STEPS = 120000;

struct StepResult
{
    std::vector<float> obs_robot1;
    std::vector<float> obs_robot2;
    float reward1 = 0.0f;
    float reward2 = 0.0f;
    bool done = false;
    int winner = 0; // 0 = draw/ongoing, 1 = robot1 wins, 2 = robot2 wins
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

private:
    void CheckCollisions();
    void CalculateRewards(StepResult& result);

    JPH::PhysicsSystem* mPhysicsSystem = nullptr;
    CombatRobotLoader* mRobotLoader = nullptr;

    CombatRobotData mRobot1;
    CombatRobotData mRobot2;
 uint32_t mEnvIndex = 0;
    int mStepCount = 0;
    bool mDone = false;
    int mObservationDim = 66;

    float mPrevHp1 = INITIAL_HP;
    float mPrevHp2 = INITIAL_HP;
};