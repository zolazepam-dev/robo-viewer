/**
 * @file CombatEnv.h
 * @brief Combat environment implementation
 * 
 * Implements IEnvironment for 1v1 robot combat.
 * Manages two robots, reward calculation, and episode logic.
 */

#pragma once

#include "IEnvironment.h"
#include "../robot/Robot.h"
#include "../robot/RobotFactory.h"
#include "../robot/RobotController.h"
#include "../physics/PhysicsWorld.h"

class CombatEnv : public IEnvironment {
public:
    CombatEnv() = default;
    ~CombatEnv() override = default;
    
    CombatEnv(const CombatEnv&) = delete;
    CombatEnv& operator=(const CombatEnv&) = delete;
    CombatEnv(CombatEnv&&) = default;
    CombatEnv& operator=(CombatEnv&&) = default;
    
    // IEnvironment interface
    void Init(uint32_t envIndex, void* physicsWorld) override;
    void Reset() override;
    void Step(const float* action1, const float* action2) override;
    void QueueActions(const float* action1, const float* action2) override;
    void GetObs(float* obs1, float* obs2) override;
    float GetReward(int robotIdx) const override;
    const VectorReward& GetVectorReward(int robotIdx) const override;
    bool IsDone() const override;
    int GetObservationDim() const override { return mObservationDim; }
    int GetActionDim() const override { return mActionDim; }
    int GetStepCount() const override { return mStepCount; }
    uint32_t GetEnvIndex() const override { return mEnvIndex; }
    
    /** @brief Get robot data (non-const for debugging) */
    Robot& GetRobot1() { return mRobot1; }
    Robot& GetRobot2() { return mRobot2; }
    const Robot& GetRobot1() const { return mRobot1; }
    const Robot& GetRobot2() const { return mRobot2; }

private:
    void CalculateRewards();
    float ComputeAirtime() const;
    float ComputeEnergyUsed(const float* actions, int actionDim) const;
    void UpdateForceSensors();
    void BuildObservationVector(float* obs, const Robot& robot,
                                 const Robot& opponent, const ForceSensorReading& forces);
    
    PhysicsWorld* mPhysicsWorld = nullptr;
    RobotFactory mRobotFactory;
    RobotController mRobotController;
    
    Robot mRobot1;
    Robot mRobot2;
    
    uint32_t mEnvIndex = 0;
    int mStepCount = 0;
    int mStepsPerEpisode = MAX_EPISODE_STEPS;
    bool mDone = false;
    int mObservationDim = DEFAULT_OBSERVATION_DIM;
    int mActionDim = DEFAULT_ACTION_DIM;
    
    JPH::RVec3 mKothPoint{0, 0, 0};
    JPH::BodyID mKothVisualId;
    
    float mPrevHP1 = INITIAL_HP;
    float mPrevHP2 = INITIAL_HP;
    float mPrevEnergy1 = 0.0f;
    float mPrevEnergy2 = 0.0f;
    
    float mAirAccumulator1 = 0.0f;
    float mAirAccumulator2 = 0.0f;
    
    VectorReward mReward1;
    VectorReward mReward2;
    
    std::array<float, VECTOR_REWARD_DIM> mPreferenceVector = {0.5f, 0.3f, 0.1f, 0.05f, 0.05f};
};
