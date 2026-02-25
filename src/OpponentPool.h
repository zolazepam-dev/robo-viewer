#pragma once

#include <vector>
#include <random>
#include <cstdint>

constexpr int MAX_POOL_SIZE = 64;

struct OpponentSnapshot
{
    std::vector<float> actorWeights;
    std::vector<float> actorBiases;
    int64_t stepCount = 0;
    float winRate = 0.0f;
    int snapshotId = 0;
};

class OpponentPool
{
public:
    OpponentPool(int maxPoolSize = MAX_POOL_SIZE);
    
    void Snapshot(const std::vector<float>& weights, const std::vector<float>& biases, int64_t stepCount);
    void SnapshotWithStats(const std::vector<float>& weights, const std::vector<float>& biases, 
                           int64_t stepCount, float winRate);
    
    bool SampleOpponent(std::vector<float>& weights, std::vector<float>& biases, std::mt19937& rng);
    bool SampleOpponentRecent(std::vector<float>& weights, std::vector<float>& biases, 
                              std::mt19937& rng, int recentN = 10);
    
    int Size() const { return mSize; }
    bool Empty() const { return mSize == 0; }
    int Capacity() const { return mMaxPoolSize; }
    
    void Clear();
    
    const OpponentSnapshot* GetSnapshot(int idx) const;
    OpponentSnapshot* GetSnapshot(int idx);
    
    void SetBiasTowardsRecent(float bias) { mRecentBias = bias; }
    float GetBiasTowardsRecent() const { return mRecentBias; }
    
    int64_t GetLatestStepCount() const;
    int64_t GetOldestStepCount() const;

private:
    std::vector<OpponentSnapshot> mPool;
    int mMaxPoolSize;
    int mSize = 0;
    int mIndex = 0;
    int mSnapshotCounter = 0;
    float mRecentBias = 0.7f;
};
