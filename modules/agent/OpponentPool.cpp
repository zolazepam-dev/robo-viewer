#include "OpponentPool.h"

#include <algorithm>

OpponentPool::OpponentPool(int maxPoolSize)
    : mMaxPoolSize(maxPoolSize)
{
    mPool.resize(mMaxPoolSize);
}

void OpponentPool::Snapshot(const std::vector<float>& weights, const std::vector<float>& biases, 
                             int64_t stepCount)
{
    mPool[mIndex].actorWeights = weights;
    mPool[mIndex].actorBiases = biases;
    mPool[mIndex].stepCount = stepCount;
    mPool[mIndex].winRate = 0.0f;
    mPool[mIndex].snapshotId = mSnapshotCounter++;
    
    mIndex = (mIndex + 1) % mMaxPoolSize;
    mSize = std::min(mSize + 1, mMaxPoolSize);
}

void OpponentPool::SnapshotWithStats(const std::vector<float>& weights, const std::vector<float>& biases,
                                      int64_t stepCount, float winRate)
{
    mPool[mIndex].actorWeights = weights;
    mPool[mIndex].actorBiases = biases;
    mPool[mIndex].stepCount = stepCount;
    mPool[mIndex].winRate = winRate;
    mPool[mIndex].snapshotId = mSnapshotCounter++;
    
    mIndex = (mIndex + 1) % mMaxPoolSize;
    mSize = std::min(mSize + 1, mMaxPoolSize);
}

bool OpponentPool::SampleOpponent(std::vector<float>& weights, std::vector<float>& biases, 
                                   std::mt19937& rng)
{
    if (mSize == 0) return false;
    
    std::uniform_int_distribution<int> dist(0, mSize - 1);
    int idx = dist(rng);
    
    weights = mPool[idx].actorWeights;
    biases = mPool[idx].actorBiases;
    
    return true;
}

bool OpponentPool::SampleOpponentRecent(std::vector<float>& weights, std::vector<float>& biases,
                                         std::mt19937& rng, int recentN)
{
    if (mSize == 0) return false;
    
    int effectiveRecent = std::min(recentN, mSize);
    
    std::uniform_real_distribution<float> biasDist(0.0f, 1.0f);
    
    int idx;
    if (biasDist(rng) < mRecentBias)
    {
        std::uniform_int_distribution<int> recentDist(0, effectiveRecent - 1);
        int offset = recentDist(rng);
        idx = (mIndex - 1 - offset + mMaxPoolSize) % mMaxPoolSize;
    }
    else
    {
        std::uniform_int_distribution<int> allDist(0, mSize - 1);
        idx = allDist(rng);
    }
    
    weights = mPool[idx].actorWeights;
    biases = mPool[idx].actorBiases;
    
    return true;
}

void OpponentPool::Clear()
{
    mSize = 0;
    mIndex = 0;
    mSnapshotCounter = 0;
    
    for (auto& snap : mPool)
    {
        snap.actorWeights.clear();
        snap.actorBiases.clear();
        snap.stepCount = 0;
        snap.winRate = 0.0f;
    }
}

const OpponentSnapshot* OpponentPool::GetSnapshot(int idx) const
{
    if (idx < 0 || idx >= mSize) return nullptr;
    return &mPool[idx];
}

OpponentSnapshot* OpponentPool::GetSnapshot(int idx)
{
    if (idx < 0 || idx >= mSize) return nullptr;
    return &mPool[idx];
}

int64_t OpponentPool::GetLatestStepCount() const
{
    if (mSize == 0) return 0;
    int latestIdx = (mIndex - 1 + mMaxPoolSize) % mMaxPoolSize;
    return mPool[latestIdx].stepCount;
}

int64_t OpponentPool::GetOldestStepCount() const
{
    if (mSize == 0) return 0;
    int oldestIdx = (mIndex - mSize + mMaxPoolSize) % mMaxPoolSize;
    return mPool[oldestIdx].stepCount;
}
