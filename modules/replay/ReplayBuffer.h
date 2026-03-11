/**
 * @file ReplayBuffer.h
 * @brief Experience replay buffer for RL training
 * 
 * Optimized for vectorized sampling with Eigen zero-copy views.
 */

#pragma once

#include "../common/Types.h"
#include <vector>
#include <random>
#include <mutex>
#include <cstdint>
#include <Eigen/Core>

class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity, int observationDim, int actionDim);
    ~ReplayBuffer() = default;

    ReplayBuffer(const ReplayBuffer&) = delete;
    ReplayBuffer& operator=(const ReplayBuffer&) = delete;

    void Add(const float* obs1, const float* obs2,
             const float* action1, const float* action2,
             float reward1, float reward2,
             const float* nextObs1, const float* nextObs2,
             bool done);

    // TD3Trainer-compatible Sample with 7 params
    bool Sample(size_t batchSize,
                float* outStates,
                float* outActions,
                float* outRewards,
                float* outNextStates,
                float* outDones,
                std::mt19937& rng);

    // Simple Sample
    bool Sample(size_t batchSize, std::mt19937& rng);

    // OPTIMIZED: Eigen-based zero-copy sampling
    bool SampleEigen(size_t batchSize,
                     Eigen::Map<Eigen::MatrixXf>& outStates,
                     Eigen::Map<Eigen::MatrixXf>& outActions,
                     Eigen::Map<Eigen::VectorXf>& outRewards,
                     Eigen::Map<Eigen::MatrixXf>& outNextStates,
                     Eigen::Map<Eigen::VectorXf>& outDones,
                     std::mt19937& rng);

    size_t Size() const { return mSize; }
    size_t Capacity() const { return mCapacity; }
    bool CanSample(size_t batchSize) const { return mSize >= batchSize; }
    void Clear();

    bool IsReady(size_t batchSize) const { return CanSample(batchSize); }

    void SampleVectorRewards(size_t batchSize,
                             float* outStates,
                             float* outActions,
                             VectorReward* outVectorRewards,
                             float* outNextStates,
                             float* outDones,
                             std::mt19937& rng);

private:
    size_t mCapacity, mSize, mHead;
    int mObservationDim, mActionDim;

    // SoA (Structure of Arrays) storage - pre-allocated to avoid heap fragmentation
    std::vector<float, AlignedAllocator<float, 32>> mObs1Storage;
    std::vector<float, AlignedAllocator<float, 32>> mObs2Storage;
    std::vector<float, AlignedAllocator<float, 32>> mAction1Storage;
    std::vector<float, AlignedAllocator<float, 32>> mAction2Storage;
    std::vector<float, AlignedAllocator<float, 32>> mReward1Storage;
    std::vector<float, AlignedAllocator<float, 32>> mReward2Storage;
    std::vector<float, AlignedAllocator<float, 32>> mNextObs1Storage;
    std::vector<float, AlignedAllocator<float, 32>> mNextObs2Storage;
    std::vector<float, AlignedAllocator<float, 32>> mDoneStorage;

    // OPTIMIZED: Pre-allocated aligned buffers for batched operations
    std::vector<float, AlignedAllocator<float, 32>> mBatchObs1, mBatchObs2;
    std::vector<float, AlignedAllocator<float, 32>> mBatchActions1, mBatchActions2;
    std::vector<float, AlignedAllocator<float, 32>> mBatchRewards1, mBatchRewards2;
    std::vector<float, AlignedAllocator<float, 32>> mBatchNextObs1, mBatchNextObs2;
    std::vector<float, AlignedAllocator<float, 32>> mBatchDones;
    size_t mCurrentBatchSize;

    mutable std::mutex mMutex;
};
