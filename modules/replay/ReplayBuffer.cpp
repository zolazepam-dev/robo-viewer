/**
 * @file ReplayBuffer.cpp
 * @brief Implementation of ReplayBuffer
 * 
 * Optimized for vectorized sampling with Eigen zero-copy views.
 */

#include "ReplayBuffer.h"
#include "modules/common/PerformanceDiagnoser.h"
#include <cstring>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>

ReplayBuffer::ReplayBuffer(size_t capacity, int observationDim, int actionDim)
    : mCapacity(capacity), mSize(0), mHead(0)
    , mObservationDim(observationDim), mActionDim(actionDim), mCurrentBatchSize(0)
{
    // Pre-allocate ALL storage buffers
    mObs1Storage.resize(capacity * observationDim);
    mObs2Storage.resize(capacity * observationDim);
    mAction1Storage.resize(capacity * actionDim);
    mAction2Storage.resize(capacity * actionDim);
    mReward1Storage.resize(capacity);
    mReward2Storage.resize(capacity);
    mNextObs1Storage.resize(capacity * observationDim);
    mNextObs2Storage.resize(capacity * observationDim);
    mDoneStorage.resize(capacity);

    size_t obsSize = capacity * observationDim * 2;
    size_t actionSize = capacity * actionDim * 2;
    
    // OPTIMIZED: Use aligned allocator for better SIMD performance
    mBatchObs1.resize(obsSize); mBatchObs2.resize(obsSize);
    mBatchActions1.resize(actionSize); mBatchActions2.resize(actionSize);
    mBatchRewards1.resize(capacity); mBatchRewards2.resize(capacity);
    mBatchNextObs1.resize(obsSize); mBatchNextObs2.resize(obsSize);
    mBatchDones.resize(capacity);
}

void ReplayBuffer::Add(const float* obs1, const float* obs2,
                        const float* action1, const float* action2,
                        float reward1, float reward2,
                        const float* nextObs1, const float* nextObs2,
                        bool done) {
    DIAGNOSE_SCOPE("ReplayBuffer: Add");
    DIAGNOSE_MUTEX_LOCK(mMutex, "ReplayBuffer: Add");
    
    size_t obsOffset = mHead * mObservationDim;
    size_t actionOffset = mHead * mActionDim;

    std::memcpy(mObs1Storage.data() + obsOffset, obs1, mObservationDim * sizeof(float));
    std::memcpy(mObs2Storage.data() + obsOffset, obs2, mObservationDim * sizeof(float));
    std::memcpy(mAction1Storage.data() + actionOffset, action1, mActionDim * sizeof(float));
    std::memcpy(mAction2Storage.data() + actionOffset, action2, mActionDim * sizeof(float));
    mReward1Storage[mHead] = reward1;
    mReward2Storage[mHead] = reward2;
    std::memcpy(mNextObs1Storage.data() + obsOffset, nextObs1, mObservationDim * sizeof(float));
    std::memcpy(mNextObs2Storage.data() + obsOffset, nextObs2, mObservationDim * sizeof(float));
    mDoneStorage[mHead] = done ? 1.0f : 0.0f;

    if (mSize < mCapacity) mSize++;
    mHead = (mHead + 1) % mCapacity;
}

bool ReplayBuffer::Sample(size_t batchSize, std::mt19937& rng) {
    DIAGNOSE_SCOPE("ReplayBuffer: SampleInternal");
    DIAGNOSE_MUTEX_LOCK(mMutex, "ReplayBuffer: Sample");
    if (mSize < batchSize) return false;
    mCurrentBatchSize = batchSize;
    std::uniform_int_distribution<size_t> dist(0, mSize - 1);
    for (size_t i = 0; i < batchSize; ++i) {
        size_t idx = dist(rng);
        size_t srcObsOffset = idx * mObservationDim;
        size_t srcActionOffset = idx * mActionDim;
        size_t dstObsOffset = i * mObservationDim;
        size_t dstActionOffset = i * mActionDim;

        std::memcpy(mBatchObs1.data() + dstObsOffset, mObs1Storage.data() + srcObsOffset, mObservationDim * sizeof(float));
        std::memcpy(mBatchObs2.data() + dstObsOffset, mObs2Storage.data() + srcObsOffset, mObservationDim * sizeof(float));
        std::memcpy(mBatchActions1.data() + dstActionOffset, mAction1Storage.data() + srcActionOffset, mActionDim * sizeof(float));
        std::memcpy(mBatchActions2.data() + dstActionOffset, mAction2Storage.data() + srcActionOffset, mActionDim * sizeof(float));
        std::memcpy(mBatchNextObs1.data() + dstObsOffset, mNextObs1Storage.data() + srcObsOffset, mObservationDim * sizeof(float));
        std::memcpy(mBatchNextObs2.data() + dstObsOffset, mNextObs2Storage.data() + srcObsOffset, mObservationDim * sizeof(float));
        mBatchRewards1[i] = mReward1Storage[idx];
        mBatchRewards2[i] = mReward2Storage[idx];
        mBatchDones[i] = mDoneStorage[idx];
    }
    return true;
}

bool ReplayBuffer::Sample(size_t batchSize, float* outStates, float* outActions,
                          float* outRewards, float* outNextStates, float* outDones,
                          std::mt19937& rng) {
    DIAGNOSE_SCOPE("ReplayBuffer: SampleBatch");
    if (!Sample(batchSize, rng)) return false;
    // ... rest of the implementation using mBatch members remains compatible
    size_t obsSize = static_cast<size_t>(batchSize) * mObservationDim * 2;
    size_t actionSize = static_cast<size_t>(batchSize) * mActionDim * 2;
    
    // OPTIMIZED: Use Eigen for vectorized memory copies
    auto statesMap = Eigen::Map<Eigen::MatrixXf>(outStates, mObservationDim * 2, batchSize);
    auto actionsMap = Eigen::Map<Eigen::MatrixXf>(outActions, mActionDim * 2, batchSize);
    auto nextStatesMap = Eigen::Map<Eigen::MatrixXf>(outNextStates, mObservationDim * 2, batchSize);
    
    for (size_t i = 0; i < batchSize; ++i) {
        size_t srcOff = i * mObservationDim;
        statesMap.col(i).head(mObservationDim) = Eigen::Map<const Eigen::VectorXf>(
            mBatchObs1.data() + srcOff, mObservationDim);
        statesMap.col(i).tail(mObservationDim) = Eigen::Map<const Eigen::VectorXf>(
            mBatchObs2.data() + srcOff, mObservationDim);
    }
    
    for (size_t i = 0; i < batchSize; ++i) {
        size_t srcOff = i * mActionDim;
        actionsMap.col(i).head(mActionDim) = Eigen::Map<const Eigen::VectorXf>(
            mBatchActions1.data() + srcOff, mActionDim);
        actionsMap.col(i).tail(mActionDim) = Eigen::Map<const Eigen::VectorXf>(
            mBatchActions2.data() + srcOff, mActionDim);
    }
    
    for (size_t i = 0; i < batchSize; ++i) {
        outRewards[i] = mBatchRewards1[i] + mBatchRewards2[i];
    }
    
    for (size_t i = 0; i < batchSize; ++i) {
        size_t srcOff = i * mObservationDim;
        nextStatesMap.col(i).head(mObservationDim) = Eigen::Map<const Eigen::VectorXf>(
            mBatchNextObs1.data() + srcOff, mObservationDim);
        nextStatesMap.col(i).tail(mObservationDim) = Eigen::Map<const Eigen::VectorXf>(
            mBatchNextObs2.data() + srcOff, mObservationDim);
    }
    
    for (size_t i = 0; i < batchSize; ++i) {
        outDones[i] = mBatchDones[i];
    }
    return true;
}

// OPTIMIZED: Eigen-based zero-copy sampling
bool ReplayBuffer::SampleEigen(size_t batchSize,
                                Eigen::Map<Eigen::MatrixXf>& outStates,
                                Eigen::Map<Eigen::MatrixXf>& outActions,
                                Eigen::Map<Eigen::VectorXf>& outRewards,
                                Eigen::Map<Eigen::MatrixXf>& outNextStates,
                                Eigen::Map<Eigen::VectorXf>& outDones,
                                std::mt19937& rng) {
    DIAGNOSE_SCOPE("ReplayBuffer: SampleEigen");
    DIAGNOSE_MUTEX_LOCK(mMutex, "ReplayBuffer: SampleEigen");
    if (mSize < batchSize) return false;
    
    // Generate random indices
    std::uniform_int_distribution<size_t> dist(0, mSize - 1);
    
    // OPTIMIZED: Vectorized sampling using Eigen
    const int totalObsDim = mObservationDim * 2;
    const int totalActionDim = mActionDim * 2;
    
    for (size_t i = 0; i < batchSize; ++i) {
        size_t idx = dist(rng);
        size_t obsOffset = idx * mObservationDim;
        size_t actionOffset = idx * mActionDim;
        
        // States
        outStates.col(i).head(mObservationDim) = Eigen::Map<const Eigen::VectorXf>(
            mObs1Storage.data() + obsOffset, mObservationDim);
        outStates.col(i).tail(mObservationDim) = Eigen::Map<const Eigen::VectorXf>(
            mObs2Storage.data() + obsOffset, mObservationDim);
        
        // Actions
        outActions.col(i).head(mActionDim) = Eigen::Map<const Eigen::VectorXf>(
            mAction1Storage.data() + actionOffset, mActionDim);
        outActions.col(i).tail(mActionDim) = Eigen::Map<const Eigen::VectorXf>(
            mAction2Storage.data() + actionOffset, mActionDim);
        
        // Rewards (combined)
        outRewards(i) = mReward1Storage[idx] + mReward2Storage[idx];
        
        // Next states
        outNextStates.col(i).head(mObservationDim) = Eigen::Map<const Eigen::VectorXf>(
            mNextObs1Storage.data() + obsOffset, mObservationDim);
        outNextStates.col(i).tail(mObservationDim) = Eigen::Map<const Eigen::VectorXf>(
            mNextObs2Storage.data() + obsOffset, mObservationDim);
        
        // Dones
        outDones(i) = mDoneStorage[idx];
    }
    
    return true;
}

void ReplayBuffer::SampleVectorRewards(size_t batchSize, float* outStates, float* outActions,
                                        VectorReward* outVectorRewards, float* outNextStates,
                                        float* outDones, std::mt19937& rng) {
    DIAGNOSE_SCOPE("ReplayBuffer: SampleVectorRewards");
    if (!Sample(batchSize, rng)) return;
    size_t obsSize = static_cast<size_t>(batchSize) * mObservationDim * 2;
    size_t actionSize = static_cast<size_t>(batchSize) * mActionDim * 2;
    for (size_t i = 0; i < batchSize; ++i) {
        size_t srcOff = i * mObservationDim, dstOff = i * mObservationDim * 2;
        std::memcpy(outStates + dstOff, mBatchObs1.data() + srcOff, mObservationDim * sizeof(float));
        std::memcpy(outStates + dstOff + mObservationDim, mBatchObs2.data() + srcOff, mObservationDim * sizeof(float));
    }
    for (size_t i = 0; i < batchSize; ++i) {
        size_t srcOff = i * mActionDim, dstOff = i * mActionDim * 2;
        std::memcpy(outActions + dstOff, mBatchActions1.data() + srcOff, mActionDim * sizeof(float));
        std::memcpy(outActions + dstOff + mActionDim, mBatchActions2.data() + srcOff, mActionDim * sizeof(float));
    }
    for (size_t i = 0; i < batchSize; ++i) {
        outVectorRewards[i].SetDamageDealt(mBatchRewards1[i]);
        outVectorRewards[i].SetDamageTaken(mBatchRewards2[i]);
    }
    for (size_t i = 0; i < batchSize; ++i) {
        size_t srcOff = i * mObservationDim, dstOff = i * mObservationDim * 2;
        std::memcpy(outNextStates + dstOff, mBatchNextObs1.data() + srcOff, mObservationDim * sizeof(float));
        std::memcpy(outNextStates + dstOff + mObservationDim, mBatchNextObs2.data() + srcOff, mObservationDim * sizeof(float));
    }
    for (size_t i = 0; i < batchSize; ++i) {
        outDones[i] = mBatchDones[i];
    }
}

void ReplayBuffer::Clear() {
    DIAGNOSE_MUTEX_LOCK(mMutex, "ReplayBuffer: Clear");
    mSize = 0; mHead = 0;
}
