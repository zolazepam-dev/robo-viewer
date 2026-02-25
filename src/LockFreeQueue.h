#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

template<typename T, size_t Capacity>
class LockFreeSPSCQueue
{
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    
public:
    LockFreeSPSCQueue() : mHead(0), mTail(0) {}
    
    bool Push(const T& item)
    {
        size_t currentTail = mTail.load(std::memory_order_relaxed);
        size_t nextTail = (currentTail + 1) & (Capacity - 1);
        
        if (nextTail == mHead.load(std::memory_order_acquire))
        {
            return false;
        }
        
        std::memcpy(&mBuffer[currentTail], &item, sizeof(T));
        mTail.store(nextTail, std::memory_order_release);
        return true;
    }
    
    bool Pop(T& item)
    {
        size_t currentHead = mHead.load(std::memory_order_relaxed);
        
        if (currentHead == mTail.load(std::memory_order_acquire))
        {
            return false;
        }
        
        std::memcpy(&item, &mBuffer[currentHead], sizeof(T));
        mHead.store((currentHead + 1) & (Capacity - 1), std::memory_order_release);
        return true;
    }
    
    bool Empty() const
    {
        return mHead.load(std::memory_order_acquire) == mTail.load(std::memory_order_acquire);
    }
    
    size_t Size() const
    {
        size_t tail = mTail.load(std::memory_order_acquire);
        size_t head = mHead.load(std::memory_order_acquire);
        return (tail - head + Capacity) & (Capacity - 1);
    }
    
    static constexpr size_t GetCapacity() { return Capacity; }

private:
    alignas(64) T mBuffer[Capacity];
    alignas(64) std::atomic<size_t> mHead;
    alignas(64) std::atomic<size_t> mTail;
};

struct TrainingBatch
{
    static constexpr int MAX_BATCH_SIZE = 256;
    static constexpr int MAX_STATE_DIM = 256;
    static constexpr int MAX_ACTION_DIM = 64;
    
    int batchSize = 0;
    int stateDim = 0;
    int actionDim = 0;
    
    alignas(32) float states[MAX_BATCH_SIZE * MAX_STATE_DIM];
    alignas(32) float actions[MAX_BATCH_SIZE * MAX_ACTION_DIM];
    alignas(32) float rewards[MAX_BATCH_SIZE];
    alignas(32) float nextStates[MAX_BATCH_SIZE * MAX_STATE_DIM];
    alignas(32) float dones[MAX_BATCH_SIZE];
    alignas(32) float logProbs[MAX_BATCH_SIZE];
    alignas(32) float klDivergences[MAX_BATCH_SIZE];
    int indices[MAX_BATCH_SIZE];
    
    void Clear()
    {
        std::memset(states, 0, sizeof(states));
        std::memset(actions, 0, sizeof(actions));
        std::memset(rewards, 0, sizeof(rewards));
        std::memset(nextStates, 0, sizeof(nextStates));
        std::memset(dones, 0, sizeof(dones));
        std::memset(logProbs, 0, sizeof(logProbs));
        std::memset(klDivergences, 0, sizeof(klDivergences));
        std::memset(indices, 0, sizeof(indices));
        batchSize = 0;
    }
};

using TrainingBatchQueue = LockFreeSPSCQueue<TrainingBatch, 16>;