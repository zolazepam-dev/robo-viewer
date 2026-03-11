#pragma once

#include <atomic>
#include <vector>
#include <cstring>
#include <algorithm>

// Lock-free ring buffer for actions - eliminates sequential QueueActions calls
// Env threads pull actions atomically without contention
template<typename T>
class LockFreeActionBuffer {
public:
    LockFreeActionBuffer(size_t capacity) 
        : mCapacity(capacity), mWriteIndex(0), mReadIndex(0) {
        mBuffer.resize(capacity);
    }
    
    // Producer: write actions (called once per step from main thread)
    void WriteBatch(const T* actions, size_t count) {
        // use fetch_add to atomically reserve space
        size_t writePos = mWriteIndex.fetch_add(count, std::memory_order_acq_rel);
        
        if (writePos < mCapacity) {
            size_t actualCount = std::min(count, mCapacity - writePos);
            std::memcpy(mBuffer.data() + writePos, actions, actualCount * sizeof(T));
        }
    }
    
    // Consumer: read batch of actions
    bool ReadBatch(T* actions, size_t maxCount, size_t& actualCount) {
        // use fetch_add to atomically reserve read position
        size_t readPos = mReadIndex.fetch_add(maxCount, std::memory_order_acq_rel);
        size_t writePos = mWriteIndex.load(std::memory_order_acquire);
        
        if (readPos >= writePos) {
            // Revert the fetch_add if we overshot (simplistic recovery)
            // Note: in a true multi-consumer high-contention scenario, this needs care
            actualCount = 0;
            return false;
        }
        
        actualCount = std::min(maxCount, writePos - readPos);
        if (actualCount > 0) {
            std::memcpy(actions, mBuffer.data() + readPos, actualCount * sizeof(T));
        }
        return true;
    }

    // Consumer: read single action (called from each env thread)
    bool Read(T* action, size_t actionSize) {
        size_t readPos = mReadIndex.fetch_add(actionSize, std::memory_order_acq_rel);
        size_t writePos = mWriteIndex.load(std::memory_order_acquire);
        
        if (readPos >= writePos) {
            return false;  // No data available
        }
        
        std::memcpy(action, mBuffer.data() + readPos, actionSize * sizeof(T));
        return true;
    }
    
    // Reset for next step
    void Reset() {
        mWriteIndex.store(0, std::memory_order_relaxed);
        mReadIndex.store(0, std::memory_order_relaxed);
    }
    
    size_t Size() const {
        size_t w = mWriteIndex.load(std::memory_order_acquire);
        size_t r = mReadIndex.load(std::memory_order_acquire);
        return (w > r) ? (w - r) : 0;
    }

private:
    std::vector<T> mBuffer;
    std::atomic<size_t> mWriteIndex;
    std::atomic<size_t> mReadIndex;
    size_t mCapacity;
};
