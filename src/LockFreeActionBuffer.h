#pragma once

#include <atomic>
#include <vector>
#include <cstring>

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
        size_t writePos = mWriteIndex.load(std::memory_order_relaxed);
        size_t actualCount = std::min(count, mCapacity - writePos);
        
        std::memcpy(mBuffer.data() + writePos, actions, actualCount * sizeof(T));
        
        mWriteIndex.store(writePos + actualCount, std::memory_order_release);
    }
    
    // Consumer: read single action (called from each env thread)
    bool Read(T* action, size_t actionSize) {
        size_t readPos = mReadIndex.load(std::memory_order_relaxed);
        size_t writePos = mWriteIndex.load(std::memory_order_acquire);
        
        if (readPos >= writePos) return false;  // No data available
        
        std::memcpy(action, mBuffer.data() + readPos, actionSize * sizeof(T));
        
        mReadIndex.store(readPos + 1, std::memory_order_release);
        return true;
    }
    
    // Reset for next step
    void Reset() {
        mWriteIndex.store(0, std::memory_order_relaxed);
        mReadIndex.store(0, std::memory_order_relaxed);
    }
    
    size_t Size() const {
        return mWriteIndex.load(std::memory_order_acquire) - 
               mReadIndex.load(std::memory_order_acquire);
    }

private:
    std::vector<T> mBuffer;
    std::atomic<size_t> mWriteIndex;
    std::atomic<size_t> mReadIndex;
    size_t mCapacity;
};
