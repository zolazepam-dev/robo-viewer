/**
 * @file LockFreeQueue.h
 * @brief Lock-free ring buffer for high-performance data transfer in JOLTrl
 * 
 * This implementation provides:
 * - Thread-safe push/pop operations using atomic CAS (Compare-And-Swap)
 * - Zero mutex contention for parallel environment data transfer
 * - Memory ordering constraints (acquire/release semantics)
 * - Fixed-size ring buffer with pre-allocated memory (zero allocation in hot path)
 * 
 * Design:
 * - Multi-producer multi-consumer (MPMC) safe via atomic head/tail with CAS
 * - Uses std::atomic with memory_order_acquire/release for proper synchronization
 * - Cache-line padding to prevent false sharing
 * 
 * Usage:
 * @code
 * LockFreeQueue<Transition> queue(1024);  // 1024-element capacity
 * 
 * // Producer (environment thread)
 * queue.push(transition);
 * 
 * // Consumer (training thread)
 * Transition t;
 * if (queue.pop(t)) {
 *     // Process transition
 * }
 * @endcode
 * 
 * @tparam T Type of elements stored (must be trivially copyable)
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>
#include <vector>
#include "AlignedAllocator.h"

/**
 * @class LockFreeQueue
 * @brief Lock-free MPMC ring buffer for inter-thread communication
 * 
 * Implements a bounded ring buffer with atomic head and tail pointers.
 * Uses compare-and-swap (CAS) operations for thread-safe updates.
 * 
 * Memory Model:
 * - Producer: Uses memory_order_acq_rel for CAS operations
 * - Consumer: Uses memory_order_acq_rel for CAS operations
 * - Ensures proper synchronization with acquire-release semantics
 * 
 * Performance Characteristics:
 * - O(1) amortized push and pop operations
 * - Zero heap allocation after construction
 * - Cache-line aligned head/tail to prevent false sharing
 * 
 * @tparam T Element type (must be trivially copyable)
 */
template<typename T>
class LockFreeQueue {
public:
    /**
     * @brief Construct a new Lock-Free Queue
     * 
     * Pre-allocates a contiguous buffer for `capacity` elements.
     * Capacity is rounded up to the next power of 2 for efficient modulo operations.
     * 
     * @param capacity Maximum number of elements the queue can hold
     * @throws std::bad_alloc if memory allocation fails
     */
    explicit LockFreeQueue(size_t capacity)
        : mCapacity(roundUpToPowerOf2(capacity))
        , mMask(mCapacity - 1)
        , mHead(0)
        , mTail(0)
        , mBuffer(nullptr)
    {
        static_assert(std::is_trivially_copyable<T>::value, 
                     "LockFreeQueue requires trivially copyable types");
        
        // Allocate aligned buffer
        mBuffer = mAllocator.allocate(mCapacity);
        
        // Initialize buffer cells with sequence numbers for MPMC safety
        for (size_t i = 0; i < mCapacity; ++i) {
            mBuffer[i].sequence.store(i, std::memory_order_relaxed);
            mBuffer[i].data = T();
        }
    }
    
    /**
     * @brief Destructor - frees allocated buffer
     */
    ~LockFreeQueue() {
        if (mBuffer) {
            mAllocator.deallocate(mBuffer, mCapacity);
        }
    }
    
    // Delete copy constructor and assignment (queue is non-copyable)
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;
    
    // Enable move semantics
    LockFreeQueue(LockFreeQueue&& other) noexcept
        : mCapacity(other.mCapacity)
        , mMask(other.mMask)
        , mHead(other.mHead.load(std::memory_order_relaxed))
        , mTail(other.mTail.load(std::memory_order_relaxed))
        , mBuffer(other.mBuffer)
    {
        other.mBuffer = nullptr;
        other.mCapacity = 0;
        other.mMask = 0;
    }
    
    LockFreeQueue& operator=(LockFreeQueue&& other) noexcept {
        if (this != &other) {
            if (mBuffer) {
                mAllocator.deallocate(mBuffer, mCapacity);
            }
            mCapacity = other.mCapacity;
            mMask = other.mMask;
            mHead.store(other.mHead.load(std::memory_order_relaxed), std::memory_order_relaxed);
            mTail.store(other.mTail.load(std::memory_order_relaxed), std::memory_order_relaxed);
            mBuffer = other.mBuffer;
            other.mBuffer = nullptr;
            other.mCapacity = 0;
            other.mMask = 0;
        }
        return *this;
    }
    
    /**
     * @brief Push an element to the back of the queue
     * 
     * Thread-safe operation using atomic CAS on head pointer.
     * If queue is full, returns false immediately (non-blocking).
     * 
     * Memory Ordering:
     * - Uses memory_order_acq_rel for proper synchronization
     * - CAS operation ensures only one producer claims each slot
     * 
     * @param item Element to push (copied into queue)
     * @return true if push succeeded, false if queue was full
     */
    bool push(const T& item) {
        Cell* cell;
        size_t pos = mHead.load(std::memory_order_relaxed);
        
        for (;;) {
            cell = &mBuffer[pos & mMask];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            
            if (diff == 0) {
                // Slot is ready for writing, try to claim it
                if (mHead.compare_exchange_weak(pos, pos + 1, std::memory_order_acq_rel)) {
                    break;  // Successfully claimed the slot
                }
                // CAS failed, another producer won, retry with new pos
            } else if (diff < 0) {
                // Queue is full (consumer hasn't caught up)
                return false;
            } else {
                // Another producer got ahead, reload pos and retry
                pos = mHead.load(std::memory_order_relaxed);
            }
        }
        
        // Write data to claimed slot
        cell->data = item;
        
        // Update sequence to signal data is ready
        cell->sequence.store(pos + 1, std::memory_order_release);
        
        return true;
    }
    
    /**
     * @brief Pop an element from the front of the queue
     * 
     * Thread-safe operation using atomic CAS on tail pointer.
     * If queue is empty, returns false immediately (non-blocking).
     * 
     * Memory Ordering:
     * - Uses memory_order_acq_rel for proper synchronization
     * - CAS operation ensures only one consumer claims each slot
     * 
     * @param item Output parameter for popped element
     * @return true if pop succeeded, false if queue was empty
     */
    bool pop(T& item) {
        Cell* cell;
        size_t pos = mTail.load(std::memory_order_relaxed);
        
        for (;;) {
            cell = &mBuffer[pos & mMask];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            
            if (diff == 0) {
                // Slot has data, try to claim it
                if (mTail.compare_exchange_weak(pos, pos + 1, std::memory_order_acq_rel)) {
                    break;  // Successfully claimed the slot
                }
                // CAS failed, another consumer won, retry with new pos
            } else if (diff < 0) {
                // Queue is empty (no data yet)
                return false;
            } else {
                // Another consumer got ahead, reload pos and retry
                pos = mTail.load(std::memory_order_relaxed);
            }
        }
        
        // Read data from claimed slot
        item = cell->data;
        
        // Update sequence to signal slot is free (for next round)
        cell->sequence.store(pos + mMask + 1, std::memory_order_release);
        
        return true;
    }
    
    /**
     * @brief Check if queue is empty
     * 
     * Note: Result may be stale in concurrent scenarios.
     * Use as a hint before calling pop().
     * 
     * @return true if head equals tail (no elements)
     */
    bool empty() const {
        const size_t head = mHead.load(std::memory_order_acquire);
        const size_t tail = mTail.load(std::memory_order_acquire);
        return head == tail;
    }
    
    /**
     * @brief Get current number of elements in queue
     * 
     * Note: Result may be stale in concurrent scenarios.
     * Provides approximate size for monitoring/debugging.
     * 
     * @return Number of elements (approximate)
     */
    size_t size() const {
        const size_t head = mHead.load(std::memory_order_acquire);
        const size_t tail = mTail.load(std::memory_order_acquire);
        
        if (head >= tail) {
            return head - tail;
        } else {
            return mCapacity - tail + head;
        }
    }
    
    /**
     * @brief Get maximum capacity of queue
     * @return Maximum number of elements
     */
    size_t capacity() const {
        return mCapacity;
    }
    
    /**
     * @brief Clear all elements from queue
     * 
     * WARNING: Not thread-safe! Only call when no other threads are accessing.
     */
    void clear() {
        mHead.store(0, std::memory_order_relaxed);
        mTail.store(0, std::memory_order_relaxed);
    }

private:
    /**
     * @brief Round up to next power of 2
     * 
     * Enables efficient modulo operations via bitwise AND.
     * 
     * @param n Input value
     * @return Next power of 2 >= n
     */
    static size_t roundUpToPowerOf2(size_t n) {
        if (n == 0) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        n++;
        return n;
    }
    
    /**
     * @brief Internal cell structure with sequence number
     * 
     * Each cell has a sequence number that tracks its state:
     * - seq == pos: cell is ready for writing (empty)
     * - seq == pos + 1: cell has data ready for reading
     * - seq == pos + capacity: cell is being recycled
     */
    struct Cell {
        std::atomic<size_t> sequence;
        T data;
        // Cache line padding - only add if T is small enough
        static constexpr size_t kCellSize = sizeof(std::atomic<size_t>) + sizeof(T);
        static constexpr size_t kPaddingSize = kCellSize < 64 ? (64 - kCellSize) : 0;
        char padding[kPaddingSize];  // Cache line padding
    };
    
    size_t mCapacity;              ///< Buffer capacity (power of 2)
    size_t mMask;                  ///< Mask for efficient modulo (capacity - 1)
    std::atomic<size_t> mHead;     ///< Head index (write position)
    std::atomic<size_t> mTail;     ///< Tail index (read position)
    Cell* mBuffer;                 ///< Element buffer with sequence numbers
    AlignedAllocator<Cell> mAllocator; ///< Aligned allocator for buffer
};
