#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

// 32-byte alignment for AVX2, 64-byte for cache lines
constexpr size_t ALIGNMENT = 64;

template<typename T>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U>;
    };

    AlignedAllocator() noexcept = default;

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        if (n > std::size_t(-1) / sizeof(T)) throw std::bad_alloc();

        size_type size = n * sizeof(T);
        void* ptr = nullptr;
        
        // Use aligned_alloc for proper SIMD alignment
        #if defined(_WIN32)
            ptr = _aligned_malloc(size, ALIGNMENT);
        #else
            ptr = aligned_alloc(ALIGNMENT, (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1));
        #endif
        
        if (!ptr) throw std::bad_alloc();
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer ptr, size_type) noexcept {
        if (ptr) {
            #if defined(_WIN32)
                _aligned_free(ptr);
            #else
                free(ptr);
            #endif
        }
    }

    template<typename U>
    bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }
    
    template<typename U>
    bool operator!=(const AlignedAllocator<U>&) const noexcept { return false; }
};

// Helper function to check alignment
template<typename T>
inline bool isAligned(const T* ptr, size_t alignment = ALIGNMENT) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// Helper to get aligned size
inline size_t alignedSize(size_t size, size_t alignment = ALIGNMENT) {
    return (size + alignment - 1) & ~(alignment - 1);
}
