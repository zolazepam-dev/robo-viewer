#pragma once

#include <cstdlib>
#include <memory>
#include <new>
#include <cstddef>
#include <limits>
#include <immintrin.h>

template<typename T, std::size_t Alignment = 32>
class AlignedAllocator
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of 2");

    template<typename U>
    struct rebind
    {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n)
    {
        if (n == 0) return nullptr;
        if (n > max_size()) throw std::bad_alloc();
        
        void* ptr = _mm_malloc(n * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept
    {
        if (ptr) _mm_free(ptr);
    }

    template<typename U, typename... Args>
    void construct(U* ptr, Args&&... args)
    {
        ::new (static_cast<void*>(ptr)) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* ptr)
    {
        ptr->~U();
    }

    std::size_t max_size() const noexcept
    {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }

    template<typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept
    {
        return true;
    }

    template<typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept
    {
        return false;
    }
};

template<typename T>
using AlignedVector32 = std::vector<T, AlignedAllocator<T, 32>>;
