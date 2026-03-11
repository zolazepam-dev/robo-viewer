// Tensor.cpp - Implementation file for CPU Tensors library

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cpuid.h>

namespace cput {

// CPU feature detection using __get_cpuid
bool hasAVX2() {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    if (!(ecx & (1 << 27))) return false;
    if (!(ecx & (1 << 28))) return false;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & (1 << 5)) != 0;
}

// Memory utilities
void* alignedAlloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
    #if defined(_WIN32)
        ptr = _aligned_malloc(size, alignment);
    #else
        ptr = std::aligned_alloc(alignment, (size + alignment - 1) & ~(alignment - 1));
    #endif
    return ptr;
}

void alignedFree(void* ptr) {
    if (ptr) {
        #if defined(_WIN32)
            _aligned_free(ptr);
        #else
            std::free(ptr);
        #endif
    }
}

} // namespace cput
