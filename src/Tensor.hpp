#pragma once

#include "AlignedAllocator.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <array>
#include <initializer_list>
#include <stdexcept>
#include <functional>

namespace cput {

enum class TensorLayout {
    RowMajor,
    ColMajor
};

template<typename T, size_t Dims = 2>
class Tensor {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using allocator_type = AlignedAllocator<T>;

    // Default constructor
    Tensor() : m_data(nullptr), m_size(0), m_capacity(0) {
        for (size_t i = 0; i < Dims; ++i) m_shape[i] = 0;
    }

    // Shape constructor
    template<typename... Args>
    Tensor(Args... dims) : m_data(nullptr), m_size(0), m_capacity(0) {
        static_assert(sizeof...(Args) == Dims, "Number of dimensions must match");
        size_type shape[Dims] = {static_cast<size_type>(dims)...};
        resize(shape);
    }

    // Array shape constructor
    explicit Tensor(const std::array<size_type, Dims>& shape) 
        : m_data(nullptr), m_size(0), m_capacity(0) {
        resize(shape.data());
    }

    // Initializer list constructor (for 1D/2D)
    Tensor(std::initializer_list<T> init) : m_data(nullptr), m_size(0), m_capacity(0) {
        if constexpr (Dims == 1) {
            m_shape[0] = init.size();
            m_size = init.size();
            m_capacity = alignedSize(m_size * sizeof(T));
            m_data = m_allocator.allocate(m_capacity / sizeof(T));
            std::copy(init.begin(), init.end(), m_data);
        }
    }

    // Copy constructor
    Tensor(const Tensor& other) : m_data(nullptr), m_size(0), m_capacity(0) {
        resize(other.m_shape);
        std::memcpy(m_data, other.m_data, m_size * sizeof(T));
    }

    // Move constructor
    Tensor(Tensor&& other) noexcept 
        : m_data(other.m_data), m_size(other.m_size), m_capacity(other.m_capacity) {
        std::memcpy(m_shape, other.m_shape, Dims * sizeof(size_type));
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_capacity = 0;
    }

    // Copy assignment
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            resize(other.m_shape);
            std::memcpy(m_data, other.m_data, m_size * sizeof(T));
        }
        return *this;
    }

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (m_data) m_allocator.deallocate(m_data, m_capacity / sizeof(T));
            m_data = other.m_data;
            m_size = other.m_size;
            m_capacity = other.m_capacity;
            std::memcpy(m_shape, other.m_shape, Dims * sizeof(size_type));
            other.m_data = nullptr;
            other.m_size = 0;
            other.m_capacity = 0;
        }
        return *this;
    }

    ~Tensor() {
        if (m_data) m_allocator.deallocate(m_data, m_capacity / sizeof(T));
    }

    // Resize
    void resize(const size_type* shape) {
        size_type new_size = 1;
        for (size_t i = 0; i < Dims; ++i) {
            m_shape[i] = shape[i];
            new_size *= shape[i];
        }
        
        if (new_size > m_capacity / sizeof(T)) {
            if (m_data) m_allocator.deallocate(m_data, m_capacity / sizeof(T));
            m_capacity = alignedSize(new_size * sizeof(T));
            m_data = m_allocator.allocate(m_capacity / sizeof(T));
        }
        m_size = new_size;
    }

    void resize(std::initializer_list<size_type> shape) {
        if (shape.size() != Dims) throw std::invalid_argument("Incorrect number of dimensions");
        resize(std::data(shape));
    }

    // Element access
    template<typename... Args>
    reference at(Args... indices) {
        static_assert(sizeof...(Args) == Dims, "Number of indices must match dimensions");
        size_type idx[Dims] = {static_cast<size_type>(indices)...};
        return m_data[flatIndex(idx)];
    }

    template<typename... Args>
    const_reference at(Args... indices) const {
        static_assert(sizeof...(Args) == Dims, "Number of indices must match dimensions");
        size_type idx[Dims] = {static_cast<size_type>(indices)...};
        return m_data[flatIndex(idx)];
    }

    // Raw pointer access (SIMD-friendly)
    pointer data() noexcept { return m_data; }
    const_pointer data() const noexcept { return m_data; }

    // Shape access
    size_type shape(size_t dim) const { return m_shape[dim]; }
    const size_type* shape() const { return m_shape; }
    size_type size() const { return m_size; }
    size_type capacity() const { return m_capacity / sizeof(T); }
    
    bool empty() const { return m_size == 0; }

    // Check if data is properly aligned
    bool isAligned() const {
        return ::isAligned(m_data, ALIGNMENT);
    }

    // Fill with value
    void fill(T value) {
        for (size_type i = 0; i < m_size; ++i) m_data[i] = value;
    }

    // Zero initialization
    void zero() {
        std::memset(m_data, 0, m_size * sizeof(T));
    }

private:
    size_type flatIndex(const size_type* indices) const {
        size_type idx = 0;
        size_type stride = 1;
        for (size_t i = Dims; i-- > 0;) {
            idx += indices[i] * stride;
            stride *= m_shape[i];
        }
        return idx;
    }

    allocator_type m_allocator;
    pointer m_data;
    size_type m_size;
    size_type m_capacity;
    size_type m_shape[Dims];
};

// Type aliases for common tensor types
using TensorF32 = Tensor<float, 2>;
using TensorF64 = Tensor<double, 2>;
using TensorI32 = Tensor<int32_t, 2>;
using TensorI8 = Tensor<int8_t, 2>;

using VectorF32 = Tensor<float, 1>;
using VectorF64 = Tensor<double, 1>;

} // namespace cput
