# Track Specification: Fix Critical Compilation and Training Bugs

## Overview

This track addresses the three critical bugs currently blocking JOLTrl development and training. These bugs prevent successful compilation and corrupt the training process.

---

## Problem Statement

### Critical Bug 1: AVX2 Tanh Implementation Failure
**File**: `src/NeuralNetwork.cpp`
**Issue**: Contains non-standard `_mm256_tanh_ps` intrinsic which does not exist in AVX2
**Impact**: Complete build failure; project cannot compile
**Current Code**:
```cpp
__m256 th = _mm256_tanh_ps ? _mm256_tanh_ps(x) : x;
```

### Critical Bug 2: KLPERBuffer Sum-Tree Indexing Broken
**File**: `src/NeuralNetwork.cpp`
**Issue**: Tree traversal starts at `idx = 0` and uses `2*idx`, which repeatedly references index 0
**Impact**: Invalid sampling behavior; PER (Prioritized Experience Replay) returns wrong samples
**Current Code**: Tree traversal logic starts at wrong index

### Critical Bug 3: Priority Precision Truncation in PER
**File**: `src/NeuralNetwork.cpp`
**Issue**: `UpdateTree(..., static_cast<int>(priority))` truncates float priorities to integers
**Impact**: Small priorities truncated to zero, corrupting sampling distribution
**Current Code**: Integer cast loses precision for priorities < 1.0

---

## Success Criteria

### Functional Requirements

1. **Build Success**
   - Project compiles without errors using `bazel build //:train --config=opt`
   - All SIMD intrinsics are standard AVX2/FMA instructions

2. **Sum-Tree Correctness**
   - Tree traversal starts at correct root index
   - Sampling returns valid leaf nodes with correct probabilities
   - Unit tests verify tree structure integrity

3. **Priority Precision**
   - Float priorities preserved without truncation
   - Sampling distribution matches priority weights
   - Unit tests verify priority-to-probability mapping

### Performance Requirements

1. **No Regression**: SPS (Steps Per Second) must not decrease by more than 5%
2. **Memory Efficiency**: No new heap allocations in hot loops
3. **SIMD Alignment**: All tensors remain 32-byte aligned

### Quality Requirements

1. **Test Coverage**: >80% code coverage for modified files
2. **Documentation**: All fixes documented in code comments
3. **Code Style**: Adheres to `conductor/code_styleguides/cpp.md`

---

## Technical Approach

### Bug 1: AVX2 Tanh Implementation

**Solution**: Implement tanh using standard AVX2 intrinsics

**Approach A**: Use exponential approximation
```cpp
// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
// Use _mm256_exp_ps for exponential
```

**Approach B**: Use polynomial approximation
```cpp
// Approximate tanh using odd polynomial
// tanh(x) ≈ x - x³/3 + 2x⁵/15 - 17x⁷/315
// Evaluate using AVX2 multiply-add
```

**Approach C**: Use lookup table with interpolation
```cpp
// Pre-compute tanh values in lookup table
// Use AVX2 for table lookup and lerp
```

**Recommended**: Approach A (exponential) for accuracy, or Approach B (polynomial) for speed

### Bug 2: Sum-Tree Indexing

**Solution**: Fix tree traversal to start at correct index

**Current (Broken)**:
```cpp
idx = 0;  // Wrong!
while (idx < tree_size) {
    idx = 2 * idx;  // Stays at 0
    // ...
}
```

**Corrected**:
```cpp
idx = 1;  // Start at root (index 1 in 1-based tree)
while (idx < internal_nodes_count) {
    idx = 2 * idx + (priority > tree[idx] ? 1 : 0);
    // ...
}
```

**Note**: Sum-tree typically uses 1-based indexing where:
- Root at index 1
- Left child: `2 * idx`
- Right child: `2 * idx + 1`
- Parent: `idx / 2`

### Bug 3: Priority Precision

**Solution**: Remove integer cast, use float throughout

**Current (Broken)**:
```cpp
UpdateTree(..., static_cast<int>(priority));  // Truncates!
```

**Corrected**:
```cpp
UpdateTree(..., priority);  // Preserve float precision
```

**Additional Fix**: Ensure tree array uses `float` or `double` type, not `int`

---

## Testing Strategy

### Unit Tests

1. **Tanh Implementation Test**
   - Compare AVX2 tanh against `std::tanh` for various inputs
   - Test boundary cases: x=0, x=±1, x=±10, x=±infinity
   - Verify max error < 1e-5

2. **Sum-Tree Test**
   - Build tree with known priorities
   - Sample 10,000 times and verify distribution matches priorities
   - Test edge cases: single element, two elements, power-of-2 elements

3. **Priority Precision Test**
   - Add samples with small priorities (0.001, 0.01, 0.1)
   - Verify sampling probability proportional to priority
   - Test priority updates preserve precision

### Integration Tests

1. **PER Buffer Integration**
   - Add 1000 transitions to replay buffer
   - Sample batch and verify priorities affect sampling frequency
   - Train for 100 steps and verify no crashes

2. **Training Loop Test**
   - Run training for 1000 steps with fixed bugs
   - Verify loss decreases (not NaN or constant)
   - Verify SPS within 5% of target

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Tanh approximation error | Low | Medium | Unit test against std::tanh |
| Sum-tree off-by-one | Medium | High | Extensive unit testing |
| Performance regression | Low | Medium | Benchmark before/after |
| Float precision issues | Low | Low | Use double for tree if needed |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Complex debugging | Medium | Low | Time-boxed investigation |
| Test failures | Low | Low | TDD approach |
| Unforeseen dependencies | Low | Medium | Code review before merge |

---

## Deliverables

1. **Fixed Source Files**
   - `src/NeuralNetwork.cpp` - All three bugs fixed
   - `src/NeuralMath.cpp` - If tanh implementation added here

2. **Test Files**
   - `src/NeuralMathTest.cpp` - Tanh and sum-tree tests
   - `src/PERBufferTest.cpp` - Priority precision tests

3. **Documentation**
   - Code comments explaining fixes
   - Updated `bug_report.md` with resolved bugs
   - Track completion summary in `plan.md`

4. **Verification**
   - All unit tests passing
   - Integration tests passing
   - Build succeeds with `bazel build //:train --config=opt`
   - Training runs for 1000+ steps without crashes

---

## Out of Scope

- Fixing High priority bugs (replay buffer transition bug, FPS display, divide-by-zero)
- Fixing Medium priority bugs (force sensor wiring, damage model, observation dimensions)
- Performance optimizations beyond bug fixes
- New feature development

---

## Dependencies

- **Bazel** 7.0+ for building
- **AVX2-capable CPU** (Intel i5-10500 or equivalent)
- **Existing test infrastructure** in `src/*Test.cpp`

---

## Acceptance Criteria

This track is complete when:

- [ ] All three critical bugs are fixed
- [ ] Project builds successfully with `bazel build //:train --config=opt`
- [ ] All new unit tests pass
- [ ] Integration test runs 1000+ training steps without crashes
- [ ] Code coverage >80% for modified files
- [ ] Code reviewed and merged to main
- [ ] `bug_report.md` updated to mark bugs as resolved
