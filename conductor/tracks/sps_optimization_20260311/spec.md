# Track Specification: SPS Performance Optimization

## Overview

This track focuses on optimizing the JOLTrl training throughput from the current baseline of ~1,000 SPS (with 128 environments) to a target of 25,000+ SPS (25x improvement). The optimization will be CPU-only and must maintain training correctness without reducing brain size.

## Problem Statement

Current training throughput of ~1,000 SPS is significantly below the product definition target of 100,000+ SPS. This bottleneck slows down research iteration and increases training time for production policies.

## Goals

### Primary Goal
- **Increase SPS from ~1,000 to 25,000+** (25x improvement) with 128 environments

### Constraints
- **CPU-only optimization** - No GPU acceleration
- **No reducing brain size** - Maintain current neural network capacity (latent dim, hidden dim)
- **Maintain correctness** - Training quality must not degrade

## Technical Approach

### Optimization Areas

1. **Lock-Free Data Structures**
   - Eliminate mutex contention in training loop
   - Replace `gSimMutex` with lock-free queues for state transfer
   - Implement wait-free ring buffers for action/observation exchange

2. **Better Thread Pinning**
   - Optimize CPU core affinity for physics workers
   - Pin Jolt job system threads to dedicated cores (1-5, 7-11)
   - Isolate main RL loop on Core 0
   - Use `pthread_setaffinity_np` for precise control

3. **SIMD Vectorization**
   - Expand AVX2/FMA optimizations in hot paths
   - Vectorize observation preprocessing
   - Optimize reward calculation with SIMD
   - Batch environment stepping with AVX2

4. **Memory Pool Optimization**
   - Reduce cache misses with better data layout
   - Implement Structure-of-Arrays (SoA) for environment states
   - Pre-allocate contiguous memory blocks for 128 envs
   - Align data to cache lines (64-byte)

## Success Criteria

### Functional Requirements
- [ ] SPS increases from ~1,000 to 25,000+ (25x improvement)
- [ ] Training converges to same or better win rate
- [ ] No increase in memory footprint per environment
- [ ] All existing tests pass

### Performance Requirements
- [ ] Lock contention reduced to <1% of step time
- [ ] Cache miss rate <5% (measured via perf)
- [ ] Thread synchronization overhead <0.1ms per step

### Quality Requirements
- [ ] Code coverage >80% for new optimization code
- [ ] No new heap allocations in hot loops
- [ ] Documentation for all new data structures

## Out of Scope

- GPU acceleration or CUDA integration
- Neural network architecture changes
- Algorithm changes (TD3 hyperparameters)
- Increasing brain size or capacity
- Multi-machine distributed training

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Lock-free bugs | Medium | High | Extensive testing, gradual rollout |
| Thread pinning conflicts | Low | Medium | Test on target hardware early |
| SIMD alignment issues | Low | Medium | Use AlignedAllocator, assert alignment |
| Cache thrashing | Medium | High | Profile with perf, iterate on layout |

## Deliverables

1. **Optimized Source Files**
   - `src/LockFreeQueue.h` - Lock-free ring buffer
   - `src/ThreadPinning.cpp` - CPU affinity management
   - `src/VectorizedEnv.cpp` - SIMD-optimized stepping
   - `src/SoAEnvironment.h` - Structure-of-arrays layout

2. **Tests**
   - `src/LockFreeQueueTest.cpp` - Concurrent queue tests
   - `src/SPSBenchmark.cpp` - Before/after comparison
   - `src/ThreadPinningTest.cpp` - Affinity verification

3. **Documentation**
   - Performance profiling report
   - Thread layout diagram
   - Memory layout documentation

## Acceptance Criteria

This track is complete when:
- [ ] SPS benchmark shows 25,000+ with 128 environments
- [ ] Training convergence matches or exceeds baseline
- [ ] All new tests pass
- [ ] Code review approved
- [ ] Performance report documented
