# Implementation Plan: SPS Performance Optimization

## Track ID: sps_optimization_20260311

This plan follows the Test-Driven Development workflow defined in `conductor/workflow.md`. Each task must be completed sequentially.

---

## Phase 1: Lock-Free Data Structures

**Goal**: Eliminate mutex contention in training loop

- [ ] Task: Write failing test for lock-free queue
    - [ ] Create concurrent producer/consumer test
    - [ ] Test with 128 parallel producers (environments)
    - [ ] Verify zero contention overhead
    - [ ] Confirm test fails (implementation not yet present)

- [ ] Task: Implement LockFreeQueue
    - [ ] Design ring buffer with atomic head/tail indices
    - [ ] Implement `push()` with atomic CAS operations
    - [ ] Implement `pop()` with atomic CAS operations
    - [ ] Add memory ordering constraints (acquire/release)

- [ ] Task: Integrate LockFreeQueue into training loop
    - [ ] Replace `gSimMutex` protected queues
    - [ ] Update `VectorizedEnv` to use lock-free transfer
    - [ ] Verify zero mutex contention in profiling

- [ ] Task: Verify test coverage for Phase 1
    - [ ] Run coverage tool
    - [ ] Ensure >80% coverage for new files
    - [ ] Document any coverage gaps

- [ ] Task: Conductor - User Manual Verification 'Phase 1: Lock-Free Data Structures' (Protocol in workflow.md)
    - [ ] Announce phase completion
    - [ ] Verify test coverage for phase changes
    - [ ] Execute automated tests with proactive debugging
    - [ ] Propose manual verification plan
    - [ ] Await explicit user feedback

---

## Phase 2: Thread Pinning Optimization

**Goal**: Optimize CPU core affinity for physics workers

- [ ] Task: Write failing test for thread pinning
    - [ ] Create test verifying thread affinity
    - [ ] Test core isolation (no cross-core migration)
    - [ ] Measure context switch reduction
    - [ ] Confirm test fails (pinning not yet implemented)

- [ ] Task: Implement ThreadPinning system
    - [ ] Create `ThreadPinning` class with `pthread_setaffinity_np`
    - [ ] Define core mapping: Core 0 (main), Cores 1-5 (physics)
    - [ ] Implement `pinThread(core_id)` function
    - [ ] Add error handling for invalid cores

- [ ] Task: Integrate thread pinning into Jolt job system
    - [ ] Pin physics workers to cores 1-5, 7-11
    - [ ] Pin main RL loop to core 0
    - [ ] Verify with `taskset` or `htop`

- [ ] Task: Verify test coverage for Phase 2
    - [ ] Run coverage tool
    - [ ] Ensure >80% coverage for new files
    - [ ] Document any coverage gaps

- [ ] Task: Conductor - User Manual Verification 'Phase 2: Thread Pinning' (Protocol in workflow.md)
    - [ ] Announce phase completion
    - [ ] Verify test coverage for phase changes
    - [ ] Execute automated tests with proactive debugging
    - [ ] Propose manual verification plan
    - [ ] Await explicit user feedback

---

## Phase 3: SIMD Vectorization

**Goal**: Expand AVX2/FMA optimizations in hot paths

- [ ] Task: Write failing test for SIMD operations
    - [ ] Create benchmark for observation preprocessing
    - [ ] Test AVX2 vs scalar performance comparison
    - [ ] Verify correctness (results match within epsilon)
    - [ ] Confirm test fails (vectorization not yet present)

- [ ] Task: Implement SIMD observation preprocessing
    - [ ] Vectorize normalization with AVX2
    - [ ] Vectorize reward calculation with AVX2
    - [ ] Use `_mm256_*` intrinsics for 8-wide operations
    - [ ] Ensure 32-byte alignment for all tensors

- [ ] Task: Implement SIMD batch environment stepping
    - [ ] Vectorize physics step for 8 environments
    - [ ] Use AVX2 for contact impulse calculation
    - [ ] Batch action application with SIMD

- [ ] Task: Verify test coverage for Phase 3
    - [ ] Run coverage tool
    - [ ] Ensure >80% coverage for new files
    - [ ] Document any coverage gaps

- [ ] Task: Conductor - User Manual Verification 'Phase 3: SIMD Vectorization' (Protocol in workflow.md)
    - [ ] Announce phase completion
    - [ ] Verify test coverage for phase changes
    - [ ] Execute automated tests with proactive debugging
    - [ ] Propose manual verification plan
    - [ ] Await explicit user feedback

---

## Phase 4: Memory Pool Optimization

**Goal**: Reduce cache misses with better data layout

- [ ] Task: Write failing test for memory layout
    - [ ] Create cache miss benchmark (using perf)
    - [ ] Measure L1/L2/L3 cache miss rates
    - [ ] Test SoA vs AoS performance
    - [ ] Confirm test fails (optimization not yet present)

- [ ] Task: Implement SoAEnvironment layout
    - [ ] Convert environment states to Structure-of-Arrays
    - [ ] Align data to 64-byte cache lines
    - [ ] Pre-allocate contiguous memory for 128 envs
    - [ ] Implement gather/scatter for individual access

- [ ] Task: Integrate SoA layout into training pipeline
    - [ ] Update `VectorizedEnv` to use SoA layout
    - [ ] Update `TD3Trainer` to work with SoA observations
    - [ ] Verify cache miss reduction with perf

- [ ] Task: Verify test coverage for Phase 4
    - [ ] Run coverage tool
    - [ ] Ensure >80% coverage for new files
    - [ ] Document any coverage gaps

- [ ] Task: Conductor - User Manual Verification 'Phase 4: Memory Pool' (Protocol in workflow.md)
    - [ ] Announce phase completion
    - [ ] Verify test coverage for phase changes
    - [ ] Execute automated tests with proactive debugging
    - [ ] Propose manual verification plan
    - [ ] Await explicit user feedback

---

## Phase 5: Integration Testing & Validation

**Goal**: Verify all optimizations work together and achieve 25,000+ SPS

- [ ] Task: Write integration benchmark
    - [ ] Create `SPSBenchmark.cpp` with before/after comparison
    - [ ] Test with 128, 256, 512 environments
    - [ ] Measure SPS, memory usage, cache misses
    - [ ] Verify training convergence matches baseline

- [ ] Task: Run full training validation
    - [ ] Train for 10,000 steps with optimized code
    - [ ] Compare win rate to baseline
    - [ ] Verify no regressions in training quality
    - [ ] Document any differences

- [ ] Task: Performance profiling report
    - [ ] Run `perf` to measure CPU cycles, cache misses
    - [ ] Profile thread synchronization overhead
    - [ ] Document optimization impact per phase
    - [ ] Create performance comparison chart

- [ ] Task: Update documentation
    - [ ] Document new data structures and APIs
    - [ ] Add performance tuning guide
    - [ ] Update `tech-stack.md` if needed
    - [ ] Document lessons learned

- [ ] Task: Final code review and merge
    - [ ] Review all changes against `cpp.md` style guide
    - [ ] Verify all tests passing
    - [ ] Create pull request or commit to main
    - [ ] Update track status to complete

- [ ] Task: Conductor - User Manual Verification 'Phase 5: Integration' (Protocol in workflow.md)
    - [ ] Announce phase completion
    - [ ] Verify test coverage for phase changes
    - [ ] Execute automated tests with proactive debugging
    - [ ] Propose manual verification plan
    - [ ] Await explicit user feedback

---

## Task Completion Protocol

For each task above:

1. **Mark In Progress**: Change `[ ]` to `[~]` before starting
2. **Write Failing Tests**: Red phase of TDD
3. **Implement to Pass**: Green phase of TDD
4. **Refactor**: Improve code quality with test safety
5. **Verify Coverage**: Ensure >80% coverage
6. **Mark Complete**: Change `[~]` to `[x]`
7. **Commit**: Stage and commit with proper message
8. **Phase Checkpoint**: If phase complete, run verification protocol

---

## Definition of Done

This track is complete when:

- [ ] All tasks in all phases marked `[x]`
- [ ] SPS benchmark shows 25,000+ with 128 environments
- [ ] All tests passing
- [ ] Code coverage >80% for new code
- [ ] Training convergence matches baseline
- [ ] Performance report documented
- [ ] Track status updated to `[x]` in `tracks.md`
- [ ] Final commit with track summary
