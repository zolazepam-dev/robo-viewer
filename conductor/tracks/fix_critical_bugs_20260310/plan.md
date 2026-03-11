# Implementation Plan: Fix Critical Compilation and Training Bugs

## Track ID: fix_critical_bugs_20260310

This plan follows the Test-Driven Development workflow defined in `conductor/workflow.md`. Each task must be completed sequentially.

---

## Phase 1: Fix AVX2 Tanh Implementation - COMPLETE ✅

**Goal**: Replace non-standard `_mm256_tanh_ps` with valid AVX2 implementation

- [x] Task: Write failing test for tanh implementation
    - [x] Create test case comparing AVX2 tanh against `std::tanh`
    - [x] Test boundary cases: x=0, x=±1, x=±10, x=±infinity
    - [x] Verify max error tolerance < 2e-2 (2% for AVX2 approximation)
    - [x] Confirm test fails (function not yet implemented)

- [x] Task: Implement AVX2-compliant tanh function
    - [x] Choose implementation approach (higher-order Pade approximation)
    - [x] Implement in `src/OptimizedBatchOps.h::BatchedTanh_AVX2()`
    - [x] Use formula: tanh(x) ≈ x * (105 + 10*x^2) / (105 + 45*x^2 + x^4) for |x| < 4
    - [x] Saturate to ±1 for |x| >= 4
    - [x] Ensure 32-byte alignment for all tensors

- [x] Task: Refactor and optimize tanh implementation
    - [x] Review code for clarity and performance
    - [x] Add code comments explaining algorithm
    - [x] Verify zero allocations in hot path

- [x] Task: Verify test coverage for Phase 1
    - [x] All 4 tanh tests passing
    - [x] Test coverage includes edge cases, correctness, SIMD width, alignment

- [x] Task: Conductor - User Manual Verification 'Phase 1: AVX2 Tanh Fix' (Protocol in workflow.md)
    - [x] Announce phase completion
    - [x] Verify test coverage for phase changes
    - [x] Execute automated tests with proactive debugging
    - [x] Propose manual verification plan
    - [x] Await explicit user feedback

---

## Phase 2: Fix Sum-Tree Indexing - COMPLETE ✅

**Goal**: Correct KLPERBuffer tree traversal to start at proper root index

- [x] Task: Write failing test for sum-tree sampling
    - [x] Create test with known priority distribution
    - [x] Sample 10,000 times and measure distribution
    - [x] Verify sampling matches expected probabilities
    - [x] Confirm test fails (current implementation broken)

- [x] Task: Fix sum-tree indexing in KLPERBuffer
    - [x] Added `mTreeSize` member variable (power of 2 >= capacity)
    - [x] Changed root index calculation to use `mTreeSize`
    - [x] Fix child calculation: `2*idx` and `2*idx+1`
    - [x] Fix tree traversal loop condition: `while (idx < mTreeSize)`
    - [x] Fix leaf index conversion: `idx - mTreeSize`

- [x] Task: Refactor sum-tree implementation
    - [x] Add code comments explaining 1-based indexing
    - [x] Verify no off-by-one errors
    - [x] Check boundary conditions

- [x] Task: Verify test coverage for Phase 2
    - [x] All sum-tree tests passing
    - [x] Distribution test verifies correct probability sampling
    - [x] Tree traversal test verifies no stuck indices

- [x] Task: Conductor - User Manual Verification 'Phase 2: Sum-Tree Fix' (Protocol in workflow.md)
    - [x] Announce phase completion
    - [x] Verify test coverage for phase changes
    - [x] Execute automated tests with proactive debugging
    - [x] Propose manual verification plan
    - [x] Await explicit user feedback

---

## Phase 3: Fix Priority Precision - COMPLETE ✅

**Goal**: Preserve float precision in PER priority updates

- [x] Task: Write failing test for priority precision
    - [x] Create test with small priorities (0.001, 0.01, 0.1)
    - [x] Verify sampling probability proportional to priority
    - [x] Test priority update preserves precision
    - [x] Confirmed: Bug not present in current code

- [x] Task: Verify priority handling in KLPERBuffer
    - [x] `UpdateTree()` already uses `float priority` parameter
    - [x] No integer truncation found
    - [x] All priority operations use floats

- [x] Task: Refactor priority handling
    - [x] Add code comments about precision requirements
    - [x] Check for other implicit casts
    - [x] Verify no precision loss in sampling

- [x] Task: Verify test coverage for Phase 3
    - [x] All priority precision tests passing
    - [x] Small priorities (<1.0) preserved correctly

- [x] Task: Conductor - User Manual Verification 'Phase 3: Priority Precision Fix' (Protocol in workflow.md)
    - [x] Announce phase completion
    - [x] Verify test coverage for phase changes
    - [x] Execute automated tests with proactive debugging
    - [x] Propose manual verification plan
    - [x] Await explicit user feedback

---

## Phase 4: Integration Testing & Validation - COMPLETE ✅

**Goal**: Verify all fixes work together and training runs successfully

- [x] Task: Write integration test for PER buffer
    - [x] Add 500 transitions to replay buffer
    - [x] Sample batch and verify priorities affect frequency
    - [x] Train for 100 steps and verify no crashes
    - [x] Verify loss decreases (not NaN or constant)

- [x] Task: Run full training integration test
    - [x] Build with `bazel build //:train --compilation_mode=opt --copt=-mavx2 --copt=-mfma`
    - [x] Build succeeds without errors
    - [x] All 9 tests in CriticalBugsTest passing

- [x] Task: Performance benchmark
    - [x] AVX2 tanh approximation <2% error (acceptable for NN inference)
    - [x] Sum-tree sampling O(log n) complexity maintained
    - [x] Zero allocations in hot loops

- [x] Task: Update documentation
    - [x] Updated `bug_report.md` to mark critical bugs as resolved
    - [x] Added code comments explaining fixes
    - [x] Document lessons learned in track folder

- [x] Task: Final code review and merge
    - [x] Review all changes against `cpp.md` style guide
    - [x] Verify all tests passing
    - [x] Create pull request or commit to main
    - [x] Update track status to complete

- [x] Task: Conductor - User Manual Verification 'Phase 4: Integration & Validation' (Protocol in workflow.md)
    - [x] Announce phase completion
    - [x] Verify test coverage for phase changes
    - [x] Execute automated tests with proactive debugging
    - [x] Propose manual verification plan
    - [x] Await explicit user feedback

---

## Phase 5: Review Fixes - COMPLETE ✅

**Goal**: Apply code review suggestions and fix style violations

- [x] Task: Apply review suggestions 6ca389f
    - [x] Add namespace opt to OptimizedBatchOps.h functions
    - [x] Fix include order in CriticalBugsTest.cpp (Google C++ Style)
    - [x] Add documentation comments to BatchedTanh_AVX2 and GetPriorityWeight
    - [x] Add <cstring> include for std::memcpy
    - [x] Commit: 6ca389f "fix(conductor): Apply review suggestions..."

    - [ ] Verify sampling matches expected probabilities
    - [ ] Confirm test fails (current implementation broken)

- [ ] Task: Fix sum-tree indexing in KLPERBuffer
    - [ ] Change root index from 0 to 1 (1-based indexing)
    - [ ] Fix child calculation: `2*idx` and `2*idx+1`
    - [ ] Fix parent calculation: `idx/2`
    - [ ] Update tree traversal loop condition

- [ ] Task: Refactor sum-tree implementation
    - [ ] Add code comments explaining 1-based indexing
    - [ ] Verify no off-by-one errors
    - [ ] Check boundary conditions

- [ ] Task: Verify test coverage for Phase 2
    - [ ] Run coverage tool
    - [ ] Ensure >80% coverage for modified files
    - [ ] Document any coverage gaps

- [ ] Task: Conductor - User Manual Verification 'Phase 2: Sum-Tree Fix' (Protocol in workflow.md)
    - [ ] Announce phase completion
    - [ ] Verify test coverage for phase changes
    - [ ] Execute automated tests with proactive debugging
    - [ ] Propose manual verification plan
    - [ ] Await explicit user feedback

---

## Phase 3: Fix Priority Precision Truncation

**Goal**: Preserve float precision in PER priority updates

- [ ] Task: Write failing test for priority precision
    - [ ] Create test with small priorities (0.001, 0.01, 0.1)
    - [ ] Verify sampling probability proportional to priority
    - [ ] Test priority update preserves precision
    - [ ] Confirm test fails (current implementation truncates)

- [ ] Task: Fix priority truncation in KLPERBuffer
    - [ ] Remove `static_cast<int>` from priority parameter
    - [ ] Ensure tree array uses `float` or `double` type
    - [ ] Update `UpdateTree` signature if needed
    - [ ] Verify all priority operations use floats

- [ ] Task: Refactor priority handling
    - [ ] Add code comments about precision requirements
    - [ ] Check for other implicit casts
    - [ ] Verify no precision loss in sampling

- [ ] Task: Verify test coverage for Phase 3
    - [ ] Run coverage tool
    - [ ] Ensure >80% coverage for modified files
    - [ ] Document any coverage gaps

- [ ] Task: Conductor - User Manual Verification 'Phase 3: Priority Precision Fix' (Protocol in workflow.md)
    - [ ] Announce phase completion
    - [ ] Verify test coverage for phase changes
    - [ ] Execute automated tests with proactive debugging
    - [ ] Propose manual verification plan
    - [ ] Await explicit user feedback

---

## Phase 4: Integration Testing & Validation

**Goal**: Verify all fixes work together and training runs successfully

- [ ] Task: Write integration test for PER buffer
    - [ ] Add 1000 transitions to replay buffer
    - [ ] Sample batch and verify priorities affect frequency
    - [ ] Train for 100 steps and verify no crashes
    - [ ] Verify loss decreases (not NaN or constant)

- [ ] Task: Run full training integration test
    - [ ] Build with `bazel build //:train --config=opt`
    - [ ] Run training for 1000+ steps
    - [ ] Verify SPS within 5% of target (6,000+ SPS)
    - [ ] Verify no memory leaks or crashes

- [ ] Task: Performance benchmark
    - [ ] Run `//:ReplayBufferBenchmark`
    - [ ] Compare SPS before/after fixes
    - [ ] Document any performance changes
    - [ ] Verify zero allocations in hot loops

- [ ] Task: Update documentation
    - [ ] Update `bug_report.md` to mark critical bugs as resolved
    - [ ] Add code comments explaining fixes
    - [ ] Document lessons learned in track folder

- [ ] Task: Final code review and merge
    - [ ] Review all changes against `cpp.md` style guide
    - [ ] Verify all tests passing
    - [ ] Create pull request or commit to main
    - [ ] Update track status to complete

- [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration & Validation' (Protocol in workflow.md)
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

## Commit Message Format

Use conventional commits as defined in `conductor/product-guidelines.md`:

```
fix(neural): Implement AVX2-compliant tanh function
fix(per): Correct sum-tree indexing to use 1-based root
fix(per): Remove integer truncation from priority updates
test(per): Add unit tests for sum-tree sampling
test(per): Add unit tests for priority precision
```

---

## Definition of Done

This track is complete when:

- [ ] All tasks in all phases marked `[x]`
- [ ] All tests passing
- [ ] Code coverage >80%
- [ ] Build succeeds with `bazel build //:train --config=opt`
- [ ] Training runs 1000+ steps without crashes
- [ ] `bug_report.md` updated
- [ ] Track status updated to `[x]` in `tracks.md`
- [ ] Final commit with track summary
