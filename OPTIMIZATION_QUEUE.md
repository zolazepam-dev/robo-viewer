## OPT-005: Reduce Physics Substeps

**Goal:** Reduce physics computation time by reducing substeps per environment step

**Current:** Physics runs at 60Hz with multiple substeps  
**Target:** Run physics at minimum viable fidelity

**Change:** Modify `VectorizedEnv::Step()` to use single physics step instead of multiple substeps

**Expected Impact:** 30-50% SPS gain  
**Risk:** Medium - may affect physics accuracy  
**Rollback:** `git revert HEAD~1`

---

## OPT-006: Batch Action Selection

**Goal:** Process all environment actions in single batch call

**Current:** Separate calls for agent1 and agent2  
**Target:** Single batch call for all actions

**Expected Impact:** 20-30% SPS gain  
**Risk:** Low - just API change  
**Rollback:** `git revert HEAD~1`

---

## OPT-007: Disable Unnecessary Logging

**Goal:** Remove all remaining fprintf/stderr logging

**Current:** Training prints step info, init messages  
**Target:** Zero logging in production

**Expected Impact:** 5-10% SPS gain  
**Risk:** Low  
**Rollback:** `git revert HEAD~1`

---

## OPT-008: Increase OpenMP Thread Count

**Goal:** Better CPU utilization

**Current:** 8 threads hardcoded  
**Target:** Use all available CPU cores

**Expected Impact:** 20-40% SPS gain  
**Risk:** Low  
**Rollback:** `git revert HEAD~1`

---

## OPT-009: Pre-allocate All Buffers

**Goal:** Zero allocation in hot path

**Current:** Some buffers allocated per-step  
**Target:** All buffers pre-allocated at startup

**Expected Impact:** 10-20% SPS gain  
**Risk:** Medium  
**Rollback:** `git revert HEAD~1`

---

## OPT-010: SIMD-Optimized Physics Step

**Goal:** Vectorize physics calculations

**Current:** Scalar physics  
**Target:** AVX2 vectorized physics

**Expected Impact:** 50-100% SPS gain  
**Risk:** High - major rewrite  
**Rollback:** `git revert HEAD~1`
