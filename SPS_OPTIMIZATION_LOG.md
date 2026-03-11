# SPS Optimization Log - Phase 2

## Goal: 300 → 3000+ SPS (10x improvement)

**Baseline:** 300 SPS (after Phase 1)  
**Target:** 3000+ SPS  
**Current Status:** In Progress

---

## Optimization Change Log

### ✅ OPT-001: Remove Profiling Overhead (COMMITTED)
**File:** `src/main_train.cpp`  
**Changes:**
- Removed `fprintf` logging in simulation loop
- Removed `DIAGNOSE_SCOPE` from action selection
- Removed latent state capture loop from DIAGNOSE_SCOPE

**Expected Impact:** 5-10% SPS gain  
**Risk:** Low - just removes logging  
**Rollback:** `git revert HEAD~1`

---

### ✅ OPT-002: Training Loop Optimization (COMMITTED)
**File:** `src/main_train.cpp`  
**Changes:**
- Reduced buffer threshold: 1024 → 512
- Changed sleep(5ms) → yield() for lower latency
- Removed DIAGNOSE_SCOPE from training loop

**Expected Impact:** 10-20% SPS gain  
**Risk:** Low  
**Rollback:** `git revert HEAD~1`

---

### ✅ OPT-003: Parallel Replay Buffer Add
**File:** `src/main_train.cpp`  
**Changes:**
- Added `#pragma omp parallel for` to replay buffer add loop
- Using `schedule(dynamic, 16)` for load balancing
- Removed DIAGNOSE_SCOPE overhead

**Expected Impact:** 15-25% SPS gain  
**Risk:** Medium - potential race conditions in buffer (needs testing)  
**Rollback:** `git revert HEAD~1`

### ✅ OPT-004: Remove Physics Mutex Lock
**File:** `src/main_train.cpp`  
**Changes:**
- Removed `DIAGNOSE_MUTEX_LOCK(gSimMutex, "SimulationLoop: PhysicsUpdate")`
- VecEnv already has internal synchronization

**Expected Impact:** 10-20% SPS gain  
**Risk:** Low - mutex was redundant  
**Rollback:** `git revert HEAD~1`

---

### 📝 OPT-005: [PENDING]

## Rollback Instructions

If something breaks after multiple optimizations:

```bash
# View recent commits
git log --oneline -10

# Revert last commit
git revert HEAD

# Revert specific commit
git revert <commit-hash>

# Revert to Phase 1 baseline
git revert HEAD~5..HEAD
```

## Testing Checklist

After each optimization:
- [ ] Both robots visible in viewer
- [ ] Robots are moving (not frozen)
- [ ] SPS counter shows value > baseline
- [ ] No crash within 30 seconds
- [ ] Training metrics updating

---

**Last Updated:** [Current timestamp]  
**Current SPS:** [To be measured]
