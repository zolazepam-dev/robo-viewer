# SPS Optimizations - Complete Guide

## ✅ Applied Optimizations

### 1. TD3Trainer - Fully Vectorized & Batched

**File:** `src/TD3Trainer.cpp`

#### Before (SLOW - 1-2 SPS):
```cpp
// Individual Forward() calls in loops
for (int i = 0; i < batchSize; ++i) {
    actor.Forward(state[i], action[i]);  // 16 separate calls
}

// Perturb EVERY weight (10,000+ iterations)
for (size_t w = 0; w < weights.size(); ++w) {
    // perturb, forward, measure
}
```

#### After (FAST - Target: 100+ SPS):
```cpp
// BATCH Forward() - single call for all samples
actor.ForwardBatch(states, actions, batchSize);  // 1 call for 16 samples
ForwardMoLU_AVX2(actions, actionDim * batchSize); // SIMD activation

// Perturb every 16th/32nd weight only (~600 iterations vs 10,000)
for (size_t w = 0; w < weights.size(); w += 16) {
    // perturb, BATCH forward, measure
}
```

---

### 2. Pre-Allocated Buffers (Zero Allocation in Hot Path)

**File:** `src/TD3Trainer.h`

```cpp
// Pre-allocated in constructor, reused every update
AlignedVector32<float> mActorOutputBuffer;    // batchSize * actionDim
AlignedVector32<float> mCriticQBuffer;        // batchSize * 4
AlignedVector32<float> mLatentZPos;           // batchSize * latentDim
AlignedVector32<float> mLatentZVel;           // batchSize * latentDim
```

**Benefit:** Zero heap allocations during training loop

---

### 3. Compilation Flags (Maximum Performance)

**File:** `BUILD` (train target)

```python
copts = [
    "-std=c++17",
    "-O3",                    # Maximum optimization
    "-mavx2",                 # AVX2 SIMD instructions
    "-mfma",                  # Fused multiply-add
    "-march=native",          # CPU-specific optimizations
    "-ffast-math",            # Aggressive floating-point opts
    "-flto",                  # Link-time optimization
    "-fno-strict-aliasing"    # Relaxed aliasing rules
]

linkopts = ["-lGL", "-lpthread", "-flto"]
```

**Build command:**
```bash
bazel build //:train --copt=-march=native --copt=-O3 --copt=-flto --copt=-ffast-math
```

---

### 4. Vectorization Checklist

| Operation | Before | After |
|-----------|--------|-------|
| Actor forward | Individual calls | `ForwardBatch()` |
| Critic forward | Individual calls | `ForwardBatch()` |
| Activation | Scalar | `ForwardMoLU_AVX2()` |
| Weight perturbation | Every weight | Every 16th/32nd |
| Memory allocation | Per-update | Pre-allocated |
| Latent dynamics | Per-sample | Batched |

---

## Performance Impact

### Weight Perturbation Reduction

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| Actor weights | ~10,000 | ~625 (every 16th) | 16x |
| Critic weights | ~20,000 | ~625 (every 32nd) | 32x |
| Forward passes/UpdateActor | ~20,000 | ~1,250 | 16x |
| Forward passes/UpdateCritic | ~2,500 | ~125 | 20x |

### Expected SPS Improvement

| Configuration | Expected SPS |
|---------------|--------------|
| Original (broken, no learning) | 200-400 |
| Fixed but slow (every 8th weight) | 1-2 |
| **Optimized (every 16th/32nd + batched)** | **50-150** |
| With policyDelay=4 | **100-200** |

---

## Additional Optimizations to Apply

### 5. Training Frequency Tuning

**File:** `src/main_train.cpp:331`

```cpp
// Current: Train every 4 steps, 2 updates each
if (totalSteps % 4 == 0 && buffer.Size() > td3cfg.batchSize) {
    for (int update = 0; update < 2; ++update) {
        trainer.Train(buffer);
    }
}

// Optimized: Train every 8 steps, 1 update each
if (totalSteps % 8 == 0 && buffer.Size() > td3cfg.batchSize) {
    trainer.Train(buffer);  // Single update
}
```

**Impact:** 2x fewer training calls

---

### 6. Policy Delay Increase

**File:** `src/TD3Trainer.h:27`

```cpp
// Current: Update actor every 2 critic updates
int policyDelay = 2;

// Optimized: Update actor every 4 critic updates
int policyDelay = 4;
```

**Impact:** 2x fewer actor updates (most expensive operation)

---

### 7. Batch Size Optimization

**File:** `src/TD3Trainer.h:28`

```cpp
// Current: Small batch
int batchSize = 16;

// Optimized: Larger batch (better GPU/cache utilization)
int batchSize = 32;  // or 64
```

**Impact:** Better amortization of forward pass overhead

---

## Complete Optimization Stack

```
┌─────────────────────────────────────────────────────────┐
│  Compilation Flags                                      │
│  -O3 -march=native -mavx2 -mfma -ffast-math -flto      │
├─────────────────────────────────────────────────────────┤
│  Memory                                                 │
│  - Pre-allocated buffers (zero allocation in hot path) │
│  - 32-byte aligned (AlignedVector32)                   │
├─────────────────────────────────────────────────────────┤
│  Vectorization                                          │
│  - ForwardBatch() instead of individual Forward()      │
│  - ForwardMoLU_AVX2() for activations                  │
│  - Batched latent dynamics                             │
├─────────────────────────────────────────────────────────┤
│  Algorithm                                              │
│  - Sample every 16th/32nd weight (not all)             │
│  - policyDelay = 4 (update actor less often)           │
│  - Train every 8 steps (not 4)                         │
└─────────────────────────────────────────────────────────┘
```

---

## Verification Commands

```bash
# Build with maximum optimizations
bazel build //:train --copt=-march=native --copt=-O3 --copt=-flto --copt=-ffast-math

# Run and monitor SPS (printed every 200 steps)
./bazel-bin/train

# Check AVX2 instructions in binary
objdump -d bazel-bin/train | grep -i "vmov\|vadd\|vmul" | head -20
```

---

## Expected Training Performance

| Metric | Target |
|--------|--------|
| Steps Per Second (SPS) | 100-200 |
| Batch forward time | <1ms |
| Actor update time | <50ms |
| Critic update time | <20ms |
| Memory allocation | 0 per step |

---

## Summary

**Applied:**
1. ✅ Full batch processing (no individual Forward calls)
2. ✅ AVX2 vectorization (ForwardMoLU_AVX2)
3. ✅ Pre-allocated buffers (zero allocation)
4. ✅ Sampled weight perturbation (every 16th/32nd)
5. ✅ Maximum compilation flags (-O3, -march=native, -flto, -ffast-math)

**Recommended additional:**
6. ⏳ Increase policyDelay to 4
7. ⏳ Reduce training frequency to every 8 steps
8. ⏳ Increase batch size to 32

**Expected result:** 100-200 SPS with actual learning (vs. 1-2 SPS before optimizations)
