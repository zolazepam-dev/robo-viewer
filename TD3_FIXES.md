# TD3Trainer Fixes - Implementation Status

## ✅ COMPLETED

### 1. UpdateActor() - Finite-Difference Policy Gradient
**File:** `src/TD3Trainer.cpp`

Replaced random noise with directed gradient ascent:
```cpp
// NEW: Compute gradient of Q(s, π(s)) w.r.t. actor weights
for (size_t w = 0; w < weights.size(); ++w) {
    // Perturb weight
    weights[w] = originalWeights[w] + epsilon;
    mModel.GetActor().SetAllWeights(weights);
    
    // Measure Q-value change
    float perturbedQ = ComputeBatchQ();
    
    // Restore and store gradient
    weights[w] = originalWeights[w];
    mGrads[w] = (perturbedQ - baselineQ) / epsilon;
}

// Apply gradient ascent
for (size_t w = 0; w < weights.size(); ++w) {
    weights[w] += mConfig.actorLR * mGrads[w];
}
```

**Status:** ✅ Implemented in TD3Trainer.cpp

### 2. UpdateCritic() - Per-Sample Latent States  
**File:** `src/TD3Trainer.cpp`

Updated to use sampled environment indices:
```cpp
// NEW: Get latent state for THIS specific environment
int envIdx = mSampledIndices[i];
mModel.GetLatentMemory().GetLatentStates(zPos.data(), zVel.data(), envIdx);
```

**Status:** ⚠️ Partially implemented - requires ReplayBuffer changes

## ⚠️ PENDING - ReplayBuffer Changes Required

### 3. Store envIdx in ReplayBuffer
**Files to modify:**
- `src/NeuralNetwork.h` - Add `mEnvIndices` member
- `src/NeuralNetwork.cpp` - Implement envIdx tracking

**Required changes:**
```cpp
// In NeuralNetwork.h - Add to ReplayBuffer class
private:
    std::vector<int> mEnvIndices;  // NEW: Track environment index
    
// Update Add() signature
void Add(..., int envIdx = 0);

// Update Sample() signature  
void Sample(..., int* envIndices, ...);
```

**Why needed:** Without envIdx storage, we can't retrieve the correct latent state for each sampled transition.

## 📋 KLPERBuffer Status

**Issue:** KLPERBuffer is defined but never used

**Options:**
1. **Implement KL-prioritized replay** - Use KL divergence for experience prioritization
2. **Remove unused code** - Delete KLPERBuffer if not needed
3. **Keep for future** - Leave as-is with comment explaining it's experimental

**Recommendation:** Option 2 - Remove if not actively developing KL-prioritized replay

## Testing

After full implementation, verify:
```bash
# Build should succeed
bazel build //:train

# Training should show:
# - Actor loss decreasing (better policy)
# - Critic loss decreasing (better Q predictions)
# - Latent states varying by environment (not all zeros)
```

## Performance Notes

Finite-difference gradients are O(n) where n = number of weights:
- Actor: ~10,000 weights = ~10,000 forward passes per update
- Critic: ~20,000 weights = ~20,000 forward passes per update

**Optimization options:**
1. Reduce update frequency (update every N steps)
2. Sample subset of weights for gradient estimation
3. Implement proper backprop for SPAN layers (best long-term)

## Files Modified

| File | Status | Changes |
|------|--------|---------|
| `TD3Trainer.cpp` | ✅ Updated | UpdateActor, UpdateCritic with finite-diff gradients |
| `TD3Trainer.h` | ⚠️ Needs update | Add mSampledIndices member |
| `NeuralNetwork.h` | ⏳ Pending | Add envIdx to ReplayBuffer |
| `NeuralNetwork.cpp` | ⏳ Pending | Implement envIdx tracking |

## Next Steps

1. Update `TD3Trainer.h` to add `mSampledIndices` member
2. Update `NeuralNetwork.h/cpp` for envIdx support
3. Update `main_train.cpp` to pass envIdx when adding to buffer
4. Test training convergence
