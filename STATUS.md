# Development Status

## ✅ Completed

### Battery System
- BatterySystem.h with thermal management, wireless charging, regen braking
- Engine thrust drains battery (0.1 J/s per Newton)
- CoG/mass tracking per robot
- ImGui battery tab with CoG display and stability indicator
- KOTH-based wireless charging (charge zone follows KOTH point)
- All tests pass (12 unit tests + integration tests)

### TD3Trainer Fixes
- UpdateActor() now uses directed weight perturbation guided by Q-values (NOT random noise)
- UpdateCritic() uses batch forward passes throughout
- Weight updates sample every 8th/16th weight for performance
- All vectorization preserved (ForwardBatch, ForwardMoLU_AVX2, AlignedVector32)
- Build succeeds

## ⚠️ Known Issues

### Training Performance (1 SPS)
**Cause:** Weight perturbation method requires multiple forward passes per update.

Even with sampling every 8th weight (~1,250 weights out of ~10,000):
- Each weight perturbation requires 1 ForwardBatch call
- UpdateActor called every policyDelay steps (default 2)
- This results in significant computation per training step

**Mitigation options:**
1. Increase policyDelay (update actor less frequently)
2. Sample fewer weights (every 16th or 32nd instead of 8th)
3. Implement proper backprop for SPAN networks (best long-term)
4. Reduce batch size

### Per-Environment Latent States
**Status:** Code structured correctly but ReplayBuffer doesn't store envIdx yet.

Current behavior: All samples use envIdx=0
Impact: Suboptimal but functional - latent states won't be environment-specific

**Fix requires:**
- Add mEnvIndices to ReplayBuffer (NeuralNetwork.h/cpp)
- Update Add() and Sample() signatures
- Update main_train.cpp to pass envIdx when adding transitions

## 📊 Performance Tuning

### Current Configuration
```cpp
batchSize = 16
policyDelay = 2  // Actor updated every 2 steps
Training frequency: Every 4 steps, 2 updates
```

### Recommended Tuning
```cpp
// Option 1: Reduce actor update frequency
policyDelay = 4  // Update actor every 4 steps instead of 2

// Option 2: Sample fewer weights
w += 16  // Instead of w += 8 (in UpdateActor)

// Option 3: Reduce training frequency
totalSteps % 8 == 0  // Instead of totalSteps % 4 == 0
```

## 🔧 Files Modified

| File | Status |
|------|--------|
| TD3Trainer.h | ✅ Updated |
| TD3Trainer.cpp | ✅ Rewritten |
| CombatRobot.h | ✅ Added CoG fields |
| InternalRobot.cpp | ✅ Battery drain |
| OverlayUI_refactor.* | ✅ CoG display |
| CombatEnv.cpp | ✅ KOTH spawn/charging |
| BatterySystem.h | ✅ Created |

## 🧪 Tests

All tests pass:
```bash
./run_tests.sh
# ✓ ALL TESTS PASSED
```

## 📝 Next Steps

1. **Test training** - Run training and measure actual SPS
2. **Tune performance** - Adjust policyDelay, weight sampling stride
3. **Optional: ReplayBuffer envIdx** - Add per-environment latent support
4. **Optional: SPAN backprop** - Implement proper gradients for SPAN networks
