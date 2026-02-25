# Source Scan Report (Bug Fixes by Urgency)

## Critical
1. **`src/NeuralNetwork.cpp` does not compile**: contains `__m256 th = _mm256_tanh_ps ? _mm256_tanh_ps(x) : x;` (`_mm256_tanh_ps` is non-standard/undefined in AVX2), causing build failure.
2. **`KLPERBuffer` sum-tree indexing appears broken** (`src/NeuralNetwork.cpp`): tree traversal starts at `idx = 0` and uses `2*idx`, which repeatedly references index `0`; this can cause invalid sampling behavior.
3. **Priority precision bug in PER** (`src/NeuralNetwork.cpp`): `UpdateTree(..., static_cast<int>(priority))` truncates float priorities to integers, often zeroing small priorities and corrupting sampling distribution.

## High
1. **Replay buffer stores wrong transition in training loop** (`src/main_train.cpp`): `buffer.Add(obs, action, reward, obs, done)` uses current state as `nextState`, likely breaking TD3 learning.
2. **FPS display bug** (`src/main_train.cpp`): `lastRenderTime` is updated before FPS calculation, so `1.0f / (currentTime - lastRenderTime)` is near-infinite/unstable.
3. **Potential divide-by-zero** (`src/main_train.cpp`): `currentAvg /= std::min(rewardIdx, 100);` when `rewardIdx == 0`.

## Medium
1. **Force sensor/contact data not wired to env outputs** (`src/CombatEnv.cpp/.h`): `StepResult.forces1/forces2` exist but are never populated; contact listener tracks impulses separately and appears unused by `CombatEnv`.
2. **Damage model likely overcounts from proximity checks** (`src/CombatEnv.cpp`): per-step distance-threshold checks can apply repeated damage without true collision validation.
3. **Inconsistent observation dimensions across modules** (`src/CombatEnv.h` vs `src/CombatRobot.h`/`src/NeuralMath.h` constants) risks silent shape mismatch.

## Low
1. **Dead/unused variables/functions** (`CombatContactListener::mActiveEnv`, `ComputeEnergyUsed`) suggest incomplete integration.
2. **`alignas(32)` on `std::vector` fields** (`src/LatentMemory.h`) does not guarantee internal data alignment; can mislead SIMD safety assumptions.
3. **Some AVX2 code paths mix aligned and unaligned loads inconsistently**, increasing fragility if alignment assumptions change.
