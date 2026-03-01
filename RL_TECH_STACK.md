# JOLTrl Reinforcement Learning Tech Stack

## What We Actually Have

---

## System Dimensions

| Parameter | Value | Source |
|-----------|-------|--------|
| **Observation Dim** | 256 | `CombatRobot.h: OBSERVATION_DIM` |
| **Action Dim** | 56 | `CombatRobot.h: ACTIONS_PER_ROBOT` |
| **Parallel Envs** | 128 | `NeuralMath.h: NUM_PARALLEL_ENVS` |
| **Latent Dim** | 16 | `TD3Trainer.h: latentDim` |
| **Hidden Dim** | 512 | `NeuralMath.h: ACTOR_HIDDEN_DIM` |
| **Batch Size** | 256 | `TD3Config` |

---

## Brain: SPAN Network

### TensorProductBSpline Layer
- **numKnots**: 4 (`SpanLayerConfig`)
- **splineDegree**: 2 (quadratic B-spline)
- AVX2-optimized forward pass

### Layer Config (from `SpanNetwork.cpp:312-323`)
```
Actor:  [state + latent] → hidden(4,2) → action(4,2)
Critic: [state + action + latent] → hidden(4,2) → Q-value(4,2)
```

### Parameter Count
- **Actor L1**: ~2,624 params
- **Actor L2**: ~2,816 params  
- **Critic L1**: ~2,688 params
- **Critic L2**: ~2,560 params
- **Total ~10,688 params** per network

---

## Latent Memory: ODE2VAE-Inspired

### SecondOrderLatentMemory (`LatentMemory.h:13`)
- `z_pos[]` - position state
- `z_vel[]` - velocity state
- Per-environment: 128 parallel latent states

### Components
- **ODE2VAEEncoder** - observation → latent
- **ODE2VAEDynamics** - computes accelerations
- **LatentMemoryManager** - orchestrates per-env latent

### Dynamics (from `LatentMemory.h:61`)
```
StepDynamicsScalar/Vectorized: z_pos += z_vel * dt
```

---

## Algorithm: TD3 (Twin Delayed DDPG)

### Implemented in `TD3Trainer.cpp`

| Component | Implementation |
|-----------|----------------|
| **Actor** | SPAN network |
| **Critic** | Twin Q-networks (Q1, Q2) |
| **Target** | Soft update (τ=0.005) |
| **Policy Delay** | Every 2 updates |
| **Exploration** | Gaussian noise |
| **γ** | 0.99 |

### Hyperparameters (`TD3Config`)
```
learningRate: 3e-4
bufferSize: 1,000,000
batchSize: 256
gamma: 0.99
tau: 0.005
policyNoise: 0.1
noiseClip: 0.5
policyDelay: 2
```

---

## Vector Rewards

### `VectorReward` struct (`NeuralNetwork.h:19`)
```cpp
struct VectorReward {
    float damage_dealt;  // HP removed from opponent
    float damage_taken;  // HP lost
    float koth;          // King of the hill / control
    float energy_used;   // Action energy cost
    float altitude;     // Height bonus
};
```

### Scalar Conversion (`NeuralNetwork.h:31`)
- `Dot(preference)` - weighted sum with preference vector
- `Scalar()` - simple sum

### Weights (from `main_train.cpp`)
```
damage_dealt: +1.0
damage_taken: -0.5
```

---

## Self-Play: OpponentPool

### `OpponentPool` (`OpponentPool.h`)
- **Pool Size**: 64 snapshots (`MAX_POOL_SIZE`)
- **Recent Bias**: 70% recent, 30% random
- **Snapshot**: actor weights + biases + winRate

### What's NOT Implemented
- ❌ ELO rating system
- ❌ Pareto front optimization
- ❌ League statistics

---

## Training Stability

| Technique | Where |
|-----------|-------|
| Target Network | `TD3Trainer::UpdateTargets` |
| Experience Replay | `ReplayBuffer` (1M) |
| Gradient Clipping | Config |
| 32-byte Alignment | `AlignedAllocator` |
| AVX2 SIMD | `NeuralMath.cpp` |

---

## Environment

- **Physics**: Jolt Physics
- **Timestep**: 1/120s
- **Sleep**: Disabled
- **CPU**: 12 threads, pinned

---

## File Map

| File | Purpose |
|------|---------|
| `SpanNetwork.cpp/h` | SPAN B-spline network |
| `LatentMemory.cpp/h` | ODE2VAE latent |
| `TD3Trainer.cpp/h` | TD3 training |
| `OpponentPool.cpp/h` | Self-play pool |
| `VectorizedEnv.cpp/h` | 128 parallel envs |
| `CombatEnv.cpp/h` | Combat environment |
| `CombatRobot.cpp/h` | Robot definition |
| `NeuralMath.cpp/h` | SIMD ops |
| `NeuralNetwork.cpp/h` | Legacy + ReplayBuffer |
