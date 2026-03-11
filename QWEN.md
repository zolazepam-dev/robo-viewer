# JOLTrl - QWEN Context Guide

## Project Overview

**JOLTrl** (Jolt Optimized Learning for Robots) is a high-performance C++ reinforcement learning framework for training autonomous combat robots. Built on the Jolt Physics engine, it achieves exceptional throughput (100,000+ steps per second) through:

- **Parallel Environment Simulation**: 128+ environments running simultaneously using "Dimensional Ghosting" technique
- **Zero-Allocation Training**: Pre-allocated memory pools eliminate GC pressure during hot loops
- **SIMD-Accelerated Networks**: Hand-optimized AVX2/FMA neural network kernels (SPAN architecture)
- **Dual Mode Operation**: Headless training for maximum SPS, OpenGL visualization for debugging

### Key Technologies

| Category | Technology |
|----------|------------|
| **Physics Engine** | Jolt Physics v5.0.0 |
| **Build System** | Bazel 7.0+ (Bzlmod) |
| **Neural Network** | Custom SPAN (B-spline) architecture |
| **RL Algorithm** | TD3 (Twin Delayed DDPG) |
| **Visualization** | OpenGL 3.3 + GLFW + Dear ImGui |
| **Math Library** | GLM + Eigen (header-only) |
| **Frontend** | React + Three.js + Bun (optional WASM viewer) |

---

## Project Structure

```
robo-viewer/
├── BUILD                      # Bazel build definitions
├── MODULE.bazel               # Bazel dependencies (Bzlmod)
├── AGENTS.md                  # Agentic development guidelines
├── DOCS.md                    # Complete documentation
├── bug_report.md              # Known bugs and issues
├── RL_TECH_STACK.md           # RL architecture reference
├── MICROBOARD.md              # MicroBoard visualization tool
├── build_and_run.sh           # Interactive build/run script
├── train_parallel.sh          # Parallel training script
│
├── src/                       # Core source files
│   ├── main_train.cpp         # Headless RL training entry point
│   ├── combat_main.cpp        # Visual debugging entry point
│   ├── PhysicsCore.h/cpp      # High-performance Jolt wrapper
│   ├── CombatEnv.h/cpp        # Combat simulation environment
│   ├── VectorizedEnv.h/cpp    # Parallel environment manager (128+ envs)
│   ├── SpanNetwork.h/cpp      # SPAN neural architecture (B-spline layers)
│   ├── NeuralMath.h/cpp       # SIMD-optimized matrix operations
│   ├── LatentMemory.h/cpp     # ODE2VAE-style second-order latent memory
│   ├── TD3Trainer.h/cpp       # Twin Delayed DDPG training loop
│   ├── OpponentPool.h/cpp     # Self-play opponent sampling
│   ├── ReplayBufferBenchmark.cpp  # Performance benchmarks
│   ├── Renderer.h/cpp         # OpenGL system (viewer only)
│   ├── OverlayUI.h/cpp        # Dear ImGui training dashboard
│   └── AlignedAllocator.h     # 32-byte aligned memory for AVX2
│
├── robots/                    # Robot configuration JSON files
│   ├── combat_bot.json        # Combat robot definition
│   └── test_bot.json          # Test robot for visualizer
│
├── third_party/               # Custom Bazel build files
│   ├── jolt.BUILD             # Optimized Jolt compilation rules
│   ├── morphologica.BUILD     # Morphologica visualization library
│   └── assimp.BUILD           # Assimp model loader
│
├── checkpoints/               # Training checkpoints (git-ignored)
├── saved_models/              # Exported trained policies (git-ignored)
└── wasm/                      # WASM frontend build (optional)
```

---

## Building and Running

### Prerequisites

- **OS**: Linux (Pop!_OS/Ubuntu tested), macOS, Windows 10+
- **Compiler**: GCC 11+, Clang 14+, or MSVC 2022+
- **Bazel**: 7.0+ (via Bazelisk recommended)
- **CPU**: Intel/AMD with AVX2 support (i5-10500 or better)
- **GPU**: OpenGL 3.3+ (viewer only)

### Quick Start

```bash
# 1. Build with maximum optimizations
bazel build //:train \
    --compilation_mode=opt \
    --copt=-march=native \
    --copt=-O3 \
    --copt=-flto \
    --copt=-ffast-math

# 2. Run training (default: 128 environments, ~6,000 SPS)
bazel run //:train --config=opt

# 3. Run with more environments (256 envs, ~12,000-15,000 SPS)
bazel run //:train --config=opt -- --envs 256
```

### Using Helper Scripts

```bash
# Interactive build and run
./build_and_run.sh

# Parallel training with configurable environments
NUM_ENVS=64 ./train_parallel.sh all

# Clean build
./build_and_run.sh --clean

# Debug build
./build_and_run.sh --debug
```

### Build Targets

| Target | Description |
|--------|-------------|
| `//:train` | Headless RL training (maximum performance) |
| `//:viewer` | OpenGL visualization for debugging |
| `//:train_headless` | Alternative headless training target |
| `//:micro_board` | Training log visualization tool |
| `//:system_test` | System integration tests |

### Build Options

```bash
# Full optimization build (recommended for training)
bazel build //:train \
    --compilation_mode=opt \
    --copt=-march=native \
    --copt=-O3 \
    --copt=-flto \
    --copt=-ffast-math

# Debug build with symbols
bazel build //:train --compilation_mode=dbg

# Fast build without optimizations
bazel build //:train --compilation_mode=fastbuild

# Clean everything
bazel clean --expunge
```

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING ENGINE (//:train)                  │
├─────────────────────────────────────────────────────────────────┤
│  VectorizedEnv ──► CombatEnv ──► CombatRobot ──► PhysicsCore    │
│        │               │               │               │         │
│        ▼               ▼               ▼               ▼         │
│  TD3Trainer ◄──► SpanNetwork ◄──► NeuralMath ◄──► Jolt        │
│        │               │                                       │
│        ▼               ▼                                       │
│  LatentMemory ◄──► OpponentPool                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VIEWER (//:viewer)                          │
├─────────────────────────────────────────────────────────────────┤
│  OpenGL/GLFW ◄──► Renderer ◄──► OverlayUI (Dear ImGui)         │
└─────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Layer | Components | Purpose |
|-------|------------|---------|
| **Physics** | `PhysicsCore`, `CombatRobot`, `RobotLoader` | Jolt Physics wrapper, robot definitions |
| **Environment** | `CombatEnv`, `VectorizedEnv`, `OpponentPool` | Combat simulation, parallel execution |
| **Neural Network** | `SpanNetwork`, `NeuralMath`, `LatentMemory` | SPAN architecture, SIMD operations |
| **Training** | `TD3Trainer`, `ReplayBuffer` | RL algorithm, experience replay |
| **Visualization** | `Renderer`, `OverlayUI` | OpenGL rendering, ImGui dashboard |
| **Utilities** | `AlignedAllocator`, `LockFreeQueue` | 32-byte aligned memory, thread-safe queues |

---

## Neural Architecture

### SPAN Network (Spline-based Polynomial Approximation Network)

JOLTrl implements a novel neural architecture using B-spline basis functions instead of traditional MLP layers.

#### TensorProductBSpline Layer

- **Purpose**: Smooth, continuous function approximation
- **Advantages**: Fewer parameters than equivalent MLP
- **Optimization**: AVX2-optimized forward pass (`ForwardAVX2()`)
- **Configuration**: Configurable knots (default: 8) and spline degree (default: 3)

#### Architecture Diagram

```
Input (Observation: ~256 dims)
    │
    ▼
┌─────────────────────────┐
│ TensorProductBSpline    │  ← B-spline basis functions
│ (stateDim → hiddenDim)  │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   LatentMemory (ODE)    │  ← Second-order dynamics
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ TensorProductBSpline    │  ← B-spline basis functions
│ (hiddenDim → actionDim) │
└─────────────────────────┘
    │
    ▼
Output (Actions: 56 dims)
```

### Latent Memory (ODE2VAE-Inspired)

The `SecondOrderLatentMemory` system provides:

- **Velocity and Acceleration States**: For temporal modeling
- **Second-Order Dynamics**: Better long-horizon credit assignment
- **Integration**: Seamlessly integrated with SPAN layers via `ForwardWithLatent()`

### SIMD Optimization

All neural network kernels are hand-optimized for AVX2/FMA:

- **8-wide Float Operations**: Maximum throughput per cycle
- **Aligned Memory Access**: 32-byte aligned buffers via `AlignedAllocator`
- **Fused Multiply-Add**: Single-instruction polynomial evaluation

---

## RL Algorithm: TD3

### Twin Delayed DDPG Implementation

| Component | Implementation |
|-----------|----------------|
| **Actor** | SPAN network with latent memory |
| **Critic** | Twin Q-networks (Q1, Q2) for overestimation prevention |
| **Target Networks** | Soft update (τ=0.005) with delayed updates |
| **Policy Delay** | Actor updated every 2 critic updates |
| **Exploration** | Gaussian noise on actions |
| **Discount (γ)** | 0.99 |

### TD3 Hyperparameters (`TD3Config`)

```cpp
struct TD3Config {
    int hiddenDim = 128;          // Hidden layer dimension
    int latentDim = 24;           // Latent space dimension
    float actorLR = 3e-4f;        // Actor learning rate
    float criticLR = 3e-4f;       // Critic learning rate
    float gamma = 0.99f;          // Discount factor
    float tau = 0.005f;           // Target network soft update
    float policyNoise = 0.2f;     // Target policy smoothing noise
    float noiseClip = 0.5f;       // Noise clipping range
    float explNoise = 0.1f;       // Exploration noise
    int policyDelay = 2;          // Policy update delay
    int batchSize = 256;          // Training batch size
    int bufferSize = 1000000;     // Replay buffer capacity
};
```

### Self-Play: OpponentPool

- **Pool Size**: 64 snapshots
- **Sampling Strategy**: 70% recent opponents, 30% random
- **Snapshot Contents**: Actor weights + biases + win rate

---

## Physics System

### Jolt Physics Integration

**CRITICAL**: Every `.cpp` file that interacts with physics **MUST** start with:

```cpp
#include <Jolt/Jolt.h>  // ABSOLUTE FIRST - before any other headers
#include <Jolt/RegisterTypes.h>
// ... other Jolt headers
// ... other includes
```

This is non-negotiable and prevents catastrophic macro expansion errors.

### Dimensional Ghosting

Parallel environments do **not** exist in separate `PhysicsSystem` instances:

- All training agents spawn into a single, unified coordinate space (0,0,0)
- Inter-robot collisions are bypassed at the broadphase level via custom `ObjectLayerPairFilter`
- No per-environment physics overhead
- Each environment gets its own exclusive object layer (starting from layer 1)

### Thread Pinning

To maximize the 12 available hardware threads and prevent OS context switching:

| Core | Threads | Assignment |
|------|---------|------------|
| Core 0 | 0, 6 | OS + main RL loop |
| Cores 1-5 | 1-5, 7-11 | Jolt worker pool |

Thread affinity is set via `pthread_setaffinity_np`.

### Sleep Mechanics

Jolt's sleep mechanics are **disabled globally**. RL agents constantly explore and must never be put to sleep:

```cpp
mAllowSleeping = false;  // See PhysicsCore.cpp
```

---

## Environment Design

### Combat Environment (`CombatEnv`)

The combat environment simulates two robots in an arena with:

- **Damage System**: Contact-based damage calculation
- **Reward Signals**: Multi-objective reward vector
- **Termination Conditions**: Health depletion, time limit, arena bounds

### Observation Space

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Base position/rotation | 6 | Robot chassis state |
| Joint positions | NUM_SATELLITES | Articulated joint angles |
| Joint velocities | NUM_SATELLITES | Angular velocities |
| IMU readings | NUM_SATELLITES × 3 | Accelerometer data |
| Force sensors | NUM_SATELLITES × 2 | Contact impulse magnitudes |
| Opponent relative | 6 | Relative position/rotation |
| Health | 1 | Remaining HP (0-100) |

**Total**: ~256 dimensions (depends on `NUM_SATELLITES` configuration)

### Action Space

| Action | Range | Description |
|--------|-------|-------------|
| Joint torques | [-1, 1] | Normalized torque per joint |

Actions are continuous and clipped to valid torque ranges per joint.

### Reward Structure

| Component | Weight | Description |
|-----------|--------|-------------|
| `damage_dealt` | +1.0 | Damage inflicted on opponent |
| `damage_taken` | -0.5 | Damage received |
| `alive_bonus` | +0.1 | Survival per step |
| `air_time` | -0.01 | Penalty for being airborne |
| `energy_efficiency` | +0.001 | Reward for efficient movement |
| `center_proximity` | +0.05 | Arena control incentive |

---

## Development Guidelines

### Code Style

| Convention | Description |
|------------|-------------|
| **C++ Standard** | C++20 (`-std=c++20`) |
| **Classes/Structs** | PascalCase (`PhysicsCore`, `SpanNetwork`) |
| **Functions/Methods** | PascalCase (`Init()`, `ResetEpisode()`) |
| **Variables** | camelCase (`physics`, `observationState`) |
| **Constants** | UPPER_CASE or kCamelCase (`NUM_PARALLEL_ENVS`) |
| **Private Members** | `m` prefix (`mPhysicsSystem`, `mJobSystem`) |

### The Zero-Allocation Mandate

Memory allocation during the training loop destroys performance:

- **Pre-allocate Everything**: All robot state tensors allocated in massive, contiguous blocks at startup
- **No Spawning/Destroying**: When episodes terminate, zero velocities and overwrite transforms (The "Necromancer" Reset)
- **Avoid STL Containers**: No `std::vector::push_back` in hot loops

### Performance Standards

At 6,000+ SPS, every microsecond counts:

- **Zero-Allocation Hot Loops**: Never use `new`, `malloc`, or `std::vector::push_back` inside `SimulationLoop` or `TrainingLoop`
- **SIMD First**: Prefer AVX2/FMA intrinsics via `NeuralMath.h`. Always verify alignment (32-byte) for tensors
- **Lock-Awareness**: Minimize `gSimMutex` hold times. Use triple-buffering pattern for visual state extraction
- **Scalability**: Support at least **2048 parallel environments**

### Error Handling

- Use early returns for initialization errors
- **No logging in hot training loops** unless fatal (I/O bottlenecks SPS)
- Use assertions for invariant checks

---

## Testing and Validation

### Agent Validation Workflow

1. **Research**: Use `grep_search` and `get_code_context_exa` to map dependencies
2. **Reproduction**: Create a minimal test case to confirm the issue
3. **Surgical Implementation**: Use `edit` for targeted edits; avoid mass-rewriting files
4. **Validation**: Run `bazel build //:train` and verify metrics

### Running Tests

```bash
# System integration test
bazel build //:system_test
bazel run //:system_test

# Replay buffer benchmark
bazel run //:ReplayBufferBenchmark
```

---

## Known Issues

See `bug_report.md` for detailed tracking. Summary:

### Critical

1. **`src/NeuralNetwork.cpp` compilation failure**: `_mm256_tanh_ps` is non-standard in AVX2
2. **`KLPERBuffer` sum-tree indexing broken**: Tree traversal starts at `idx = 0`, causing invalid sampling
3. **Priority precision in PER**: Float priorities truncated to integers, corrupting distribution

### High Priority

1. **Replay buffer stores wrong transition**: Uses current state as `nextState` in `main_train.cpp`
2. **FPS display bug**: `lastRenderTime` updated before FPS calculation
3. **Potential divide-by-zero**: When `rewardIdx == 0`

### Medium Priority

1. **Force sensor data not wired**: `StepResult.forces1/forces2` never populated
2. **Damage model overcounts**: Per-step distance checks without collision validation
3. **Inconsistent observation dimensions**: Shape mismatch between modules

---

## Troubleshooting

### Build Failures

**Problem**: Build fails with SIMD-related errors
```bash
# Solution: Ensure CPU supports AVX2
cat /proc/cpuinfo | grep avx2

# If missing, remove AVX2 flags from BUILD file
```

**Problem**: Jolt Physics include errors
```bash
# Solution: Verify include order - Jolt/Jolt.h must be FIRST
# Check that third_party/jolt.BUILD is correctly configured
```

### Runtime Issues

**Problem**: Segmentation fault during training
```bash
# Enable core dumps for debugging
ulimit -c unlimited

# Run with debug build
bazel run //:train --compilation_mode=dbg
```

**Problem**: Low Steps Per Second (SPS)
```bash
# Check thread pinning
top -H -p $(pgrep train)

# Verify AVX2 is enabled
bazel build //:train --copt=-march=native --verbose_failures
```

**Problem**: Viewer fails to open window
```bash
# Check OpenGL support
glxinfo | grep "OpenGL version"

# Ensure display is available
export DISPLAY=:0
```

---

## Useful Commands

```bash
# Format C++ code
clang-format -i src/*.cpp src/*.h

# Generate compile_commands.json for LSP
bazel run //:refresh_compile_commands

# View training logs with MicroBoard
cat logs.csv | bazel run //:micro_board

# Profile training performance
bazel run //:train --config=opt -- --envs 64
```

---

## Key Files for Context

| File | Purpose |
|------|---------|
| `src/PhysicsCore.h/cpp` | Jolt Physics wrapper with dimensional ghosting |
| `src/CombatEnv.h/cpp` | Combat environment with reward calculation |
| `src/VectorizedEnv.h/cpp` | Parallel environment manager (128+ envs) |
| `src/SpanNetwork.h/cpp` | B-spline neural network architecture |
| `src/NeuralMath.h/cpp` | SIMD-optimized matrix operations |
| `src/LatentMemory.h/cpp` | ODE2VAE-style latent memory |
| `src/TD3Trainer.h/cpp` | TD3 training loop implementation |
| `src/TD3Config` | Training hyperparameters |
| `AGENTS.md` | Agentic development guidelines |
| `DOCS.md` | Complete documentation |
| `RL_TECH_STACK.md` | RL architecture reference |
| `bug_report.md` | Known bugs and issues |

---

## Current Status

**Phase**: Scalable Environments & Domain Randomization
**Last Updated**: March 9, 2026

---

## Quick Reference

### System Dimensions

| Parameter | Value |
|-----------|-------|
| **Observation Dim** | ~256 |
| **Action Dim** | 56 |
| **Parallel Envs** | 128 (configurable) |
| **Latent Dim** | 24 |
| **Hidden Dim** | 128 |
| **Batch Size** | 256 |
| **Physics Timestep** | 1/120s |
| **Target SPS** | 100,000+ |

### Network Parameters

- **Total per network**: ~10,688 params
- **Actor L1**: ~2,624 params
- **Actor L2**: ~2,816 params
- **Critic L1**: ~2,688 params
- **Critic L2**: ~2,560 params
