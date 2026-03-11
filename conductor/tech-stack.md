# Technology Stack

## Overview

JOLTrl is built on a high-performance C++ technology stack optimized for reinforcement learning training throughput. This document lists all core technologies, their versions, and their roles in the system.

---

## Core Stack

### Language & Standards

| Component | Version | Purpose |
|-----------|---------|---------|
| **C++ Standard** | C++20 (primary), C++17 (compatible) | Core language with concepts, ranges, and coroutines |
| **SIMD Extensions** | AVX2 + FMA | Hand-optimized neural network kernels |
| **Threading** | pthreads + OpenMP | CPU affinity control and parallel loops |

### Build System

| Component | Version | Purpose |
|-----------|---------|---------|
| **Bazel** | 7.0+ | Primary build system with Bzlmod |
| **Bzlmod** | Enabled | Modern dependency management |
| **hedron_compile_commands** | Latest | compile_commands.json generation for LSP |

---

## Physics & Simulation

### Physics Engine

| Component | Version | Purpose |
|-----------|---------|---------|
| **Jolt Physics** | v5.0.0 | High-fidelity rigid body dynamics |
| **Custom Patches** | Job system threading | Modified for RL workloads (sleep disabled) |

### Physics Integration

- **Custom Object Layer Filter**: Broadphase collision filtering for dimensional ghosting
- **Per-Environment Gravity Control**: `bodyInterface.SetGravityFactor()` for domain randomization
- **Contact Listener**: Custom `CombatContactListener` for impulse-based damage calculation
- **Thread Pinning**: 12 hardware threads pinned via `pthread_setaffinity_np`

---

## Neural Network Stack

### Core Architecture

| Component | Version | Purpose |
|-----------|---------|---------|
| **SPAN** | Custom | Spline-based Polynomial Approximation Network |
| **TensorProductBSpline** | Custom | B-spline basis function layers (knots=4-8, degree=2-3) |
| **RFF Layers** | cpu-tensors library | Random Fourier Features for feature embedding |

### Neural Network Components

- **Actor Network**: `[state + latent] → hidden(4,2) → action(4,2)`
- **Critic Network**: `[state + action + latent] → hidden(4,2) → Q-value(4,2)`
- **Latent Memory**: ODE2VAE-inspired second-order dynamics
- **Total Parameters**: ~10,688 per network

### Math Libraries

| Component | Version | Purpose |
|-----------|---------|---------|
| **Eigen** | 3.4+ (header-only) | Matrix operations with AVX2 enabled |
| **GLM** | 1.0.1 | OpenGL mathematics for visualization |
| **Custom NeuralMath** | Hand-written | AVX2 intrinsics (`_mm256_*`) for forward pass |

### Memory Management

- **AlignedAllocator<32>**: 32-byte aligned memory for AVX2 tensors
- **Pre-allocated Pools**: Zero allocation during training loops
- **Structure-of-Arrays (SoA)**: SIMD-friendly memory layout

---

## Reinforcement Learning

### Algorithm

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| **TD3** | Twin Delayed DDPG | Primary RL algorithm |
| **Twin Q-Networks** | Q1, Q2 | Overestimation prevention |
| **Target Networks** | Soft update (τ=0.005) | Stable training |
| **Policy Delay** | Every 2 updates | Actor-critic balance |
| **Exploration Noise** | Gaussian (σ=0.1) | Action space exploration |

### Training Components

| Component | Purpose |
|-----------|---------|
| **Prioritized Experience Replay** | Efficient sampling with sum-tree |
| **OpponentPool** | Self-play with 64-snapshot pool (70% recent + 30% random) |
| **LatentMemory** | Second-order ODE dynamics for temporal modeling |
| **Domain Randomization** | Physics parameter randomization during reset |

### Hyperparameters

```
Actor/Critic LR: 3e-4
Gamma (discount): 0.99
Tau (soft update): 0.005
Batch Size: 256
Buffer Size: 1,000,000
Policy Noise: 0.2
Noise Clip: 0.5
Policy Delay: 2
```

---

## Visualization Stack

### Rendering

| Component | Version | Purpose |
|-----------|---------|---------|
| **OpenGL** | 3.3 Core Profile | 3D rendering API |
| **GLFW** | 3.4.0 | Window management and input |
| **GLEW** | 2.2.0 | OpenGL extension loading |
| **Dear ImGui** | 1.92.2 | Immediate mode GUI for dashboards |

### Visualization Features

- **Triple-Buffered State Extraction**: Non-blocking visual updates
- **Two-Pass Rendering**: Opaque → Transparent for arena boundaries
- **LOD System**: Level-of-detail for high environment counts
- **Debug Renderer**: Jolt physics visualization (`JPH_DEBUG_RENDERER`)

### Optional Frontend

| Component | Version | Purpose |
|-----------|---------|---------|
| **Bun** | 1.3.9+ | JavaScript runtime for WASM frontend |
| **React** | Latest | UI framework for web viewer |
| **Three.js** | Latest | 3D rendering in browser |
| **TypeScript** | Latest | Type-safe frontend development |

---

## Environment & Robot System

### Environment Design

- **CombatEnv**: Two-robot arena combat simulation
- **VectorizedEnv**: Parallel environment manager (128-2048 envs)
- **Dimensional Ghosting**: Unified coordinate space with collision filtering

### Robot Configuration

- **JSON-Based**: MJCF-style robot definitions
- **Articulated Bodies**: Base + satellites (configurable count)
- **Sensor Suite**:
  - Base position/rotation (6 dim)
  - Joint positions/velocities (per satellite)
  - IMU readings (accelerometer × 3 per satellite)
  - Force sensors (contact impulse × 2 per satellite)
  - Opponent relative state (6 dim)
  - Health (1 dim)

**Total Observation**: ~256 dimensions

### Action Space

- **Joint Torques**: Normalized [-1, 1] per joint
- **Total Actions**: 56 dimensions (depends on satellite count)

---

## System Dimensions

| Parameter | Value | Source |
|-----------|-------|--------|
| **Observation Dim** | 256 | `CombatRobot.h: OBSERVATION_DIM` |
| **Action Dim** | 56 | `CombatRobot.h: ACTIONS_PER_ROBOT` |
| **Parallel Envs** | 128 (configurable to 2048) | `NeuralMath.h: NUM_PARALLEL_ENVS` |
| **Latent Dim** | 16-24 | `TD3Trainer.h: latentDim` |
| **Hidden Dim** | 128-512 | Configurable per network |
| **Batch Size** | 256 | `TD3Config` |
| **Physics Timestep** | 1/120s or 1/240s | Configurable |
| **Target SPS** | 100,000+ | Performance goal |

---

## Third-Party Dependencies

### Bazel Dependencies (MODULE.bazel)

```python
bazel_dep(name = "rules_cc", version = "0.2.17")
bazel_dep(name = "bazel_skylib", version = "1.7.1")
bazel_dep(name = "glfw", version = "3.4.0")
bazel_dep(name = "glm", version = "1.0.1")
bazel_dep(name = "glew", version = "2.2.0")
bazel_dep(name = "nlohmann_json", version = "3.11.3")
bazel_dep(name = "imgui", version = "1.92.2")
bazel_dep(name = "tinyxml2", version = "10.0.0")
```

### External Archives

- **Jolt Physics**: v5.0.0 (GitHub release)
- **Morphologica**: v4.1 (visualization library)
- **Assimp**: v5.3.1 (model loader, headers only)

### Local Libraries

- **cpu-tensors**: Custom library for RFF (Random Fourier Features) tensors
  - Location: `/home/cammyz/cpu-tensors/include/RFF.hpp`
  - Purpose: Random Fourier Features layers for neural network embedding

---

## Performance Optimizations

### Compiler Flags

```bash
-std=c++17
-O3
-mavx2
-mfma
-march=native
-ffast-math
-flto
-fno-strict-aliasing
-fopenmp
-DEIGEN_ENABLE_AVX2
-DEIGEN_DONT_PARALLELIZE
```

### Linker Flags

```bash
-lGL
-lpthread
-lgomp  # OpenMP
-flto    # Link-time optimization
```

### Threading Model

- **Main Thread**: OS + main RL loop (Core 0, Thread 0,6)
- **Physics Workers**: Jolt job pool (Cores 1-5, Threads 1-5, 7-11)
- **Background Training**: Separate thread for async training loop

---

## Development Tools

### Code Quality

| Tool | Purpose |
|------|---------|
| **clang-format** | Code formatting (`.clang-format`) |
| **LSP** | C++ language server via compile_commands.json |
| **GDB** | Debugging with core dumps |

### Testing

| Tool | Purpose |
|------|---------|
| **Custom Test Framework** | Integration tests in `src/*Test.cpp` |
| **ReplayBufferBenchmark** | Performance benchmarks |
| **SPSPerformanceTest** | Throughput validation |

### Visualization

| Tool | Purpose |
|------|---------|
| **MicroBoard** | Training log visualization |
| **Dear ImGui Dashboard** | Real-time training metrics |
| **OpenGL Viewer** | 3D environment visualization |

---

## Known Limitations

### Current Bugs

1. **AVX2 Tanh**: `_mm256_tanh_ps` is non-standard; requires custom implementation
2. **Sum-Tree Indexing**: `KLPERBuffer` tree traversal starts at idx=0 (broken)
3. **Priority Truncation**: Float priorities truncated to integers in PER
4. **Replay Buffer**: Stores current state as `nextState` (learning broken)
5. **FPS Display**: `lastRenderTime` updated before calculation

### Technical Debt

- Force sensor data not wired to environment outputs
- Damage model may overcount from proximity checks
- Inconsistent observation dimensions across modules
- Mixed aligned/unaligned AVX2 loads

---

## Future Technology Considerations

### Potential Additions

- **CUDA Support**: GPU-accelerated physics and training
- **Distributed Training**: Multi-machine scaling with MPI or NCCL
- **Alternative Algorithms**: SAC, PPO, MAPPO for multi-agent
- **Model-Based RL**: World models for sample efficiency
- **Quantization**: INT8 inference for deployment

### Deprecation Candidates

- **OpenGL**: Consider Vulkan or Metal for modern rendering
- **GLFW**: Consider SDL2 for better cross-platform support
- **Custom SPAN**: Evaluate against standard MLP for maintainability
