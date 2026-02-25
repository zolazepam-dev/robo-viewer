# JOLTrl - High-Performance Robot Combat RL Environment

A bespoke C++ reinforcement learning environment for training autonomous combat robots using the Jolt Physics engine. Optimized for 6-core/12-thread CPU architectures with zero-allocation training loops and SIMD-accelerated neural networks.

## Overview

JOLTrl (Jolt Optimized Learning for Robots) is a high-throughput training system for robot combat scenarios. It features:

- **Parallel Environment Simulation**: 128+ environments running simultaneously
- **SPAN/ODE2VAE Neural Architecture**: Tensor-product B-spline networks with latent memory
- **Zero-Allocation Training**: Pre-allocated memory pools, no GC pressure during training
- **AVX2/FMA SIMD**: Hand-optimized neural network kernels
- **Headless + Visual Modes**: Train at maximum SPS or visualize with OpenGL

## System Architecture

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

## Core Components

### Physics Layer
| Component | Purpose |
|-----------|---------|
| `PhysicsCore` | Jolt Physics wrapper with thread pinning & CPU affinity |
| `CombatRobot` | Robot body definition, joint control, sensor reading |
| `RobotLoader` | JSON-based robot configuration loading |

### RL Environment
| Component | Purpose |
|-----------|---------|
| `CombatEnv` | Combat arena simulation (2 robots, damage, rewards) |
| `VectorizedEnv` | Parallel environment management (128+ instances) |
| `OpponentPool` | Opponent sampling for self-play curriculum |

### Neural Networks
| Component | Purpose |
|-----------|---------|
| `SpanNetwork` | SPAN (Spline-based Polynomial Approximation Network) |
| `TensorProductBSpline` | B-spline basis function layer with AVX2 kernels |
| `LatentMemory` | Second-order ODE latent memory (ODE2VAE-inspired) |
| `NeuralMath` | SIMD-optimized matrix operations |
| `TD3Trainer` | Twin Delayed Deep Deterministic Policy Gradient |

### Utilities
| Component | Purpose |
|-----------|---------|
| `AlignedAllocator` | 32-byte aligned memory for AVX2 |
| `LockFreeQueue` | Thread-safe experience replay buffer |

## Build Requirements

- **OS**: Linux (Pop!_OS tested), macOS, Windows
- **Compiler**: GCC 11+, Clang 14+, or MSVC 2022+
- **Bazel**: 6.0+ with Bzlmod enabled
- **CPU**: Intel/AMD with AVX2 support (Intel i5-10500 or better)
- **GPU**: OpenGL 3.3+ capable (for viewer only)
- **RAM**: 16GB+ recommended for 128 parallel environments

## Build Commands

```bash
# Training build (max performance - use this for training)
bazel build //:train --copt=-march=native --copt=-O3 --copt=-flto --copt=-ffast-math

# Run training
bazel run //:train --config=opt

# Visualizer build (for debugging physics/layout)
bazel build //:viewer
bazel run //:viewer

# System test (validate SIMD alignment)
bazel run //:system_test

# Full clean
bazel clean --expunge

# Format code
clang-format -i src/*.cpp src/*.h
```

## Training Configuration

The training engine supports command-line arguments and configuration via `TrainingConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numParallelEnvs` | 128 | Number of simultaneous environments |
| `checkpointInterval` | 50000 | Steps between checkpoints |
| `maxSteps` | 10000000 | Total training steps |
| `checkpointDir` | "checkpoints" | Checkpoint output directory |
| `loadCheckpoint` | "" | Resume from checkpoint path |

## Neural Architecture

### SPAN Network (Spline-based Polynomial Approximation)

The project implements a novel neural architecture combining:

1. **Tensor-Product B-Splines**: Replace traditional MLP layers with B-spline basis functions
   - Configurable knots and spline degree
   - Smooth, continuous function approximation
   - Fewer parameters than equivalent MLP

2. **Second-Order Latent Memory**: ODE2VAE-inspired latent dynamics
   - Velocity and acceleration states for temporal modeling
   - Better long-horizon credit assignment

3. **AVX2-Optimized Kernels**: Hand-written SIMD for forward passes
   - 8-wide float operations
   - Aligned memory access patterns

## Key Design Principles

### 1. Zero-Allocation Mandate
All memory is pre-allocated at startup. During training:
- No `new`/`delete` or `malloc`/`free`
- No `std::vector::push_back` in hot loops
- Fixed-size buffers reused across steps

### 2. The "Necromancer" Reset
When episodes terminate, bodies are not destroyed:
- Zero out linear/angular velocities
- Manually overwrite transform to spawn position
- Eliminates physics body creation overhead

### 3. Thread Pinning
CPU affinity is strictly controlled:
- Core 0 (Threads 0, 6): OS + main RL loop
- Cores 1-5 (Threads 1-5, 7-11): Jolt worker pool

### 4. Dimensional Ghosting
All parallel environments share a single `PhysicsSystem`:
- Robots coexist at 0,0,0 in unified space
- Collisions bypassed via custom `ObjectLayerPairFilter`
- No per-environment physics overhead

## Robot Configuration

Robots are defined in JSON format (`robots/combat_bot.json`):

```json
{
  "name": "CombatBot",
  "base": {
    "shape": "box",
    "dimensions": [0.5, 0.2, 0.8],
    "mass": 5.0
  },
  "joints": [
    {
      "name": "shoulder_left",
      "type": "revolute",
      "axis": [0, 1, 0],
      "limits": [-90, 90]
    }
  ],
  "sensors": ["force", "imu", "joint_angle"]
}
```

## Observation Space

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Base position/rotation | 6 | Robot chassis state |
| Joint positions | NUM_SATELLITES | Articulated joint angles |
| Joint velocities | NUM_SATELLITES | Angular velocities |
| IMU readings | NUM_SATELLITES * 3 | Accelerometer data |
| Force sensors | NUM_SATELLITES * 2 | Contact impulse magnitudes |
| Opponent relative | 6 | Relative position/rotation |
| Health | 1 | Remaining HP (0-100) |

**Total**: ~50+ dimensions (depends on NUM_SATELLITES configuration)

## Action Space

| Action | Range | Description |
|--------|-------|-------------|
| Joint torques | [-1, 1] | Normalized torque per joint |

Actions are continuous and clipped to valid torque ranges per joint.

## Reward Structure

Vectorized rewards provide multi-objective feedback:

| Component | Weight | Description |
|-----------|--------|-------------|
| `damage_dealt` | +1.0 | Damage inflicted on opponent |
| `damage_taken` | -1.0 | Damage received |
| `alive_bonus` | +0.1 | Survival per step |
| `air_time` | -0.01 | Penalty for being airborne |
| `energy_efficiency` | +0.001 | Reward for efficient movement |
| `center_proximity` | +0.05 | Arena control incentive |

## Project Structure

```
robo-viewer/
├── BUILD                      # Bazel build definitions
├── MODULE.bazel               # Bazel dependencies (Bzlmod)
├── MODULE.bazel.lock          # Locked dependency versions
├── src/
│   ├── main_train.cpp         # Training entry point
│   ├── main.cpp               # Viewer entry point
│   ├── PhysicsCore.{h,cpp}    # Physics engine wrapper
│   ├── CombatRobot.{h,cpp}    # Robot definition
│   ├── CombatEnv.{h,cpp}      # Combat simulation
│   ├── VectorizedEnv.{h,cpp}  # Parallel environment manager
│   ├── SpanNetwork.{h,cpp}    # SPAN neural architecture
│   ├── NeuralMath.{h,cpp}     # SIMD math kernels
│   ├── TD3Trainer.{h,cpp}     # RL training loop
│   ├── LatentMemory.{h,cpp}   # ODE2VAE latent states
│   ├── OpponentPool.{h,cpp}   # Self-play opponent sampling
│   ├── Renderer.{h,cpp}       # OpenGL visualization
│   ├── OverlayUI.{h,cpp}      # Dear ImGui training overlay
│   ├── AlignedAllocator.h     # AVX2-aligned memory
│   ├── LockFreeQueue.h        # Thread-safe queues
│   └── system_test.cpp        # SIMD validation tests
├── robots/
│   ├── combat_bot.json        # Combat robot definition
│   └── test_bot.json          # Test robot definition
├── third_party/
│   └── jolt.BUILD             # Jolt Physics build rules
├── checkpoints/               # Saved model checkpoints
└── saved_models/              # Exported trained policies
```

## Development

### Code Style
- C++20 standard
- PascalCase for classes/functions
- camelCase for variables
- `m` prefix for private members
- No logging in hot training loops

### Physics Requirements
- `#include <Jolt/Jolt.h>` must be first in every physics file
- Disable Jolt sleep mechanics globally
- Never allocate in physics callbacks

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Jolt Physics](https://github.com/jrouwe/JoltPhysics) - High-performance physics engine
- [Dear ImGui](https://github.com/ocornut/imgui) - Immediate mode GUI
- SPAN architecture inspired by Kolmogorov-Arnold Networks and ODE2VAE research
