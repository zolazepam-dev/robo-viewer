# JOLTrl - Complete Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Building the Project](#building-the-project)
6. [Running Training](#running-training)
7. [Running the Viewer](#running-the-viewer)
8. [Configuration](#configuration)
9. [Neural Architecture](#neural-architecture)
10. [Physics System](#physics-system)
11. [Environment Design](#environment-design)
12. [Robot Configuration](#robot-configuration)
13. [Project Structure](#project-structure)
14. [Development Guidelines](#development-guidelines)
15. [Troubleshooting](#troubleshooting)
16. [Known Issues](#known-issues)

---

## Introduction

**JOLTrl** (Jolt Optimized Learning for Robots) is a high-performance C++ reinforcement learning environment designed for training autonomous combat robots. Built from the ground up with the Jolt Physics engine, it delivers exceptional throughput through:

- **Parallel Environment Simulation**: 128+ environments running simultaneously
- **Zero-Allocation Training**: Pre-allocated memory pools eliminate GC pressure
- **SIMD-Accelerated Networks**: Hand-optimized AVX2/FMA neural network kernels
- **Novel Neural Architecture**: SPAN (Spline-based Polynomial Approximation Network) with ODE2VAE-inspired latent memory
- **Dual Mode Operation**: Headless training for maximum SPS, OpenGL visualization for debugging

### Key Performance Characteristics

| Metric | Target |
|--------|--------|
| Steps Per Second (SPS) | 100,000+ with 128 envs |
| Parallel Environments | 128 (configurable) |
| Physics Timestep | 1/120s or 1/240s |
| CPU Optimization | 6-core/12-thread (Intel i5-10500) |
| Memory Allocation | Zero during training |

---

## Quick Start

```bash
# 1. Build training binary with maximum optimizations
bazel build //:train --copt=-march=native --copt=-O3 --copt=-flto --copt=-ffast-math

# 2. Run training
bazel run //:train --config=opt

# 3. (Optional) Run visualizer for debugging
bazel build //:viewer
bazel run //:viewer
```

### Using Helper Scripts

```bash
# Build and run with interactive options
./build_and_run.sh

# Parallel training with configurable environments
./train_parallel.sh all

# Clean build
./build_and_run.sh --clean
```

---

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

### Component Overview

| Layer | Components | Purpose |
|-------|------------|---------|
| **Physics** | `PhysicsCore`, `CombatRobot`, `RobotLoader` | Jolt Physics wrapper, robot definitions |
| **Environment** | `CombatEnv`, `VectorizedEnv`, `OpponentPool` | Combat simulation, parallel execution |
| **Neural Network** | `SpanNetwork`, `NeuralMath`, `LatentMemory` | SPAN architecture, SIMD operations |
| **Training** | `TD3Trainer`, `LockFreeQueue` | RL algorithm, experience replay |
| **Visualization** | `Renderer`, `OverlayUI` | OpenGL rendering, ImGui dashboard |
| **Utilities** | `AlignedAllocator` | 32-byte aligned memory for AVX2 |

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Linux (Pop!_OS tested), macOS, Windows 10+ | Linux (Ubuntu 22.04+) |
| **Compiler** | GCC 11+, Clang 14+, MSVC 2022+ | GCC 12+ |
| **Bazel** | 6.0+ | 7.0+ |
| **CPU** | Intel/AMD with AVX2 (i5-10500 or better) | 6-core/12-thread+ |
| **GPU** | OpenGL 3.3+ (viewer only) | Any modern GPU |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 2GB | SSD recommended |

### Dependencies

All dependencies are managed via Bazel Bzlmod:

- **Jolt Physics** (v5.0.0) - High-performance physics engine
- **GLFW** (v3.4.0) - Window/context management
- **GLEW** (v2.2.0) - OpenGL extension loading
- **GLM** (v1.0.1) - OpenGL mathematics
- **Dear ImGui** (v1.92.2) - Immediate mode GUI
- **nlohmann_json** (v3.11.3) - JSON parsing

### Installation Steps

1. **Install Bazel** (if not already installed):
   ```bash
   # Ubuntu/Debian
   sudo apt install bazel

   # Or via Bazelisk (recommended)
   curl -LO https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
   chmod +x bazelisk-linux-amd64
   sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd robo-viewer
   ```

3. **Verify installation**:
   ```bash
   bazel build //:system_test
   bazel run //:system_test
   ```

---

## Building the Project

### Training Build (Maximum Performance)

The training binary requires aggressive compiler optimizations to unlock SIMD instructions:

```bash
# Full optimization build
bazel build //:train \
    --copt=-march=native \
    --copt=-O3 \
    --copt=-flto \
    --copt=-ffast-math

# Run training
bazel run //:train --config=opt
```

### Viewer Build

```bash
# Standard build with OpenGL support
bazel build //:viewer
bazel run //:viewer
```

### Build Options

| Flag | Description |
|------|-------------|
| `--copt=-march=native` | Optimize for host CPU architecture |
| `--copt=-O3` | Maximum optimization level |
| `--copt=-flto` | Link-time optimization |
| `--copt=-ffast-math` | Aggressive floating-point optimizations |
| `--compilation_mode=dbg` | Debug build with symbols |
| `--compilation_mode=fastbuild` | Fast build without optimizations |
| `--verbose_failures` | Detailed build error messages |

### Clean Build

```bash
# Standard clean
bazel clean

# Full clean (including cached dependencies)
bazel clean --expunge
```

### Code Formatting

```bash
# Format all C++ source files
clang-format -i src/*.cpp src/*.h
```

---

## Running Training

### Basic Training

```bash
# Run with default settings (128 parallel environments)
bazel run //:train --config=opt
```

### Using the Training Script

```bash
# Clean, build, and run with defaults
./train_parallel.sh all

# Run with custom environment count
NUM_ENVS=16 ./train_parallel.sh run

# Start fresh without loading checkpoint
LOAD_LATEST=false ./train_parallel.sh run
```

### Command-Line Arguments

The training binary accepts the following parameters:

| Argument | Default | Description |
|----------|---------|-------------|
| `--envs` | 128 | Number of parallel environments |
| `--checkpoint-interval` | 50000 | Steps between checkpoints |
| `--max-steps` | 10000000 | Total training steps |
| `--checkpoint-dir` | "checkpoints" | Checkpoint output directory |
| `--load-latest` | true | Load latest checkpoint on start |

### Example: Custom Training Configuration

```bash
bazel run //:train --config=opt -- \
    --envs 64 \
    --checkpoint-interval 25000 \
    --max-steps 5000000 \
    --checkpoint-dir ./my_checkpoints
```

### Training Output

During training, the system displays:

- **Steps Per Second (SPS)**: Training throughput metric
- **Episode Rewards**: Average reward over recent episodes
- **Loss Metrics**: TD3 critic/actor loss values
- **Checkpoint Status**: Save/load notifications

---

## Running the Viewer

The viewer provides OpenGL-based visualization for debugging physics and robot behavior:

```bash
# Build and run viewer
bazel build //:viewer
bazel run //:viewer
```

### Viewer Controls

| Key | Action |
|-----|--------|
| `ESC` | Exit viewer |
| `Space` | Pause/Resume simulation |
| `R` | Reset environment |
| `Mouse Drag` | Rotate camera |
| `Scroll` | Zoom in/out |
| `WASD` | Move camera |

### Debug Features

- Real-time physics visualization
- Joint constraint display
- Contact force indicators
- Robot state overlays (via Dear ImGui)

---

## Configuration

### Training Configuration

Training parameters are defined in `TrainingConfig` (see `main_train.cpp`):

```cpp
struct TrainingConfig {
    int numParallelEnvs = 128;
    int checkpointInterval = 50000;
    int maxSteps = 10000000;
    std::string checkpointDir = "checkpoints";
    std::string loadCheckpoint = "";
    bool loadLatest = true;
};
```

### Physics Configuration

Physics parameters are tuned in `PhysicsCore`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Timestep | 1/120s | Physics simulation step |
| Substeps | 1-2 | Collision substeps per frame |
| Allow Sleeping | false | Disabled for RL (constant exploration) |

### Neural Network Configuration

SPAN network architecture is configured via layer definitions:

```cpp
std::vector<SpanLayerConfig> configs = {
    {stateDim, hiddenDim, 8, 3},    // input -> hidden
    {hiddenDim, actionDim, 8, 3}     // hidden -> output
};
```

| Parameter | Description |
|-----------|-------------|
| `numKnots` | Number of B-spline knots (default: 8) |
| `splineDegree` | Polynomial degree (default: 3) |

---

## Neural Architecture

### SPAN Network (Spline-based Polynomial Approximation Network)

JOLTrl implements a novel neural architecture that replaces traditional MLP layers with B-spline basis functions.

#### TensorProductBSpline Layer

- **Purpose**: Smooth, continuous function approximation
- **Advantages**: Fewer parameters than equivalent MLP
- **Optimization**: AVX2-optimized forward pass (`ForwardAVX2()`)
- **Configuration**: Configurable knots and spline degree

#### Architecture Diagram

```
Input (Observation)
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
Output (Actions)
```

### Latent Memory (ODE2VAE-Inspired)

The `LatentMemory` / `SecondOrderLatentMemory` system provides:

- **Velocity and Acceleration States**: For temporal modeling
- **Second-Order Dynamics**: Better long-horizon credit assignment
- **Integration**: Seamlessly integrated with SPAN layers via `ForwardWithLatent()`

### SIMD Optimization

All neural network kernels are hand-optimized for AVX2/FMA:

- **8-wide Float Operations**: Maximum throughput per cycle
- **Aligned Memory Access**: 32-byte aligned buffers via `AlignedAllocator`
- **Fused Multiply-Add**: Single-instruction polynomial evaluation

---

## Physics System

### Jolt Physics Integration

The physics system is built on Jolt Physics with several customizations:

#### Include Order (CRITICAL)

Every `.cpp` file that interacts with physics **MUST** start with:

```cpp
#include <Jolt/Jolt.h>  // ABSOLUTE FIRST - before any other headers
#include <Jolt/RegisterTypes.h>
// ... other Jolt headers
// ... other includes
```

This is non-negotiable and prevents catastrophic macro expansion errors.

#### Sleep Mechanics

Jolt's sleep mechanics are **disabled globally**. RL agents constantly explore and must never be put to sleep by the broadphase:

```cpp
mAllowSleeping = false;  // See PhysicsCore.cpp
```

### Thread Pinning

To maximize the 12 available hardware threads and prevent OS context switching:

| Core | Threads | Assignment |
|------|---------|------------|
| Core 0 | 0, 6 | OS + main RL loop |
| Cores 1-5 | 1-5, 7-11 | Jolt worker pool |

Thread affinity is set via `pthread_setaffinity_np`.

### Dimensional Ghosting

Parallel environments do **not** exist in separate `PhysicsSystem` instances:

- All training agents spawn into a single, unified 0,0,0 coordinate space
- Inter-robot collisions are bypassed at the broadphase level via custom `ObjectLayerPairFilter`
- No per-environment physics overhead

---

## Environment Design

### Combat Environment (`CombatEnv`)

The combat environment simulates two robots in an arena with:

- **Damage System**: Contact-based damage calculation
- **Reward Signals**: Multi-objective reward vector
- **Termination Conditions**: Health depletion, time limit, arena bounds

### Vectorized Environment (`VectorizedEnv`)

Manages 128+ parallel environment instances:

- **Shared Memory**: All environments share contiguous memory blocks
- **Lock-Free Queues**: Thread-safe experience collection
- **Batched Operations**: SIMD-accelerated observation/action processing

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

**Total**: ~50+ dimensions (depends on `NUM_SATELLITES` configuration)

### Action Space

| Action | Range | Description |
|--------|-------|-------------|
| Joint torques | [-1, 1] | Normalized torque per joint |

Actions are continuous and clipped to valid torque ranges per joint.

### Reward Structure

| Component | Weight | Description |
|-----------|--------|-------------|
| `damage_dealt` | +1.0 | Damage inflicted on opponent |
| `damage_taken` | -1.0 | Damage received |
| `alive_bonus` | +0.1 | Survival per step |
| `air_time` | -0.01 | Penalty for being airborne |
| `energy_efficiency` | +0.001 | Reward for efficient movement |
| `center_proximity` | +0.05 | Arena control incentive |

---

## Robot Configuration

Robots are defined in JSON format. Example: `robots/combat_bot.json`:

```json
{
  "core": {
    "radius": 0.5,
    "mass": 13.0
  },
  "satellites": [
    { "id": 1, "offset_angle": 0, "elevation": 0, "distance": 1.4 },
    { "id": 2, "offset_angle": 72, "elevation": 0, "distance": 1.4 },
    { "id": 3, "offset_angle": 144, "elevation": 0, "distance": 1.4 },
    { "id": 4, "offset_angle": 216, "elevation": 0, "distance": 1.4 },
    { "id": 5, "offset_angle": 288, "elevation": 0, "distance": 1.4 },
    { "id": 6, "offset_angle": 0, "elevation": 45, "distance": 1.4 },
    { "id": 7, "offset_angle": 90, "elevation": 45, "distance": 1.4 },
    { "id": 8, "offset_angle": 180, "elevation": 45, "distance": 1.4 },
    { "id": 9, "offset_angle": 270, "elevation": 45, "distance": 1.4 },
    { "id": 10, "offset_angle": 0, "elevation": -45, "distance": 1.4 },
    { "id": 11, "offset_angle": 90, "elevation": -45, "distance": 1.4 },
    { "id": 12, "offset_angle": 180, "elevation": -45, "distance": 1.4 },
    { "id": 13, "offset_angle": 270, "elevation": -45, "distance": 1.4 }
  ],
  "joints": {
    "hinge_damping": 0.8,
    "hinge_armature": 0.5,
    "motor_torque": 450.0,
    "slide_range": [0.0, 0.5]
  }
}
```

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `core.radius` | Main body radius |
| `core.mass` | Total robot mass |
| `satellites` | Array of satellite attachments |
| `satellites[].offset_angle` | Angular offset (degrees) |
| `satellites[].elevation` | Vertical angle (degrees) |
| `satellites[].distance` | Distance from core |
| `joints.hinge_damping` | Joint damping coefficient |
| `joints.hinge_armature` | Joint armature value |
| `joints.motor_torque` | Maximum motor torque |
| `joints.slide_range` | Sliding joint limits |

---

## Project Structure

```
robo-viewer/
├── BUILD                      # Bazel build definitions
├── MODULE.bazel               # Bazel dependencies (Bzlmod)
├── MODULE.bazel.lock          # Locked dependency versions
├── AGENTS.md                  # Development guidelines
├── README.md                  # Project overview
├── DOCS.md                    # This file - complete documentation
├── bug_report.md              # Known bugs and issues
├── build_and_run.sh           # Interactive build/run script
├── train_parallel.sh          # Parallel training script
├── build_and_run_utility.sh   # Utility build script
├── config_debug               # Debug configuration
├── debugrec                   # Debug recording directory
├── kilo-code/                 # Code generation artifacts
├── src/
│   ├── main_train.cpp         # Headless RL entry point
│   ├── combat_main.cpp        # Visual debugging entry point
│   ├── PhysicsCore.h/cpp      # High-performance Jolt wrapper
│   ├── CombatRobot.h/cpp      # Robot body definition and control
│   ├── CombatEnv.h/cpp        # Combat simulation environment
│   ├── VectorizedEnv.h/cpp    # Parallel environment manager (128+ envs)
│   ├── SpanNetwork.h/cpp      # SPAN neural architecture (B-spline layers)
│   ├── NeuralMath.h/cpp       # SIMD-optimized matrix operations
│   ├── NeuralNetwork.h/cpp    # Legacy/utility neural network code
│   ├── LatentMemory.h/cpp     # ODE2VAE-style second-order latent memory
│   ├── TD3Trainer.h/cpp       # Twin Delayed DDPG training loop
│   ├── OpponentPool.h/cpp     # Self-play opponent sampling
│   ├── RobotLoader.h/cpp      # Minimal JSON loader for robot constraints
│   ├── Renderer.h/cpp         # OpenGL system (viewer only)
│   ├── OverlayUI.h/cpp        # Dear ImGui training dashboard
│   ├── AlignedAllocator.h     # 32-byte aligned memory for AVX2
│   └── LockFreeQueue.h        # Thread-safe lock-free queues
├── robots/
│   ├── combat_bot.json        # Combat robot definition
│   └── test_bot.json          # Test robot for visualizer
├── third_party/
│   └── jolt.BUILD             # Custom optimized Jolt compilation rules
├── checkpoints/               # Training checkpoints (git-ignored)
├── saved_models/              # Exported trained policies (git-ignored)
└── mujoco_robot_combat.xml    # MuJoCo reference model
```

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

### Memory Management

**The Zero-Allocation Mandate**: Memory allocation during the training loop destroys performance.

- **Pre-allocate Everything**: All robot state tensors allocated in massive, contiguous blocks at startup
- **No Spawning/Destroying**: When episodes terminate, zero velocities and overwrite transforms (The "Necromancer" Reset)
- **Avoid STL Containers**: No `std::vector::push_back` in hot loops

### Error Handling

- Use early returns for initialization errors
- **No logging in hot training loops** unless fatal (I/O bottlenecks SPS)
- Use assertions for invariant checks

### Adding New SPAN Layers

```cpp
std::vector<SpanLayerConfig> configs = {
    {stateDim, hiddenDim, 8, 3},    // input -> hidden
    {hiddenDim, actionDim, 8, 3}     // hidden -> output
};
SpanNetwork network;
network.Init(configs, rng);
```

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

### Performance Issues

**Problem**: Training slower than expected
- Verify `--copt=-march=native` is used
- Check parallel environment count (default: 128)
- Ensure no logging in hot loops
- Verify thread pinning is working

**Problem**: High memory usage
- Reduce `numParallelEnvs` in config
- Check for accidental allocations in training loop
- Verify `AlignedAllocator` is used for SIMD buffers

---

## Known Issues

See `bug_report.md` for detailed bug tracking. Summary:

### Critical

1. **`NeuralNetwork.cpp` compilation failure**: `_mm256_tanh_ps` is non-standard in AVX2
2. **KLPERBuffer sum-tree indexing**: Tree traversal starts at `idx = 0`, causing invalid sampling
3. **Priority precision in PER**: Float priorities truncated to integers, corrupting distribution

### High Priority

1. **Replay buffer stores wrong transition**: Uses current state as `nextState` in `main_train.cpp`
2. **FPS display bug**: `lastRenderTime` updated before FPS calculation
3. **Potential divide-by-zero**: When `rewardIdx == 0`

### Medium Priority

1. **Force sensor data not wired**: `StepResult.forces1/forces2` never populated
2. **Damage model overcounts**: Per-step distance checks without collision validation
3. **Inconsistent observation dimensions**: Shape mismatch between modules

### Low Priority

1. **Dead code**: Unused variables/functions throughout
2. **`alignas(32)` on `std::vector`**: Does not guarantee internal data alignment
3. **Mixed aligned/unaligned AVX2 loads**: Increases fragility

---

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Jolt Physics](https://github.com/jrouwe/JoltPhysics) - High-performance physics engine
- [Dear ImGui](https://github.com/ocornut/imgui) - Immediate mode GUI
- SPAN architecture inspired by Kolmogorov-Arnold Networks and ODE2VAE research

---

*Last updated: February 25, 2026*
