# AGENTS.md - JOLTrl Development Guidelines

## Quick Reference

```bash
# Build and run training (primary workflow)
bazel build //:train --copt=-march=native --copt=-O3 --copt=-flto --copt=-ffast-math
bazel run //:train --config=opt

# Build and run visualizer (debugging)
bazel build //:viewer
bazel run //:viewer

# Format all code
clang-format -i src/*.cpp src/*.h

# Clean build cache
bazel clean --expunge
```

## Project Overview

JOLTrl is a high-performance, bespoke C++ reinforcement learning environment using the Jolt Physics engine. The
architecture is strictly optimized for a 6-core/12-thread CPU architecture (Intel i5-10500).

The system operates under a strict dichotomy:

1. **The Training Engine (`//:train`):** Headless, zero-allocation RL training with SPAN neural networks
2. **The Viewer (`//:viewer`):** OpenGL/GLFW visualizer for physics debugging and constraint validation

## Build Commands

### Training Build (Max Performance)

Every training build must leverage aggressive compiler optimizations to unlock SIMD instructions (AVX2/FMA).

```bash
# Primary training build - ALWAYS use these flags
bazel build //:train --copt=-march=native --copt=-O3 --copt=-flto --copt=-ffast-math
bazel run //:train --config=opt

# Viewer build (OpenGL visualization)
bazel build //:viewer
bazel run //:viewer

# System test (validate SIMD alignment)
bazel run //:system_test

# Clean builds
bazel clean                               # Standard clean
bazel clean --expunge                     # Full clean including cached dependencies

# Code formatting
clang-format -i src/*.cpp src/*.h
```

## Architectural Doctrines for Reinforcement Learning
1. The Zero-Allocation Mandate
Memory allocation during the training loop destroys performance.

Pre-allocate Everything: All robot state tensors (positions, velocities, joint angles) must be allocated in massive, contiguous blocks at startup.

No Spawning/Destroying: When an episode terminates, do not destroy the JPH::Body. You must mathematically zero out its linear/angular velocities and manually overwrite its transform back to the initial state (The "Necromancer" Reset).

2. Thread Pinning & CPU Affinity
To maximize the 12 available hardware threads and prevent the Pop!_OS kernel from swapping our cache context:

Core 0 (Threads 0 & 6) is reserved for the OS background tasks and the main RL policy loop.

Cores 1-5 (Threads 1-5, 7-11) are strictly pinned to Jolt's worker thread pool using pthread_setaffinity_np.

3. Dimensional Ghosting
Parallel environments do not exist in separate PhysicsSystem instances.

All training agents are spawned into a single, unified 0,0,0 coordinate space.

Inter-robot collisions are bypassed mathematically at the broadphase level using heavily customized ObjectLayerPairFilter configurations.

4. The Physics Loop (Fidelity over Frames)
Simulation is completely uncoupled from real-time rendering.

The training loop spins as fast as the CPU allows.

Timestep: To ensure mathematical stability when the RL policy applies erratic joint torques, the internal physics step must be high fidelity (e.g., 1/120f or 1/240f).

Substepping: Keep collision substeps extremely low (1 or 2 max) to maintain a massive Steps-Per-Second (SPS) throughput.

Code Style Guidelines
C++ Standards
C++17 standard enforced (copts = ["-std=c++17"] in BUILD file).

Raw pointers are permitted only for interfacing with Jolt's pre-allocated memory pools or zero-copy bridges.

Avoid standard library containers (like std::vector::push_back) inside the hot physics loop.

Naming Conventions
Classes/Structs: PascalCase (e.g., PhysicsCore, TensorBridge)

Functions/Methods: PascalCase (e.g., Init(), ResetEpisode(), ApplyTorque())

Variables: camelCase (e.g., physics, observationState)

Constants: UPPER_CASE or kCamelCase (e.g., NUM_PARALLEL_ENVS, kObservationDim)

Private Members: m prefix (e.g., mPhysicsSystem, mJobSystem)

### Error Handling
Use early returns for initialization errors.

Logging inside the hot training loop (//:train) is strictly prohibited unless it is a fatal assertion. I/O operations will bottleneck the SPS.

## Neural Architecture: SPAN Network

The project implements **SPAN** (Spline-based Polynomial Approximation Network), a novel architecture:

### TensorProductBSpline Layer
- Replaces traditional MLP layers with B-spline basis functions
- Configurable: `numKnots`, `splineDegree`
- AVX2-optimized forward pass (`ForwardAVX2()`)
- Fewer parameters than equivalent MLP for same expressiveness

### Latent Memory (ODE2VAE-inspired)
- `LatentMemory` / `SecondOrderLatentMemory` classes
- Tracks velocity and acceleration states
- Better long-horizon temporal modeling
- Integrated with SPAN layers via `SpanNetwork::ForwardWithLatent()`

### Adding New SPAN Layers
```cpp
std::vector<SpanLayerConfig> configs = {
    {stateDim, hiddenDim, 8, 3},    // input -> hidden
    {hiddenDim, actionDim, 8, 3}     // hidden -> output
};
SpanNetwork network;
network.Init(configs, rng);
```

## Project Structure

```
robo-viewer/
├── BUILD                      # Bazel build definitions (//:train, //:viewer)
├── MODULE.bazel               # Bazel dependencies (Bzlmod)
├── MODULE.bazel.lock          # Locked dependency versions
├── AGENTS.md                  # This file - development guidelines
├── README.md                  # Project documentation
├── src/
│   ├── main_train.cpp         # Headless RL entry point (Zero graphics)
│   ├── main.cpp               # Visual debugging entry point (//:viewer)
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
│   ├── LockFreeQueue.h        # Thread-safe lock-free queues
│   └── system_test.cpp        # SIMD alignment validation tests
├── robots/
│   ├── combat_bot.json        # Combat robot definition
│   └── test_bot.json          # Test robot for visualizer
├── third_party/
│   └── jolt.BUILD             # Custom optimized Jolt compilation rules
├── checkpoints/               # Training checkpoints (created at runtime)
└── saved_models/              # Exported trained policies
```
## Physics Engine Requirements (STRICT)

### Include Order
Every .cpp file that interacts with the physics engine MUST start with:
```cpp
#include <Jolt/Jolt.h>  // ABSOLUTE FIRST - before any other headers
#include <Jolt/RegisterTypes.h>
// ... other Jolt headers
// ... other includes
```
This is non-negotiable and prevents catastrophic macro expansion errors.

### Sleep Mechanics
Disable Jolt's "sleep" mechanics globally. RL agents are constantly exploring and must never be put to sleep by the broadphase. Check `PhysicsCore.cpp` for the `mAllowSleeping = false` configuration.