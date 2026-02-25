# AGENTS.md - JOLTrl Development Guidelines

## Project Overview

JOLTrl is a high-performance, bespoke C++ reinforcement learning environment using the Jolt Physics engine. The
architecture is strictly optimized for a 6-core/12-thread CPU architecture (Intel i5-10500).

The system operates under a strict dichotomy:

1. **The Training Engine (`//:train`):** A headless, zero-allocation mathematical matrix cruncher designed to saturate
   the CPU with parallel simulation steps.
2. **The Viewer (`//:viewer`):** A lightweight OpenGL/GLFW visualizer utilized exclusively for sanity-checking
   constraints and debugging physical layouts on the integrated graphics.

## Build Commands

### Training Build (Max Performance)

Every training build must leverage aggressive compiler optimizations to unlock SIMD instructions (AVX2/FMA).

```bash
bazel build //:train --copt=-march=native --copt=-O3 --copt=-flto --copt=-ffast-math 
bazel run //:train --config=opt

bazel build //:viewer                           # Build the visualizer
bazel run //:viewer                             # Run to sanity-check joints/physics

bazel clean --expunge                           # Full clean including cached dependencies
clang-format -i src/*.cpp src/*.h               # Format C++ files

Architectural Doctrines for Reinforcement Learning
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

Error Handling
Use early returns for initialization errors.

Logging inside the hot training loop (//:train) is strictly prohibited unless it is a fatal assertion. I/O operations will bottleneck the SPS.

Project Structure
Plaintext
robo-viewer/
├── BUILD                 # Main Bazel build file (defines //:train and //:viewer)
├── MODULE.bazel          # Bazel dependencies (rules_cc locked to correct version)
├── src/
│   ├── main_train.cpp    # Headless RL entry point (Zero graphics)
│   ├── main_viewer.cpp   # Visual debugging entry point
│   ├── PhysicsCore.h/cpp # High-performance Jolt wrapper
│   ├── RlBridge.h/cpp    # Zero-copy memory bridge for policy network
│   ├── RobotLoader.h/cpp # Minimal JSON loader for constraints
│   └── Renderer.h/cpp    # OpenGL system (compiled ONLY for //:viewer target)
└── third_party/
    └── jolt.BUILD        # Custom optimized Jolt compilation rules
Physics Engine Requirements (Strict)
Every .cpp file that interacts with the physics engine MUST start with #include <Jolt/Jolt.h> as the absolute first include before any other headers. This is non-negotiable and prevents catastrophic macro expansion errors.

Disable Jolt's "sleep" mechanics globally. RL agents are constantly exploring and must never be put to sleep by the broadphase.