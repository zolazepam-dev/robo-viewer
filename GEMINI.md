# JOLTrl - Project Overview & Guidelines

JOLTrl is a high-performance reinforcement learning (RL) project focused on autonomous robot combat. It combines a bespoke C++ physics simulation engine with a modern React-based telemetry and control interface.

## System Architecture

The project is split into two primary domains:

1.  **Core Training Engine (C++)**: A high-throughput simulation environment leveraging the Jolt Physics engine and SIMD-optimized neural networks (SPAN).
2.  **Telemetry Frontend (React/Bun)**: A modern web interface for monitoring training progress, visualizing robot states, and controlling training parameters.

---

## 🛠 Building and Running

### 1. C++ Training & Simulation (Bazel)
The project uses Bazel for building the high-performance C++ components. Aggressive SIMD (AVX2/FMA) optimizations are mandatory for training performance.

-   **Build Training Engine**:
    ```bash
    bazel build //:train --compilation_mode=opt --copt=-march=native
    ```
-   **Run Headless Training**:
    ```bash
    ./train_parallel.sh run
    ```
-   **Run Visualizer (Viewer)**:
    ```bash
    bazel run //:viewer
    ```
-   **Run System Tests**:
    ```bash
    bazel run //:system_test
    ```

### 2. Telemetry Frontend (Bun/React)
The frontend is built with React, TailwindCSS, and Shadcn UI, powered by the Bun runtime.

-   **Install Dependencies**:
    ```bash
    bun install
    ```
-   **Start Development Server**:
    ```bash
    bun run dev
    ```
-   **Build for Production**:
    ```bash
    bun run build
    ```

---

## 🏗 Project Structure

-   `src/`: Core source code.
    -   `main_train.cpp`: Entry point for the hybrid training/visualizer.
    -   `VectorizedEnv.h/cpp`: Manages parallel physics environments.
    -   `SpanNetwork.h/cpp`: Implementation of Spline-based Polynomial Approximation Networks.
    -   `PhysicsCore.h/cpp`: Wrapper for Jolt Physics.
    -   `App.tsx`: Main React entry point.
-   `robots/`: JSON definitions for robot configurations.
-   `third_party/`: External dependencies (Jolt, ImGui, etc.).
-   `checkpoints/`: Directory for saved training weights.

---

## 📜 Development Conventions

### Physics & Performance (C++)
-   **Zero-Allocation Mandate**: Avoid memory allocation in the hot training loop. Pre-allocate all state tensors.
-   **SIMD Optimization**: Use `AlignedVector32` and avoid unaligned loads. Use `_mm256_loadu_ps` for safe access if alignment isn't guaranteed.
-   **Jolt Include Order**: `Jolt/Jolt.h` **MUST** be the first include in any file using the physics engine.
-   **Coordinate Space**: All environments share a single global coordinate space; use filters to prevent inter-environment collisions.

### Frontend (React/TypeScript)
-   **Styling**: Use TailwindCSS utility classes. Avoid custom CSS unless necessary.
-   **Components**: Use the Shadcn UI components located in `src/components/ui`.
-   **Runtime**: Use Bun for all package management and execution tasks.

---

## 🧪 Testing & Validation
-   Always run `bazel run //:system_test` after making changes to the SIMD math or memory structures.
-   Use `bazel run //:test_json_load` to verify robot configuration files.
-   Verify frontend HMR by running `bun run dev` and editing `src/App.tsx`.
