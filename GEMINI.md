# JOLTrl - Project Overview & Guidelines

JOLTrl is a high-performance reinforcement learning (RL) project focused on autonomous robot combat. It combines a bespoke C++ physics simulation engine with a modern React-based telemetry and control interface.

## System Architecture

The project is split into two primary domains:

1.  **Core Training Engine (C++)**: A high-throughput simulation environment leveraging the Jolt Physics engine and SIMD-optimized neural networks (SPAN/MLP).
2.  **Telemetry Frontend (React/Bun)**: A modern web interface for monitoring training progress, visualizing robot states, and controlling training parameters.

### Core Components
- **Physics Core**: Wrapper around Jolt Physics, managing collision layers, body creation, and step simulation.
- **Vectorized Environment**: Manages multiple parallel physics environments for high-throughput training.
- **Combat Engine**: Logic for robot state, sensors (raycasts, contact), and reward calculation.
- **RL Suite**: Implementation of TD3 (Twin Delayed DDPG) with prioritized replay buffers and SIMD-accelerated neural networks.

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
    bazel run //:train_headless -- --envs=128 --max-steps=1000000
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

## Agentic Capabilities & MCP

The project is designed to be agent-friendly, with integrated Model Context Protocol (MCP) support. 

- **Config Location**: `.ai/mcp/mcp_config.json`
- **Capabilities**:
    - **Debugging**: Integrated GDB and Ripgrep for deep code analysis.
    - **Research**: Multiple search engines (Brave, Exa, Tavily) for technical documentation and state-of-the-art RL research.
    - **Automation**: Playwright/Puppeteer for end-to-end telemetry testing.
    - **Memory**: Persistent agent memory and database support (SQLite/DuckDB).

For more details on agent development, see `AGENTS.md`.

---

## 🏗 Project Structure

-   `src/`: Core source code.
    -   `main_train.cpp`: Entry point for the visualizer and interactive training.
    -   `main_train_headless.cpp`: Optimized entry point for high-speed cluster training.
    -   `VectorizedEnv.h/cpp`: Manages parallel physics environments.
    -   `CombatEnv.h/cpp`: Specific environment logic for robot combat.
    -   `CombatRobot.h/cpp`: High-level robot actor with component management.
    -   `PhysicsCore.h/cpp`: Low-level wrapper for Jolt Physics.
    -   `TD3Trainer.h/cpp`: Core RL algorithm implementation.
    -   `NeuralNetwork.h/cpp`: SIMD-optimized MLP implementation.
    -   `SpanNetwork.h/cpp`: Implementation of Spline-based Polynomial Approximation Networks.
    -   `NeuralMath.h/cpp`: Hand-optimized SIMD kernels for neural ops.
    -   `Renderer.h/cpp`: OpenGL-based visualizer engine.
    -   `App.tsx`: Main React entry point for telemetry.
-   `robots/`: JSON definitions for robot configurations (morphology, sensors, joints).
-   `third_party/`: External dependencies (Jolt, ImGui, Glm, etc.).
-   `checkpoints/`: Directory for saved training weights.

---

## 📜 Development Conventions

### Physics & Performance (C++)
-   **Zero-Allocation Mandate**: Avoid memory allocation in the hot training loop. Pre-allocate all state tensors.
-   **SIMD Optimization**: Use `AlignedVector32` and avoid unaligned loads. Use `_mm256_loadu_ps` for safe access if alignment isn't guaranteed.
-   **Jolt Include Order**: `Jolt/Jolt.h` **MUST** be the first include in any file using the physics engine.
-   **Coordinate Space**: All environments share a single global coordinate space; use filters to prevent inter-environment collisions.

### Reinforcement Learning & Neural Networks
-   **Tensor Alignment**: Ensure all weight and activation buffers are 32-byte aligned for AVX2.
-   **Precision**: Use `float` (FP32) for all neural calculations; avoid `double` unless strictly required for stability.
-   **Reward Shaping**: All reward signals must be normalized to a reasonable range (typically [-1, 1] or [0, 1]) to ensure stable gradients.

### Frontend (React/TypeScript)
-   **Styling**: Use TailwindCSS utility classes. Avoid custom CSS unless necessary.
-   **Components**: Use the Shadcn UI components located in `src/components/ui`.
-   **Runtime**: Use Bun for all package management and execution tasks.

---

## 🧪 Testing & Validation
-   Always run `bazel run //:system_test` after making changes to the SIMD math or memory structures.
-   Use `bazel run //:test_json_load` to verify robot configuration files.
-   Use `bazel run //:end_to_end_test` for a full smoke test of the training loop.
-   Verify frontend HMR by running `bun run dev` and editing `src/App.tsx`.
