# Swift JOLTrl Pipeline

This is a separate subfolder of the JOLTrl project, remade using the Swift language. It leverages Swift's C++ interoperability to use the Jolt Physics engine while providing a modern, safe, and performant Swift-native interface for reinforcement learning.

## Structure

- `Sources/Physics/`: Swift wrapper for Jolt Physics (C++ Interop).
- `Sources/Neural/`: Swift implementation of the SPAN (Spline-based Polynomial Approximation Network) neural network.
- `Sources/Robot/`: Parsers for loading robot configurations (JSON) like `bouncy_orbiter.json`.
- `Sources/Environment/`: Combat simulation logic (1v1 arenas, rewards, resetting).
- `Sources/Training/`: TD3 (Twin Delayed DDPG) reinforcement learning algorithm implementation.
- `Sources/main.swift`: The main training loop that loads the robot, initializes the environment, and trains the agent.

## Building and Running

This subfolder is set up as a standalone Bazel workspace. To build it, you'll need the Swift toolchain (5.9+) and Bazel.

```bash
cd swift_joltrl
bazel build //Sources:swift_train
bazel run //Sources:swift_train
```

## Implementation Status

This is a **functional skeleton** of the pipeline. 
- **Physics**: The wrapper structure is complete, but the actual Jolt C++ calls are mocked in the source code (commented out) because a full build requires a configured Swift C++ toolchain and linking against the Jolt library binary.
- **Training**: The TD3 algorithm structure is implemented, but the backpropagation step is mocked as it requires a full automatic differentiation engine.
- **Simulation**: The combat environment logic is implemented to spawn robots and calculate rewards based on the loaded JSON definition.

## Key Features

- **Swift-Native SPAN**: The neural network logic is re-implemented in Swift, allowing for better type safety and integration with Swift's `simd` and `Accelerate` frameworks.
- **C++ Interop**: Directly calls Jolt Physics C++ APIs without a manual C wrapper, reducing overhead and maintenance.
- **Modern Concurrency**: Swift's structured concurrency (async/await) can be easily integrated for parallel environment stepping and asynchronous telemetry reporting.
