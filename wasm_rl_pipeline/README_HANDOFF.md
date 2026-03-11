# WebAssembly RL Pipeline Project Handoff Document

## Project Overview
This project implements a parallel reinforcement learning pipeline using WebAssembly with Jolt physics integration. The system supports 1v1 combat training with 1 rendered environment and multiple headless instances for parallel processing.

## Current Status
- **Project Structure**: Established in `/media/cammyz/EverythingHere/robo-viewer/wasm_rl_pipeline`
- **Core Implementation**: C++ source files with mock Jolt physics interface
- **Build System**: CMake configured for Emscripten
- **Key Components**: Physics bindings, vectorized environments, training coordinator

## Technical Architecture

### Core Components
1. **Jolt Physics Bindings** (`jolt_physics_bindings.cpp`)
   - Physics world simulation
   - Robot state management
   - Combat environment stepping

2. **Vectorized Environment** (`VectorizedEnvironment` class)
   - Parallel environment execution
   - Observation/reward handling
   - Headless/rendered mode switching

3. **Training Coordinator** (`CombatTrainingCoordinator` class)
   - 1v1 combat training management
   - Environment lifecycle control
   - Action/observation coordination

4. **Web Worker Manager** (`WebWorkerManager` class)
   - Parallel task execution
   - Thread pool management
   - Asynchronous processing

## Build Requirements

### Dependencies
- **Emscripten SDK**: WebAssembly compiler toolchain
- **Node.js**: Required for Emscripten
- **C++17**: Standard library support
- **Jolt Physics**: Physics engine (to be integrated)

### Current Issues
1. **Emscripten Installation**: `em++` command not found
2. **CMake Configuration**: Requires Emscripten compiler
3. **Jolt Physics Integration**: Mock interface needs actual bindings

## Installation Steps

### 1. Install Node.js (if not present)
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 2. Install Emscripten SDK
```bash
# Download and install Emscripten
curl -L https://github.com/emscripten-core/emsdk/archive/master.tar.gz | tar xz
cd emsdk-master
./emsdk install latest
./emsdk activate latest
```

### 3. Configure Environment
```bash
source emsdk-master/emsdk_env.sh
```

### 4. Build Project
```bash
cd /media/cammyz/EverythingHere/robo-viewer/wasm_rl_pipeline
emcmake cmake ..
emmake make
```

## File Structure
```
wasm_rl_pipeline/
├── src/                     # Main source files
├── wasm_module/            # WebAssembly module
│   ├── jolt_physics_bindings.cpp
│   ├── rl_pipeline.cpp
│   ├── vectorized_env.cpp
│   ├── combat_env.cpp
│   ├── training_coordinator.cpp
│   └── webworker_manager.cpp
├── build/                 # Build output
└── CMakeLists.txt         # Build configuration
```

## Key Implementation Details

### Physics System
- PhysicsWorld struct for simulation state
- RobotState struct for agent data
- CombatEnvState for environment management

### Environment Management
- Vectorized execution for parallel processing
- Headless vs rendered mode switching
- Observation/action space handling (256-dim state, 56-dim action)

### Training Pipeline
- 1v1 combat training coordinator
- Reward calculation and health tracking
- Termination condition handling

## Next Steps

1. **Resolve Emscripten Installation**
   - Install Node.js and Emscripten SDK
   - Configure build environment

2. **Complete Jolt Physics Integration**
   - Replace mock interface with actual Jolt bindings
   - Implement physics callbacks

3. **Test WebAssembly Module**
   - Compile and test basic functionality
   - Validate parallel execution

4. **Performance Optimization**
   - Optimize memory usage
   - Improve parallel processing efficiency

## Troubleshooting

### Common Issues
- **em++ not found**: Emscripten not installed or not in PATH
- **CMake errors**: Requires Emscripten compiler
- **Memory issues**: Adjust MAXIMUM_MEMORY in CMakeLists.txt
- **Performance problems**: Check thread pool configuration

### Debug Commands
```bash
# Check Emscripten installation
em++ --version

# Verify build environment
emcmake --version

# Test compilation
em++ -v
```

## Documentation Links
- [Emscripten Documentation](https://emscripten.org/docs/)
- [Jolt Physics Documentation](https://joltphysics.org/)
- [WebAssembly MDN](https://developer.mozilla.org/en-US/docs/WebAssembly)

## Contact Information
For questions or issues regarding this project:
- Project maintainer: [Your Name]
- Last updated: 2026-03-08
- Repository: /media/cammyz/EverythingHere/robo-viewer/wasm_rl_pipeline

---

**Note**: This document assumes basic familiarity with C++, WebAssembly, and reinforcement learning concepts. For detailed implementation specifics, refer to the source code comments and function documentation.