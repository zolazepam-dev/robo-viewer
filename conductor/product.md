# Initial Concept

**JOLTrl** (Jolt Optimized Learning for Robots) is a high-performance C++ reinforcement learning framework for training autonomous combat robots. Built on the Jolt Physics engine, it achieves exceptional throughput (100,000+ steps per second) through parallel environment simulation, zero-allocation training loops, and SIMD-accelerated neural network kernels.

## Core Vision

Enable rapid, scalable reinforcement learning research and development for robotic combat scenarios by providing:
- Massively parallel physics simulation (128-2048 environments)
- Custom neural architecture optimized for throughput
- Real-time visual debugging capabilities
- Self-play opponent pool for robust policy training

## Target Users

- **RL Researchers**: Academic and industry researchers developing new RL algorithms
- **Game Developers**: Studios implementing AI for combat/robot games
- **Robotics Engineers**: Engineers simulating multi-agent robotic systems

---

# Product Definition

## Overview

**JOLTrl** (Jolt Optimized Learning for Robots) is a production-grade, high-performance C++ reinforcement learning framework specifically designed for training autonomous combat robots. By leveraging the Jolt Physics engine and custom SIMD-optimized neural network architectures, JOLTrl delivers unprecedented training throughput while maintaining the flexibility needed for research and development.

## Problem Statement

Traditional RL frameworks struggle with:
- **Low throughput**: Single-environment or poorly parallelized simulation bottlenecks
- **Memory allocation overhead**: GC pressure and heap fragmentation during training loops
- **Limited physics fidelity**: Simplified physics engines that don't transfer to real-world scenarios
- **Poor debugging experience**: Lack of real-time visualization for training diagnostics

## Solution

JOLTrl addresses these challenges through:

### 1. Maximum Performance
- **100,000+ Steps Per Second (SPS)** through massively parallel environment simulation
- **SIMD-Accelerated Neural Networks**: Hand-optimized AVX2/FMA kernels for SPAN (B-spline) architecture
- **Zero-Allocation Hot Loops**: Pre-allocated memory pools eliminate GC pressure during training
- **Thread-Aware Design**: Optimized for 6-core/12-thread CPUs with proper core affinity

### 2. Production Ready
- **Robust Training Pipelines**: TD3 (Twin Delayed DDPG) with checkpointing and model export
- **Self-Play Opponent Pool**: 64-snapshot opponent sampling for robust policy training
- **Prioritized Experience Replay**: Efficient sampling with proper tree-based data structures
- **Comprehensive Logging**: Training metrics, performance diagnostics, and MicroBoard visualization

### 3. Easy Extensibility
- **Modular Architecture**: Clean separation between physics, environment, neural network, and training modules
- **Custom Robot Definitions**: JSON-based robot configuration with MJCF-style specifications
- **Pluggable RL Algorithms**: Framework supports TD3, with easy extension to SAC, PPO, etc.
- **Bazel Build System**: Modern dependency management with Bzlmod

### 4. Visual Debugging
- **OpenGL 3.3 Renderer**: Real-time 3D visualization of training environments
- **Dear ImGui Dashboard**: Live training metrics, hyperparameter tuning, and environment controls
- **Triple-Buffered State Extraction**: Non-blocking visual state updates for smooth rendering
- **Optional WASM Frontend**: React + Three.js web-based viewer for remote monitoring

## Key Features

### Parallel Environment Simulation
- **Dimensional Ghosting Technique**: 128-2048 environments running in unified coordinate space
- **Object Layer Filtering**: Custom broadphase collision filtering eliminates inter-environment collisions
- **Per-Environment Physics**: Independent robot instances with shared physics system

### Neural Architecture: SPAN
- **TensorProductBSpline Layers**: B-spline basis functions instead of traditional MLP
- **ODE2VAE-Inspired Latent Memory**: Second-order dynamics for temporal modeling
- **~10K Parameters**: Compact networks optimized for inference speed
- **32-Byte Aligned Memory**: AVX2-optimized with AlignedAllocator

### Training System
- **TD3 Algorithm**: Twin Delayed DDPG with target network soft updates
- **Latent Memory Integration**: Second-order latent states for better credit assignment
- **OpponentPool Self-Play**: 70% recent + 30% random opponent sampling
- **Domain Randomization**: Physics parameter randomization during environment reset

## Success Metrics

### Primary Metrics
1. **Throughput**: Steps per second (SPS) exceeding 100,000 with 128 environments
2. **Training Quality**: Policy win rate >70% against scripted opponents after 1M steps

### Secondary Metrics
3. **Ease of Use**: Time to first trained policy <30 minutes for new users
4. **Scalability**: Linear SPS scaling up to 2048 parallel environments
5. **Memory Efficiency**: Zero heap allocations during training loop

## Differentiators

### 1. Jolt Physics Integration
- High-fidelity rigid body dynamics optimized for RL workloads
- Custom job system with thread pinning for deterministic performance
- Sleep mechanics disabled for continuous agent exploration

### 2. Zero-Allocation Design
- All tensors pre-allocated in massive contiguous blocks at startup
- Structure-of-Arrays (SoA) layout for SIMD-friendly memory access
- No `std::vector::push_back`, `new`, or `malloc` in hot loops

### 3. Combat-Focused Architecture
- Damage system with contact-based impulse calculation
- Multi-objective reward structure (damage dealt, survival, energy efficiency)
- Arena-based combat with boundary control incentives
- Self-play curriculum through opponent pool sampling

## Use Cases

### Research Applications
- Multi-agent RL algorithm development
- Sim-to-real transfer for robotic combat
- Curriculum learning and self-play studies
- Physics-based character control

### Industry Applications
- Game AI for combat/robot games
- Autonomous drone swarm training
- Robotic manipulation in competitive scenarios
- Defense and security simulations

## Non-Goals

- **General-Purpose RL**: JOLTrl is specialized for physics-based combat, not Atari/GridWorld
- **GPU Training**: Focus is on CPU-only training with SIMD optimization
- **Real-Time Inference**: Optimized for training throughput, not deployment latency
- **Multi-GPU Scaling**: Single-machine, multi-core design

## Future Roadmap

### Phase 1: Core Stability
- Fix critical compilation bugs (AVX2 tanh, sum-tree indexing)
- Resolve replay buffer transition storage bugs
- Complete force sensor integration

### Phase 2: Performance Scaling
- Support 2048+ parallel environments
- Implement lock-free replay buffer
- Optimize latent memory vectorization

### Phase 3: Algorithm Extensions
- SAC (Soft Actor-Critic) implementation
- PPO for discrete action spaces
- Distributed training across multiple machines

### Phase 4: Tooling & UX
- Web-based WASM viewer with live telemetry
- Automated hyperparameter tuning
- Policy distillation for deployment
