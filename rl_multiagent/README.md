# Multi-Agent Reinforcement Learning Pipeline

A complete C++20 Multi-Agent RL system where 4 specialized agents learn to collaborate on feature development tasks.

## Overview

This project implements a bespoke reinforcement learning system from scratch with:

- **4 Specialized Agents**: Environment Architect, Core RL Implementer, Parallelization Engineer, Testing Specialist
- **Actor-Critic Architecture**: REINFORCE with baseline (A2C-style)
- **Parallel Environment Execution**: Thread pool for concurrent episode execution
- **Real-time Monitoring**: Console renderer and CSV metrics logging
- **Zero External RL Dependencies**: Pure C++20 standard library implementation

## Project Structure

```
rl_multiagent/
├── include/
│   ├── environment.hpp    # Development environment simulation
│   ├── neural_net.hpp     # Neural networks (Actor, Critic, Layers)
│   ├── rl_algo.hpp        # RL algorithms (REINFORCE, Trajectory Buffer)
│   └── parallel.hpp       # Parallel infrastructure (ThreadPool, Workers)
├── src/
│   ├── main.cpp           # Main training entry point
│   └── *.cpp              # Stub files (header-only implementation)
├── tests/
│   ├── test_environment.cpp   # Environment tests (10 tests)
│   ├── test_neural_net.cpp    # Neural network tests (17 tests)
│   ├── test_rl_algo.cpp       # RL algorithm tests (13 tests)
│   ├── test_parallel.cpp      # Parallel infrastructure tests (14 tests)
│   └── test_integration.cpp   # Integration tests (7 tests)
├── logs/
│   └── metrics.csv        # Training metrics output
├── CMakeLists.txt         # Build configuration
├── TODO.md                # Task tracker (maintained by agents)
└── README.md              # This file
```

## Build Instructions

### Prerequisites

- **Compiler**: GCC 11+, Clang 14+, or MSVC 2022+
- **CMake**: 3.16+
- **C++ Standard**: C++20
- **Threading**: pthread (Linux/macOS) or std::thread (Windows)

### Building

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build all targets
make -j4

# Run all tests
./test_env && ./test_nn && ./test_rl && ./test_parallel && ./test_integration
```

### Alternative: Direct Compilation

```bash
# Compile main training executable
g++ -std=c++20 -pthread -I include src/main.cpp -o marl_train

# Compile individual tests
g++ -std=c++20 -I include tests/test_environment.cpp -o test_env
g++ -std=c++20 -I include tests/test_neural_net.cpp -o test_nn
g++ -std=c++20 -I include tests/test_rl_algo.cpp -o test_rl
g++ -std=c++20 -pthread -I include tests/test_parallel.cpp -o test_parallel
g++ -std=c++20 -pthread -I include tests/test_integration.cpp -o test_integration
```

## Usage

### Training

```bash
# Default training (4 environments, 100 episodes)
./marl_train

# Custom configuration
./marl_train --envs 8 --episodes 200 --steps 500 --actor-lr 0.05 --critic-lr 0.05

# Show help
./marl_train --help
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--envs N` | 4 | Number of parallel environments |
| `--episodes N` | 100 | Total training episodes |
| `--steps N` | 1000 | Max steps per episode |
| `--actor-lr F` | 0.01 | Actor learning rate |
| `--critic-lr F` | 0.01 | Critic learning rate |
| `--gamma F` | 0.99 | Discount factor |

## Architecture

### Environment (`FeatureDevEnv`)

Simulated development environment where agents collaborate:

- **State**: File contents, TODO.md tasks, step count
- **Actions**: READ, EDIT, RUN_COMMAND, CLAIM, COMPLETE, BLOCK, NOOP
- **Roles**: ARCHITECT, CORE_RL, PARALLEL, TESTING
- **Rewards**: Task completion, coordination bonuses, efficiency penalties

### Neural Networks

**Actor Network** (Policy):
```
Input (state: 5 dims) → Hidden (64, ReLU) → Output (7 actions, Softmax)
```

**Critic Network** (Value):
```
Input (state: 5 dims) → Hidden (64, ReLU) → Output (1 value)
```

### RL Algorithm: REINFORCE with Baseline

1. **Collect Trajectories**: Run episodes, store (state, action, reward, next_state)
2. **Calculate Returns**: G_t = Σ γ^k * r_{t+k}
3. **Compute Advantages**: A_t = G_t - V(s_t)
4. **Update Actor**: ∇J = E[A_t * ∇log π(a_t|s_t)]
5. **Update Critic**: Minimize TD error: (G_t - V(s_t))²

### Parallel Infrastructure

```
┌─────────────────────────────────────────┐
│           ThreadPool (4 threads)         │
├──────────┬──────────┬──────────┬────────┤
│ Worker 0 │ Worker 1 │ Worker 2 │ Worker 3│
│   Env 0  │   Env 1  │   Env 2  │  Env 3 │
└────┬─────┴────┬─────┴────┬─────┴───┬────┘
     │          │          │         │
     └──────────┴────┬─────┴─────────┘
                     │
              Trajectory Aggregator
                     │
              RL Agent Update
```

## Test Suite

### Test Coverage

| Module | Tests | Description |
|--------|-------|-------------|
| Environment | 10 | State structs, enums, file system, TODO parser |
| Neural Net | 17 | Math ops, layers, Actor, Critic |
| RL Algo | 13 | Trajectory buffer, REINFORCE, integration |
| Parallel | 14 | ThreadPool, workers, aggregator, logger |
| Integration | 7 | Full training runs, reward trends |
| **Total** | **61** | |

### Running Tests

```bash
cd build

# Run all tests individually
./test_env        # Environment tests
./test_nn         # Neural network tests
./test_rl         # RL algorithm tests
./test_parallel   # Parallel infrastructure tests
./test_integration # Integration tests (10+ episodes)

# Or use CTest
ctest --output-on-failure
```

## Output

### Console Output

```
╔════════════════════════════════════════╗
║  Multi-Agent RL Pipeline v1.0          ║
║  4 Agents Collaborating on Dev Tasks   ║
╚════════════════════════════════════════╝

Configuration:
  Parallel Environments: 4
  Episodes: 100
  Max Steps: 1000
  Actor LR: 0.01
  Critic LR: 0.01
  Gamma: 0.99

Starting parallel training with 4 environments...

=== Episode 0 ===
  Reward: -0.41
  Steps: 20
  Avg Reward: -0.41
  Tasks Done: 0
  Time: 0s
========================

=== Episode 10 ===
  Reward: 1.23
  Steps: 45
  Avg Reward: 0.52
  Tasks Done: 2
  Time: 2s
========================

========================================
TRAINING COMPLETE
========================================
Total Episodes: 100
Final Avg Reward: 2.345
Total Time: 15s
========================================
```

### Metrics CSV

`logs/metrics.csv`:
```csv
episode,reward,steps,duration_ms,tasks_completed,avg_reward
0,-0.4100,20,15.00,0,-0.4100
1,0.2300,35,18.50,1,-0.0900
2,1.0500,50,22.30,1,0.2900
...
```

### Learning Curve

`logs/learning_curve.csv`:
```csv
episode,reward,rolling_avg
0,-0.4100,-0.4100
1,0.2300,-0.0900
2,1.0500,0.2900
...
```

## Success Criteria

✅ **All 61 unit tests pass**
✅ **Integration test runs 10+ episodes**
✅ **Parallel environments execute concurrently**
✅ **Metrics logged to CSV**
✅ **Code compiles with `g++ -std=c++20 -pthread`**
✅ **README documents build and usage**
✅ **TODO.md tracks all tasks**

## Tuning Tips

### If rewards are not improving:

1. **Increase learning rates**: `--actor-lr 0.05 --critic-lr 0.05`
2. **Run more episodes**: `--episodes 200`
3. **Reduce step penalty**: Modify `STEP_PENALTY` in `environment.hpp`
4. **Increase task completion reward**: Modify `TASK_COMPLETE_REWARD`

### For faster training:

1. **More parallel environments**: `--envs 8`
2. **Fewer steps per episode**: `--steps 200`
3. **Reduce logging frequency**: Modify `log_interval` in code

## Known Limitations

1. **Simple State Representation**: Currently uses 5 hand-crafted features
2. **Discrete Actions Only**: 7 action types with heuristic targeting
3. **No Task Dependencies**: BLOCK action exists but dependencies not enforced
4. **Synchronous Updates**: All environments wait for slowest (can be improved)

## Future Enhancements

- [ ] PPO or A3C for more stable learning
- [ ] Curriculum learning (start with fewer tasks)
- [ ] Attention-based state representation
- [ ] Asynchronous advantage actor-critic (A3C)
- [ ] Self-play for competitive task allocation
- [ ] Persistent task embeddings

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `make test`
4. Submit a pull request

---

**Built with C++20 ❤️ | Zero RL Libraries | Pure Standard Library**
