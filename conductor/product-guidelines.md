# Product Guidelines

## Overview

This document defines the core identity, communication style, and development principles for JOLTrl. All contributors should adhere to these guidelines to maintain consistency and quality across the project.

---

## 1. Code Style & Conventions

### 1.1 Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| **Classes/Structs** | PascalCase | `PhysicsCore`, `SpanNetwork`, `TD3Trainer` |
| **Functions/Methods** | PascalCase | `Init()`, `ResetEpisode()`, `Forward()` |
| **Variables (Local)** | camelCase | `physics`, `observation`, `batchSize` |
| **Variables (Member)** | `m` + PascalCase | `mPhysicsSystem`, `mJobSystem`, `mAllowSleeping` |
| **Constants** | UPPER_CASE or kCamelCase | `NUM_PARALLEL_ENVS`, `kMaxBatchSize` |
| **Templates** | PascalCase with `T` suffix | `MatrixT`, `VectorT` |
| **Private Members** | `m` prefix (enforced) | `mActorNetwork`, `mCriticNetwork` |

### 1.2 File Organization

- **Header Files** (`.h`): Declarations only, minimal inline implementations
- **Source Files** (`.cpp`): Implementations, include corresponding header first
- **Include Order**:
  1. Corresponding header (for `.cpp` files)
  2. Jolt Physics headers (MUST be first for macro safety)
  3. Standard library headers
  4. Third-party headers
  5. Project headers

### 1.3 The Jolt Include Mandate

**CRITICAL**: Every `.cpp` file that interacts with physics MUST start with:

```cpp
#include <Jolt/Jolt.h>  // ABSOLUTE FIRST - before any other headers
#include <Jolt/RegisterTypes.h>
// ... other Jolt headers
// ... other includes
```

This is non-negotiable and prevents catastrophic macro expansion errors.

---

## 2. Performance Standards

### 2.1 The Zero-Allocation Mandate

Memory allocation during the training loop destroys performance:

- **Pre-allocate Everything**: All robot state tensors allocated in massive, contiguous blocks at startup
- **No Spawning/Destroying**: When episodes terminate, zero velocities and overwrite transforms (The "Necromancer" Reset)
- **Avoid STL Containers**: No `std::vector::push_back` in hot loops
- **Use SoA Layout**: Structure-of-Arrays for SIMD-friendly access patterns

### 2.2 SIMD Optimization Requirements

- **AVX2/FMA First**: Prefer intrinsics via `NeuralMath.h` over scalar operations
- **32-Byte Alignment**: All tensors must use `AlignedAllocator<32>`
- **Verify Alignment**: Use `assert(reinterpret_cast<uintptr_t>(ptr) % 32 == 0)` in debug builds
- **8-Wide Operations**: Maximize throughput with `_mm256_*` intrinsics

### 2.3 Threading & Concurrency

- **Asynchronous Training**: Training MUST happen in a background thread to prevent simulation hitches
- **Nested Parallelism**: Disable library-internal threading (`EIGEN_DONT_PARALLELIZE`)
- **Thread Pinning**: Set CPU affinity for physics worker threads via `pthread_setaffinity_np`
- **Lock Minimization**: Use `gSimMutex` with minimal hold times; prefer triple-buffering

### 2.4 Scalability Requirements

- **Support 2048+ Environments**: Visual buffers and triple-buffering arrays sized appropriately
- **Linear SPS Scaling**: Doubling environments should ~double SPS (within 10% overhead)
- **Memory Efficiency**: <1GB RAM for 1024 environments

---

## 3. Documentation Standards

### 3.1 Code Comments

- **Why, Not What**: Focus on rationale, not mechanics
- **High-Value Only**: Comment complex logic, not obvious operations
- **No User Communication**: Never talk to users via comments
- **Preserve Existing Comments**: Do not edit comments separate from code changes

### 3.2 API Documentation

- **Header File Docs**: All public APIs documented in `.h` files
- **Parameter Descriptions**: Explain units, ranges, and constraints
- **Example Usage**: Provide snippets for complex interfaces
- **Performance Notes**: Document allocation behavior and thread safety

### 3.3 User-Facing Documentation

- **DOCS.md**: Complete user guide with quick start, architecture, and troubleshooting
- **QWEN.md**: Agent context guide with project structure and key files
- **AGENTS.md**: Development guidelines for AI agents
- **bug_report.md**: Known issues organized by severity

---

## 4. Testing Requirements

### 4.1 Test Coverage

- **Critical Paths**: All RL training loops must have integration tests
- **SIMD Kernels**: Neural network forward pass verified against scalar reference
- **Physics Interactions**: Environment step function tested for determinism
- **Edge Cases**: Boundary conditions, zero values, maximum values

### 4.2 Test Structure

```cpp
// Test file naming: <Component>Test.cpp
// Test structure:
TEST(ComponentName, TestCaseName) {
    // Arrange
    // Act
    // Assert
}
```

### 4.3 Performance Tests

- **SPS Benchmarks**: Run `//:ReplayBufferBenchmark` to verify throughput
- **Memory Profiling**: Verify zero allocations in hot loops
- **Scaling Tests**: Test with 128, 256, 512, 1024 environments

---

## 5. Error Handling

### 5.1 Initialization Errors

- **Early Returns**: Use early returns for initialization failures
- **Descriptive Messages**: Include component name and failure reason
- **No Logging in Hot Loops**: I/O bottlenecks SPS; only fatal errors logged

### 5.2 Runtime Assertions

- **Invariant Checks**: Use `assert()` for internal consistency
- **Debug Builds Only**: Assertions compiled out in release
- **Alignment Checks**: Verify SIMD buffer alignment in debug

### 5.3 Recovery Strategies

- **Graceful Degradation**: Continue training with reduced features if optional components fail
- **Checkpoint Recovery**: Auto-reload from last valid checkpoint on crash
- **NaN Detection**: Detect and halt on NaN/Inf in neural network outputs

---

## 6. API Design Principles

### 6.1 Interface Design

- **Minimal Surfaces**: Expose only what users need
- **Value Semantics**: Prefer pass-by-value for small structs
- **Const Correctness**: Mark all non-mutating methods `const`
- **RAII**: Resource acquisition is initialization; no manual cleanup

### 6.2 Configuration

- **Struct-Based Config**: Group related parameters in config structs
- **Sensible Defaults**: All config fields have production-ready defaults
- **Validation**: Check config values at initialization, not runtime

### 6.3 Extensibility

- **Virtual Interfaces**: Abstract base classes for pluggable components
- **Factory Pattern**: Use factories for object creation
- **Dependency Injection**: Pass dependencies via constructor

---

## 7. Commit Message Conventions

### 7.1 Format

```
<type>(<scope>): <subject>

<body - optional>

<footer - optional>
```

### 7.2 Types

- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `refactor`: Code restructuring (no behavior change)
- `docs`: Documentation updates
- `test`: Test additions or modifications
- `chore`: Build/config/tooling changes
- `conductor`: Conductor framework changes

### 7.3 Examples

```
feat(environment): Add domain randomization for gravity and friction

Implemented per-environment physics randomization in CombatEnv::Reset().
Randomizes gravity (0.8-1.2g), friction (0.5-1.5), and restitution (0.3-0.7).

Fixes #42

perf(neural): Optimize SPAN layer forward pass with AVX2 intrinsics

Replaced scalar B-spline evaluation with _mm256_* intrinsics.
Achieves 4.2x speedup for layer forward pass.

chore(conductor): Mark track 'Fix AVX2 Tanh Implementation' as complete
```

---

## 8. Code Review Guidelines

### 8.1 Review Checklist

- [ ] Zero allocations in hot loops
- [ ] SIMD alignment verified (32-byte)
- [ ] Jolt includes are first (if physics code)
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Performance impact measured (SPS)

### 8.2 Review Response Time

- **Critical Bugs**: <4 hours
- **PR Reviews**: <24 hours
- **Documentation**: <48 hours

### 8.3 Merge Requirements

- **CI Passing**: All builds and tests must pass
- **One Approval**: Minimum one reviewer approval
- **No Conflicts**: Branch must be up-to-date with main

---

## 9. User Experience Principles

### 9.1 Training UX

- **Progress Feedback**: Real-time SPS, reward, and loss metrics
- **Checkpoint Transparency**: Clear indication of save/load events
- **Error Recovery**: Automatic resume from last checkpoint

### 9.2 Visualization UX

- **Non-Blocking Rendering**: Training continues during visualization
- **Configurable Detail**: LOD system for high environment counts
- **Camera Controls**: Intuitive fly-through and follow modes

### 9.3 Configuration UX

- **JSON Configs**: Human-readable robot and environment definitions
- **CLI Overrides**: Command-line flags for common parameters
- **Validation Errors**: Clear messages for invalid configurations

---

## 10. Brand Voice & Tone

### 10.1 Technical Communication

- **Precise**: Use exact terminology (SPS, latent dim, B-spline)
- **Concise**: No filler; get to the point
- **Direct**: Active voice, imperative mood for instructions

### 10.2 Error Messages

- **Actionable**: Tell users what to do next
- **Specific**: Include component, parameter, and value
- **Helpful**: Link to relevant documentation

### 10.3 Documentation Tone

- **Professional**: Formal but approachable
- **Assumptive**: Assume technical competence
- **Example-Driven**: Show, don't just tell

---

## 11. Security Guidelines

### 11.1 Code Security

- **No Secrets in Code**: API keys, credentials in environment variables only
- **Input Validation**: Validate all JSON config inputs
- **Buffer Safety**: Use `std::array` or `std::vector` with bounds checking

### 11.2 Model Security

- **Checkpoint Validation**: Verify checkpoint integrity before loading
- **Export Sanitization**: Remove internal state from exported policies

---

## 12. Accessibility Considerations

### 12.1 Visual Accessibility

- **Color-Blind Safe**: Use color + shape for status indicators
- **Configurable UI**: Font size, contrast, and layout options
- **Keyboard Navigation**: All viewer controls accessible via keyboard

### 12.2 Documentation Accessibility

- **Screen Reader Friendly**: Proper heading hierarchy in markdown
- **Alt Text**: Describe diagrams and charts
- **Clear Language**: Avoid idioms and cultural references
