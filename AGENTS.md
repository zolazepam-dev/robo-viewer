# JOLTrl - Agentic Development Guide 🤖

## 🌟 Living Document Protocol
This document is a **living entity**. It evolves alongside the JOLTrl codebase. 
- **Agents:** When you implement a major architectural change or discover a foundational pattern, you **must** propose an update to this file.
- **Users:** Please prompt agents with update proposals whenever you feel the project's "mental model" has shifted.

---

## 🏗 Agentic Architecture
JOLTrl is designed to be "Agent-First." The codebase prioritizes explicit state management, SIMD-optimized math, and a strictly decoupled C++/React architecture to ensure AI agents can reason about and modify the system effectively.

### Core Agentic Capabilities (MCP)
The project integrates with the **Model Context Protocol (MCP)** to grant agents deep system access:
- **Debugging**: Integrated GDB and Ripgrep for deep code analysis.
- **Research**: Access to Brave, Exa, and technical documentation for state-of-the-art RL research.
- **Validation**: Playwright/Puppeteer for end-to-end telemetry testing.
- **Persistence**: SQLite/DuckDB for long-term agent memory and training meta-analysis.

### 🤖 Autonomous Agent Systems

JOLTrl includes two complementary autonomous agent systems:

#### 1. Autonomous Build-Test-Debug Agent

Continuous build monitoring and automated repair:

**Components:**
- **`scripts/autonomous_build_agent.py`** - Pattern-based diagnosis
- **`scripts/watch_build.sh`** - Continuous monitoring
- **`scripts/launch_agent.sh`** - Interactive launcher

**Capabilities:**
- Real-time build monitoring
- Pattern recognition (7+ error types)
- Automated diagnosis and fix proposals
- Iterative repair (up to 5 attempts)
- Comprehensive logging

**Usage:**
```bash
# Quick diagnostic
python3 scripts/autonomous_build_agent.py

# Continuous monitoring
./scripts/watch_build.sh &
```

**Known Error Patterns:**
| Pattern | Severity | Description |
|---------|----------|-------------|
| `AVX2_TANH_ERROR` | CRITICAL | `_mm256_tanh_ps` non-standard in AVX2 |
| `JOLT_INCLUDE_ERROR` | CRITICAL | Jolt headers not first |
| `LINKER_ERROR` | HIGH | Missing symbols |
| `SIGNATURE_MISMATCH` | HIGH | Function signature mismatch |
| `UNDECLARED_IDENTIFIER` | MEDIUM | Missing include |
| `SUM_TREE_INDEX_ERROR` | CRITICAL | Sum-tree idx starts at 0 |
| `PRIORITY_TRUNCATION` | HIGH | Float→int priority loss |

See `AUTONOMOUS_AGENT_DEPLOYMENT.md` for details.

#### 2. Test-Driven Autonomous Refactoring Agent

Test-first autonomous code modernization:

**Components:**
- **`scripts/test_driven_refactor_agent.py`** - Main refactoring agent
- **`scripts/run_refactoring_task.sh`** - Interactive task menu
- **`src/NeuralMathTest.cpp`** - Example test suite

**Capabilities:**
- **Test-First Development**: Writes comprehensive tests before refactoring
- **File Discovery**: Maps all affected files using glob patterns
- **Baseline Establishment**: Records current test state
- **Incremental Refactoring**: Makes small changes with test validation
- **Failure Analysis**: Categorizes test failures (assertion, segfault, timeout)
- **Regression Prevention**: Verifies no functionality breaks
- **Iterative Improvement**: Up to 20 iterations per task

**Workflow:**
```
1. Discovery     → Map affected files
2. Test Gen      → Write comprehensive tests
3. Baseline      → Run tests, record state
4. Iterate       → Make change → Run tests → Analyze → Fix
5. Verify        → Confirm goal achieved, no regressions
```

**Usage:**
```bash
# Interactive task menu
./scripts/run_refactoring_task.sh

# Direct command
python3 scripts/test_driven_refactor_agent.py \
    --goal "Replace _mm256_tanh_ps with AVX2 implementation" \
    --test-file "src/NeuralMathTest.cpp" \
    --max-iterations 10
```

**Pre-configured Tasks:**
1. Fix AVX2 Tanh Implementation (Critical)
2. Fix KLPERBuffer Sum-Tree Indexing (Critical)
3. Fix Priority Truncation in PER (High)
4. Fix Replay Buffer Transition Bug (High)
5. Optimize NeuralMath with AVX2 (Performance)
6. Modernize TD3Trainer API (Code Quality)

**Test Structure:**
- Baseline tests (document current behavior)
- Correctness tests (verify desired behavior)
- Edge case tests (boundary conditions)
- Performance tests (benchmarks)
- Alignment tests (memory alignment)
- Regression tests (prevent degradation)

See `TEST_DRIVEN_REFACTOR_DEPLOYMENT.md` for complete guide.

---

## 🛠 Engineering Standards for Agents

### 1. The Performance Mandate
At 6,000+ SPS, every microsecond counts. Agents must adhere to these rules:
- **Zero-Allocation Hot Loops**: Never use `new`, `malloc`, or `std::vector::push_back` inside the `SimulationLoop` or `TrainingLoop`. Use pre-allocated SoA (Structure of Arrays) patterns.
- **SIMD First**: Prefer AVX2/FMA intrinsics via `NeuralMath.h`. Always verify alignment (32-byte) for tensors.
- **Lock-Awareness**: Minimize `gSimMutex` hold times. Use the triple-buffering pattern in `main_train.cpp` for visual state extraction.
- **Scalability**: Support at least **2048 parallel environments** by ensuring visual buffers and triple-buffering arrays are appropriately sized.

### 2. Threading & Concurrency
- **Asynchronous Training**: Training MUST happen in a background thread (`TrainingLoop`) to prevent simulation hitches.
- **Nested Parallelism**: Disable library-internal threading (e.g., `EIGEN_DONT_PARALLELIZE`) to prevent core over-subscription and context-switch stutters.

### 3. Robust RL Environment Design
- **Domain Randomization (DR)**: Every environment should support physics randomization (gravity, friction, restitution) during `Reset()`. Use `bodyInterface.SetGravityFactor` for per-robot gravity control.
- **Transparent Boundaries**: When rendering, use two-pass rendering (Opaque -> Transparent) to ensure arena boundaries (walls) are visible but do not obscure the robots.
- **Damage Mechanics**: Damage should be calculated based on relative velocities and collision intensities. Ensure multipliers are tuned for a noticeable HP delta (usually 100x larger than raw impulse values).

### 4. Diagnostic-Driven Development
- **PerformanceDiagnoser**: Always use `DIAGNOSE_SCOPE` and `DIAGNOSE_MUTEX_LOCK` when modifying hot paths. 
- **UI Explanability**: All user-exposed settings in `OverlayUI` should include descriptive tooltips using `ImGui::SetTooltip()`.

---

## 🧪 Agent Validation Workflow
When tasked with a feature or bug fix, agents should follow this cycle:
1. **Research**: Use `grep_search` and `get_code_context_exa` to map dependencies.
2. **Reproduction**: Create a minimal test case (e.g., `src/ReplayBufferBenchmark.cpp`) to confirm the issue.
3. **Surgical Implementation**: Use `replace` for targeted edits; avoid mass-rewriting files unless refactoring.
4. **Validation**: Run `bazel build //:train` and verify metrics via the `PerformanceDiagnoser`.

---

## 💡 Prompting Proposals
If you notice this guide is missing a new convention (e.g., a new sensor implementation pattern or a specific Jolt interface quirk), please use the following prompt:
> *"Based on our recent changes to [Module], please propose an update to AGENTS.md to document the new [Convention/Standard]."*

---

**Current Status:** Phase 4 - Scalable Environments & Domain Randomization.
**Last Updated:** March 9, 2026
