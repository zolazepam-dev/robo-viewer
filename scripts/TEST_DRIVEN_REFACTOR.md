# Test-Driven Autonomous Refactoring Agent

## Overview

This agent writes comprehensive tests first, then autonomously refactors code to pass those tests while maintaining all existing functionality. It can safely make multi-file changes across the codebase, running the full test suite after each modification to ensure no regressions.

## 🎯 Use Cases

- **Safe Refactoring**: Modernize legacy code without breaking functionality
- **Performance Optimization**: Improve performance with test-backed validation
- **Bug Fixes**: Fix bugs with test-driven development
- **API Modernization**: Update APIs while maintaining backward compatibility
- **Code Cleanup**: Remove technical debt with confidence

## 🚀 Quick Start

```bash
# Basic refactoring task
python3 scripts/test_driven_refactor_agent.py \
    --goal "Refactor SpanNetwork to use AVX2-optimized batch operations"

# With custom test file
python3 scripts/test_driven_refactor_agent.py \
    --goal "Modernize ReplayBuffer to use lock-free data structures" \
    --test-file "src/ReplayBufferTest.cpp"

# With specific affected files
python3 scripts/test_driven_refactor_agent.py \
    --goal "Optimize VectorizedEnv physics stepping" \
    --affected-files src/VectorizedEnv.cpp src/VectorizedEnv.h src/PhysicsCore.cpp \
    --max-iterations 30
```

## 📋 Workflow

### Step 1: Discovery
```
┌─────────────────────────────────────┐
│  Map all affected files             │
│  - Scan src/*.cpp, src/*.h          │
│  - Identify dependencies            │
│  - Create file dependency graph     │
└─────────────────────────────────────┘
```

### Step 2: Test Generation
```
┌─────────────────────────────────────┐
│  Write comprehensive tests          │
│  - Capture current behavior         │
│  - Define desired improvements      │
│  - Document success criteria        │
└─────────────────────────────────────┘
```

### Step 3: Baseline
```
┌─────────────────────────────────────┐
│  Establish baseline                 │
│  - Run existing test suite          │
│  - Record passing/failing tests     │
│  - Document known issues            │
└─────────────────────────────────────┘
```

### Step 4-6: Iterative Refactoring
```
┌─────────────────────────────────────┐
│  Iteration Loop                     │
│  ├─► Make incremental change        │
│  ├─► Run tests                      │
│  ├─► Analyze failures               │
│  ├─► Fix code OR adjust tests       │
│  └─► Verify no regressions          │
└─────────────────────────────────────┘
```

### Step 7: Completion
```
┌─────────────────────────────────────┐
│  Verify goal achieved               │
│  - All tests passing                │
│  - No regressions                   │
│  - Success criteria met             │
└─────────────────────────────────────┘
```

## 🔧 Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--goal` | Refactoring goal description (required) | - |
| `--test-file` | Test file to create/use | `src/RefactoringTest.cpp` |
| `--max-iterations` | Maximum refactoring iterations | 20 |
| `--affected-files` | Files that may be affected | `src/*.cpp src/*.h` |

## 📊 Example Refactoring Goals

### 1. AVX2 Optimization
```bash
python3 scripts/test_driven_refactor_agent.py \
    --goal "Replace scalar math operations with AVX2 SIMD intrinsics in NeuralMath.cpp" \
    --test-file "src/NeuralMathTest.cpp" \
    --max-iterations 25
```

### 2. Memory Layout Optimization
```bash
python3 scripts/test_driven_refactor_agent.py \
    --goal "Convert AoS to SoA layout in VectorizedEnv for better cache utilization" \
    --affected-files src/VectorizedEnv.h src/VectorizedEnv.cpp \
    --test-file "src/VectorizedEnvTest.cpp"
```

### 3. API Modernization
```bash
python3 scripts/test_driven_refactor_agent.py \
    --goal "Modernize TD3Trainer API to use std::optional instead of raw pointers" \
    --affected-files src/TD3Trainer.h src/TD3Trainer.cpp \
    --test-file "src/TD3TrainerTest.cpp"
```

### 4. Bug Fix with TDD
```bash
python3 scripts/test_driven_refactor_agent.py \
    --goal "Fix sum-tree indexing bug in KLPERBuffer (idx starts at 0 instead of 1)" \
    --affected-files src/NeuralNetwork.cpp \
    --test-file "src/KLPERBufferTest.cpp"
```

### 5. Performance Optimization
```bash
python3 scripts/test_driven_refactor_agent.py \
    --goal "Optimize ReplayBuffer sampling with pre-allocated batches" \
    --affected-files src/NeuralNetwork.cpp src/NeuralNetwork.h \
    --test-file "src/ReplayBufferTest.cpp" \
    --max-iterations 30
```

## 🧪 Test File Structure

The agent generates test files following this structure:

```cpp
// Auto-generated test for: [Goal Title]
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <string>

// Test suite
class RefactoringTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup test fixtures
    }

    void TearDown() override {
        // Cleanup
    }
};

// Test current behavior (baseline)
TEST_F(RefactoringTest, CurrentBehaviorBaseline) {
    // Document current behavior
    EXPECT_TRUE(true) << "Baseline test";
}

// Test desired behavior (after refactoring)
TEST_F(RefactoringTest, DesiredBehaviorAfterRefactoring) {
    // This test may fail initially
    // Success criteria documented here
}

// Performance regression tests
TEST_F(RefactoringTest, PerformanceNoRegression) {
    // Ensure performance doesn't degrade
}

// Edge case tests
TEST_F(RefactoringTest, EdgeCasesHandled) {
    // Test edge cases
}
```

## 📈 Agent Capabilities

### File Discovery
- Glob pattern matching for affected files
- Dependency graph analysis
- Cross-file reference tracking

### Test Generation
- Automatic test scaffolding
- Behavior documentation
- Success criteria definition
- Edge case identification

### Incremental Refactoring
- Single Responsibility Principle application
- Method extraction
- Class restructuring
- API modernization
- Performance optimization

### Test Execution
- Bazel test integration
- Timeout handling
- Output parsing
- Failure categorization

### Failure Analysis
- Assertion failure detection
- Segfault identification
- Timeout analysis
- Suggested fixes

### Regression Prevention
- Baseline comparison
- Test count monitoring
- Performance metric tracking
- Behavior verification

## 🔍 Failure Categorization

The agent categorizes test failures to decide on fix strategy:

| Failure Type | Severity | Fix Strategy |
|--------------|----------|--------------|
| `ASSERTION_FAILURE` | HIGH | Check test expectations or implementation |
| `SEGFAULT` | CRITICAL | Check memory access and null pointers |
| `TIMEOUT` | HIGH | Check for infinite loops or deadlocks |
| `COMPILATION_ERROR` | CRITICAL | Fix syntax or type errors |
| `LINKER_ERROR` | CRITICAL | Fix symbol definitions |

## 📝 Logging and Audit

All agent activity is logged:

| Log File | Content |
|----------|---------|
| `refactoring_agent_log.txt` | Detailed agent activity with timestamps |
| `test_results/test_run_XXX.txt` | Individual test run outputs |
| `refactoring_summary.txt` | Final summary of changes made |

## 🎓 Best Practices

### For Safe Refactoring

1. **Always start with tests** - Never refactor without test coverage
2. **Small increments** - Make one change at a time
3. **Run tests frequently** - After every change
4. **Document decisions** - Log why changes were made
5. **Verify no regressions** - Compare against baseline

### When to Use

- ✅ Large-scale code modernization
- ✅ Performance optimization with correctness guarantees
- ✅ API refactoring with backward compatibility
- ✅ Bug fixes with clear reproduction cases
- ✅ Technical debt reduction

### When NOT to Use

- ❌ Adding new features (use regular development)
- ❌ Exploratory refactoring (unclear goals)
- ❌ Time-critical fixes (manual is faster)
- ❌ Cosmetic changes only (no behavior change)

## 🔮 Advanced Features

### Multi-File Coordination

The agent can coordinate changes across multiple files:

```python
# Example: Coordinated API change
affected_files = [
    "src/TD3Trainer.h",
    "src/TD3Trainer.cpp",
    "src/main_train.cpp",
    "src/VectorizedEnv.cpp"
]
```

### Performance Validation

Integrate with benchmarks:

```bash
# Run benchmarks after refactoring
bazel run //:ReplayBufferBenchmark
```

### Git Integration

Automatic commit generation:

```bash
# Stage changes
git add src/*.cpp src/*.h

# Commit with message
git commit -m "Refactor: [goal description]"
```

## 🆘 Troubleshooting

### Agent Stuck in Loop
**Symptom**: Repeatedly making same change

**Solution**:
1. Check `refactoring_agent_log.txt` for iteration count
2. Agent stops after `--max-iterations`
3. Manual intervention may be needed for complex changes

### Tests Not Running
**Symptom**: Agent reports test execution failure

**Solution**:
1. Verify Bazel test target exists: `bazel query 'tests(//...)'`
2. Check test timeout: increase with `--test_timeout=600`
3. Ensure test dependencies are available

### False Positive Regressions
**Symptom**: Agent reports regression when none exists

**Solution**:
1. Review baseline test output
2. Check if test count comparison is accurate
3. Adjust regression detection logic

## 📚 Related Documentation

- `AGENTS.md` - Agentic development guidelines
- `AUTONOMOUS_AGENT_DEPLOYMENT.md` - Build agent deployment
- `bug_report.md` - Known bugs to fix
- `DOCS.md` - Complete project documentation

## 🎯 Success Metrics

The agent is successful when:

1. ✓ All tests pass after refactoring
2. ✓ No regressions detected
3. ✓ Refactoring goal achieved
4. ✓ Code is more maintainable
5. ✓ Performance maintained or improved
6. ✓ Documentation updated

---

**Version**: 1.0.0  
**Deployed**: March 9, 2026  
**Status**: Ready for Production Use
