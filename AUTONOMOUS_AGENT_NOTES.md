# Autonomous Build Agent - Session Notes

## Current Session: March 9, 2026

### Agent Setup Complete ✓

The autonomous build-test-debug loop agent has been successfully deployed with the following components:

1. **`scripts/autonomous_build_agent.py`** - Main Python agent with pattern-based diagnosis
2. **`scripts/autonomous_build_agent.sh`** - Bash wrapper version
3. **`scripts/watch_build.sh`** - Continuous monitoring with auto-trigger
4. **`scripts/launch_agent.sh`** - Interactive quick-start launcher

### Known Critical Bugs (Pre-identified)

Based on `bug_report.md`, the agent should prioritize these issues:

#### 1. CRITICAL: `_mm256_tanh_ps` Compilation Failure
- **Location**: `src/OptimizedBatchOps.h:212`
- **Issue**: Non-standard AVX2 intrinsic (actually AVX-512)
- **Current Code**:
  ```cpp
  __m256 result = _mm256_tanh_ps(x);  // AVX-512, fallback for AVX2:
  // Manual tanh for AVX2:
  __m256 exp_2x = _mm256_exp_ps(_mm256_mul_ps(x, _mm256_set1_ps(2.0f)));
  ```
- **Fix**: Remove line 212, the manual implementation below is correct
- **Pattern**: `AVX2_TANH_ERROR`

#### 2. CRITICAL: `KLPERBuffer` Sum-Tree Indexing
- **Location**: `src/NeuralNetwork.cpp` (KLPERBuffer class)
- **Issue**: Tree traversal starts at `idx = 0`, causing infinite loop
- **Pattern**: `SUM_TREE_INDEX_ERROR`

#### 3. HIGH: Priority Truncation in PER
- **Location**: `src/NeuralNetwork.cpp`
- **Issue**: `static_cast<int>(priority)` loses precision
- **Pattern**: `PRIORITY_TRUNCATION`

#### 4. HIGH: Wrong Transition in Replay Buffer
- **Location**: `src/main_train.cpp`
- **Issue**: `buffer.Add(obs, action, reward, obs, done)` uses current state as nextState
- **Impact**: Breaks TD3 learning algorithm

#### 5. HIGH: FPS Display Bug
- **Location**: `src/main_train.cpp`
- **Issue**: `lastRenderTime` updated before FPS calculation
- **Pattern**: Timing/calculation order error

### Agent Diagnostic Patterns

The agent recognizes these error signatures:

```python
ERROR_PATTERNS = {
    'AVX2_TANH_ERROR': {
        'pattern': r'_mm256_tanh_ps',
        'fix': 'Implement custom tanh using polynomial approximation'
    },
    'JOLT_INCLUDE_ERROR': {
        'pattern': r'Jolt/Jolt\.h.*must be first',
        'fix': 'Add #include <Jolt/Jolt.h> as first include'
    },
    'LINKER_ERROR': {
        'pattern': r'undefined reference to',
        'fix': 'Check BUILD file dependencies'
    },
    'SIGNATURE_MISMATCH': {
        'pattern': r'no matching function for call',
        'fix': 'Verify function declaration matches definition'
    },
    'UNDECLARED_IDENTIFIER': {
        'pattern': r'use of undeclared identifier',
        'fix': 'Add required header or forward declaration'
    },
    'SUM_TREE_INDEX_ERROR': {
        'pattern': r'idx\s*=\s*0.*2\s*\*\s*idx',
        'fix': 'Start tree traversal at idx = 1'
    },
    'PRIORITY_TRUNCATION': {
        'pattern': r'static_cast<int>\(priority\)',
        'fix': 'Use float priorities throughout sum-tree'
    }
}
```

### Recommended Fix Order

Based on dependency analysis:

1. **Fix `_mm256_tanh_ps` first** - Blocks all other compilation
2. **Fix sum-tree indexing** - Required for PER to function
3. **Fix priority truncation** - Improves PER accuracy
4. **Fix replay buffer transition** - Critical for learning
5. **Fix FPS display** - Quality of life improvement

### Manual Fix: `_mm256_tanh_ps` (Blocking Issue)

**File**: `src/OptimizedBatchOps.h`
**Lines**: 207-223

**Current Code**:
```cpp
inline void BatchedTanh_AVX2(float* X, int size) {
    const int simd_size = size - (size % 8);

    for (int i = 0; i < simd_size; i += 8) {
        __m256 x = _mm256_loadu_ps(X + i);
        __m256 result = _mm256_tanh_ps(x);  // AVX-512, fallback for AVX2:
        // Manual tanh for AVX2:
        __m256 exp_2x = _mm256_exp_ps(_mm256_mul_ps(x, _mm256_set1_ps(2.0f)));
        __m256 one = _mm256_set1_ps(1.0f);
        result = _mm256_div_ps(_mm256_sub_ps(exp_2x, one), _mm256_add_ps(exp_2x, one));
        _mm256_storeu_ps(X + i, result);
    }
    // ... scalar tail
}
```

**Fixed Code**:
```cpp
inline void BatchedTanh_AVX2(float* X, int size) {
    const int simd_size = size - (size % 8);

    for (int i = 0; i < simd_size; i += 8) {
        __m256 x = _mm256_loadu_ps(X + i);
        // Manual tanh for AVX2 using exponential identity:
        // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        __m256 exp_2x = _mm256_exp_ps(_mm256_mul_ps(x, _mm256_set1_ps(2.0f)));
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 result = _mm256_div_ps(
            _mm256_sub_ps(exp_2x, one),
            _mm256_add_ps(exp_2x, one)
        );
        _mm256_storeu_ps(X + i, result);
    }

    for (int i = simd_size; i < size; i++) {
        X[i] = std::tanh(X[i]);
    }
}
```

**Alternative (Better Performance)**: Use Pade approximation from `NeuralMath.cpp`:
```cpp
inline void BatchedTanh_AVX2(float* X, int size) {
    const int simd_size = size - (size % 8);
    
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 clampHi = _mm256_set1_ps(10.0f);
    const __m256 clampLo = _mm256_set1_ps(-10.0f);
    const __m256 pade_a = _mm256_set1_ps(0.275f);
    const __m256 pade_b = _mm256_set1_ps(0.664f);

    for (int i = 0; i < simd_size; i += 8) {
        __m256 x = _mm256_loadu_ps(X + i);
        x = _mm256_min_ps(x, clampHi);
        x = _mm256_max_ps(x, clampLo);

        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 num = _mm256_fmadd_ps(pade_a, x2, one);
        num = _mm256_mul_ps(x, num);
        __m256 den = _mm256_fmadd_ps(pade_b, x2, one);

        __m256 rcp = _mm256_rcp_ps(den);
        rcp = _mm256_mul_ps(rcp, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(den, rcp)));

        _mm256_storeu_ps(X + i, _mm256_mul_ps(num, rcp));
    }

    for (int i = simd_size; i < size; i++) {
        X[i] = std::tanh(X[i]);
    }
}
```

### Agent Usage Examples

#### Quick Diagnostic
```bash
cd /media/cammyz/EverythingHere/robo-viewer
python3 scripts/autonomous_build_agent.py
```

#### Continuous Monitoring
```bash
./scripts/watch_build.sh
```

#### Interactive Session
```bash
./scripts/launch_agent.sh
# Select mode 3 for interactive debugging
```

### Integration with Qwen Code

The agent uses these Qwen Code tools:
- `grep_search` - Locate error sources
- `read_file` - Understand context
- `edit` - Apply targeted fixes
- `run_shell_command` - Execute builds and tests
- `todo_write` - Track fix attempts
- `task` - Delegate complex multi-file repairs

### Performance Metrics

Target build performance:
- **Clean build**: < 2 minutes
- **Incremental build**: < 30 seconds
- **Agent iteration**: < 5 minutes (including diagnosis)

### Next Steps

1. Apply manual fix to `src/OptimizedBatchOps.h` (see above)
2. Run build to verify fix
3. Allow agent to diagnose remaining issues
4. Iterate through bug fix list

### Log Files

- `agent_log.txt` - Python agent detailed log
- `watch_agent.log` - Watch script activity
- `build_output.txt` - Last build output
- `test_output.txt` - Last test output

---

**Agent Status**: Deployed and Ready
**Next Action**: Apply manual fix to unblock compilation
**Expected Outcome**: Build succeeds, agent can diagnose runtime issues
