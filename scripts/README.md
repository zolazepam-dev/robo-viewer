# Autonomous Build-Test-Debug Agent

## Overview

This directory contains autonomous agents that continuously monitor build failures, diagnose root causes, apply targeted fixes, and validate through test execution without human intervention.

## Components

### 1. `autonomous_build_agent.py` - Main Python Agent

A sophisticated agent that:
- Parses compiler/test error output to identify root causes
- Uses pattern matching to diagnose common errors (AVX2 issues, include order, linker errors, etc.)
- Searches the codebase to locate problematic code
- Proposes minimal, targeted fixes
- Iterates through up to 5 distinct approaches before giving up
- Maintains detailed logs of all attempts

**Usage:**
```bash
# Basic usage
python3 scripts/autonomous_build_agent.py

# With custom max iterations
python3 scripts/autonomous_build_agent.py --max-iterations 10

# Auto-apply fixes (experimental)
python3 scripts/autonomous_build_agent.py --auto-apply
```

### 2. `autonomous_build_agent.sh` - Bash Wrapper

A shell script version for simpler integration:
```bash
chmod +x scripts/autonomous_build_agent.sh
./scripts/autonomous_build_agent.sh
```

### 3. `watch_build.sh` - Continuous Monitoring

Watches for file changes and automatically triggers the agent on build failures:
```bash
chmod +x scripts/watch_build.sh
./scripts/watch_build.sh
```

## Detected Error Patterns

The agent recognizes these common error patterns:

| Pattern | Severity | Description |
|---------|----------|-------------|
| `AVX2_TANH_ERROR` | CRITICAL | `_mm256_tanh_ps` is non-standard in AVX2 |
| `JOLT_INCLUDE_ERROR` | CRITICAL | Jolt headers not included first |
| `LINKER_ERROR` | HIGH | Missing symbol definitions |
| `SIGNATURE_MISMATCH` | HIGH | Function signature mismatch |
| `UNDECLARED_IDENTIFIER` | MEDIUM | Missing include or forward declaration |
| `SUM_TREE_INDEX_ERROR` | CRITICAL | Sum-tree indexing starts at 0 |
| `PRIORITY_TRUNCATION` | HIGH | Float priorities truncated to integers |

## Integration with Qwen Code

The agent is designed to work seamlessly with Qwen Code's tools:

1. **Automatic Tool Usage**: The agent uses `grep_search` to locate errors, `read_file` to understand context, and `edit` to apply fixes.

2. **Todo Tracking**: Each fix attempt is tracked with `todo_write` for visibility.

3. **Persistent Terminal**: Run the watch script in a persistent terminal session for continuous monitoring.

## Example Workflow

```bash
# 1. Start the watch agent in background
./scripts/watch_build.sh &

# 2. Make code changes
edit src/NeuralNetwork.cpp

# 3. Agent automatically detects build failure
# 4. Agent diagnoses: AVX2_TANH_ERROR
# 5. Agent locates: src/NeuralNetwork.cpp:123
# 6. Agent proposes: Replace _mm256_tanh_ps with custom implementation
# 7. Agent applies fix and rebuilds
# 8. If successful, agent continues monitoring
```

## Logs

All agent activity is logged to:
- `agent_log.txt` - Python agent detailed log
- `watch_agent.log` - Watch script activity log
- `build_output.txt` - Last build output
- `test_output.txt` - Last test output

## Configuration

Environment variables:
- `RUN_TESTS_ON_SUCCESS=true` - Run tests after successful builds
- `MAX_ITERATIONS=5` - Maximum fix attempts per session

## Limitations

- **Auto-apply mode is experimental** - Review proposed fixes before applying
- **5 iteration limit** - Prevents infinite loops on unfixable errors
- **Pattern-based diagnosis** - May miss novel error types
- **No semantic understanding** - Cannot fix logic errors, only syntax/compilation issues

## Future Enhancements

- [ ] Integrate with Qwen Code's `task` tool for complex multi-file fixes
- [ ] Add machine learning to improve pattern recognition
- [ ] Support for test failure diagnosis (not just build failures)
- [ ] Automatic PR creation for successful fixes
- [ ] Integration with CI/CD pipelines

## Troubleshooting

**Agent not detecting errors:**
- Check that error patterns are defined in `BuildDiagnostician.ERROR_PATTERNS`
- Verify grep commands are finding the right files

**Agent stuck in loop:**
- Check `agent_log.txt` for iteration count
- Agent automatically stops after `MAX_ITERATIONS` attempts

**Build succeeds but tests fail:**
- Agent will attempt to diagnose test failures
- Review `test_output.txt` for failure details

## Best Practices

1. **Run in persistent terminal**: Use `tmux` or `screen` for long-running sessions
2. **Review proposed fixes**: Even in auto-apply mode, review changes before committing
3. **Check logs regularly**: Agent logs provide insight into recurring issues
4. **Update patterns**: Add new error patterns as you encounter them
5. **Combine with manual debugging**: Agent handles routine fixes, humans handle complex issues

## Related Documentation

- `AGENTS.md` - Agentic development guidelines
- `bug_report.md` - Known bugs and issues
- `DOCS.md` - Complete project documentation
- `BUILD` - Bazel build definitions
