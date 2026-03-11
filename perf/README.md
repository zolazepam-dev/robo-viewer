# Robo-Viewer Performance Testing Framework

This directory contains the performance testing infrastructure for the robo-viewer project.

## Overview

The performance testing framework provides tools to measure:
- Build time benchmarking
- Runtime performance
- Memory usage analysis
- CPU utilization monitoring

## Usage

### Building and Running Performance Tests

1. **Build all performance tests:**
   ```bash
   bazel build //perf:all
   ```

2. **Run a specific performance test:**
   ```bash
   bazel run //perf:build_time_benchmark
   ```

3. **Run with custom duration (default is 10 seconds):**
   ```bash
   bazel run //perf:viewer_runtime_perf --define perf_duration=30
   ```

### Available Performance Tests

| Test Name | Description |
|-----------|-------------|
| `build_time_benchmark` | Measures build time for the train target |
| `viewer_runtime_perf` | Measures runtime performance of the viewer |
| `memory_usage_analysis` | Monitors memory usage during execution |
| `cpu_utilization_monitor` | Monitors CPU utilization during execution |
| `jolt_physics_perf` | Focuses on Jolt Physics engine performance |

## Customizing Performance Tests

To add new performance tests:

1. Add a new target to `perf/BUILD`
2. Define the test parameters (target, duration, etc.)
3. Optionally create a custom test implementation in `perf_test.bzl`

## Advanced Usage

For more detailed profiling, you can use Bazel's built-in profiling features:

```bash
# Generate a build profile
bazel build //:train --profile=build_profile.json

# Analyze the profile
bazel info profile
```

## Integration with CI

The performance tests can be integrated into CI pipelines to monitor performance regressions over time.

## Requirements

- Bazel 6.0+
- Linux/macOS (for memory/CPU monitoring features)
- Standard development tools (gcc/clang, make, etc.)