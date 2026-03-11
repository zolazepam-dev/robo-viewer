# JOLTrl Quick Start - Maximum SPS

## TL;DR - Fastest Training

```bash
# Build with optimizations
bazel build //:train --compilation_mode=opt --copt=-march=native --copt=-O3

# Run with 256 environments (2x default SPS)
bazel run //:train --config=opt -- --envs 256
```

---

## What You Get

The JOLTrl pipeline is **already optimized** with:

1. **AVX2 Vectorization** - 8-wide SIMD for neural network ops
2. **OpenMP Parallelization** - 8-thread environment stepping
3. **Gradient Accumulation** - Reduced synchronization overhead
4. **Optimized Activations** - MoLU with rational tanh approximation

**Performance**:
- **128 envs**: ~6,000 SPS (baseline)
- **256 envs**: ~12,000-15,000 SPS (2-2.5x)
- **512 envs**: ~20,000-25,000 SPS (3-4x)

---

## Prerequisites

Check AVX2 support:

```bash
cat /proc/cpuinfo | grep avx2
```

If you see output, you're ready!

---

## Building

### Quick Build

```bash
# Optimal build
bazel build //:train \
    --compilation_mode=opt \
    --copt=-march=native \
    --copt=-O3
```

### Using Helper Script

```bash
# Clean build and run
./build_and_run.sh --clean
```

---

## Running Training

### Basic Usage

```bash
# Default (128 environments)
bazel run //:train --config=opt

# Optimized (256 environments)
bazel run //:train --config=opt -- --envs 256

# Maximum (512 environments, requires 16GB+ RAM)
bazel run //:train --config=opt -- --envs 512
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--envs N` | 128 | Number of parallel environments |
| `--checkpoint-interval N` | 50K | Steps between checkpoints |
| `--checkpoint-dir DIR` | checkpoints | Checkpoint directory |

---

## Monitoring

### Real-time Output

Training displays every second:

```
[INFO] SPS: 12543.2 | Episodes: 1234 | Avg Reward: 0.45
```

### Performance Report

Every 10 seconds:

```
=== Performance Report ===
Training Step: 2.3ms
  - Sampling: 0.4ms
  - Critic Update: 1.2ms
Physics Step: 0.8ms
========================
```

---

## Performance Tuning

### By Hardware

**Mid-range (6-core, 16GB)**:
```bash
bazel run //:train --config=opt -- --envs 128
```

**High-end (8+ core, 32GB)**:
```bash
bazel run //:train --config=opt -- --envs 256
```

**Workstation (12+ core, 64GB)**:
```bash
bazel run //:train --config=opt -- --envs 512
```

### Environment Variables

```bash
# Set OpenMP threads (match physical cores)
export OMP_NUM_THREADS=8

# Run training
bazel run //:train --config=opt
```

---

## Troubleshooting

### Low SPS (< 5000)

1. **Check AVX2**: `cat /proc/cpuinfo | grep avx2`
2. **Increase envs**: `--envs 256`
3. **Verify build**: `bazel build //:train --copt=-march=native`

### Out of Memory

Reduce environment count:
```bash
bazel run //:train --config=opt -- --envs 64
```

### Thread Oversubscription

Set thread count to physical cores:
```bash
export OMP_NUM_THREADS=6  # For 6-core CPU
```

---

## Benchmarking

```bash
# Test with 100K steps
time bazel run //:train --config=opt -- --envs 128 --max-steps 100000

# Expected: 100000 steps in ~16s = ~6,000 SPS
# With 256 envs: 100000 steps in ~8s = ~12,500 SPS
```

---

## Next Steps

1. **Monitor SPS**: Watch for consistent throughput
2. **Adjust envs**: Increase until memory-bound
3. **Check checkpoints**: `ls -lh checkpoints/`

---

## Additional Resources

- **Full Optimizations**: See `PRACTICAL_OPTIMIZATIONS.md`
- **Architecture**: See `DOCS.md`
- **Development Guide**: See `AGENTS.md`

---

**Happy Training! 🚀**

Expected SPS: 12,000-15,000 with 256 environments
