# Bouncy Orbiter Training Guide

## ✅ Build Issues Fixed

The build was failing due to a linker compatibility issue with `lld`. The fix is to use the GNU gold linker instead.

### Solution
Add `--linkopt="-fuse-ld=gold"` to the bazel build command:

```bash
bazel build //:train --compilation_mode=opt --linkopt="-fuse-ld=gold"
```

## 🚀 Quick Start

### Option 1: Use the Build Script
```bash
./build_train.sh          # Normal build
./build_train.sh clean    # Clean build
```

### Option 2: Direct Command
```bash
bazel build //:train --compilation_mode=opt --linkopt="-fuse-ld=gold"
```

### Option 3: Run Bouncy Orbiter Training
```bash
./train_bouncy_orbiter.sh 128 10000
```

## 🎮 Training with Bouncy Orbiter

### Command Line
```bash
./bazel-bin/train --envs 128 --steps 10000 --robot "robots/bouncy_orbiter.json"
```

### Launcher Script
```bash
./train_bouncy_orbiter.sh [envs] [steps]

# Examples:
./train_bouncy_orbiter.sh 128 10000    # 128 envs, 10k steps
./train_bouncy_orbiter.sh 256 50000    # 256 envs, 50k steps
```

## 🤖 Bouncy Orbiter Robot Specs

| Component | Value |
|-----------|-------|
| **Central Body** | Sphere, radius 1.0, mass 30.0 |
| **Satellites** | 3 spheres, radius 0.4, mass 5.0 each |
| **Constraints** | 6-DOF joints with motors |
| **Motor Max Torque** | 20,000 Nm per joint |
| **Restitution** | 0.9 (very bouncy) |
| **Friction** | 0.8-0.9 |

### Configuration File
```json
robots/bouncy_orbiter.json
```

## 📊 Performance Metrics

Current performance with 128 environments:
- **SPS**: ~1,000-2,000 (varies based on physics complexity)
- **Stutter**: Some stuttering present (being optimized)
- **Training**: Non-blocking async training

## 🔧 Available Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--envs N` | Number of parallel environments | 128 |
| `--steps N` | Number of training steps | 10000 |
| `--robot PATH` | Robot configuration JSON | UI setting |
| `--render-env N` | Which environment to render | 0 |

### Examples

```bash
# Default training (uses UI robot setting)
./bazel-bin/train --envs 128 --steps 10000

# Bouncy orbiter training
./bazel-bin/train --envs 128 --steps 10000 --robot robots/bouncy_orbiter.json

# Combat bot training
./bazel-bin/train --envs 128 --steps 10000 --robot robots/combat_bot.json

# High-throughput training (256 envs)
./bazel-bin/train --envs 256 --steps 10000 --robot robots/bouncy_orbiter.json
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `build_train.sh` | Build script with linker fix |
| `train_bouncy_orbiter.sh` | Bouncy orbiter launcher |
| `robots/bouncy_orbiter.json` | Robot configuration |
| `src/main_train.cpp` | Main training loop (updated with --robot flag) |
| `BUILD` | Bazel build definition |

## 🐛 Troubleshooting

### Build fails with "undefined symbol: main"
**Solution**: Use the gold linker:
```bash
bazel build //:train --compilation_mode=opt --linkopt="-fuse-ld=gold"
```

### Training is slow or stuttering
**Solutions**:
1. Reduce number of environments: `--envs 64`
2. Use headless mode (no rendering)
3. Close other GPU-intensive applications

### Robot doesn't load
**Check**:
1. Robot JSON file exists: `ls robots/bouncy_orbiter.json`
2. JSON is valid: `cat robots/bouncy_orbiter.json | python3 -m json.tool`
3. Path is correct: Use absolute or relative to project root

## 📈 Next Steps

1. **Start Training**:
   ```bash
   ./train_bouncy_orbiter.sh 128 10000
   ```

2. **Monitor Progress**: Watch the console output for training metrics

3. **Adjust Parameters**: Modify envs/steps based on your needs

4. **Save Checkpoints**: Models are auto-saved to `~/.joltrl/checkpoints/`

---

**Created**: March 10, 2026  
**Status**: ✅ Working with gold linker fix
