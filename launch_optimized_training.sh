#!/bin/bash
# ============================================================================
# Launch Optimized Training with Viewer
# ============================================================================
# Quick script to start the optimized RL training pipeline with visualization
#
# Usage:
#   ./launch_optimized_training.sh [options]
#
# Options:
#   --envs N                Number of environments (default: 256)
#   --batch-size N          Batch size (default: 1024)
#   --max-steps N           Maximum steps (default: 10000000)
#   --load-latest           Load latest checkpoint
#   --pause                 Start paused
#   --help                  Show this help
# ============================================================================

set -e

# Default configuration
NUM_ENVS=256
NUM_PHYSICS=8
BATCH_SIZE=1024
ACCUMULATION_STEPS=4
MAX_STEPS=10000000
CHECKPOINT_DIR="checkpoints"
LOAD_LATEST=""
PAUSE=""
TIME_SCALE=1.0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --physics-systems)
            NUM_PHYSICS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --accumulation-steps)
            ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --load-latest)
            LOAD_LATEST="--load-latest"
            shift
            ;;
        --load)
            LOAD_LATEST="--load $2"
            shift 2
            ;;
        --pause)
            PAUSE="--pause-on-start"
            shift
            ;;
        --time-scale)
            TIME_SCALE="$2"
            shift 2
            ;;
        --help|-h)
            head -20 "$0" | tail -17
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         OPTIMIZED RL TRAINING - LAUNCHING                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Environments:        %-42s ║\n" "$NUM_ENVS"
printf "║  Physics Systems:     %-42s ║\n" "$NUM_PHYSICS"
printf "║  Batch Size:          %-42s ║\n" "$BATCH_SIZE"
printf "║  Accumulation Steps:  %-42s ║\n" "$ACCUMULATION_STEPS"
printf "║  Max Steps:           %-42s ║\n" "$MAX_STEPS"
printf "║  Time Scale:          %-42s ║\n" "$TIME_SCALE"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Expected SPS: 1000-5000 (20-50x improvement)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if Bazel is available
if ! command -v bazel &> /dev/null; then
    echo "[Error] Bazel is not installed or not in PATH"
    echo "Please install Bazel or use the pre-built binary if available"
    exit 1
fi

# Build the optimized training viewer
echo "[Build] Building optimized training viewer..."
bazel build //:train_optimized_viewer || {
    echo "[Error] Build failed. Check BUILD.optimized for configuration."
    exit 1
}

# Run the training
echo "[Run] Starting training..."
echo ""
echo "Controls:"
echo "  SPACE    - Pause/Resume"
echo "  ESC      - Exit"
echo "  W/S      - Zoom in/out"
echo "  A/D      - Rotate camera"
echo ""

bazel run //:train_optimized_viewer -- \
    --envs "$NUM_ENVS" \
    --physics-systems "$NUM_PHYSICS" \
    --batch-size "$BATCH_SIZE" \
    --accumulation-steps "$ACCUMULATION_STEPS" \
    --max-steps "$MAX_STEPS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --time-scale "$TIME_SCALE" \
    $LOAD_LATEST \
    $PAUSE
