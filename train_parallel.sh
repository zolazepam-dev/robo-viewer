#!/bin/bash
# Ultra-Safe Combat Training Script
# Runs vectorized training with minimal environments to verify stability

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# SAFE MODE: Minimal load for visual debugging
DEFAULT_ENVS=10 # 2 envs = 4 robots total
DEFAULT_CHECKPOINT_INTERVAL=10000
DEFAULT_MAX_STEPS=10000000
DEFAULT_CHECKPOINT_DIR="checkpoints"
DEFAULT_MODE="rendered"  # "rendered" or "headless"

NUM_ENVS="${NUM_ENVS:-$DEFAULT_ENVS}"
MODE="${MODE:-$DEFAULT_MODE}"

# Select target based on mode
if [ "$MODE" = "rendered" ]; then
    TARGET="train"
    MAX_SAFE_ENVS=16  # Rendering is slower, use fewer envs
else
    TARGET="train_headless"
    MAX_SAFE_ENVS=96
fi

# Validate environment count
if [ "$NUM_ENVS" -gt "$MAX_SAFE_ENVS" ]; then
    echo "WARNING: NUM_ENVS=$NUM_ENVS exceeds safe limit of $MAX_SAFE_ENVS for $MODE mode"
    echo "System may crash due to Jolt Physics capacity constraints"
    echo "Recommend using NUM_ENVS=$MAX_SAFE_ENVS or lower"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echono dud li
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-$DEFAULT_CHECKPOINT_INTERVAL}"
MAX_STEPS="${MAX_STEPS:-$DEFAULT_MAX_STEPS}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DEFAULT_CHECKPOINT_DIR}"
LOAD_LATEST="${LOAD_LATEST:-false}"

clean() {
    echo "Cleaning build artifacts..."
    bazel clean --expunge
}

build() {
    echo "Building $TARGET with mandatory AVX2/FMA Optimizations..."
    bazel build ":$TARGET" \
        --compilation_mode=opt \
        --copt=-march=native \
        --copt=-O3 \
        --copt=-flto
}

run() {
    local args=(
        --envs "$NUM_ENVS"
        --checkpoint-interval "$CHECKPOINT_INTERVAL"
        --max-steps "$MAX_STEPS"
        --checkpoint-dir "$CHECKPOINT_DIR"
    )

    if [ "$LOAD_LATEST" = "true" ]; then
        args+=(--load-latest)
    fi

    echo "=========================================="
    echo " COMBAT TRAINING INITIATED"
    echo "=========================================="
    echo " Mode: $MODE"
    echo " Target: $TARGET"
    echo " Parallel Envs: ${NUM_ENVS}"
    echo " Total Robots: $((NUM_ENVS * 2)) (2 per env)"
    echo " Expected Dims: actionDim=56, stateDim=240"
    echo " Expected totalActionDim=$((NUM_ENVS * 2 * 56))"
    echo " Max Steps: $MAX_STEPS"
    echo " Checkpoint Every: $CHECKPOINT_INTERVAL steps"
    echo "=========================================="

    # Use bazel run so it finds the config.json files properly
    bazel run ":$TARGET" --compilation_mode=opt -- "${args[@]}"

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo " TRAINING CRASHED (exit code: $exit_code)"
        echo "=========================================="
        echo "Common causes:"
        echo "  1. NUM_ENVS > 96 (Jolt capacity limit)"
        echo "  2. Observation dimension mismatch (check BuildObservationVector)"
        echo "  3. Memory alignment issue (AVX2 requirements)"
        echo ""
        echo "Try reducing NUM_ENVS or check recent code changes."
        exit $exit_code
    fi
}

usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo "Commands: clean, build, run, all"
}

main() {
    local cmd="${1:-all}"
    shift || true

    case "$cmd" in
        clean) clean ;;
        build) build ;;
        run)   run   ;;
        all)
            clean
            build
            run
            ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"