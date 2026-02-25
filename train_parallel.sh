#!/bin/bash
# Parallel Combat Training Script
# Runs vectorized training with 1 visible env + N headless envs
# Checkpoints saved periodically, best policy auto-loaded

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_ENVS=8
DEFAULT_CHECKPOINT_INTERVAL=50000
DEFAULT_MAX_STEPS=10000000
DEFAULT_CHECKPOINT_DIR="checkpoints"

NUM_ENVS="${NUM_ENVS:-$DEFAULT_ENVS}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-$DEFAULT_CHECKPOINT_INTERVAL}"
MAX_STEPS="${MAX_STEPS:-$DEFAULT_MAX_STEPS}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DEFAULT_CHECKPOINT_DIR}"
LOAD_LATEST="${LOAD_LATEST:-true}"

TARGET="train"
BINARY_PATH="bazel-bin/$TARGET"

clean() {
    echo "Cleaning build artifacts..."
    bazel clean --expunge
}

build() {
    echo "Building $TARGET..."
    bazel build ":$TARGET" --compilation_mode=opt
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
    
    echo "Running $TARGET with ${NUM_ENVS} parallel environments..."
    echo "Checkpoint interval: $CHECKPOINT_INTERVAL steps"
    echo "Max steps: $MAX_STEPS"
    echo "Checkpoint dir: $CHECKPOINT_DIR"
    echo "Load latest: $LOAD_LATEST"
    echo ""
    
    if [ -f "$BINARY_PATH" ]; then
        "$BINARY_PATH" "${args[@]}"
    else
        echo "Binary not found, running with bazel..."
        bazel run ":$TARGET" -- "${args[@]}"
    fi
}

usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  clean     Clean build artifacts (expunge)"
    echo "  build     Build the parallel_train target"
    echo "  run       Run training (requires build first)"
    echo "  all       Clean, build, and run (default)"
    echo ""
    echo "Environment Variables:"
    echo "  NUM_ENVS             Number of parallel environments (default: $DEFAULT_ENVS)"
    echo "  CHECKPOINT_INTERVAL  Steps between checkpoints (default: $DEFAULT_CHECKPOINT_INTERVAL)"
    echo "  MAX_STEPS            Maximum training steps (default: $DEFAULT_MAX_STEPS)"
    echo "  CHECKPOINT_DIR       Directory for checkpoints (default: $DEFAULT_CHECKPOINT_DIR)"
    echo "  LOAD_LATEST          Load latest checkpoint on start (default: true)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Clean, build, run with defaults"
    echo "  $0 run                      # Run with existing binary"
    echo "  NUM_ENVS=16 $0 all          # Clean, build, run with 16 parallel envs"
    echo "  LOAD_LATEST=false $0 run    # Start fresh (don't load checkpoint)"
}

main() {
    local cmd="${1:-all}"
    shift || true
    
    case "$cmd" in
        clean)
            clean
            ;;
        build)
            build
            ;;
        run)
            run
            ;;
        all)
            clean
            build
            run
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            echo "Unknown command: $cmd"
            usage
            exit 1
            ;;
    esac
}

main "$@"
