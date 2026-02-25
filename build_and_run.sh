#!/bin/bash

# Robo-Viewer Build and Run Script
# This script provides a clean build and launch interface for the combat training viewer

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
${GREEN}Robo-Viewer Build and Run Script${NC}

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -c, --clean         Clean build (removes all build artifacts)
    -b, --build-only    Build only, don't run
    -r, --run-only      Run only, don't build
    -v, --viewer        Run regular viewer instead of combat trainer
    --verbose           Verbose build output
    --debug             Build with debug symbols
    --no-optimizations  Build without optimizations

EXAMPLES:
    $0                  # Build and run combat trainer
    $0 -c               # Clean build and run
    $0 -v               # Build and run regular viewer
    $0 -b               # Build only, don't run
    $0 --clean --debug  # Clean debug build and run

EOF
}

# Default options
CLEAN_BUILD=false
BUILD_ONLY=false
RUN_ONLY=false
USE_VIEWER=false
VERBOSE=false
DEBUG=false
NO_OPTIMIZATIONS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -b|--build-only)
            BUILD_ONLY=true
            shift
            ;;
        -r|--run-only)
            RUN_ONLY=true
            shift
            ;;
        -v|--viewer)
            USE_VIEWER=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --no-optimizations)
            NO_OPTIMIZATIONS=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if bazel is installed
if ! command -v bazel &> /dev/null; then
    print_error "Bazel is not installed or not in PATH"
    echo "Please install Bazel: https://bazel.build/install"
    exit 1
fi

# Set target based on viewer type
if [ "$USE_VIEWER" = true ]; then
    TARGET="viewer"
    BINARY_PATH="bazel-bin/viewer"
else
    TARGET="combat_train"
    BINARY_PATH="bazel-bin/combat_train"
fi

print_info "Target: //:$TARGET"

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_info "Cleaning build artifacts..."
    bazel clean --expunge
    print_success "Clean complete"
fi

# Build if not run-only
if [ "$RUN_ONLY" = false ]; then
    print_info "Building $TARGET..."

    # Build command
    BUILD_CMD="bazel build //:$TARGET"

    # Add build flags
    BUILD_FLAGS=""

    if [ "$VERBOSE" = true ]; then
        BUILD_FLAGS="$BUILD_FLAGS --verbose_failures"
    fi

    if [ "$DEBUG" = true ]; then
        BUILD_FLAGS="$BUILD_FLAGS --compilation_mode=dbg"
        print_info "Building with debug symbols"
    elif [ "$NO_OPTIMIZATIONS" = true ]; then
        BUILD_FLAGS="$BUILD_FLAGS --compilation_mode=fastbuild"
        print_info "Building without optimizations"
    else
        BUILD_FLAGS="$BUILD_FLAGS --compilation_mode=opt"
        print_info "Building with optimizations"
    fi

    # Execute build
    if eval "$BUILD_CMD $BUILD_FLAGS"; then
        print_success "Build complete"
    else
        print_error "Build failed"
        exit 1
    fi
fi

# Check if binary exists
if [ ! -f "$BINARY_PATH" ]; then
    print_error "Binary not found: $BINARY_PATH"
    print_info "Try building first: $0 -b"
    exit 1
fi

# Run if not build-only
if [ "$BUILD_ONLY" = false ]; then
    print_info "Launching $TARGET..."
    echo ""
    echo "=========================================="
    echo "  Press CTRL+C to stop"
    echo "  Press ESC in window to exit"
    echo "=========================================="
    echo ""

    # Run the binary
    "./$BINARY_PATH"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        print_success "Exited successfully"
    elif [ $EXIT_CODE -eq 130 ]; then
        print_info "Interrupted by user (CTRL+C)"
    else
        print_error "Exited with code: $EXIT_CODE"

        # Check for common issues
        if [ $EXIT_CODE -eq 139 ]; then
            print_warning "Segmentation fault detected"
            print_info "Try running with: ulimit -c unlimited && $0"
            print_info "This will generate a core dump for debugging"
        fi

        exit $EXIT_CODE
    fi
fi

print_success "Done!"
