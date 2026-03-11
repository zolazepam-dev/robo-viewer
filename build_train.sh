#!/bin/bash
# JOLTrl Build Script with Linker Fix
# Usage: ./build_train.sh [clean]

set -e

cd /media/cammyz/EverythingHere/robo-viewer

echo "╔════════════════════════════════════════════════════════╗"
echo "║  JOLTrl Build Script                                   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo

if [[ "$1" == "clean" ]]; then
    echo "Cleaning build cache..."
    bazel clean
    echo
fi

echo "Building training binary with gold linker..."
echo "Note: Using -fuse-ld=gold to fix linking issues"
echo

bazel build //:train \
    --compilation_mode=opt \
    --linkopt="-fuse-ld=gold"

echo
echo "╔════════════════════════════════════════════════════════╗"
echo "║  Build Complete!                                       ║"
echo "╚════════════════════════════════════════════════════════╝"
echo
echo "Binary location: bazel-bin/train"
echo
echo "Usage examples:"
echo "  ./bazel-bin/train --envs 128 --steps 10000"
echo "  ./bazel-bin/train --envs 256 --steps 10000 --robot robots/bouncy_orbiter.json"
echo
echo "Or use the launcher scripts:"
echo "  ./train_bouncy_orbiter.sh 128 10000"
echo
