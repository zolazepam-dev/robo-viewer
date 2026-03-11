#!/bin/bash
# Launch training with Bouncy Orbiter robot
# Usage: ./train_bouncy_orbiter.sh [envs] [steps]

set -e

cd /media/cammyz/EverythingHere/robo-viewer

# Configuration
ROBOT="bouncy_orbiter"
ENVS=${1:-128}
STEPS=${2:-10000}

echo "╔════════════════════════════════════════════════════════╗"
echo "║  JOLTrl Training - Bouncy Orbiter                      ║"
echo "╚════════════════════════════════════════════════════════╝"
echo
echo "Robot: $ROBOT"
echo "Environments: $ENVS"
echo "Steps: $STEPS"
echo

# Check if binary exists, build if needed
if [[ ! -f "./bazel-bin/train" ]]; then
    echo "Building training binary..."
    bazel build //:train --compilation_mode=opt --linkopt="-fuse-ld=gold"
fi

# Run training with bouncy orbiter robot
./bazel-bin/train --envs $ENVS --steps $STEPS --robot "robots/bouncy_orbiter.json"

echo
echo "Training complete!"
