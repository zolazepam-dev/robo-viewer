#!/bin/bash

echo "=== MicroBoard Test ==="
echo "Building..."
bazel build //:micro_board

echo ""
echo "=== Running MicroBoard with sample data ==="
echo -e "Step,Tag,Value
1,Loss,1.5
2,Loss,0.9
3,Loss,0.4
4,Accuracy,0.6
5,Accuracy,0.75
6,Accuracy,0.85" | bazel-bin/micro_board