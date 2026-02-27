#!/bin/bash

# Create a test log file
TEST_LOG=$(mktemp /tmp/microboard_test_log.XXXXXX.csv)
cat > "$TEST_LOG" <<EOF
Step,Tag,Value
1,Loss,1.5
2,Loss,0.9
3,Loss,0.4
4,Accuracy,0.6
5,Accuracy,0.75
6,Accuracy,0.85
EOF

echo "=== MicroBoard Test with File Input ==="
echo "Test log file created: $TEST_LOG"
echo
echo "=== Running MicroBoard ==="
bazel run //:micro_board -- "$TEST_LOG"

# Clean up
rm -f "$TEST_LOG"