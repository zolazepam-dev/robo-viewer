#!/bin/bash

echo "=== Testing MicroBoard GUI ==="
echo "Creating test data..."

# Create test data file
TEST_DATA=$(mktemp /tmp/microboard_test.XXXXXX.csv)
cat > "$TEST_DATA" <<EOF
Step,Tag,Value
1,Loss,1.5
2,Loss,0.9
3,Loss,0.4
4,Loss,0.35
5,Loss,0.28
6,Loss,0.22
10,Accuracy,0.6
20,Accuracy,0.75
30,Accuracy,0.82
40,Accuracy,0.87
50,Accuracy,0.91
60,Accuracy,0.94
EOF

echo ""
echo "Running micro_board_gui with test data..."
echo "A GUI window should open showing training graphs."
echo "Close the window when done viewing."
echo ""

./micro_board_gui "$TEST_DATA"

# Clean up
rm -f "$TEST_DATA"

echo ""
echo "Test complete!"