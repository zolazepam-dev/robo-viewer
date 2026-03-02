#!/bin/bash
# Battery System Test Runner

echo "╔════════════════════════════════════════════╗"
echo "║     BATTERY SYSTEM TEST SUITE             ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Build tests
echo "Building tests..."

echo ""
echo "═══════════════════════════════════════════"
echo "  TEST 1: Standalone Battery Unit Tests"
echo "═══════════════════════════════════════════"
g++ -std=c++17 -O2 src/battery_standalone_test.cpp -o battery_test 2>&1 | tail -3
./battery_test
TEST1_RESULT=$?

echo ""
echo "═══════════════════════════════════════════"
echo "  TEST 2: KOTH Charging Integration"
echo "═══════════════════════════════════════════"
g++ -std=c++17 -O2 src/integration_test.cpp -o integration_test 2>&1 | tail -3
./integration_test
TEST2_RESULT=$?

echo ""
echo "═══════════════════════════════════════════"
echo "  TEST 3: Engine Battery Drain & CoG"
echo "═══════════════════════════════════════════"
g++ -std=c++17 -O2 src/engine_battery_test.cpp -o engine_battery_test 2>&1 | tail -3
./engine_battery_test
TEST3_RESULT=$?

echo ""
echo "═══════════════════════════════════════════"
echo "  SUMMARY"
echo "═══════════════════════════════════════════"

if [ $TEST1_RESULT -eq 0 ] && [ $TEST2_RESULT -eq 0 ] && [ $TEST3_RESULT -eq 0 ]; then
    echo -e "\033[32m✓ ALL TESTS PASSED\033[0m"
    echo ""
    echo "Battery system verified:"
    echo "  ✓ Wireless charging (KOTH-based, 5m range)"
    echo "  ✓ Regenerative braking (30% efficiency)"
    echo "  ✓ Engine thrust drains battery (0.1 J/s per N)"
    echo "  ✓ CoG tracking and mass calculation"
    echo "  ✓ Power requests and thermal limits"
    exit 0
else
    echo -e "\033[31m✗ SOME TESTS FAILED\033[0m"
    [ $TEST1_RESULT -ne 0 ] && echo "  - battery_standalone_test failed"
    [ $TEST2_RESULT -ne 0 ] && echo "  - integration_test failed"
    [ $TEST3_RESULT -ne 0 ] && echo "  - engine_battery_test failed"
    exit 1
fi
