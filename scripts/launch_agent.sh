#!/bin/bash
# Quick-start script for autonomous build agent
# Sets up environment and launches the agent

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     JOLTrl Autonomous Build-Test-Debug Agent            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Select mode:"
echo "  1) Watch mode (continuous monitoring)"
echo "  2) Single run (diagnose and propose fixes)"
echo "  3) Interactive (step-by-step debugging)"
echo "  4) Quick build check"
echo ""
read -p "Choose mode [1-4]: " mode

case $mode in
    1)
        echo ""
        echo "Starting watch mode..."
        echo "The agent will monitor for build failures and auto-repair."
        echo "Press Ctrl+C to stop."
        echo ""
        ./watch_build.sh
        ;;
    2)
        echo ""
        echo "Running single diagnostic pass..."
        echo ""
        python3 autonomous_build_agent.py --max-iterations 5
        ;;
    3)
        echo ""
        echo "Starting interactive mode..."
        echo "The agent will propose fixes and wait for approval."
        echo ""
        python3 autonomous_build_agent.py --max-iterations 10 --auto-apply
        ;;
    4)
        echo ""
        echo "Running quick build check..."
        echo ""
        if bazel build //:train --compilation_mode=opt 2>&1 | tee build_status.txt; then
            echo "✓ Build successful!"
            exit 0
        else
            echo "✗ Build failed!"
            echo ""
            read -p "Launch autonomous agent to diagnose? [y/N]: " launch
            if [ "$launch" = "y" ] || [ "$launch" = "Y" ]; then
                python3 autonomous_build_agent.py
            fi
        fi
        ;;
    *)
        echo "Invalid mode selected."
        exit 1
        ;;
esac
