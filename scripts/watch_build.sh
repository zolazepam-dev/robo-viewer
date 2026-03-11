#!/bin/bash
# Watch script for autonomous build monitoring
# Triggers the autonomous agent when build failures are detected

set -e

WATCH_INTERVAL=5  # seconds
LOG_FILE="watch_agent.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log "${BLUE}Starting Build Watch Agent${NC}"
log "Monitoring for build failures every ${WATCH_INTERVAL}s"
log "Press Ctrl+C to stop"

# Initial build to establish baseline
log "${YELLOW}Running initial build check...${NC}"

if bazel build //:train --compilation_mode=opt 2>&1 | tee build_status.txt; then
    log "${GREEN}✓ Initial build successful${NC}"
else
    log "${RED}✗ Initial build failed - launching autonomous agent${NC}"
    python3 scripts/autonomous_build_agent.py
    exit $?
fi

# Main watch loop
while true; do
    sleep $WATCH_INTERVAL
    
    # Check if source files changed
    if find src/ -name "*.cpp" -o -name "*.h" -newer build_status.txt 2>/dev/null | grep -q .; then
        log "${YELLOW}Source changes detected - rebuilding...${NC}"
        
        if bazel build //:train --compilation_mode=opt 2>&1 | tee build_status.txt; then
            log "${GREEN}✓ Build successful${NC}"
            
            # Optionally run tests
            if [ "$RUN_TESTS_ON_SUCCESS" = "true" ]; then
                log "${BLUE}Running tests...${NC}"
                if bazel run //:system_test 2>&1 | tee test_status.txt; then
                    log "${GREEN}✓ All tests passed${NC}"
                else
                    log "${RED}✗ Tests failed - launching agent${NC}"
                    python3 scripts/autonomous_build_agent.py
                fi
            fi
        else
            log "${RED}✗ Build failed - launching autonomous agent${NC}"
            python3 scripts/autonomous_build_agent.py
            
            # After agent runs, check if build is fixed
            if bazel build //:train --compilation_mode=opt 2>&1 | tee build_status.txt; then
                log "${GREEN}✓ Agent successfully repaired build${NC}"
            else
                log "${RED}✗ Agent could not fix build - manual intervention required${NC}"
            fi
        fi
        
        # Update timestamp
        touch build_status.txt
    fi
done
