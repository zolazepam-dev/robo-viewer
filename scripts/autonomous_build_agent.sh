#!/bin/bash
# Autonomous Build-Test-Debug Loop Agent
# This script continuously monitors build failures, diagnoses root causes,
# applies targeted fixes, and validates through test execution.

set -e

# Configuration
MAX_ITERATIONS=5
LOG_FILE="build_agent_log.txt"
ERROR_PATTERNS_FILE="error_patterns.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize log
echo "=== Autonomous Build Agent Started: $(date) ===" > "$LOG_FILE"

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log "${BLUE}Starting Autonomous Build-Test-Debug Loop${NC}"
log "Max iterations: $MAX_ITERATIONS"

# Track iteration count
iteration=0
last_error=""

while [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))
    log "\n${YELLOW}=== Iteration $iteration/$MAX_ITERATIONS ===${NC}"
    
    # Step 1: Attempt build
    log "${BLUE}Attempting build...${NC}"
    
    # Run build and capture output
    if bazel build //:train \
        --compilation_mode=opt \
        --copt=-march=native \
        --copt=-O3 \
        --copt=-flto \
        --copt=-ffast-math \
        2>&1 | tee build_output.txt; then
        log "${GREEN}✓ Build succeeded!${NC}"
        
        # Step 2: Run tests if build succeeds
        log "${BLUE}Running system tests...${NC}"
        if bazel run //:system_test 2>&1 | tee test_output.txt; then
            log "${GREEN}✓ All tests passed!${NC}"
            log "${GREEN}=== Autonomous Agent Completed Successfully ===${NC}"
            exit 0
        else
            log "${RED}✗ Tests failed${NC}"
            # Extract test failures
            grep -A 10 "FAILED\|Error\|failed" test_output.txt > test_failures.txt || true
            last_error=$(cat test_failures.txt)
        fi
    else
        log "${RED}✗ Build failed${NC}"
        # Extract compiler errors
        grep -A 5 "error:\|Error\|failed" build_output.txt > build_errors.txt || true
        last_error=$(cat build_errors.txt)
    fi
    
    # Step 3: Diagnose root cause
    log "${BLUE}Diagnosing root cause...${NC}"
    
    # Analyze error patterns
    diagnosis=""
    
    # Check for common error patterns
    if echo "$last_error" | grep -q "_mm256_tanh_ps"; then
        diagnosis="AVX2_TANH_ERROR: _mm256_tanh_ps is non-standard. Need to implement custom tanh using AVX2 intrinsics."
        log "${YELLOW}Detected: $diagnosis${NC}"
        
        # Search for the problematic code
        log "${BLUE}Locating problematic code...${NC}"
        grep -rn "_mm256_tanh_ps" src/ || true
        
        # Propose fix
        log "${BLUE}Proposing fix: Replace _mm256_tanh_ps with custom implementation${NC}"
        
    elif echo "$last_error" | grep -q "Jolt/Jolt.h"; then
        diagnosis="JOLT_INCLUDE_ERROR: Jolt headers not included first. Must add #include <Jolt/Jolt.h> as first include."
        log "${YELLOW}Detected: $diagnosis${NC}"
        
    elif echo "$last_error" | grep -q "undefined reference"; then
        diagnosis="LINKER_ERROR: Missing symbol definitions. Check BUILD file dependencies."
        log "${YELLOW}Detected: $diagnosis${NC}"
        
    elif echo "$last_error" | grep -q "no matching function"; then
        diagnosis="SIGNATURE_MISMATCH: Function signature doesn't match declaration."
        log "${YELLOW}Detected: $diagnosis${NC}"
        
    elif echo "$last_error" | grep -q "use of undeclared"; then
        diagnosis="UNDECLARED_IDENTIFIER: Missing include or forward declaration."
        log "${YELLOW}Detected: $diagnosis${NC}"
        
    else
        diagnosis="UNKNOWN_ERROR: Manual review required."
        log "${RED}Detected: $diagnosis${NC}"
        echo "$last_error" >> "$LOG_FILE"
    fi
    
    # Step 4: Apply fix (if pattern matched)
    if [ "$diagnosis" != "UNKNOWN_ERROR: Manual review required." ]; then
        log "${BLUE}Attempting automated fix...${NC}"
        # Here we would call the AI agent to apply fixes
        # For now, log the diagnosis for manual intervention
        log "${YELLOW}Fix requires manual intervention. Diagnosis: $diagnosis${NC}"
    fi
    
    # Check if we should continue
    if [ $iteration -ge $MAX_ITERATIONS ]; then
        log "\n${RED}=== Max iterations reached. Manual intervention required. ===${NC}"
        log "Review $LOG_FILE for detailed error history."
        exit 1
    fi
    
    log "${BLUE}Waiting for manual fix application before retry...${NC}"
    read -p "Press Enter after applying fixes to continue, or Ctrl+C to abort..." -r
done

log "${GREEN}=== Autonomous Agent Session Complete ===${NC}"
