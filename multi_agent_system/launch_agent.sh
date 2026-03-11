#!/bin/bash
#
# Multi-Agent System - Agent Launcher
#
# Initializes an agent session with:
# - Unique session ID generation
# - Role-based context setup
# - Current project state display
# - Available tasks for role
# - Session registration
#
# Usage: ./launch_agent.sh <role>
# Roles: arch (architect), be (backend), fe (frontend), qa (QA)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║         Multi-Agent AI Collaboration System               ║"
    echo "║                    Agent Launcher                         ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

print_section() {
    echo -e "${BLUE}━━━ $1 ${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

show_usage() {
    echo "Usage: $0 <role>"
    echo ""
    echo "Roles:"
    echo "  arch  - System Architect"
    echo "  be    - Backend Developer"
    echo "  fe    - Frontend Developer"
    echo "  qa    - QA/Test Specialist"
    echo ""
    echo "Examples:"
    echo "  $0 arch    # Launch architect agent"
    echo "  $0 be      # Launch backend agent"
    echo ""
    echo "The script will:"
    echo "  1. Generate a unique session ID"
    echo "  2. Display available tasks for your role"
    echo "  3. Show current project state"
    echo "  4. Register your agent session"
    echo "  5. Provide context for AI agent"
    echo ""
}

# ============================================================================
# Validation
# ============================================================================

if [ $# -lt 1 ]; then
    print_error "Missing role argument"
    echo ""
    show_usage
    exit 1
fi

ROLE="$1"

# Validate role
case "$ROLE" in
    arch|Architect|ARCHITECT|architect)
        ROLE="arch"
        ROLE_NAME="System Architect"
        ;;
    be|backend|Backend|BACKEND)
        ROLE="be"
        ROLE_NAME="Backend Developer"
        ;;
    fe|frontend|Frontend|FRONTEND)
        ROLE="fe"
        ROLE_NAME="Frontend Developer"
        ;;
    qa|QA|qa|Quality|quality|QUALITY)
        ROLE="qa"
        ROLE_NAME="QA/Test Specialist"
        ;;
    *)
        print_error "Invalid role: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac

# ============================================================================
# Check Prerequisites
# ============================================================================

print_header

print_section "Checking Prerequisites"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi
print_success "Python found: $PYTHON_CMD"

# Check agent_tools.py exists
if [ ! -f "agent_tools.py" ]; then
    print_error "agent_tools.py not found in current directory"
    exit 1
fi
print_success "Agent tools module found"

# Check TODO.md exists
if [ ! -f "TODO.md" ]; then
    print_error "TODO.md not found. Is this a valid project directory?"
    exit 1
fi
print_success "Task tracker found"

# Create necessary directories
mkdir -p locks agents state

print_success "Directory structure ready"

# ============================================================================
# Generate Session ID
# ============================================================================

print_section "Generating Session ID"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_ID="agent_${TIMESTAMP}_${ROLE}"

print_success "Session ID: ${SESSION_ID}"
print_info "Role: ${ROLE_NAME}"

# ============================================================================
# Display Project State
# ============================================================================

print_section "Current Project State"

# Get state from current_state.json
if [ -f "state/current_state.json" ]; then
    # Use Python to parse JSON
    TASK_INFO=$($PYTHON_CMD -c "
import json
try:
    with open('state/current_state.json') as f:
        state = json.load(f)
    tc = state.get('task_counts', {})
    print(f\"Unclaimed: {tc.get('unclaimed', 0)}\")
    print(f\"Claimed: {tc.get('claimed', 0)}\")
    print(f\"In Progress: {tc.get('in_progress', 0)}\")
    print(f\"Blocked: {tc.get('blocked', 0)}\")
    print(f\"Completed: {tc.get('completed', 0)}\")
    print(f\"Active Agents: {state.get('active_agents', 0)}\")
except Exception as e:
    print('State file unreadable')
")
    echo "$TASK_INFO"
else
    print_warning "State file not found (first run)"
fi

# ============================================================================
# Display Available Tasks
# ============================================================================

print_section "Available Tasks for ${ROLE_NAME}"

AVAILABLE_TASKS=$($PYTHON_CMD << EOF
import sys
sys.path.insert(0, '.')
from agent_tools import get_available_tasks

tasks = get_available_tasks("$ROLE")
if tasks:
    for task in tasks[:10]:  # Show first 10
        print(f"  • {task}")
    if len(tasks) > 10:
        print(f"  ... and {len(tasks) - 10} more")
else:
    print("  No available tasks for your role")
EOF
)

echo "$AVAILABLE_TASKS"

# ============================================================================
# Display Current Blockers
# ============================================================================

print_section "Current Blockers"

if [ -f "state/blockers.md" ]; then
    # Check for active blockers
    BLOCKER_COUNT=$(grep -c "^## BLOCKER-" state/blockers.md 2>/dev/null || echo "0")
    ACTIVE_COUNT=$(grep -c "Status: Active" state/blockers.md 2>/dev/null || echo "0")
    
    if [ "$ACTIVE_COUNT" -gt 0 ]; then
        print_warning "$ACTIVE_COUNT active blocker(s) found"
        echo ""
        grep -A 5 "^## BLOCKER-" state/blockers.md | head -20
    else
        print_success "No active blockers"
    fi
else
    print_info "Blockers file not initialized"
fi

# ============================================================================
# Register Session
# ============================================================================

print_section "Registering Agent Session"

$PYTHON_CMD << EOF
import sys
sys.path.insert(0, '.')
from agent_tools import register_session, initialize_project

# Initialize if needed
initialize_project()

# Register session
result = register_session("$SESSION_ID", "$ROLE")
if result:
    print("Session registered successfully")
else:
    print("Warning: Session registration failed")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    print_error "Failed to register session"
    exit 1
fi

print_success "Agent session registered"

# ============================================================================
# Display Role Information
# ============================================================================

print_section "Role: ${ROLE_NAME}"

if [ -f "roles/role_${ROLE}.md" ]; then
    # Extract key responsibilities
    echo "Key Responsibilities:"
    grep -E "^### [0-9]+\." roles/role_${ROLE}.md | head -4 | sed 's/### /  /'
    echo ""
else
    print_warning "Role definition file not found"
fi

# ============================================================================
# Generate Agent Context Prompt
# ============================================================================

print_section "Agent Context Prompt"

echo -e "${YELLOW}Copy the following prompt into your Qwen Code instance:${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat << PROMPT
You are an AI agent participating in a multi-agent collaboration system.

## Your Identity
- **Session ID**: ${SESSION_ID}
- **Role**: ${ROLE_NAME}
- **Launch Time**: $(date +"%Y-%m-%d %H:%M:%S")

## Your Responsibilities
As a ${ROLE_NAME}, you are responsible for:
$(grep -E "^- \[" "roles/role_${ROLE}.md" 2>/dev/null | head -5 || echo "- See roles/role_${ROLE}.md for full responsibilities")

## Current Project State
- Review TODO.md for task board
- Check state/current_state.json for metrics
- Read agents/message_board.md for team communication
- Review state/blockers.md for any blocking issues

## Getting Started
1. Claim an available task using agent_tools.py:
   \`\`\`python
   from agent_tools import claim_task, start_task, complete_task
   
   # Claim a task
   claim_task("TASK-ID", "${SESSION_ID}", "${ROLE}")
   
   # Start working
   start_task("TASK-ID", "${SESSION_ID}")
   
   # When done
   complete_task("TASK-ID", "Completion notes", "${SESSION_ID}")
   \`\`\`

2. Communicate with team via message board:
   \`\`\`python
   from agent_tools import send_message
   send_message("${SESSION_ID}", "Starting work on TASK-ID", "INFO")
   \`\`\`

3. Follow the agent protocol in agent_protocol.md

## Files to Know
- \`TODO.md\` - Task tracker
- \`agent_protocol.md\` - Coordination protocol
- \`roles/role_${ROLE}.md\` - Your role definition
- \`agents/message_board.md\` - Team communication
- \`agent_tools.py\` - Your utility functions

## Important Rules
1. Always claim tasks before working on them
2. Release file locks promptly
3. Update task status as you progress
4. Communicate blockers immediately
5. Document decisions and handoffs

Good luck!
PROMPT

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# Quick Reference Card
# ============================================================================

print_section "Quick Reference Card"

cat << 'QUICKREF'
Common Operations:
─────────────────────────────────────────────────────────────────────
Claim a task:     claim_task("TASK-ID", agent_id, role)
Start task:       start_task("TASK-ID", agent_id)
Complete task:    complete_task("TASK-ID", "notes", agent_id)
Block task:       block_task("TASK-ID", "blocked_by", "reason", agent_id)
Send message:     send_message(agent_id, "message", "TYPE")
Get tasks:        get_available_tasks(role)
Read task:        read_task("TASK-ID")

Message Types: INFO, HELP, HANDOFF, BLOCKER, DECISION, COMPLETE
─────────────────────────────────────────────────────────────────────
QUICKREF

# ============================================================================
# Final Instructions
# ============================================================================

print_section "Next Steps"

echo "1. Copy the context prompt above"
echo "2. Paste it into your Qwen Code instance"
echo "3. Review available tasks and claim one"
echo "4. Begin work!"
echo ""
print_info "Session heartbeat will timeout after 5 minutes of inactivity"
print_info "Run './launch_agent.sh $ROLE' again to refresh heartbeat"
echo ""
print_success "Agent launcher complete! Good luck, ${ROLE_NAME}!"
echo ""

# Export session ID for convenience
export AGENT_SESSION_ID="$SESSION_ID"
export AGENT_ROLE="$ROLE"
