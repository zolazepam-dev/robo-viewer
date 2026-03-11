#!/bin/bash
# Multi-Agent System Demo Script
# This script demonstrates how multiple AI agents collaborate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     Multi-Agent AI Collaboration System Demo            ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to pause between sections
pause() {
    echo -e "${YELLOW}Press Enter to continue...${NC}"
    read -r
    echo ""
}

# Section 1: Show current state
echo -e "${BLUE}=== Step 1: Current Project State ===${NC}"
echo ""
echo "Checking TODO.md for available tasks..."
echo ""
cat TODO.md | head -25
pause

# Section 2: Show active agents
echo -e "${BLUE}=== Step 2: Active Agent Sessions ===${NC}"
echo ""
if [ -f "agents/session_registry.json" ]; then
    echo "Active agents:"
    cat agents/session_registry.json | python3 -m json.tool 2>/dev/null || cat agents/session_registry.json
else
    echo "No active agents yet"
fi
pause

# Section 3: Demonstrate agent launch
echo -e "${BLUE}=== Step 3: Launch Architect Agent ===${NC}"
echo ""
echo "Running: ./launch_agent.sh arch"
echo ""
echo -e "${YELLOW}This would generate a session ID and context prompt${NC}"
echo -e "${YELLOW}for the Architect agent.${NC}"
echo ""
echo "Example output:"
echo "  Session ID: agent_20260309_143022_arch"
echo "  Role: System Architect"
echo "  Available tasks: ARCH-001, ARCH-002, ARCH-003, ARCH-004"
echo ""
pause

# Section 4: Show agent tools
echo -e "${BLUE}=== Step 4: Agent Tools Available ===${NC}"
echo ""
echo "Agents can use these Python functions:"
echo ""
echo -e "${GREEN}from agent_tools import:${NC}"
echo "  - claim_task(task_id, agent_id, role)"
echo "  - start_task(task_id, agent_id)"
echo "  - complete_task(task_id, notes, agent_id)"
echo "  - block_task(task_id, blocked_by, reason, agent_id)"
echo "  - send_message(agent_id, message, msg_type, tags)"
echo "  - read_messages(agent_id, tags)"
echo "  - acquire_lock(filepath, agent_id)"
echo "  - release_lock(filepath, agent_id)"
echo "  - get_available_tasks(role)"
echo "  - get_task_dependencies(task_id)"
echo ""
pause

# Section 5: Demonstrate task claiming
echo -e "${BLUE}=== Step 5: Task Claiming Demo ===${NC}"
echo ""
echo "Let's simulate an agent claiming a task..."
echo ""

# Create a test agent session
TEST_AGENT="demo_agent_$(date +%Y%m%d_%H%M%S)_arch"
echo "Test Agent ID: ${CYAN}$TEST_AGENT${NC}"
echo ""

# Use Python to claim a task
python3 << EOF
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from agent_tools import claim_task, get_available_tasks

# Show available tasks
print("Available tasks for architect:")
tasks = get_available_tasks("arch")
for task in tasks[:5]:
    print(f"  - {task}")

# Claim first available task
if tasks:
    task_id = tasks[0]
    print(f"\nClaiming {task_id}...")
    success = claim_task(task_id, "$TEST_AGENT", "arch")
    if success:
        print(f"✓ Successfully claimed {task_id}")
    else:
        print(f"✗ Failed to claim {task_id} (already claimed?)")
EOF

echo ""
pause

# Section 6: Show file locking
echo -e "${BLUE}=== Step 6: File Locking Demo ===${NC}"
echo ""
echo "Demonstrating file lock acquisition..."
echo ""

python3 << EOF
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from agent_tools import acquire_lock, release_lock, check_lock

test_file = "example_project/README.md"

print(f"Test file: {test_file}")
print("")

# Check current lock status
locked = check_lock(test_file)
if locked:
    print(f"File is locked by: {locked['locked_by']}")
else:
    print(f"File is unlocked")

# Try to acquire lock
print(f"\nAcquiring lock...")
if acquire_lock(test_file, "$TEST_AGENT"):
    print(f"✓ Lock acquired!")
    
    # Try to acquire again (should fail)
    print(f"\nTrying to acquire again (should fail)...")
    if acquire_lock(test_file, "another_agent"):
        print("✗ Unexpectedly acquired lock!")
    else:
        print("✓ Lock correctly rejected - file already locked")
    
    # Release lock
    print(f"\nReleasing lock...")
    release_lock(test_file, "$TEST_AGENT")
    print("✓ Lock released")
else:
    print(f"✗ Failed to acquire lock (already locked?)")
EOF

echo ""
pause

# Section 7: Show message board
echo -e "${BLUE}=== Step 7: Message Board Demo ===${NC}"
echo ""
echo "Agents communicate through the message board..."
echo ""

python3 << EOF
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from agent_tools import send_message, read_messages

# Send a message
print("Sending message to team...")
send_message(
    "$TEST_AGENT",
    "ARCH-001 complete! Architecture docs updated in docs/architecture.md",
    "HANDOFF",
    tags=["architecture", "handoff", "backend"],
    related_tasks=["ARCH-001"]
)
print("✓ Message sent")

# Read messages
print("\nReading messages with tag 'architecture'...")
messages = read_messages(tags=["architecture"])
print(f"Found {len(messages)} messages")
for msg in messages[-3:]:  # Show last 3 messages
    print(f"\n  From: {msg['agent_id']}")
    print(f"  Type: {msg['type']}")
    print(f"  Message: {msg['message'][:100]}...")
EOF

echo ""
pause

# Section 8: Show current state
echo -e "${BLUE}=== Step 8: Updated Project State ===${NC}"
echo ""
echo "Current TODO.md status:"
echo ""
cat TODO.md | head -25
echo ""

# Section 9: Cleanup
echo -e "${BLUE}=== Demo Cleanup ===${NC}"
echo ""
echo "Cleaning up demo agent session..."
python3 << EOF
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from agent_tools import release_all_locks, unregister_session

# Release any locks held by demo agent
print("Releasing locks...")
release_all_locks("$TEST_AGENT")

# Unregister session
print("Unregistering session...")
unregister_session("$TEST_AGENT")

print("✓ Cleanup complete")
EOF

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Demo Complete!                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "To run a real multi-agent session:"
echo ""
echo "  1. Open multiple Qwen Code windows"
echo "  2. In each window run: ${CYAN}./launch_agent.sh <role>${NC}"
echo "  3. Agents will coordinate through TODO.md and message board"
echo ""
echo "See ${CYAN}PRACTICAL_GUIDE.md${NC} for detailed instructions."
echo ""
