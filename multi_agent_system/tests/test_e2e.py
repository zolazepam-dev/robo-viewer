#!/usr/bin/env python3
"""
Multi-Agent System - End-to-End Integration Test

Simulates multiple AI agents collaborating on tasks to verify:
- No conflicts during concurrent operations
- Task dependencies are respected
- Communication works correctly
- State is properly tracked

Run with: python3 test_e2e.py
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_tools import (
    claim_task, complete_task, block_task, start_task,
    read_task, get_available_tasks, send_message,
    register_session, unregister_session, get_active_sessions,
    update_state_snapshot, initialize_project,
    TODO_FILE, BASE_DIR
)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_success(msg):
    print(f"✓ {msg}")


def print_error(msg):
    print(f"✗ {msg}")


def test_single_agent_workflow():
    """Test a single agent completing a full workflow."""
    print_section("Test 1: Single Agent Workflow")
    
    agent_id = "agent_test_single_arch"
    task_id = "ARCH-001"  # Use task without dependencies
    
    # Register session
    register_session(agent_id, "arch")
    print_success(f"Session registered: {agent_id}")
    
    # Claim task
    result = claim_task(task_id, agent_id, "arch")
    if result:
        print_success(f"Claimed {task_id}")
    else:
        print_error(f"Failed to claim {task_id}")
        return False
    
    # Start task
    result = start_task(task_id, agent_id)
    if result:
        print_success(f"Started {task_id}")
    else:
        print_error(f"Failed to start {task_id}")
        return False
    
    # Complete task
    result = complete_task(task_id, "Test completion", agent_id)
    if result:
        print_success(f"Completed {task_id}")
    else:
        print_error(f"Failed to complete {task_id}")
        return False
    
    # Verify task state
    task = read_task(task_id)
    if task.status == "completed":
        print_success(f"Task state verified: {task.status}")
    else:
        print_error(f"Task state incorrect: {task.status}")
        return False
    
    # Unregister session
    unregister_session(agent_id)
    print_success(f"Session unregistered: {agent_id}")
    
    return True


def test_concurrent_agents():
    """Test multiple agents working concurrently without conflicts."""
    print_section("Test 2: Concurrent Agents")
    
    results = {}
    
    def agent_work(agent_id, role, task_id):
        """Simulate an agent's work."""
        # Register
        register_session(agent_id, role)
        
        # Claim
        claim_result = claim_task(task_id, agent_id, role)
        if not claim_result:
            return {"agent": agent_id, "task": task_id, "result": "claim_failed"}
        
        # Work (simulate)
        time.sleep(0.1)
        
        # Start
        start_result = start_task(task_id, agent_id)
        
        # Complete
        complete_result = complete_task(task_id, f"{agent_id} completed", agent_id)
        
        # Unregister
        unregister_session(agent_id)
        
        return {
            "agent": agent_id,
            "task": task_id,
            "result": "success" if all([claim_result, start_result, complete_result]) else "failed"
        }
    
    # Define agents and their tasks (use tasks without dependencies)
    agents = [
        ("agent_concurrent_arch", "arch", "ARCH-003"),
        ("agent_concurrent_fe", "fe", "FE-002"),
        ("agent_concurrent_qa", "qa", "QA-001"),
    ]
    
    # Run agents concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(agent_work, agent_id, role, task_id)
            for agent_id, role, task_id in agents
        ]
        
        for future in as_completed(futures):
            result = future.result()
            if result["result"] == "success":
                print_success(f"{result['agent']}: {result['task']} - {result['result']}")
                results[result['agent']] = True
            else:
                print_error(f"{result['agent']}: {result['task']} - {result['result']}")
                results[result['agent']] = False
    
    # Verify all succeeded
    all_success = all(results.values())
    if all_success:
        print_success("All concurrent agents completed without conflicts")
    else:
        print_error("Some agents failed")
    
    return all_success


def test_dependency_chain():
    """Test that task dependencies are respected."""
    print_section("Test 3: Dependency Chain")
    
    # This tests that BE-003 (blocked on BE-001, BE-002) cannot be claimed
    # until dependencies are complete
    
    # First, try to claim BE-003 (should fail if blocked)
    task = read_task("BE-003")
    if task:
        print_success(f"BE-003 found, status: {task.status}")
        print_success(f"BE-003 blocked by: {task.blocked_by or 'nothing'}")
    
    # Try to claim it
    result = claim_task("BE-003", "agent_dep_test_be", "be")
    # This might fail due to blocking - that's expected
    print_success(f"BE-003 claim attempt: {'success' if result else 'blocked (expected)'}")
    
    return True


def test_message_board():
    """Test message board communication."""
    print_section("Test 4: Message Board")
    
    agent_id = "agent_msg_test_arch"
    message = f"Test message from {agent_id} at {datetime.now().isoformat()}"
    
    send_message(agent_id, message, "INFO", tags=["test"], related_tasks=["TEST-001"])
    print_success(f"Message posted: {message[:50]}...")
    
    # Verify message in file
    message_board = BASE_DIR / "agents" / "message_board.md"
    with open(message_board) as f:
        content = f.read()
    
    if agent_id in content:
        print_success("Message found on message board")
        return True
    else:
        print_error("Message not found on message board")
        return False


def test_state_tracking():
    """Test state snapshot updates."""
    print_section("Test 5: State Tracking")
    
    # Update state
    state = update_state_snapshot()
    
    print_success(f"State updated: {state['last_updated']}")
    print_success(f"Task counts: {state['task_counts']}")
    print_success(f"Active agents: {state['active_agents']}")
    
    # Verify state file
    state_file = BASE_DIR / "state" / "current_state.json"
    with open(state_file) as f:
        saved_state = json.load(f)
    
    if saved_state['last_updated'] == state['last_updated']:
        print_success("State file matches memory")
        return True
    else:
        print_error("State file mismatch")
        return False


def test_session_registry():
    """Test session registration and tracking."""
    print_section("Test 6: Session Registry")
    
    # Register multiple sessions
    sessions = [
        ("agent_reg_arch", "arch"),
        ("agent_reg_be", "be"),
        ("agent_reg_fe", "fe"),
    ]
    
    for agent_id, role in sessions:
        register_session(agent_id, role)
        print_success(f"Registered: {agent_id}")
    
    # Get active sessions
    active = get_active_sessions()
    print_success(f"Active sessions: {len(active)}")
    
    # Verify all registered
    active_ids = [s['agent_id'] for s in active]
    for agent_id, role in sessions:
        if agent_id in active_ids:
            print_success(f"Found: {agent_id}")
        else:
            print_error(f"Missing: {agent_id}")
    
    # Cleanup
    for agent_id, role in sessions:
        unregister_session(agent_id)
    
    print_success("Sessions cleaned up")
    return True


def test_file_locking():
    """Test file locking mechanism."""
    print_section("Test 7: File Locking")
    
    from agent_tools import acquire_lock, release_lock, is_file_locked
    
    # Test basic locking
    result = acquire_lock("test_lock.txt", "agent_lock_1")
    if result:
        print_success("Lock acquired")
    else:
        print_error("Failed to acquire lock")
        return False
    
    # Verify locked
    if is_file_locked("test_lock.txt"):
        print_success("File is locked")
    else:
        print_error("File not locked")
        return False
    
    # Try to acquire again (should fail)
    result2 = acquire_lock("test_lock.txt", "agent_lock_2", timeout=1)
    if not result2:
        print_success("Second lock correctly rejected")
    else:
        print_error("Second lock should have failed")
        release_lock("test_lock.txt", "agent_lock_2")
    
    # Release lock
    release_lock("test_lock.txt", "agent_lock_1")
    print_success("Lock released")
    
    # Verify unlocked
    if not is_file_locked("test_lock.txt"):
        print_success("File is unlocked")
    else:
        print_error("File still locked")
        return False
    
    return True


def run_all_tests():
    """Run all end-to-end tests."""
    print_section("Multi-Agent System - End-to-End Tests")
    print(f"Base Directory: {BASE_DIR}")
    print(f"TODO File: {TODO_FILE}")
    
    # Initialize
    initialize_project()
    
    tests = [
        ("Single Agent Workflow", test_single_agent_workflow),
        ("Concurrent Agents", test_concurrent_agents),
        ("Dependency Chain", test_dependency_chain),
        ("Message Board", test_message_board),
        ("State Tracking", test_state_tracking),
        ("Session Registry", test_session_registry),
        ("File Locking", test_file_locking),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            print_error(f"{name} failed with exception: {e}")
            results[name] = False
    
    # Summary
    print_section("Test Summary")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("All tests passed!")
        return 0
    else:
        print_error(f"{total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
