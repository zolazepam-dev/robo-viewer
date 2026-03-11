#!/usr/bin/env python3
"""
Multi-Agent System - Agent Tools Tests

Comprehensive test suite for agent_tools.py covering:
- File locking mechanisms
- Task management operations
- Session registry
- Message board
- State management
- Concurrent access patterns

Run with: pytest tests/test_agent_tools.py -v
"""

import pytest
import os
import json
import time
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_tools import (
    # File locking
    acquire_lock, release_lock, is_file_locked, get_lock_holder,
    acquire_file_lock,
    
    # Task management
    claim_task, complete_task, block_task, start_task, unblock_task,
    read_task, get_available_tasks, get_task_dependencies, get_dependent_tasks,
    
    # Session management
    register_session, unregister_session, update_heartbeat, get_active_sessions,
    
    # Communication
    send_message, log_handoff,
    
    # State
    update_state_snapshot,
    
    # Utilities
    generate_session_id, validate_session_id, get_role_from_session_id,
    initialize_project,
    
    # Configuration
    TODO_FILE, SESSION_REGISTRY, MESSAGE_BOARD, LOCKS_DIR,
    BASE_DIR, AGENTS_DIR, STATE_DIR
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def test_dir(tmp_path):
    """Create a temporary test directory structure."""
    # Create test structure
    test_base = tmp_path / "test_multi_agent"
    test_base.mkdir()
    
    (test_base / "locks").mkdir()
    (test_base / "agents").mkdir()
    (test_base / "state").mkdir()
    (test_base / "roles").mkdir()
    
    # Create minimal TODO.md
    todo_content = """# Multi-Agent System - Task Tracker

**Last Updated**: 2026-03-09

---

## Task Summary

| Status | Count |
|--------|-------|
| Unclaimed | 3 |
| Claimed | 0 |
| In Progress | 0 |
| Blocked | 0 |
| Completed | 0 |
| **Total** | **3** |

---

## Tasks

| Task ID | Role | Description | Status | Claimed By | Blocked By | Priority | Acceptance Criteria |
|---------|------|-------------|--------|------------|------------|----------|---------------------|
| ARCH-001 | Architect | Design system architecture | unclaimed | | | P0 | Architecture diagram in docs/ |
| BE-001 | Backend | Implement core API | unclaimed | | ARCH-001 | P0 | API endpoints functional |
| QA-001 | QA | Write unit tests | unclaimed | | BE-001 | P0 | Coverage >80% |

---
"""
    
    (test_base / "TODO.md").write_text(todo_content)
    
    # Create empty message board
    (test_base / "agents" / "message_board.md").write_text("# Agent Message Board\n\n---\n\n")
    
    # Create empty handoff log
    (test_base / "agents" / "handoff_log.md").write_text("# Handoff Log\n\n---\n\n")
    
    # Create session registry
    (test_base / "agents" / "session_registry.json").write_text(
        json.dumps({"sessions": [], "last_updated": datetime.now().isoformat()})
    )
    
    # Create state file
    (test_base / "state" / "current_state.json").write_text(
        json.dumps({
            "last_updated": datetime.now().isoformat(),
            "task_counts": {"unclaimed": 3, "claimed": 0, "in_progress": 0, "blocked": 0, "completed": 0},
            "active_agents": 0,
            "current_blockers": 0
        })
    )
    
    # Create blockers file
    (test_base / "state" / "blockers.md").write_text("# Current Blockers\n\n---\n\n*No active blockers.*\n")
    
    return test_base


@pytest.fixture
def mock_paths(test_dir, monkeypatch):
    """Mock the path constants to use test directory."""
    monkeypatch.setattr('agent_tools.BASE_DIR', test_dir)
    monkeypatch.setattr('agent_tools.LOCKS_DIR', test_dir / "locks")
    monkeypatch.setattr('agent_tools.AGENTS_DIR', test_dir / "agents")
    monkeypatch.setattr('agent_tools.STATE_DIR', test_dir / "state")
    monkeypatch.setattr('agent_tools.TODO_FILE', test_dir / "TODO.md")
    monkeypatch.setattr('agent_tools.SESSION_REGISTRY', test_dir / "agents" / "session_registry.json")
    monkeypatch.setattr('agent_tools.MESSAGE_BOARD', test_dir / "agents" / "message_board.md")
    monkeypatch.setattr('agent_tools.HANDOFF_LOG', test_dir / "agents" / "handoff_log.md")
    monkeypatch.setattr('agent_tools.CURRENT_STATE', test_dir / "state" / "current_state.json")
    monkeypatch.setattr('agent_tools.BLOCKERS_FILE', test_dir / "state" / "blockers.md")
    return test_dir


@pytest.fixture
def clean_locks(mock_paths):
    """Ensure locks directory is clean before and after test."""
    locks_dir = mock_paths / "locks"
    locks_dir.mkdir(exist_ok=True)
    
    # Clean before
    for f in locks_dir.iterdir():
        f.unlink()
    
    yield
    
    # Clean after
    for f in locks_dir.iterdir():
        f.unlink()


# ============================================================================
# File Locking Tests
# ============================================================================

class TestFileLocking:
    """Test file locking mechanisms."""
    
    def test_acquire_lock_success(self, clean_locks):
        """Test successful lock acquisition."""
        result = acquire_lock("test_file.txt", "agent_test_001")
        assert result is True
        assert (clean_locks / "test_file.txt.lock").exists()
    
    def test_acquire_lock_twice_same_agent(self, clean_locks):
        """Test same agent can acquire lock twice."""
        result1 = acquire_lock("test_file.txt", "agent_test_001")
        result2 = acquire_lock("test_file.txt", "agent_test_001")
        assert result1 is True
        # Second acquire should fail (lock exists)
        assert result2 is False
    
    def test_acquire_lock_different_agent_timeout(self, clean_locks):
        """Test different agent times out waiting for lock."""
        # First agent acquires lock
        result1 = acquire_lock("test_file.txt", "agent_test_001")
        assert result1 is True
        
        # Second agent should timeout
        start = time.time()
        result2 = acquire_lock("test_file.txt", "agent_test_002", timeout=1)
        elapsed = time.time() - start
        
        assert result2 is False
        assert elapsed >= 1.0  # Should have waited for timeout
    
    def test_release_lock_success(self, clean_locks):
        """Test successful lock release."""
        acquire_lock("test_file.txt", "agent_test_001")
        result = release_lock("test_file.txt", "agent_test_001")
        assert result is True
        assert not (clean_locks / "test_file.txt.lock").exists()
    
    def test_release_lock_wrong_agent(self, clean_locks):
        """Test agent cannot release another agent's lock."""
        acquire_lock("test_file.txt", "agent_test_001")
        result = release_lock("test_file.txt", "agent_test_002")
        assert result is False
        # Lock should still exist
        assert (clean_locks / "test_file.txt.lock").exists()
    
    def test_is_file_locked(self, clean_locks):
        """Test file lock status check."""
        assert is_file_locked("test_file.txt") is False
        
        acquire_lock("test_file.txt", "agent_test_001")
        assert is_file_locked("test_file.txt") is True
        
        release_lock("test_file.txt", "agent_test_001")
        assert is_file_locked("test_file.txt") is False
    
    def test_get_lock_holder(self, clean_locks):
        """Test getting lock holder information."""
        assert get_lock_holder("test_file.txt") is None
        
        acquire_lock("test_file.txt", "agent_test_001")
        holder = get_lock_holder("test_file.txt")
        assert holder == "agent_test_001"
    
    def test_stale_lock_detection(self, clean_locks):
        """Test stale lock is detected and broken."""
        # Create a stale lock manually
        lock_file = clean_locks / "stale_test.lock"
        stale_time = "2020-01-01T00:00:00"
        lock_data = {
            "locked_by": "old_agent",
            "locked_at": stale_time,
            "operation": "edit",
            "pid": 12345
        }
        lock_file.write_text(json.dumps(lock_data))
        
        # Should be able to acquire (stale lock broken)
        result = acquire_lock("stale_test.lock", "new_agent", timeout=1)
        assert result is True
    
    def test_context_manager_lock(self, clean_locks):
        """Test file lock context manager."""
        with acquire_file_lock("test_file.txt", "agent_test_001"):
            assert is_file_locked("test_file.txt") is True
        
        # Lock should be released
        assert is_file_locked("test_file.txt") is False
    
    def test_context_manager_lock_timeout(self, clean_locks):
        """Test context manager raises on timeout."""
        acquire_lock("test_file.txt", "agent_test_001")
        
        with pytest.raises(TimeoutError):
            with acquire_file_lock("test_file.txt", "agent_test_002", operation="edit"):
                pass  # Should not reach here


# ============================================================================
# Task Management Tests
# ============================================================================

class TestTaskManagement:
    """Test task management operations."""
    
    def test_read_task_success(self, mock_paths):
        """Test reading a task from TODO.md."""
        task = read_task("ARCH-001")
        assert task is not None
        assert task.task_id == "ARCH-001"
        assert task.role == "Architect"
        assert task.status == "unclaimed"
    
    def test_read_task_not_found(self, mock_paths):
        """Test reading non-existent task."""
        task = read_task("INVALID-999")
        assert task is None
    
    def test_claim_task_success(self, mock_paths, clean_locks):
        """Test successful task claiming."""
        result = claim_task("ARCH-001", "agent_test_arch", "arch")
        assert result is True
        
        # Verify task updated
        task = read_task("ARCH-001")
        assert task.status == "claimed"
        assert task.claimed_by == "agent_test_arch"
    
    def test_claim_already_claimed_task(self, mock_paths, clean_locks):
        """Test claiming already claimed task fails."""
        # First claim
        claim_task("ARCH-001", "agent_test_arch", "arch")
        
        # Second claim should fail
        result = claim_task("ARCH-001", "agent_test_arch2", "arch")
        assert result is False
    
    def test_claim_task_wrong_role(self, mock_paths, clean_locks):
        """Test claiming task with wrong role fails."""
        # Backend trying to claim architect task
        result = claim_task("ARCH-001", "agent_test_be", "be")
        assert result is False
    
    def test_claim_task_architect_override(self, mock_paths, clean_locks):
        """Test architect can claim any role's tasks."""
        result = claim_task("BE-001", "agent_test_arch", "arch")
        assert result is True
    
    def test_claim_blocked_task(self, mock_paths, clean_locks):
        """Test claiming blocked task fails if blocker not complete."""
        # BE-001 is blocked by ARCH-001
        result = claim_task("BE-001", "agent_test_be", "be")
        assert result is False
    
    def test_start_task_success(self, mock_paths, clean_locks):
        """Test starting a claimed task."""
        claim_task("ARCH-001", "agent_test_arch", "arch")
        result = start_task("ARCH-001", "agent_test_arch")
        assert result is True
        
        task = read_task("ARCH-001")
        assert task.status == "in_progress"
    
    def test_start_task_not_claimed_by_agent(self, mock_paths, clean_locks):
        """Test starting task not claimed by agent fails."""
        claim_task("ARCH-001", "agent_test_arch", "arch")
        result = start_task("ARCH-001", "agent_test_arch2")
        assert result is False
    
    def test_complete_task_success(self, mock_paths, clean_locks):
        """Test completing a task."""
        claim_task("ARCH-001", "agent_test_arch", "arch")
        start_task("ARCH-001", "agent_test_arch")
        result = complete_task("ARCH-001", "Implementation complete", "agent_test_arch")
        assert result is True
        
        task = read_task("ARCH-001")
        assert task.status == "completed"
        assert task.completion_notes == "Implementation complete"
    
    def test_block_task_success(self, mock_paths, clean_locks):
        """Test blocking a task."""
        claim_task("ARCH-001", "agent_test_arch", "arch")
        result = block_task("ARCH-001", "EXTERNAL", "Waiting on external API", "agent_test_arch")
        assert result is True
        
        task = read_task("ARCH-001")
        assert task.status == "blocked"
        assert task.blocked_by == "EXTERNAL"
    
    def test_unblock_task_success(self, mock_paths, clean_locks):
        """Test unblocking a task."""
        claim_task("ARCH-001", "agent_test_arch", "arch")
        block_task("ARCH-001", "EXTERNAL", "Waiting", "agent_test_arch")
        result = unblock_task("ARCH-001", "agent_test_arch")
        assert result is True
        
        task = read_task("ARCH-001")
        assert task.status == "claimed"
        assert task.blocked_by == ""
    
    def test_get_available_tasks(self, mock_paths):
        """Test getting list of available tasks."""
        available = get_available_tasks()
        assert "ARCH-001" in available
        assert "BE-001" in available
        assert "QA-001" in available
    
    def test_get_available_tasks_filtered_by_role(self, mock_paths):
        """Test getting available tasks filtered by role."""
        available = get_available_tasks("arch")
        assert "ARCH-001" in available
        assert "BE-001" not in available
    
    def test_get_task_dependencies(self, mock_paths):
        """Test getting task dependencies."""
        deps = get_task_dependencies("BE-001")
        assert "ARCH-001" in deps
    
    def test_get_dependent_tasks(self, mock_paths):
        """Test getting tasks that depend on a task."""
        dependents = get_dependent_tasks("ARCH-001")
        assert "BE-001" in dependents


# ============================================================================
# Concurrent Access Tests
# ============================================================================

class TestConcurrentAccess:
    """Test concurrent access patterns."""
    
    def test_concurrent_task_claims(self, mock_paths, clean_locks):
        """Test concurrent task claims - only one should succeed."""
        results = []
        
        def try_claim(agent_id):
            return claim_task("ARCH-001", agent_id, "arch")
        
        # Try to claim same task from 5 agents concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(try_claim, f"agent_{i}") for i in range(5)]
            results = [f.result() for f in as_completed(futures)]
        
        # Only one should succeed
        success_count = sum(1 for r in results if r)
        assert success_count == 1
        
        # Verify task is claimed
        task = read_task("ARCH-001")
        assert task.status == "claimed"
    
    def test_concurrent_lock_acquisition(self, clean_locks):
        """Test concurrent lock acquisition is safe."""
        lock_results = []
        
        def try_acquire(agent_id):
            return acquire_lock("concurrent_test.txt", agent_id, timeout=2)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(try_acquire, f"agent_{i}") for i in range(10)]
            lock_results = [f.result() for f in as_completed(futures)]
        
        # Only one should succeed immediately
        success_count = sum(1 for r in lock_results if r)
        assert success_count == 1
    
    def test_stress_test_file_locking(self, clean_locks):
        """Stress test file locking with many iterations."""
        success_count = 0
        iterations = 100
        
        for i in range(iterations):
            agent_id = f"agent_stress_{i}"
            
            # Acquire and release
            if acquire_lock("stress_test.txt", agent_id):
                if release_lock("stress_test.txt", agent_id):
                    success_count += 1
        
        # All iterations should succeed
        assert success_count == iterations


# ============================================================================
# Session Registry Tests
# ============================================================================

class TestSessionRegistry:
    """Test session registry operations."""
    
    def test_register_session(self, mock_paths, clean_locks):
        """Test registering a new session."""
        result = register_session("agent_test_arch", "arch")
        assert result is True
        
        # Verify session in registry
        with open(mock_paths / "agents" / "session_registry.json") as f:
            registry = json.load(f)
        
        session_ids = [s['agent_id'] for s in registry['sessions']]
        assert "agent_test_arch" in session_ids
    
    def test_unregister_session(self, mock_paths, clean_locks):
        """Test unregistering a session."""
        register_session("agent_test_arch", "arch")
        result = unregister_session("agent_test_arch")
        assert result is True
        
        # Verify session removed
        with open(mock_paths / "agents" / "session_registry.json") as f:
            registry = json.load(f)
        
        session_ids = [s['agent_id'] for s in registry['sessions']]
        assert "agent_test_arch" not in session_ids
    
    def test_update_heartbeat(self, mock_paths, clean_locks):
        """Test updating session heartbeat."""
        register_session("agent_test_arch", "arch")
        
        time.sleep(0.1)  # Small delay
        result = update_heartbeat("agent_test_arch")
        assert result is True
        
        # Verify heartbeat updated
        with open(mock_paths / "agents" / "session_registry.json") as f:
            registry = json.load(f)
        
        session = next(s for s in registry['sessions'] if s['agent_id'] == "agent_test_arch")
        assert session['last_heartbeat'] > session['started_at']
    
    def test_get_active_sessions(self, mock_paths, clean_locks):
        """Test getting active sessions."""
        register_session("agent_test_arch", "arch")
        register_session("agent_test_be", "be")
        
        sessions = get_active_sessions()
        assert len(sessions) == 2


# ============================================================================
# Message Board Tests
# ============================================================================

class TestMessageBoard:
    """Test message board operations."""
    
    def test_send_message(self, mock_paths, clean_locks):
        """Test sending a message."""
        send_message("agent_test_arch", "Test message content", "INFO")
        
        with open(mock_paths / "agents" / "message_board.md") as f:
            content = f.read()
        
        assert "agent_test_arch" in content
        assert "Test message content" in content
    
    def test_send_message_with_tags(self, mock_paths, clean_locks):
        """Test sending message with tags."""
        send_message("agent_test_arch", "Test message", "INFO", 
                    tags=["test", "example"], related_tasks=["ARCH-001"])
        
        with open(mock_paths / "agents" / "message_board.md") as f:
            content = f.read()
        
        assert "#test" in content or "**Tags**: test example" in content
        assert "ARCH-001" in content


# ============================================================================
# Handoff Log Tests
# ============================================================================

class TestHandoffLog:
    """Test handoff log operations."""
    
    def test_log_handoff(self, mock_paths, clean_locks):
        """Test logging a handoff."""
        handoff_id = log_handoff(
            from_agent="agent_test_arch",
            to_agent="agent_test_be",
            related_tasks=["ARCH-001", "BE-001"],
            context="Architecture complete, ready for implementation",
            artifacts=["docs/architecture.md"],
            notes="Pay attention to section 3.2"
        )
        
        assert handoff_id.startswith("HANDOFF-")
        
        with open(mock_paths / "agents" / "handoff_log.md") as f:
            content = f.read()
        
        assert handoff_id in content
        assert "agent_test_arch" in content
        assert "agent_test_be" in content


# ============================================================================
# State Management Tests
# ============================================================================

class TestStateManagement:
    """Test state management operations."""
    
    def test_update_state_snapshot(self, mock_paths, clean_locks):
        """Test updating state snapshot."""
        state = update_state_snapshot()
        
        assert "last_updated" in state
        assert "task_counts" in state
        assert "active_agents" in state
    
    def test_state_snapshot_after_task_claim(self, mock_paths, clean_locks):
        """Test state snapshot reflects task claims."""
        # Initial state
        state1 = update_state_snapshot()
        initial_unclaimed = state1['task_counts']['unclaimed']
        
        # Claim a task
        claim_task("ARCH-001", "agent_test_arch", "arch")
        
        # Updated state
        state2 = update_state_snapshot()
        
        assert state2['task_counts']['unclaimed'] == initial_unclaimed - 1
        assert state2['task_counts']['claimed'] == 1


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_generate_session_id_format(self):
        """Test session ID format is correct."""
        session_id = generate_session_id("arch")
        assert session_id.startswith("agent_")
        assert session_id.endswith("_arch")
        
        # Should match pattern
        assert validate_session_id(session_id) is True
    
    def test_validate_session_id_valid(self):
        """Test validating valid session IDs."""
        assert validate_session_id("agent_20260309_143022_arch") is True
        assert validate_session_id("agent_20260309_150045_be") is True
        assert validate_session_id("agent_20260309_160000_fe") is True
        assert validate_session_id("agent_20260309_170000_qa") is True
    
    def test_validate_session_id_invalid(self):
        """Test validating invalid session IDs."""
        assert validate_session_id("invalid_id") is False
        assert validate_session_id("agent_invalid_arch") is False
        assert validate_session_id("agent_20260309_143022_unknown") is False
    
    def test_get_role_from_session_id(self):
        """Test extracting role from session ID."""
        assert get_role_from_session_id("agent_20260309_143022_arch") == "arch"
        assert get_role_from_session_id("agent_20260309_150045_be") == "be"
        assert get_role_from_session_id("agent_20260309_160000_fe") == "fe"
        assert get_role_from_session_id("agent_20260309_170000_qa") == "qa"
        assert get_role_from_session_id("invalid_id") is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_task_workflow(self, mock_paths, clean_locks):
        """Test complete task workflow from claim to completion."""
        agent_id = "agent_test_arch"
        
        # Claim
        assert claim_task("ARCH-001", agent_id, "arch") is True
        task = read_task("ARCH-001")
        assert task.status == "claimed"
        
        # Start
        assert start_task("ARCH-001", agent_id) is True
        task = read_task("ARCH-001")
        assert task.status == "in_progress"
        
        # Complete
        assert complete_task("ARCH-001", "Done!", agent_id) is True
        task = read_task("ARCH-001")
        assert task.status == "completed"
    
    def test_dependency_chain_workflow(self, mock_paths, clean_locks):
        """Test workflow with task dependencies."""
        # Step 1: Architect claims and completes ARCH-001
        assert claim_task("ARCH-001", "agent_arch", "arch") is True
        assert start_task("ARCH-001", "agent_arch") is True
        assert complete_task("ARCH-001", "Architecture done", "agent_arch") is True
        
        # Step 2: Backend can now claim BE-001 (was blocked by ARCH-001)
        assert claim_task("BE-001", "agent_be", "be") is True
        
        # Step 3: Backend completes BE-001
        assert start_task("BE-001", "agent_be") is True
        assert complete_task("BE-001", "API done", "agent_be") is True
        
        # Step 4: QA can now claim QA-001 (was blocked by BE-001)
        assert claim_task("QA-001", "agent_qa", "qa") is True
    
    def test_multi_agent_no_conflict_workflow(self, mock_paths, clean_locks):
        """Test multiple agents working without conflicts."""
        results = {}
        
        # Each agent claims their appropriate task
        results['arch'] = claim_task("ARCH-001", "agent_arch", "arch")
        results['be'] = claim_task("BE-001", "agent_be", "be")  # Will fail - blocked
        results['qa'] = claim_task("QA-001", "agent_qa", "qa")  # Will fail - blocked
        
        # Only architect should succeed initially
        assert results['arch'] is True
        assert results['be'] is False  # Blocked by ARCH-001
        assert results['qa'] is False  # Blocked by BE-001


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_task_id(self, mock_paths, clean_locks):
        """Test handling empty task ID."""
        result = claim_task("", "agent_test", "arch")
        assert result is False
    
    def test_none_agent_id(self, mock_paths, clean_locks):
        """Test handling None agent ID."""
        result = claim_task("ARCH-001", None, "arch")
        assert result is False
    
    def test_invalid_role(self, mock_paths, clean_locks):
        """Test handling invalid role."""
        result = claim_task("ARCH-001", "agent_test", "invalid_role")
        assert result is False
    
    def test_missing_todo_file(self, tmp_path):
        """Test handling missing TODO.md file."""
        # Don't create TODO.md
        result = read_task("ARCH-001")
        assert result is None
    
    def test_corrupted_lock_file(self, clean_locks):
        """Test handling corrupted lock file."""
        # Create corrupted lock file
        lock_file = clean_locks / "corrupted.lock"
        lock_file.write_text("not valid json {{{")
        
        # Should handle gracefully
        result = acquire_lock("corrupted.lock", "agent_test", timeout=1)
        assert result is True


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance-related tests."""
    
    def test_lock_acquisition_speed(self, clean_locks):
        """Test lock acquisition is fast."""
        start = time.time()
        
        for i in range(100):
            acquire_lock(f"perf_test_{i}.txt", "agent_test")
            release_lock(f"perf_test_{i}.txt", "agent_test")
        
        elapsed = time.time() - start
        
        # Should complete 100 lock cycles in under 5 seconds
        assert elapsed < 5.0, f"Lock operations too slow: {elapsed}s"
    
    def test_task_read_performance(self, mock_paths):
        """Test task reading is fast."""
        start = time.time()
        
        for _ in range(1000):
            read_task("ARCH-001")
        
        elapsed = time.time() - start
        
        # Should read 1000 times in under 2 seconds
        assert elapsed < 2.0, f"Task reads too slow: {elapsed}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
