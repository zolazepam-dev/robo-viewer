#!/usr/bin/env python3
"""
Multi-Agent System - Agent Tools

Core utility functions for agent coordination, task management, and file locking.
All operations are atomic and thread-safe.

Usage:
    from agent_tools import claim_task, complete_task, block_task, send_message
    
    # Claim a task
    success = claim_task("TASK-001", "agent_20260309_143022_arch", "arch")
    
    # Complete a task
    success = complete_task("TASK-001", "Implementation complete")
    
    # Send a message
    send_message("agent_20260309_143022_arch", "Task complete!")
"""

import os
import json
import fcntl
import time
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent
LOCKS_DIR = BASE_DIR / "locks"
AGENTS_DIR = BASE_DIR / "agents"
STATE_DIR = BASE_DIR / "state"
TODO_FILE = BASE_DIR / "TODO.md"
SESSION_REGISTRY = AGENTS_DIR / "session_registry.json"
MESSAGE_BOARD = AGENTS_DIR / "message_board.md"
HANDOFF_LOG = AGENTS_DIR / "handoff_log.md"
CURRENT_STATE = STATE_DIR / "current_state.json"
BLOCKERS_FILE = STATE_DIR / "blockers.md"

LOCK_TIMEOUT = 30  # seconds
SESSION_TIMEOUT = 300  # seconds (5 minutes)


# ============================================================================
# Data Classes
# ============================================================================

class TaskStatus(Enum):
    UNCLAIMED = "unclaimed"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    REOPENED = "reopened"


class Role(Enum):
    ARCHITECT = "arch"
    BACKEND = "be"
    FRONTEND = "fe"
    QA = "qa"
    ANY = "any"


@dataclass
class Task:
    task_id: str
    role: str
    description: str
    status: str
    claimed_by: str
    blocked_by: str
    priority: str
    acceptance_criteria: str
    claimed_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    completion_notes: Optional[str] = None
    blocked_at: Optional[str] = None
    block_reason: Optional[str] = None
    unblocked_at: Optional[str] = None


@dataclass
class LockInfo:
    locked_by: str
    locked_at: str
    operation: str
    pid: int


@dataclass
class AgentSession:
    agent_id: str
    role: str
    started_at: str
    last_heartbeat: str
    status: str
    current_task: Optional[str] = None


# ============================================================================
# File Locking
# ============================================================================

def _ensure_locks_dir():
    """Create locks directory if it doesn't exist."""
    LOCKS_DIR.mkdir(exist_ok=True)


def _get_lock_file(filename: str) -> Path:
    """Get the lock file path for a given filename."""
    _ensure_locks_dir()
    return LOCKS_DIR / f"{Path(filename).name}.lock"


def _is_lock_stale(lock_file: Path, timeout: int = LOCK_TIMEOUT) -> bool:
    """Check if a lock is stale (held longer than timeout)."""
    if not lock_file.exists():
        return False
    
    try:
        with open(lock_file, 'r') as f:
            lock_data = json.load(f)
        
        locked_at = datetime.fromisoformat(lock_data['locked_at'])
        elapsed = (datetime.now() - locked_at).total_seconds()
        return elapsed > timeout
    except (json.JSONDecodeError, KeyError, ValueError):
        # Corrupted lock file, treat as stale
        return True


def _break_lock(lock_file: Path) -> bool:
    """Break a stale lock."""
    try:
        if lock_file.exists():
            lock_file.unlink()
        return True
    except OSError:
        return False


def acquire_lock(filename: str, agent_id: str, operation: str = "edit", timeout: int = LOCK_TIMEOUT) -> bool:
    """
    Acquire an exclusive lock on a file.
    
    Args:
        filename: Name of the file to lock
        agent_id: ID of the agent acquiring the lock
        operation: Type of operation (edit, read, etc.)
        timeout: Maximum time to wait for lock (seconds)
    
    Returns:
        True if lock acquired, False if timeout
    """
    lock_file = _get_lock_file(filename)
    start_time = time.time()
    
    while True:
        try:
            # Try to create lock file atomically
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w') as f:
                lock_info = LockInfo(
                    locked_by=agent_id,
                    locked_at=datetime.now().isoformat(),
                    operation=operation,
                    pid=os.getpid()
                )
                json.dump(asdict(lock_info), f, indent=2)
            return True
            
        except FileExistsError:
            # Check if lock is stale
            if _is_lock_stale(lock_file, timeout):
                if _break_lock(lock_file):
                    continue
            
            # Check if we've timed out
            if time.time() - start_time > timeout:
                return False
            
            # Wait and retry
            time.sleep(0.1)


def release_lock(filename: str, agent_id: str) -> bool:
    """
    Release a lock on a file.
    
    Args:
        filename: Name of the file to unlock
        agent_id: ID of the agent releasing the lock
    
    Returns:
        True if lock released, False if not owned by agent
    """
    lock_file = _get_lock_file(filename)
    
    if not lock_file.exists():
        return True  # Lock already released
    
    try:
        with open(lock_file, 'r') as f:
            lock_data = json.load(f)
        
        if lock_data.get('locked_by') != agent_id:
            # Don't release another agent's lock
            return False
        
        lock_file.unlink()
        return True
        
    except (json.JSONDecodeError, KeyError):
        # Corrupted lock file, remove it
        if lock_file.exists():
            lock_file.unlink()
        return True


def is_file_locked(filename: str) -> bool:
    """Check if a file is currently locked."""
    lock_file = _get_lock_file(filename)
    
    if not lock_file.exists():
        return False
    
    if _is_lock_stale(lock_file):
        _break_lock(lock_file)
        return False
    
    return True


def get_lock_holder(filename: str) -> Optional[str]:
    """Get the agent ID holding the lock on a file."""
    lock_file = _get_lock_file(filename)
    
    if not lock_file.exists():
        return None
    
    try:
        with open(lock_file, 'r') as f:
            lock_data = json.load(f)
        return lock_data.get('locked_by')
    except (json.JSONDecodeError, KeyError):
        return None


# ============================================================================
# TODO.md Parsing and Manipulation
# ============================================================================

def _parse_todo_md(content: str) -> List[Dict[str, Any]]:
    """Parse TODO.md markdown table into list of task dictionaries."""
    tasks = []
    lines = content.split('\n')
    
    in_task_table = False
    headers = []
    
    for line in lines:
        line = line.strip()
        
        # Find the task table header
        if line.startswith('| Task ID |') and 'Role' in line:
            in_task_table = True
            headers = [h.strip() for h in line.split('|')[1:-1]]
            continue
        
        # Skip separator line
        if in_task_table and line.startswith('|---'):
            continue
        
        # Parse task rows
        if in_task_table and line.startswith('|'):
            if line.startswith('| ---') or line.startswith('| **'):
                in_task_table = False
                continue
            
            values = [v.strip() for v in line.split('|')[1:-1]]
            
            if len(values) >= len(headers):
                task = dict(zip(headers, values))
                
                # Only include actual task rows (have valid task ID)
                if task.get('Task ID', '').startswith(('ARCH-', 'BE-', 'FE-', 'QA-', 'DOCS-')):
                    tasks.append(task)
    
    return tasks


def _tasks_to_markdown(tasks: List[Dict[str, Any]], summary: Dict[str, int]) -> str:
    """Convert task list back to markdown table format."""
    # Read original file to preserve structure
    with open(TODO_FILE, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    new_lines = []
    in_task_table = False
    table_written = False

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Update summary counts
        if 'Task Summary' in line:
            new_lines.append(line)
            # Find and update the summary table
            j = i + 1
            while j < len(lines) and j < i + 10:
                if '| Unclaimed |' in lines[j]:
                    new_lines.append(f'| Unclaimed | {summary.get("unclaimed", 0)} |')
                elif '| Claimed |' in lines[j]:
                    new_lines.append(f'| Claimed | {summary.get("claimed", 0)} |')
                elif '| In Progress |' in lines[j]:
                    new_lines.append(f'| In Progress | {summary.get("in_progress", 0)} |')
                elif '| Blocked |' in lines[j]:
                    new_lines.append(f'| Blocked | {summary.get("blocked", 0)} |')
                elif '| Completed |' in lines[j]:
                    new_lines.append(f'| Completed | {summary.get("completed", 0)} |')
                elif '| **Total** |' in lines[j]:
                    new_lines.append(f'| **Total** | **{sum(summary.values())}** |')
                else:
                    new_lines.append(lines[j])
                j += 1
            continue
        
        # Find the task table header
        if line_stripped.startswith('| Task ID |') and 'Role' in line_stripped:
            in_task_table = True
            new_lines.append(line)
            # Add separator
            new_lines.append('|---------|------|-------------|--------|------------|------------|----------|---------------------|')
            
            # Add task rows
            for task in tasks:
                row = f"| {task.get('Task ID', '')} | {task.get('Role', '')} | {task.get('Description', '')} | {task.get('Status', 'unclaimed')} | {task.get('Claimed By', '')} | {task.get('Blocked By', '')} | {task.get('Priority', '')} | {task.get('Acceptance Criteria', '')} |"
                new_lines.append(row)
            
            table_written = True
            # Skip old task rows until we hit next section
            continue
        
        # Skip old task rows
        if in_task_table and not table_written:
            if line_stripped.startswith('| ---') or line_stripped.startswith('| **') or line_stripped.startswith('##'):
                in_task_table = False
                new_lines.append(line)
            continue
        
        new_lines.append(line)
    
    return '\n'.join(new_lines)


def _calculate_summary(tasks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate task summary counts."""
    summary = {
        'unclaimed': 0,
        'claimed': 0,
        'in_progress': 0,
        'blocked': 0,
        'completed': 0
    }
    
    for task in tasks:
        status = task.get('Status', 'unclaimed').lower()
        if status in summary:
            summary[status] += 1
    
    return summary


def _get_task_row_index(content: str, task_id: str) -> int:
    """Get the line index of a task in TODO.md."""
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        if f'| {task_id} |' in line or f'|{task_id}|' in line:
            return i
    
    return -1


def read_task(task_id: str) -> Optional[Task]:
    """
    Read a task from TODO.md.
    
    Args:
        task_id: The task ID to read
    
    Returns:
        Task object or None if not found
    """
    if not TODO_FILE.exists():
        return None
    
    with open(TODO_FILE, 'r') as f:
        content = f.read()
    
    tasks = _parse_todo_md(content)
    
    for task_data in tasks:
        if task_data.get('Task ID') == task_id:
            return Task(
                task_id=task_data.get('Task ID', ''),
                role=task_data.get('Role', ''),
                description=task_data.get('Description', ''),
                status=task_data.get('Status', 'unclaimed'),
                claimed_by=task_data.get('Claimed By', ''),
                blocked_by=task_data.get('Blocked By', ''),
                priority=task_data.get('Priority', ''),
                acceptance_criteria=task_data.get('Acceptance Criteria', ''),
                claimed_at=task_data.get('Claimed At'),
                started_at=task_data.get('Started At'),
                completed_at=task_data.get('Completed At'),
                completion_notes=task_data.get('Completion Notes'),
                blocked_at=task_data.get('Blocked At'),
                block_reason=task_data.get('Block Reason'),
                unblocked_at=task_data.get('Unblocked At')
            )
    
    return None


def _update_task_in_todo(task: Task) -> bool:
    """Update a task in TODO.md file."""
    if not TODO_FILE.exists():
        return False
    
    with acquire_file_lock("TODO.md", task.claimed_by or "system"):
        with open(TODO_FILE, 'r') as f:
            content = f.read()
        
        tasks = _parse_todo_md(content)
        
        # Find and update the task
        task_found = False
        for i, task_data in enumerate(tasks):
            if task_data.get('Task ID') == task.task_id:
                tasks[i]['Status'] = task.status
                if task.claimed_by:
                    tasks[i]['Claimed By'] = task.claimed_by
                if task.blocked_by:
                    tasks[i]['Blocked By'] = task.blocked_by
                task_found = True
                break
        
        if not task_found:
            return False
        
        # Calculate new summary
        summary = _calculate_summary(tasks)
        
        # Write updated content
        new_content = _tasks_to_markdown(tasks, summary)
        
        with open(TODO_FILE, 'w') as f:
            f.write(new_content)
        
        return True


# ============================================================================
# Context Manager for File Locking
# ============================================================================

class FileLock:
    """Context manager for file locking."""
    
    def __init__(self, filename: str, agent_id: str, operation: str = "edit"):
        self.filename = filename
        self.agent_id = agent_id
        self.operation = operation
        self.acquired = False
    
    def __enter__(self):
        self.acquired = acquire_lock(self.filename, self.agent_id, self.operation)
        if not self.acquired:
            raise TimeoutError(f"Could not acquire lock on {self.filename}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            release_lock(self.filename, self.agent_id)
        return False


def acquire_file_lock(filename: str, agent_id: str, operation: str = "edit"):
    """
    Context manager for acquiring file locks.
    
    Usage:
        with acquire_file_lock("TODO.md", agent_id):
            # Do file operations
            pass
    """
    return FileLock(filename, agent_id, operation)


# ============================================================================
# Core Task Management Functions
# ============================================================================

def claim_task(task_id: str, agent_id: str, role: str) -> bool:
    """
    Claim a task for an agent.
    
    Args:
        task_id: The task ID to claim (e.g., "ARCH-001")
        agent_id: The agent's session ID
        role: The agent's role (arch, be, fe, qa)
    
    Returns:
        True if task claimed successfully, False otherwise
    """
    # Validate inputs
    if not task_id or not agent_id or not role:
        return False
    
    # Read current task state
    task = read_task(task_id)
    if not task:
        print(f"Error: Task {task_id} not found")
        return False
    
    # Check if already claimed
    if task.status.lower() not in ['unclaimed', 'reopened']:
        print(f"Error: Task {task_id} is already {task.status}")
        return False
    
    # Validate role compatibility
    task_role = task.role.lower().strip()
    agent_role = role.lower().strip()
    
    # Map role abbreviations
    role_map = {
        'architect': 'arch',
        'backend': 'be', 
        'frontend': 'fe',
        'qa': 'qa',
        'any': 'any'
    }
    
    task_role = role_map.get(task_role, task_role)
    agent_role = role_map.get(agent_role, agent_role)
    
    if task_role != 'any' and agent_role != task_role:
        # Allow architect to override
        if agent_role != 'arch':
            print(f"Error: Task {task_id} requires {task_role} role, got {agent_role}")
            return False
    
    # Check for dependencies
    if task.blocked_by and task.blocked_by.strip():
        # Check if blocking task is complete
        blocker = read_task(task.blocked_by.strip())
        if blocker and blocker.status.lower() != 'completed':
            print(f"Error: Task {task_id} is blocked by {task.blocked_by}")
            return False
    
    # Acquire lock and update task
    with acquire_file_lock("TODO.md", agent_id):
        # Re-read task to ensure no race condition
        task = read_task(task_id)
        if not task or task.status.lower() not in ['unclaimed', 'reopened']:
            return False
        
        # Update task state
        task.status = 'claimed'
        task.claimed_by = agent_id
        task.claimed_at = datetime.now().isoformat()
        
        # Write update
        if not _update_task_in_todo(task):
            return False
    
    # Update session registry
    _update_agent_task(agent_id, task_id)
    
    # Log the claim
    _log_task_event("CLAIM", task_id, agent_id)
    
    return True


def start_task(task_id: str, agent_id: str) -> bool:
    """
    Mark a task as in progress.
    
    Args:
        task_id: The task ID to start
        agent_id: The agent's session ID
    
    Returns:
        True if task started successfully, False otherwise
    """
    task = read_task(task_id)
    if not task:
        return False
    
    if task.claimed_by != agent_id:
        print(f"Error: Task {task_id} is not claimed by {agent_id}")
        return False
    
    with acquire_file_lock("TODO.md", agent_id):
        task = read_task(task_id)
        if not task or task.status != 'claimed':
            return False
        
        task.status = 'in_progress'
        task.started_at = datetime.now().isoformat()
        
        if not _update_task_in_todo(task):
            return False
    
    _log_task_event("START", task_id, agent_id)
    return True


def complete_task(task_id: str, notes: str, agent_id: Optional[str] = None) -> bool:
    """
    Mark a task as completed.
    
    Args:
        task_id: The task ID to complete
        notes: Completion notes
        agent_id: Optional agent ID (defaults to current claim holder)
    
    Returns:
        True if task completed successfully, False otherwise
    """
    task = read_task(task_id)
    if not task:
        return False
    
    if agent_id and task.claimed_by != agent_id:
        print(f"Error: Task {task_id} is not claimed by {agent_id}")
        return False
    
    actual_agent = agent_id or task.claimed_by
    
    with acquire_file_lock("TODO.md", actual_agent):
        task = read_task(task_id)
        if not task or task.status not in ['claimed', 'in_progress']:
            return False
        
        task.status = 'completed'
        task.completed_at = datetime.now().isoformat()
        task.completion_notes = notes
        
        if not _update_task_in_todo(task):
            return False
    
    # Clear agent's current task
    _update_agent_task(actual_agent, None)
    
    # Log completion
    _log_task_event("COMPLETE", task_id, actual_agent, notes)
    
    # Post to message board
    send_message(actual_agent, f"**{task_id} Complete**: {task.description}\n\n**Notes**: {notes}")
    
    return True


def block_task(task_id: str, blocked_by: str, reason: str, agent_id: Optional[str] = None) -> bool:
    """
    Mark a task as blocked.
    
    Args:
        task_id: The task ID to block
        blocked_by: What is blocking this task (task ID or external factor)
        reason: Reason for the block
        agent_id: Optional agent ID
    
    Returns:
        True if task blocked successfully, False otherwise
    """
    task = read_task(task_id)
    if not task:
        return False
    
    if agent_id and task.claimed_by != agent_id:
        print(f"Error: Task {task_id} is not claimed by {agent_id}")
        return False
    
    actual_agent = agent_id or task.claimed_by
    
    with acquire_file_lock("TODO.md", actual_agent):
        task = read_task(task_id)
        if not task or task.status not in ['claimed', 'in_progress']:
            return False
        
        task.status = 'blocked'
        task.blocked_by = blocked_by
        task.blocked_at = datetime.now().isoformat()
        task.block_reason = reason
        
        if not _update_task_in_todo(task):
            return False
    
    # Log blocker
    _add_blocker(task_id, blocked_by, reason, actual_agent)
    
    # Log the block
    _log_task_event("BLOCK", task_id, actual_agent, reason)
    
    return True


def unblock_task(task_id: str, agent_id: str) -> bool:
    """
    Remove block from a task and return to claimed state.
    
    Args:
        task_id: The task ID to unblock
        agent_id: The agent's session ID
    
    Returns:
        True if task unblocked successfully, False otherwise
    """
    task = read_task(task_id)
    if not task:
        return False
    
    if task.status != 'blocked':
        print(f"Error: Task {task_id} is not blocked")
        return False
    
    with acquire_file_lock("TODO.md", agent_id):
        task = read_task(task_id)
        if task.status != 'blocked':
            return False
        
        task.status = 'claimed'
        task.unblocked_at = datetime.now().isoformat()
        task.blocked_by = ''
        task.block_reason = ''
        
        if not _update_task_in_todo(task):
            return False
    
    # Remove from blockers
    _remove_blocker(task_id)
    
    _log_task_event("UNBLOCK", task_id, agent_id)
    return True


def get_available_tasks(role: Optional[str] = None) -> List[str]:
    """
    Get list of available (unclaimed) tasks.
    
    Args:
        role: Optional role filter (only return tasks for this role)
    
    Returns:
        List of task IDs
    """
    if not TODO_FILE.exists():
        return []
    
    with open(TODO_FILE, 'r') as f:
        content = f.read()
    
    tasks = _parse_todo_md(content)
    available = []
    
    for task_data in tasks:
        status = task_data.get('Status', '').lower()
        if status not in ['unclaimed', 'reopened']:
            continue
        
        task_role = task_data.get('Role', '').lower()
        
        # Filter by role if specified
        if role:
            if task_role == 'any' or task_role == role.lower():
                available.append(task_data.get('Task ID', ''))
        else:
            available.append(task_data.get('Task ID', ''))
    
    return available


def get_task_dependencies(task_id: str) -> List[str]:
    """
    Get list of tasks that this task depends on.
    
    Args:
        task_id: The task ID to check
    
    Returns:
        List of dependency task IDs
    """
    task = read_task(task_id)
    if not task:
        return []
    
    dependencies = []
    
    if task.blocked_by and task.blocked_by.strip():
        # Check if blocked_by is a task ID
        if any(task.blocked_by.strip().startswith(prefix) for prefix in ['ARCH-', 'BE-', 'FE-', 'QA-', 'DOCS-']):
            dependencies.append(task.blocked_by.strip())
    
    return dependencies


def get_dependent_tasks(task_id: str) -> List[str]:
    """
    Get list of tasks that depend on this task.
    
    Args:
        task_id: The task ID to check
    
    Returns:
        List of dependent task IDs
    """
    if not TODO_FILE.exists():
        return []
    
    with open(TODO_FILE, 'r') as f:
        content = f.read()
    
    tasks = _parse_todo_md(content)
    dependents = []
    
    for task_data in tasks:
        blocked_by = task_data.get('Blocked By', '').strip()
        if blocked_by == task_id:
            dependents.append(task_data.get('Task ID', ''))
    
    return dependents


# ============================================================================
# Session Registry
# ============================================================================

def _ensure_session_registry():
    """Create session registry file if it doesn't exist."""
    AGENTS_DIR.mkdir(exist_ok=True)
    if not SESSION_REGISTRY.exists():
        with open(SESSION_REGISTRY, 'w') as f:
            json.dump({"sessions": [], "last_updated": datetime.now().isoformat()}, f, indent=2)


def register_session(agent_id: str, role: str) -> bool:
    """
    Register a new agent session.
    
    Args:
        agent_id: The agent's session ID
        role: The agent's role
    
    Returns:
        True if registration successful
    """
    _ensure_session_registry()
    
    with acquire_file_lock("session_registry.json", agent_id):
        with open(SESSION_REGISTRY, 'r') as f:
            registry = json.load(f)
        
        # Check if already registered
        for session in registry.get('sessions', []):
            if session.get('agent_id') == agent_id:
                # Update heartbeat
                session['last_heartbeat'] = datetime.now().isoformat()
                session['status'] = 'active'
                break
        else:
            # Add new session
            new_session = asdict(AgentSession(
                agent_id=agent_id,
                role=role,
                started_at=datetime.now().isoformat(),
                last_heartbeat=datetime.now().isoformat(),
                status='active'
            ))
            registry['sessions'].append(new_session)
        
        registry['last_updated'] = datetime.now().isoformat()
        
        with open(SESSION_REGISTRY, 'w') as f:
            json.dump(registry, f, indent=2)
    
    return True


def unregister_session(agent_id: str) -> bool:
    """
    Unregister an agent session.
    
    Args:
        agent_id: The agent's session ID
    
    Returns:
        True if unregistration successful
    """
    _ensure_session_registry()
    
    with acquire_file_lock("session_registry.json", agent_id):
        with open(SESSION_REGISTRY, 'r') as f:
            registry = json.load(f)
        
        # Remove session
        registry['sessions'] = [
            s for s in registry.get('sessions', [])
            if s.get('agent_id') != agent_id
        ]
        
        registry['last_updated'] = datetime.now().isoformat()
        
        with open(SESSION_REGISTRY, 'w') as f:
            json.dump(registry, f, indent=2)
    
    return True


def update_heartbeat(agent_id: str) -> bool:
    """
    Update agent session heartbeat.
    
    Args:
        agent_id: The agent's session ID
    
    Returns:
        True if heartbeat updated
    """
    _ensure_session_registry()
    
    with acquire_file_lock("session_registry.json", agent_id):
        with open(SESSION_REGISTRY, 'r') as f:
            registry = json.load(f)
        
        for session in registry.get('sessions', []):
            if session.get('agent_id') == agent_id:
                session['last_heartbeat'] = datetime.now().isoformat()
                session['status'] = 'active'
                break
        
        registry['last_updated'] = datetime.now().isoformat()
        
        with open(SESSION_REGISTRY, 'w') as f:
            json.dump(registry, f, indent=2)
    
    return True


def _update_agent_task(agent_id: str, task_id: Optional[str]):
    """Update the current task for an agent in session registry."""
    _ensure_session_registry()
    
    with open(SESSION_REGISTRY, 'r') as f:
        registry = json.load(f)
    
    for session in registry.get('sessions', []):
        if session.get('agent_id') == agent_id:
            session['current_task'] = task_id
            break
    
    registry['last_updated'] = datetime.now().isoformat()
    
    with open(SESSION_REGISTRY, 'w') as f:
        json.dump(registry, f, indent=2)


def get_active_sessions() -> List[Dict[str, Any]]:
    """Get list of active agent sessions."""
    _ensure_session_registry()
    
    with open(SESSION_REGISTRY, 'r') as f:
        registry = json.load(f)
    
    # Filter out stale sessions
    active = []
    for session in registry.get('sessions', []):
        last_heartbeat = datetime.fromisoformat(session.get('last_heartbeat', ''))
        elapsed = (datetime.now() - last_heartbeat).total_seconds()
        
        if elapsed < SESSION_TIMEOUT:
            active.append(session)
        else:
            session['status'] = 'stale'
    
    return active


# ============================================================================
# Message Board
# ============================================================================

def _ensure_message_board():
    """Create message board file if it doesn't exist."""
    AGENTS_DIR.mkdir(exist_ok=True)
    if not MESSAGE_BOARD.exists():
        with open(MESSAGE_BOARD, 'w') as f:
            f.write("# Agent Message Board\n\n")
            f.write("---\n\n")
            f.write("*No messages yet.*\n")


def send_message(agent_id: str, message: str, message_type: str = "INFO", 
                 tags: Optional[List[str]] = None, related_tasks: Optional[List[str]] = None) -> None:
    """
    Post a message to the message board.
    
    Args:
        agent_id: The agent's session ID
        message: The message content
        message_type: Type of message (INFO, HELP, HANDOFF, BLOCKER, DECISION, COMPLETE)
        tags: Optional list of tags
        related_tasks: Optional list of related task IDs
    """
    _ensure_message_board()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with acquire_file_lock("message_board.md", agent_id):
        with open(MESSAGE_BOARD, 'r') as f:
            content = f.read()
        
        # Build message
        message_lines = [
            f"## {timestamp} {agent_id} {message_type}",
            "",
            message,
            ""
        ]
        
        if tags:
            message_lines.append(f"**Tags**: {' '.join(tags)}")
        
        if related_tasks:
            message_lines.append(f"**Related Tasks**: {', '.join(related_tasks)}")
        
        message_lines.append("")
        message_lines.append("---")
        message_lines.append("")
        
        new_message = '\n'.join(message_lines)
        
        # Remove "*No messages yet.*" if present
        content = content.replace("*No messages yet.*\n", "")
        
        # Prepend new message
        new_content = content + new_message
        
        with open(MESSAGE_BOARD, 'w') as f:
            f.write(new_content)


# ============================================================================
# Handoff Log
# ============================================================================

def _ensure_handoff_log():
    """Create handoff log file if it doesn't exist."""
    AGENTS_DIR.mkdir(exist_ok=True)
    if not HANDOFF_LOG.exists():
        with open(HANDOFF_LOG, 'w') as f:
            f.write("# Handoff Log\n\n")
            f.write("---\n\n")
            f.write("*No handoffs yet.*\n")


def log_handoff(from_agent: str, to_agent: str, related_tasks: List[str],
                context: str, artifacts: List[str], notes: str) -> str:
    """
    Log a task handoff between agents.
    
    Args:
        from_agent: Agent handing off
        to_agent: Agent receiving
        related_tasks: List of related task IDs
        context: Handoff context
        artifacts: List of artifact paths
        notes: Additional notes
    
    Returns:
        Handoff ID
    """
    _ensure_handoff_log()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    handoff_id = f"HANDOFF-{_get_next_handoff_id()}"
    
    with acquire_file_lock("handoff_log.md", from_agent):
        with open(HANDOFF_LOG, 'r') as f:
            content = f.read()
        
        # Remove "*No handoffs yet.*" if present
        content = content.replace("*No handoffs yet.*\n", "")
        
        handoff_entry = f"""
### {handoff_id}

**From**: {from_agent}
**To**: {to_agent}
**Timestamp**: {timestamp}
**Related Tasks**: {', '.join(related_tasks)}
**Context**: 
{context}
**Artifacts**: {', '.join(artifacts)}
**Notes**: {notes}

---

"""
        
        new_content = content + handoff_entry
        
        with open(HANDOFF_LOG, 'w') as f:
            f.write(new_content)
    
    return handoff_id


def _get_next_handoff_id() -> str:
    """Get the next sequential handoff ID."""
    if not HANDOFF_LOG.exists():
        return "001"
    
    with open(HANDOFF_LOG, 'r') as f:
        content = f.read()
    
    # Find all HANDOFF-XXX entries
    matches = re.findall(r'HANDOFF-(\d+)', content)
    
    if not matches:
        return "001"
    
    next_id = max(int(m) for m in matches) + 1
    return f"{next_id:03d}"


# ============================================================================
# State Management
# ============================================================================

def _ensure_state_files():
    """Create state directory and files if they don't exist."""
    STATE_DIR.mkdir(exist_ok=True)
    
    if not CURRENT_STATE.exists():
        with open(CURRENT_STATE, 'w') as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "task_counts": {
                    "unclaimed": 0,
                    "claimed": 0,
                    "in_progress": 0,
                    "blocked": 0,
                    "completed": 0
                },
                "active_agents": 0,
                "current_blockers": 0
            }, f, indent=2)
    
    if not BLOCKERS_FILE.exists():
        with open(BLOCKERS_FILE, 'w') as f:
            f.write("# Current Blockers\n\n")
            f.write("---\n\n")
            f.write("*No active blockers.*\n")


def update_state_snapshot():
    """Update the current state snapshot."""
    _ensure_state_files()
    
    # Calculate task counts
    if TODO_FILE.exists():
        with open(TODO_FILE, 'r') as f:
            content = f.read()
        tasks = _parse_todo_md(content)
        task_counts = _calculate_summary(tasks)
    else:
        task_counts = {k: 0 for k in ['unclaimed', 'claimed', 'in_progress', 'blocked', 'completed']}
    
    # Get active agents
    active_agents = len(get_active_sessions())
    
    # Count blockers
    if BLOCKERS_FILE.exists():
        with open(BLOCKERS_FILE, 'r') as f:
            content = f.read()
        current_blockers = content.count('## BLOCKER-')
    else:
        current_blockers = 0
    
    state = {
        "last_updated": datetime.now().isoformat(),
        "task_counts": task_counts,
        "active_agents": active_agents,
        "current_blockers": current_blockers
    }
    
    with acquire_file_lock("current_state.json", "system"):
        with open(CURRENT_STATE, 'w') as f:
            json.dump(state, f, indent=2)
    
    return state


def _log_task_event(event_type: str, task_id: str, agent_id: str, notes: str = ""):
    """Log a task event to the state update log."""
    # This could be expanded to maintain a full event log
    update_state_snapshot()


def _add_blocker(task_id: str, blocked_by: str, reason: str, agent_id: str):
    """Add a blocker to the blockers file."""
    _ensure_state_files()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blocker_id = f"BLOCKER-{_get_next_blocker_id()}"
    
    with acquire_file_lock("blockers.md", agent_id):
        with open(BLOCKERS_FILE, 'r') as f:
            content = f.read()
        
        # Remove "*No active blockers.*" if present
        content = content.replace("*No active blockers.*\n", "")
        
        blocker_entry = f"""
## {blocker_id}

**Task**: {task_id}
**Blocked By**: {blocked_by}
**Reason**: {reason}
**Reported By**: {agent_id}
**Timestamp**: {timestamp}
**Status**: Active

---

"""
        
        new_content = content + blocker_entry
        
        with open(BLOCKERS_FILE, 'w') as f:
            f.write(new_content)
    
    update_state_snapshot()


def _remove_blocker(task_id: str):
    """Remove a blocker from the blockers file."""
    if not BLOCKERS_FILE.exists():
        return
    
    with acquire_file_lock("blockers.md", "system"):
        with open(BLOCKERS_FILE, 'r') as f:
            content = f.read()
        
        # Find and mark blocker as resolved
        lines = content.split('\n')
        new_lines = []
        in_blocker = False
        current_task = None
        
        for line in lines:
            if line.startswith('## BLOCKER-'):
                in_blocker = True
                current_task = None
            
            if in_blocker and line.startswith('**Task**:'):
                current_task = line.split('**Task**:')[1].strip()
            
            if in_blocker and line.startswith('**Status**: Active'):
                if current_task == task_id:
                    new_lines.append('**Status**: Resolved')
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
            
            if line.startswith('---') and in_blocker:
                in_blocker = False
        
        with open(BLOCKERS_FILE, 'w') as f:
            f.write('\n'.join(new_lines))
    
    update_state_snapshot()


def _get_next_blocker_id() -> str:
    """Get the next sequential blocker ID."""
    if not BLOCKERS_FILE.exists():
        return "001"
    
    with open(BLOCKERS_FILE, 'r') as f:
        content = f.read()
    
    matches = re.findall(r'BLOCKER-(\d+)', content)
    
    if not matches:
        return "001"
    
    next_id = max(int(m) for m in matches) + 1
    return f"{next_id:03d}"


# ============================================================================
# Utility Functions
# ============================================================================

def generate_session_id(role: str) -> str:
    """
    Generate a unique session ID for an agent.
    
    Args:
        role: The agent's role (arch, be, fe, qa)
    
    Returns:
        Session ID in format: agent_YYYYMMDD_HHMMSS_<role>
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"agent_{timestamp}_{role}"


def validate_session_id(session_id: str) -> bool:
    """Validate a session ID format."""
    pattern = r'^agent_\d{8}_\d{6}_(arch|be|fe|qa)$'
    return bool(re.match(pattern, session_id))


def get_role_from_session_id(session_id: str) -> Optional[str]:
    """Extract role from session ID."""
    if not validate_session_id(session_id):
        return None
    return session_id.split('_')[-1]


def initialize_project():
    """Initialize all required files and directories for the multi-agent system."""
    # Create directories
    for dir_path in [LOCKS_DIR, AGENTS_DIR, STATE_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # Initialize files
    _ensure_session_registry()
    _ensure_message_board()
    _ensure_handoff_log()
    _ensure_state_files()
    
    print("Multi-agent system initialized successfully!")
    print(f"  - Locks directory: {LOCKS_DIR}")
    print(f"  - Agents directory: {AGENTS_DIR}")
    print(f"  - State directory: {STATE_DIR}")
    print(f"  - TODO file: {TODO_FILE}")


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    # Initialize the system
    initialize_project()
    
    # Example usage
    print("\n=== Example Usage ===\n")
    
    # Generate session ID
    session_id = generate_session_id("arch")
    print(f"Generated session ID: {session_id}")
    
    # Get available tasks
    available = get_available_tasks()
    print(f"Available tasks: {available}")
    
    # Try to claim a task
    if available:
        task_id = available[0]
        print(f"\nAttempting to claim {task_id}...")
        
        if claim_task(task_id, session_id, "arch"):
            print(f"✓ Successfully claimed {task_id}")
            
            # Start the task
            if start_task(task_id, session_id):
                print(f"✓ Started {task_id}")
                
                # Complete the task
                if complete_task(task_id, "Example completion", session_id):
                    print(f"✓ Completed {task_id}")
        
        # Send a message
        send_message(session_id, f"Completed {task_id} as a test", "COMPLETE", 
                    ["example"], [task_id])
        print(f"✓ Message posted to message board")
    
    print("\n=== System State ===\n")
    state = update_state_snapshot()
    print(json.dumps(state, indent=2))
