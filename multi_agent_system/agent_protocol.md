# Multi-Agent Coordination Protocol

## Overview

This protocol defines how AI agents coordinate through shared files to complete software development tasks without conflicts. Each agent runs in a separate Qwen Code instance and communicates asynchronously through the file system.

---

## Session ID Format

**Format**: `agent_YYYYMMDD_HHMMSS_<role>`

**Examples**:
- `agent_20260309_143022_arch` - Architect agent started March 9, 2026 at 14:30:22
- `agent_20260309_150045_be` - Backend developer agent
- `agent_20260309_151230_fe` - Frontend developer agent
- `agent_20260309_160000_qa` - QA testing agent

**Components**:
- `agent_` - Prefix identifying this as an agent session
- `YYYYMMDD` - Date (year, month, day)
- `HHMMSS` - Time (hour, minute, second) in 24-hour format
- `<role>` - Role abbreviation (arch, be, fe, qa)

---

## Task Lifecycle

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ UNCLAIMED   │────►│  CLAIMED    │────►│ IN_PROGRESS │────►│  COMPLETED  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       ▲                                       │
       │                                       ▼
       │                                 ┌─────────────┐
       └─────────────────────────────────│  BLOCKED    │
                                         └─────────────┘
```

### State Transitions

| From | To | Trigger | Required Fields |
|------|-----|---------|-----------------|
| UNCLAIMED | CLAIMED | Agent claims task | `claimed_by`, `claimed_at` |
| CLAIMED | IN_PROGRESS | Agent starts work | `started_at` |
| IN_PROGRESS | COMPLETED | Agent finishes task | `completed_at`, `completion_notes` |
| IN_PROGRESS | BLOCKED | Agent encounters blocker | `blocked_by`, `block_reason`, `blocked_at` |
| BLOCKED | IN_PROGRESS | Blocker resolved | `unblocked_at` |
| BLOCKED | UNCLAIMED | Task abandoned | `abandoned_at`, `abandon_reason` |
| COMPLETED | UNCLAIMED | Task reopened | `reopened_at`, `reopen_reason` |

---

## Task Claiming Protocol

### Step 1: Check Task Availability

Before claiming, agent MUST:
1. Read `TODO.md` and verify task status is `unclaimed`
2. Check `agents/session_registry.json` for conflicting claims (within last 30 seconds)
3. Verify role matches task requirements (or agent has override permission)

### Step 2: Atomic Claim Operation

```python
# Pseudocode for atomic claim
def claim_task(task_id, agent_id, role):
    acquire_file_lock("TODO.md")
    try:
        # Re-verify task is unclaimed (double-check pattern)
        task = read_task(task_id)
        if task.status != "unclaimed":
            return False
        
        # Update task
        task.status = "claimed"
        task.claimed_by = agent_id
        task.claimed_at = current_timestamp()
        task.claimed_role = role
        write_task(task)
        
        # Log claim event
        log_event("CLAIM", task_id, agent_id)
        
        return True
    finally:
        release_file_lock("TODO.md")
```

### Step 3: Registration

After successful claim:
1. Update `agents/session_registry.json` with task assignment
2. Post message to `agents/message_board.md` announcing claim
3. Update `state/current_state.json` task counters

---

## Conflict Resolution

### File Lock Conflicts

**Timeout**: 30 seconds maximum lock hold time

**Resolution Strategy**:
1. **First-Come-First-Served**: Agent holding lock has priority
2. **Lock Timeout**: If lock held >30s without activity, next agent can force-release
3. **Stale Lock Detection**: Check `session_registry.json` - if agent session expired, lock can be broken

### Task Claim Conflicts

If two agents attempt to claim same task simultaneously:

1. **Atomic Check**: File locking ensures only one succeeds
2. **Losing Agent**: Must:
   - Read updated TODO.md
   - Select alternative task
   - Log conflict in `message_board.md`

### Role Conflicts

If agent claims task outside their role:

1. **Validation**: `agent_tools.py` validates role compatibility
2. **Override**: Only Architect role can override role constraints
3. **Escalation**: Log to `message_board.md` with `@arch` mention

---

## Communication Through Files

### Message Board Protocol

**Location**: `agents/message_board.md`

**Message Format**:
```markdown
## [TIMESTAMP] [AGENT_ID] [TYPE]

Message content here.

**Tags**: #coordination #help-needed #info
**Related Tasks**: TASK-001, TASK-002
```

**Message Types**:
- `INFO` - General information
- `HELP` - Request for assistance
- `HANDOFF` - Task handoff to another agent
- `BLOCKER` - Blocking issue announcement
- `DECISION` - Architectural decision
- `COMPLETE` - Task completion announcement

### Handoff Protocol

When transferring work between agents:

1. **Initiating Agent**:
   - Complete current work to stable checkpoint
   - Write handoff entry in `agents/handoff_log.md`
   - Post `HANDOFF` message on message board
   - Tag receiving agent's role (e.g., `@backend`)

2. **Receiving Agent**:
   - Acknowledge handoff within 5 minutes
   - Review `handoff_log.md` entry
   - Claim follow-up tasks
   - Post acknowledgment on message board

**Handoff Log Entry**:
```markdown
### HANDOFF-[SEQUENTIAL_ID]

**From**: agent_20260309_143022_arch
**To**: agent_20260309_150045_be
**Timestamp**: 2026-03-09 15:45:00
**Related Tasks**: TASK-001, TASK-002
**Context**: 
  - Completed: Architecture design
  - In Progress: API specification 80% complete
  - Next Steps: Implement REST endpoints
**Artifacts**: docs/architecture.md, api/spec.yaml
**Notes**: Pay attention to authentication flow in section 3.2
```

---

## File Locking Mechanism

### Lock File Format

**Location**: `locks/<filename>.lock`

```json
{
  "locked_by": "agent_20260309_143022_arch",
  "locked_at": "2026-03-09T14:30:22Z",
  "operation": "edit",
  "pid": 12345
}
```

### Lock Acquisition

```python
def acquire_lock(filename, agent_id, timeout=30):
    lock_file = f"locks/{filename}.lock"
    start_time = time.time()
    
    while True:
        try:
            # Try to create lock file atomically
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w') as f:
                json.dump({
                    "locked_by": agent_id,
                    "locked_at": datetime.now().isoformat(),
                    "operation": "edit",
                    "pid": os.getpid()
                }, f)
            return True
        except FileExistsError:
            # Check if lock is stale
            if is_lock_stale(lock_file, timeout):
                break_lock(lock_file)
                continue
            
            # Wait and retry
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.5)
```

### Lock Release

```python
def release_lock(filename, agent_id):
    lock_file = f"locks/{filename}.lock"
    try:
        with open(lock_file) as f:
            lock_data = json.load(f)
        
        if lock_data["locked_by"] != agent_id:
            raise PermissionError("Cannot release another agent's lock")
        
        os.remove(lock_file)
    except FileNotFoundError:
        pass  # Lock already released
```

---

## State Machine

### Agent State

```python
class AgentState(Enum):
    INITIALIZING = "initializing"
    IDLE = "idle"
    WORKING = "working"
    BLOCKED = "blocked"
    OFFLINE = "offline"
```

### Task State

```python
class TaskState(Enum):
    UNCLAIMED = "unclaimed"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    REOPENED = "reopened"
```

---

## Error Handling

### Common Error Scenarios

| Error | Recovery Action |
|-------|-----------------|
| Lock timeout | Force-release stale lock, log warning |
| Task already claimed | Select alternative task, notify user |
| File corruption | Restore from last known good state in `state/` |
| Agent crash | Mark tasks as unclaimed after session timeout |
| Race condition | Retry with exponential backoff |

### Error Logging

All errors MUST be logged to:
1. `state/blockers.md` - If blocking progress
2. `agents/message_board.md` - If affecting other agents
3. Console output - For immediate visibility

---

## Security Considerations

### Agent Authentication

- Session ID serves as authentication token
- Agents can only modify files within their permissions
- Role constraints enforced at protocol level

### File Permissions

| Role | Read | Write | Lock Break | Override |
|------|------|-------|------------|----------|
| Architect | All | All | Yes | Yes |
| Backend | All | Backend tasks, shared | No | No |
| Frontend | All | Frontend tasks, shared | No | No |
| QA | All | Test files, bug reports | No | No |

---

## Performance Optimization

### Batch Operations

When updating multiple tasks:
1. Acquire single lock on TODO.md
2. Perform all updates
3. Release lock
4. Post single summary message

### Caching

Agents SHOULD:
- Cache TODO.md state for up to 30 seconds
- Invalidate cache on lock acquisition
- Refresh cache after completing task

---

## Protocol Version

**Current Version**: 1.0.0

**Changelog**:
- 1.0.0 (2026-03-09): Initial protocol definition

---

## Appendix: Quick Reference

### Session ID Generation
```bash
echo "agent_$(date +%Y%m%d_%H%M%S)_${ROLE}"
```

### Task Status Values
- `unclaimed` - Available for claiming
- `claimed` - Reserved by agent
- `in_progress` - Actively being worked on
- `blocked` - Waiting on dependency
- `completed` - Done and verified

### Role Abbreviations
- `arch` - Architect
- `be` - Backend Developer
- `fe` - Frontend Developer
- `qa` - QA/Test Specialist

### File Lock Locations
```
multi_agent_system/
└── locks/
    ├── TODO.md.lock
    ├── message_board.md.lock
    └── current_state.json.lock
```
