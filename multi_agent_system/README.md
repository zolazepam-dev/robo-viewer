# Multi-Agent AI Collaboration System

A practical file-based coordination system enabling multiple AI agents (running in separate Qwen Code instances) to collaborate on software development tasks without conflicts.

## 🎯 Overview

This system provides:
- **Task Management**: Claim, track, and complete tasks with dependency handling
- **File Locking**: Prevent concurrent edit conflicts
- **Role-Based Access**: Architect, Backend, Frontend, and QA roles
- **Async Communication**: Message boards, handoff logs, decision tracking
- **State Management**: Real-time project state snapshots

## 🚀 Quick Start

### 1. Launch an Agent Session

```bash
cd multi_agent_system
./launch_agent.sh <role>
```

Available roles:
- `arch` - System Architect
- `be` - Backend Developer  
- `fe` - Frontend Developer
- `qa` - QA/Test Specialist

### 2. Copy the Context Prompt

The launcher will display a context prompt. Copy it into your Qwen Code instance.

### 3. Claim a Task

```python
from agent_tools import claim_task, start_task, complete_task

# Claim an available task
claim_task("ARCH-001", "agent_20260309_143022_arch", "arch")

# Start working
start_task("ARCH-001", "agent_20260309_143022_arch")

# When done
complete_task("ARCH-001", "Implementation complete", "agent_20260309_143022_arch")
```

### 4. Communicate with Team

```python
from agent_tools import send_message

send_message("agent_20260309_143022_arch", 
             "ARCH-001 complete! Ready for backend implementation", 
             "COMPLETE",
             tags=["handoff"],
             related_tasks=["ARCH-001", "BE-001"])
```

## 📁 Project Structure

```
multi_agent_system/
├── agent_protocol.md       # Coordination protocol
├── TODO.md                 # Task tracker
├── agent_tools.py          # Core utility functions
├── launch_agent.sh         # Agent launcher script
├── README.md               # This file
│
├── roles/                  # Role definitions
│   ├── role_architect.md
│   ├── role_backend.md
│   ├── role_frontend.md
│   └── role_testing.md
│
├── agents/                 # Communication channels
│   ├── message_board.md    # Async messaging
│   ├── session_registry.json # Active sessions
│   └── handoff_log.md      # Task handoffs
│
├── state/                  # Project state tracking
│   ├── current_state.json  # State snapshot
│   ├── decision_log.md     # Architectural decisions
│   └── blockers.md         # Current blockers
│
├── locks/                  # File locks (auto-created)
│
├── tests/                  # Test suite
│   └── test_agent_tools.py
│
└── example_project/        # Sample project for agents
    ├── README.md
    ├── TODO.md
    └── src/
```

## 🎭 Agent Roles

### System Architect (`arch`)
- High-level design decisions
- Task decomposition and planning
- Unblocking team members
- Technical leadership

### Backend Developer (`be`)
- API implementation
- Database design
- Business logic
- Integration with external services

### Frontend Developer (`fe`)
- UI component development
- User experience implementation
- API integration
- Responsive design

### QA/Test Specialist (`qa`)
- Unit, integration, and E2E tests
- Quality validation
- Bug identification
- Test infrastructure

## 🔧 Core Functions

### Task Management

```python
from agent_tools import (
    claim_task,      # Claim a task
    start_task,      # Mark task as in progress
    complete_task,   # Mark task as completed
    block_task,      # Mark task as blocked
    unblock_task,    # Remove block from task
    read_task,       # Read task details
    get_available_tasks,  # Get unclaimed tasks
    get_task_dependencies, # Get task dependencies
)
```

### Session Management

```python
from agent_tools import (
    register_session,    # Register agent session
    unregister_session,  # Unregister session
    update_heartbeat,    # Update session heartbeat
    get_active_sessions, # Get active agent sessions
)
```

### Communication

```python
from agent_tools import (
    send_message,    # Post to message board
    log_handoff,     # Log task handoff
)
```

### File Locking

```python
from agent_tools import (
    acquire_lock,        # Acquire exclusive lock
    release_lock,        # Release lock
    is_file_locked,      # Check if file is locked
    acquire_file_lock,   # Context manager for locking
)
```

## 📋 Task Lifecycle

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

## 🔒 File Locking

The system uses atomic file locking to prevent conflicts:

```python
# Using context manager (recommended)
with acquire_file_lock("TODO.md", agent_id):
    # Safe to edit TODO.md
    pass

# Manual locking
if acquire_lock("file.txt", agent_id):
    try:
        # Edit file
        pass
    finally:
        release_lock("file.txt", agent_id)
```

**Lock Properties**:
- Timeout: 30 seconds
- Stale lock detection: Automatic
- Atomic acquisition: Yes (O_CREAT | O_EXCL)

## 📊 Session IDs

Each agent gets a unique session ID:

**Format**: `agent_YYYYMMDD_HHMMSS_<role>`

**Examples**:
- `agent_20260309_143022_arch` - Architect
- `agent_20260309_150045_be` - Backend
- `agent_20260309_160000_fe` - Frontend
- `agent_20260309_170000_qa` - QA

## 🧪 Running Tests

```bash
cd multi_agent_system
python -m pytest tests/test_agent_tools.py -v
```

**Test Coverage**:
- File locking mechanisms
- Task management operations
- Concurrent access patterns
- Session registry
- Communication channels

## 📖 Example Workflow

### Multi-Agent Collaboration

```
1. Architect launches: ./launch_agent.sh arch
   - Claims ARCH-001 (design architecture)
   - Creates docs/architecture.md
   - Completes task, posts to message board

2. Backend launches: ./launch_agent.sh be
   - Sees ARCH-001 complete
   - Claims BE-001 (implement API)
   - Implements endpoints
   - Hands off to frontend

3. Frontend launches: ./launch_agent.sh fe
   - Reviews API documentation
   - Claims FE-001 (build UI)
   - Implements components
   - Requests QA testing

4. QA launches: ./launch_agent.sh qa
   - Writes integration tests
   - Validates all functionality
   - Marks project complete
```

## 🛠 Troubleshooting

### Common Issues

**Issue**: "Could not acquire lock"
- **Cause**: Another agent holds the lock
- **Solution**: Wait and retry, or check for stale locks

**Issue**: "Task already claimed"
- **Cause**: Another agent claimed the task
- **Solution**: Select a different available task

**Issue**: "Session not found"
- **Cause**: Session timed out (5 minutes)
- **Solution**: Re-run `launch_agent.sh` to refresh

**Issue**: "Task is blocked"
- **Cause**: Dependency not complete
- **Solution**: Wait for blocking task to complete

### Debug Commands

```bash
# Check active sessions
python -c "from agent_tools import get_active_sessions; print(get_active_sessions())"

# Check file lock status
python -c "from agent_tools import is_file_locked; print(is_file_locked('TODO.md'))"

# Get available tasks for role
python -c "from agent_tools import get_available_tasks; print(get_available_tasks('arch'))"

# View current state
cat state/current_state.json | python -m json.tool
```

## 📝 Best Practices

### For AI Agents

1. **Always claim tasks first** - Never work on unclaimed tasks
2. **Release locks promptly** - Don't hold locks longer than necessary
3. **Communicate frequently** - Post updates to message board
4. **Document decisions** - Use decision log for architectural choices
5. **Report blockers immediately** - Don't stay blocked silently

### For Humans

1. **Observe before intervening** - Let agents work through problems
2. **Use message board for feedback** - Keep communication async
3. **Review decision log** - Understand architectural evolution
4. **Check blockers file** - Identify systemic issues

## 🔗 Integration with Qwen Code

This system is designed for Qwen Code but can work with any AI assistant that:
- Can execute Python code
- Can read/write files
- Can run shell commands

### Setup for Multiple Instances

1. Open multiple Qwen Code instances
2. Launch different roles in each:
   - Instance 1: `./launch_agent.sh arch`
   - Instance 2: `./launch_agent.sh be`
   - Instance 3: `./launch_agent.sh fe`
   - Instance 4: `./launch_agent.sh qa`
3. Each agent works independently, coordinating through files

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `agent_protocol.md` | Detailed coordination protocol |
| `roles/role_*.md` | Role-specific responsibilities |
| `TODO.md` | Task tracker with dependencies |
| `state/decision_log.md` | Architectural decisions |
| `state/blockers.md` | Current blocking issues |
| `agents/message_board.md` | Team communication |
| `agents/handoff_log.md` | Task handoffs |

## 🎯 Example Project

The `example_project/` directory contains a sample Task Management API project for agents to build:

- **Backend**: FastAPI REST API
- **Frontend**: React TypeScript UI
- **Database**: SQLite
- **Tests**: Pytest + React Testing Library

See `example_project/README.md` for details.

## 📈 Metrics

Track project health via `state/current_state.json`:

```json
{
  "task_counts": {
    "unclaimed": 5,
    "claimed": 2,
    "in_progress": 3,
    "blocked": 1,
    "completed": 10
  },
  "active_agents": 4,
  "current_blockers": 1
}
```

## 🔐 Security Considerations

- Session IDs serve as authentication
- File permissions enforced by role
- Lock breaking requires authorization
- All operations logged for audit

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Test changes with multiple agents
2. Update documentation
3. Add tests for new features
4. Follow existing code style

---

**Built for the future of AI-assisted software development** 🚀
