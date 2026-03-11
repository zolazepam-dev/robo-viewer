# 🚀 Practical Multi-Agent Setup - Quick Guide

## What This Is

A system for **multiple AI agents** (you + me in separate Qwen Code windows) to **collaborate on real projects** without stepping on each other's toes.

---

## 🎯 The Problem

You want **4 AI agents** working together:
- **Agent 1 (Architect)**: Designs system architecture
- **Agent 2 (Backend)**: Implements APIs and databases
- **Agent 3 (Frontend)**: Builds UI components
- **Agent 4 (QA)**: Writes tests and validates

But we can't all edit the same files at the same time without conflicts!

---

## ✅ The Solution

**File-based coordination** with:
- **TODO.md** - Task tracker with claiming system
- **File Locks** - Prevent concurrent edits
- **Message Board** - Async communication
- **Role Constraints** - Only architects can claim architecture tasks

---

## 🛠 How to Actually Use This

### Step 1: Open Multiple Qwen Code Windows

You need **separate Qwen Code instances** (different browser windows or tabs):

```
Window 1: Qwen Code - Architect Agent
Window 2: Qwen Code - Backend Agent  
Window 3: Qwen Code - Frontend Agent
Window 4: Qwen Code - QA Agent
```

### Step 2: Launch Each Agent

In **Window 1** (Architect):
```bash
cd /media/cammyz/EverythingHere/robo-viewer/multi_agent_system
./launch_agent.sh arch
```

Copy the context prompt it gives you and paste it back into Window 1.

In **Window 2** (Backend):
```bash
cd /media/cammyz/EverythingHere/robo-viewer/multi_agent_system
./launch_agent.sh be
```

Repeat for Frontend (`fe`) and QA (`qa`).

### Step 3: Each Agent Claims Tasks

**Architect Agent** (Window 1):
```python
from agent_tools import claim_task, complete_task

# Claim architecture tasks
claim_task("ARCH-001", "agent_20260309_143022_arch", "arch")
claim_task("ARCH-002", "agent_20260309_143022_arch", "arch")

# Work on them...

# Mark complete
complete_task("ARCH-001", "Architecture designed", "agent_20260309_143022_arch")
```

**Backend Agent** (Window 2):
```python
from agent_tools import claim_task, get_available_tasks

# Wait for ARCH-001 to complete, then claim backend tasks
available = get_available_tasks("be")  # Shows BE-001, BE-002, etc.
claim_task("BE-001", "agent_20260309_150045_be", "be")
```

### Step 4: Communicate Through Message Board

**Architect** leaves a message:
```python
from agent_tools import send_message

send_message(
    "agent_20260309_143022_arch",
    "Architecture complete! API specs in docs/api.md. Ready for backend implementation.",
    "HANDOFF",
    tags=["backend", "ready"],
    related_tasks=["ARCH-001"]
)
```

**Backend Agent** checks messages:
```python
from agent_tools import read_messages

messages = read_messages(tags=["backend"])
# Sees architect's message and knows to start
```

---

## 📋 Example Workflow

### Phase 1: Architecture (Agent 1)

```
Window 1 - Architect Agent:
1. Claims ARCH-001 (Design system architecture)
2. Creates docs/architecture.md
3. Designs API structure
4. Completes ARCH-001
5. Sends message: "Architecture done!"
```

### Phase 2: Backend (Agent 2)

```
Window 2 - Backend Agent:
1. Sees ARCH-001 complete in TODO.md
2. Claims BE-001 (Implement user API)
3. Reads docs/architecture.md for specs
4. Creates src/api/users.py
5. Completes BE-001
6. Sends message: "User API ready for frontend!"
```

### Phase 3: Frontend (Agent 3)

```
Window 3 - Frontend Agent:
1. Sees BE-001 complete
2. Claims FE-001 (Build user UI)
3. Reads API docs
4. Creates src/components/UserList.tsx
5. Completes FE-001
```

### Phase 4: QA (Agent 4)

```
Window 4 - QA Agent:
1. Sees BE-001 and FE-001 complete
2. Claims TST-001 (Write integration tests)
3. Creates tests/test_users.py
4. Finds bug! Creates blocker for BE-001
5. Backend fixes bug
6. QA completes TST-001
```

---

## 🔒 File Locking (Prevents Conflicts)

When an agent edits a file:

```python
from agent_tools import acquire_lock, release_lock

# Lock the file before editing
if acquire_lock("src/api/users.py", "agent_20260309_150045_be"):
    # Safe to edit - no one else can lock it
    # ... edit file ...
    release_lock("src/api/users.py", "agent_20260309_150045_be")
else:
    # Someone else is editing - wait or work on something else
    print("File locked by agent_20260309_143022_arch")
```

---

## 🎭 Role Definitions

### Architect (`arch`)
- **Skills**: System design, API specification, database schema
- **Can Claim**: ARCH-* tasks
- **Override**: Can claim any task if blocked
- **Example Tasks**:
  - ARCH-001: Design system architecture
  - ARCH-002: Define API contracts
  - ARCH-003: Database schema design

### Backend Developer (`be`)
- **Skills**: API implementation, database queries, business logic
- **Can Claim**: BE-* tasks
- **Example Tasks**:
  - BE-001: Implement user authentication API
  - BE-002: Create database models
  - BE-003: Add caching layer

### Frontend Developer (`fe`)
- **Skills**: UI components, state management, API integration
- **Can Claim**: FE-* tasks
- **Example Tasks**:
  - FE-001: Build user list component
  - FE-002: Implement login form
  - FE-003: Add routing

### QA Specialist (`qa`)
- **Skills**: Test writing, bug finding, validation
- **Can Claim**: TST-* tasks
- **Example Tasks**:
  - TST-001: Write unit tests for user API
  - TST-002: Integration tests for login flow
  - TST-003: Performance testing

---

## 📊 Monitoring Progress

Check TODO.md anytime:
```bash
cat TODO.md
```

Check active agents:
```bash
cat agents/session_registry.json
```

Check messages:
```bash
cat agents/message_board.md
```

Check blockers:
```bash
cat state/blockers.md
```

---

## 🎯 Real Example: Building a Task Management API

### Starting State
```
TODO.md has 19 tasks:
- 4 Architecture tasks (ARCH-001 to ARCH-004)
- 6 Backend tasks (BE-001 to BE-006)
- 5 Frontend tasks (FE-001 to FE-005)
- 4 Testing tasks (TST-001 to TST-004)
```

### After 1 Hour (4 Agents Working)
```
TODO.md shows:
✅ ARCH-001, ARCH-002, ARCH-003, ARCH-004 - Complete
✅ BE-001, BE-002, BE-003 - Complete
🔄 BE-004 - In Progress (agent_20260309_150045_be)
⏳ FE-001 - Blocked by BE-003
⏳ TST-001 - Blocked by BE-001
```

### After 3 Hours
```
✅ All 19 tasks complete!
Project built successfully.
Tests passing.
```

---

## 🆘 Troubleshooting

### Problem: Two agents edit same file

**Solution**: File locking prevents this! Second agent gets:
```
ERROR: File locked by agent_20260309_143022_arch
Lock acquired at: 2026-03-09 14:30:22
Timeout: 300 seconds
```

### Problem: Agent claims task but goes idle

**Solution**: Stale lock detection (5 minute timeout)
```python
# Locks auto-release after timeout
# Other agents can force-release stale locks
```

### Problem: Task dependencies not clear

**Solution**: Check TODO.md "Blocked By" column
```markdown
| FE-001 | Frontend | Build user UI | unclaimed | | BE-003 |
```
FE-001 blocked until BE-003 complete.

---

## 💡 Best Practices

1. **Always claim tasks first** - Don't start work without claiming
2. **Lock files before editing** - Prevent conflicts
3. **Leave detailed completion notes** - Help other agents
4. **Use message board for handoffs** - Async communication
5. **Update blockers immediately** - Don't sit on blocked tasks
6. **Check session registry** - See who else is active

---

## 🚀 Try It Now!

### Single Agent Test
```bash
cd multi_agent_system
./launch_agent.sh arch
# Follow the prompts
# Claim and complete ARCH-001
```

### Multi-Agent Simulation (You Play All Roles)
```bash
# Terminal 1 - Architect
./launch_agent.sh arch

# Terminal 2 - Backend
./launch_agent.sh be

# Terminal 3 - Frontend
./launch_agent.sh fe

# Terminal 4 - QA
./launch_agent.sh qa
```

### Real Multi-Agent (Multiple Qwen Instances)
1. Open 4 browser windows with Qwen Code
2. In each window, run `./launch_agent.sh <role>`
3. Paste the context prompt
4. Watch them collaborate!

---

## 📚 What's Next?

The system is ready! You can:

1. **Use the example project** in `example_project/`
2. **Create your own project** with custom TODO.md
3. **Add more roles** (DevOps, Security, etc.)
4. **Extend agent_tools.py** with custom functions

The framework is generic - use it for any software project!

---

**Built with ❤️ for Multi-Agent Collaboration**
