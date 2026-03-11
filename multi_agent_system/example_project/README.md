# Example Project: Task Management API

**Purpose**: Demonstrate multi-agent collaboration on a real software project  
**Goal**: Build a RESTful task management API with frontend dashboard  
**Tech Stack**: Python FastAPI backend, React frontend, SQLite database

---

## Project Overview

Build a simple task management system where users can:
- Create, read, update, and delete tasks
- Assign tasks to users
- Track task status and priority
- View tasks on a dashboard

This project is designed to be completed by multiple AI agents working in parallel.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React UI      │────►│  FastAPI Backend│────►│   SQLite DB     │
│   (Frontend)    │     │   (REST API)    │     │   (Storage)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## File Structure

```
example_project/
├── README.md              # This file
├── TODO.md                # Task tracker (pre-populated)
├── src/
│   ├── backend/
│   │   ├── main.py        # FastAPI application
│   │   ├── models.py      # Database models
│   │   ├── schemas.py     # Pydantic schemas
│   │   ├── database.py    # Database connection
│   │   └── api/
│   │       ├── tasks.py   # Task endpoints
│   │       └── users.py   # User endpoints
│   └── frontend/
│       ├── App.tsx        # Main React component
│       ├── components/
│       │   ├── TaskList.tsx
│       │   ├── TaskForm.tsx
│       │   └── Dashboard.tsx
│       └── api/
│           └── client.ts  # API client
├── tests/
│   ├── test_api.py        # API tests
│   └── test_frontend.tsx  # Component tests
└── docs/
    ├── api.md             # API documentation
    └── architecture.md    # Architecture decisions
```

---

## Getting Started

### For Agents

1. **Launch your agent session**:
   ```bash
   cd multi_agent_system
   ./launch_agent.sh <your_role>
   ```

2. **Review the TODO.md** for available tasks

3. **Claim a task** using `agent_tools.py`:
   ```python
   from agent_tools import claim_task
   claim_task("ARCH-001", "agent_YYYYMMDD_HHMMSS_arch", "arch")
   ```

4. **Complete the task** and mark it done:
   ```python
   from agent_tools import complete_task
   complete_task("ARCH-001", "Implementation complete", "agent_YYYYMMDD_HHMMSS_arch")
   ```

### For Humans

1. Observe how AI agents collaborate
2. Intervene only when agents are stuck
3. Review completed work
4. Provide feedback via message board

---

## Success Criteria

The project is complete when:
- [ ] All P0 tasks completed
- [ ] API endpoints functional and tested
- [ ] Frontend displays tasks correctly
- [ ] All tests passing (>80% coverage)
- [ ] Documentation complete

---

## Agent Workflow Example

```
1. Architect claims ARCH-001 (design architecture)
   ↓ (completes task, posts decision log)
2. Backend claims BE-001 (implement API)
   ↓ (depends on ARCH-001)
3. Frontend claims FE-001 (build UI components)
   ↓ (depends on BE-001 API spec)
4. QA claims QA-001 (write tests)
   ↓ (validates all work)
5. Project complete!
```

---

## Communication

Agents should communicate via:
- **Message Board**: `../agents/message_board.md`
- **Handoff Log**: `../agents/handoff_log.md`
- **Decision Log**: `../state/decision_log.md`
- **Blockers**: `../state/blockers.md`

---

## Notes

- This is a simplified example for demonstration
- Real projects would have more complex requirements
- The focus is on agent coordination, not project complexity
- Feel free to extend with additional features
