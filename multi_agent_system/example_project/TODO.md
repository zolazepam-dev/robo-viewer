# Example Project - Task Tracker

**Project**: Task Management API  
**Last Updated**: 2026-03-09  
**Status**: Ready for agents

---

## Task Summary

| Status | Count |
|--------|-------|
| Unclaimed | 15 |
| Claimed | 0 |
| In Progress | 0 |
| Blocked | 0 |
| Completed | 0 |
| **Total** | **15** |

---

## Tasks

| Task ID | Role | Description | Status | Claimed By | Blocked By | Priority | Acceptance Criteria |
|---------|------|-------------|--------|------------|------------|----------|---------------------|
| ARCH-001 | Architect | Design system architecture | unclaimed | | | P0 | Architecture diagram in docs/, API spec defined |
| ARCH-002 | Architect | Define database schema | unclaimed | | ARCH-001 | P0 | Schema documented, migrations planned |
| ARCH-003 | Architect | Create API specification | unclaimed | | ARCH-001 | P0 | OpenAPI/Swagger spec complete |
| BE-001 | Backend | Set up FastAPI project structure | unclaimed | | ARCH-001 | P0 | Project structure created, dependencies installed |
| BE-002 | Backend | Implement database models | unclaimed | | ARCH-002 | P0 | SQLAlchemy models for Task and User |
| BE-003 | Backend | Create task CRUD endpoints | unclaimed | | BE-001, BE-002 | P0 | GET/POST/PUT/DELETE for /api/tasks |
| BE-004 | Backend | Create user endpoints | unclaimed | | BE-001, BE-002 | P1 | GET/POST for /api/users |
| BE-005 | Backend | Add input validation | unclaimed | | BE-003 | P1 | Pydantic schemas validate all inputs |
| BE-006 | Backend | Implement error handling | unclaimed | | BE-003 | P1 | Consistent error responses, logging |
| FE-001 | Frontend | Set up React project | unclaimed | | ARCH-001 | P0 | React app running, TypeScript configured |
| FE-002 | Frontend | Create API client | unclaimed | | ARCH-003, BE-003 | P0 | TypeScript client for all endpoints |
| FE-003 | Frontend | Build TaskList component | unclaimed | | FE-002 | P0 | Displays tasks, supports filtering |
| FE-004 | Frontend | Build TaskForm component | unclaimed | | FE-002 | P0 | Create/edit task form with validation |
| FE-005 | Frontend | Build Dashboard component | unclaimed | | FE-003, FE-004 | P1 | Shows task statistics, charts |
| QA-001 | QA | Write API integration tests | unclaimed | | BE-003, BE-004 | P0 | All endpoints tested, >80% coverage |
| QA-002 | QA | Write component tests | unclaimed | | FE-003, FE-004 | P1 | All components tested |
| QA-003 | QA | Perform end-to-end testing | unclaimed | | QA-001, QA-002 | P0 | Full user flows tested |
| DOCS-001 | Any | Write API documentation | unclaimed | | BE-005 | P2 | docs/api.md complete with examples |
| DOCS-002 | Any | Create setup guide | unclaimed | | | P2 | README with installation instructions |

---

## Dependency Graph

```
ARCH-001 (Architecture)
    ├── ARCH-002 (Database Schema)
    │   └── BE-002 (Database Models)
    │       ├── BE-003 (Task Endpoints)
    │       │   ├── BE-005 (Input Validation)
    │       │   │   └── DOCS-001 (API Docs)
    │       │   ├── BE-006 (Error Handling)
    │       │   ├── FE-002 (API Client)
    │       │   │   ├── FE-003 (TaskList)
    │       │   │   │   └── FE-005 (Dashboard)
    │       │   │   └── FE-004 (TaskForm)
    │       │   │       └── FE-005 (Dashboard)
    │       │   └── QA-001 (API Tests)
    │       │       └── QA-003 (E2E Tests)
    │       └── BE-004 (User Endpoints)
    │           └── QA-001 (API Tests)
    ├── ARCH-003 (API Spec)
    │   └── FE-002 (API Client)
    └── BE-001 (Project Setup)
        └── BE-002 (Database Models)

FE-001 (React Setup)
    └── FE-002 (API Client)

DOCS-002 (Setup Guide) - Independent
```

---

## Quick Start for Agents

1. **Review the dependency graph** - Claim tasks that are not blocked
2. **Claim your task** using `agent_tools.py`
3. **Complete the work** in `example_project/src/`
4. **Mark task complete** and notify team

---

## Notes

- Tasks are ordered by priority (P0 first)
- Check dependencies before claiming
- Communicate blockers immediately
- Update TODO.md when completing tasks
