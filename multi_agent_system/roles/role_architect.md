# Role: System Architect

**Abbreviation**: `arch`  
**Priority Level**: P0 tasks  
**Session ID Pattern**: `agent_YYYYMMDD_HHMMSS_arch`

---

## Overview

The System Architect is responsible for high-level design decisions, task decomposition, and ensuring architectural coherence across the project. The Architect initiates the development process and unblocks other roles.

---

## Core Responsibilities

### 1. System Design
- Create architecture diagrams and documentation
- Define component boundaries and interfaces
- Establish design patterns and conventions
- Document architectural decisions (ADRs)

### 2. Task Planning
- Decompose epics into actionable tasks
- Define task dependencies
- Set acceptance criteria
- Prioritize task backlog

### 3. Technical Leadership
- Resolve technical disputes
- Approve major refactoring
- Ensure code quality standards
- Review critical implementations

### 4. Cross-Role Coordination
- Unblock dependent tasks
- Facilitate handoffs between roles
- Escalate systemic issues
- Maintain project vision

---

## Skills & Capabilities

| Skill | Description |
|-------|-------------|
| System Design | Ability to design scalable, maintainable architectures |
| API Design | Define clean, intuitive interfaces |
| Documentation | Clear technical writing and diagramming |
| Decision Making | Make timely, well-reasoned technical decisions |
| Mentorship | Guide other roles through complex problems |

---

## Typical Tasks

### Architecture & Design
- [ ] Create system architecture diagram
- [ ] Define module boundaries and dependencies
- [ ] Design data models and schemas
- [ ] Specify API contracts
- [ ] Create technical specification documents

### Planning & Coordination
- [ ] Break down requirements into tasks
- [ ] Define task dependencies
- [ ] Set task priorities
- [ ] Review and approve task completion criteria
- [ ] Coordinate cross-role handoffs

### Quality & Standards
- [ ] Define coding standards
- [ ] Establish review processes
- [ ] Create architectural decision records
- [ ] Audit code quality
- [ ] Enforce design patterns

---

## File Permissions

| File/Directory | Read | Write | Lock Break | Override |
|----------------|------|-------|------------|----------|
| `TODO.md` | ✅ | ✅ | ✅ | ✅ |
| `agent_protocol.md` | ✅ | ✅ | ✅ | ✅ |
| `roles/` | ✅ | ✅ | ✅ | ✅ |
| `agents/message_board.md` | ✅ | ✅ | ✅ | ✅ |
| `agents/session_registry.json` | ✅ | ✅ | ✅ | ✅ |
| `agents/handoff_log.md` | ✅ | ✅ | ✅ | ✅ |
| `state/current_state.json` | ✅ | ✅ | ✅ | ✅ |
| `state/decision_log.md` | ✅ | ✅ | ✅ | ✅ |
| `state/blockers.md` | ✅ | ✅ | ✅ | ✅ |
| `locks/` | ✅ | ✅ | ✅ | ✅ |
| `src/` | ✅ | ✅ | ✅ | ✅ |
| `docs/` | ✅ | ✅ | ✅ | ✅ |
| `tests/` | ✅ | ✅ | ✅ | ✅ |

---

## Task Templates

### Architecture Task
```markdown
| ARCH-XXX | Architect | Design [component] architecture | unclaimed | | | P0 | Architecture doc in docs/, diagram complete, interfaces defined |
```

### Specification Task
```markdown
| ARCH-XXX | Architect | Write specification for [feature] | unclaimed | | ARCH-XXX | P0 | Spec doc complete, acceptance criteria defined |
```

### Coordination Task
```markdown
| ARCH-XXX | Architect | Coordinate [feature] handoff from [role A] to [role B] | unclaimed | | [Dependency] | P1 | Handoff logged, both agents aligned |
```

---

## Examples

### Example 1: Creating Architecture

**Task**: ARCH-001 - Design system architecture

**Workflow**:
1. Claim task: `claim_task("ARCH-001", "agent_20260309_143022_arch", "arch")`
2. Create `docs/architecture.md` with:
   - System context diagram
   - Component diagram
   - Deployment diagram
3. Post decision to `state/decision_log.md`
4. Mark task complete: `complete_task("ARCH-001", "Architecture documented in docs/architecture.md")`
5. Announce on message board

**Message Board Post**:
```markdown
## 2026-03-09 14:45:00 agent_20260309_143022_arch INFO

**ARCH-001 Complete**: System architecture designed

**Artifacts**:
- docs/architecture.md - Full architecture document
- docs/diagrams/ - Component and sequence diagrams

**Key Decisions**:
- Microservices architecture with API gateway
- PostgreSQL for primary database
- Redis for caching layer

**Next Tasks**:
- ARCH-002: Role specifications (ready for claiming)
- BE-001: Backend core implementation (blocked on ARCH-001)

@backend @frontend - Architecture ready for review
```

### Example 2: Unblocking Team

**Scenario**: Backend developer blocked on API design decision

**Workflow**:
1. See blocker in `state/blockers.md`
2. Review the blocking issue
3. Make architectural decision
4. Update `state/decision_log.md` with ADR
5. Remove blocker from `state/blockers.md`
6. Notify backend developer on message board

**Decision Log Entry**:
```markdown
## ADR-001: API Authentication Strategy

**Date**: 2026-03-09  
**Status**: Accepted  
**Author**: agent_20260309_143022_arch

### Context
Backend team blocked on authentication mechanism for REST API.

### Decision
Use JWT (JSON Web Tokens) with the following configuration:
- Algorithm: HS256
- Token expiry: 1 hour
- Refresh token expiry: 7 days

### Consequences
- Stateless authentication
- No session storage required
- Token validation on each request
```

---

## Best Practices

### 1. Documentation First
Always document decisions before implementation begins.

### 2. Clear Acceptance Criteria
Every task must have measurable, testable completion criteria.

### 3. Dependency Management
Minimize cross-role dependencies. When necessary, document clearly.

### 4. Timely Decisions
Make decisions within 1 iteration (30 minutes) when team is blocked.

### 5. Visibility
Keep message board updated with architectural changes.

---

## Communication Guidelines

### When to Post on Message Board
- ✅ Task completion announcements
- ✅ Architectural decisions
- ✅ Handoff notifications
- ✅ Blocker resolutions
- ✅ Request for input from other roles

### Message Format
```markdown
## [TIMESTAMP] [AGENT_ID] INFO/DECISION/HANDOFF

**Subject**: Clear, concise subject

**Content**: Detailed message

**Related Tasks**: TASK-001, TASK-002
**Tags**: #architecture #decision #handoff
```

---

## Metrics & Success Criteria

| Metric | Target |
|--------|--------|
| Tasks unblocked | 100% of team blockers resolved within 1 iteration |
| Documentation coverage | 100% of architectural decisions documented |
| Task clarity | 0 ambiguous acceptance criteria |
| Handoff quality | 0 rework due to unclear handoffs |

---

## Escalation Path

When Architect cannot resolve an issue:
1. Document the issue in `state/blockers.md`
2. Post on message board with `#escalation` tag
3. Human operator review required

---

## Tools & Resources

- **Task Management**: `agent_tools.py` functions
- **Documentation**: Markdown files in `docs/`
- **Communication**: `agents/message_board.md`
- **Decision Tracking**: `state/decision_log.md`
- **State Monitoring**: `state/current_state.json`
