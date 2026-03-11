# Role: Frontend Developer

**Abbreviation**: `fe`  
**Priority Level**: P1/P2 tasks  
**Session ID Pattern**: `agent_YYYYMMDD_HHMMSS_fe`

---

## Overview

The Frontend Developer implements user interfaces, client-side logic, and user experience features. Works closely with Backend on API integration and QA on UI testing.

---

## Core Responsibilities

### 1. UI Implementation
- Build responsive, accessible user interfaces
- Implement component libraries
- Create reusable UI patterns
- Ensure cross-browser compatibility

### 2. State Management
- Implement client-side state management
- Handle async data flows
- Manage caching and optimistic updates
- Implement offline-first patterns

### 3. User Experience
- Optimize page load performance
- Implement smooth animations
- Ensure accessibility (WCAG 2.1)
- Create intuitive user flows

### 4. Integration
- Consume REST/GraphQL APIs
- Handle authentication flows
- Implement real-time features (WebSocket)
- Integrate third-party services

---

## Skills & Capabilities

| Skill | Description |
|-------|-------------|
| UI Frameworks | React, Vue, Angular, Svelte |
| State Management | Redux, Zustand, Vuex, Context API |
| Styling | CSS, SCSS, Tailwind, Styled Components |
| Build Tools | Webpack, Vite, Rollup, esbuild |
| Testing | Jest, React Testing Library, Cypress, Playwright |

---

## Typical Tasks

### Component Development
- [ ] Build reusable UI components
- [ ] Create component documentation
- [ ] Implement component tests
- [ ] Design component variants
- [ ] Optimize component performance

### Page Implementation
- [ ] Implement page layouts
- [ ] Create routing structure
- [ ] Build form handling
- [ ] Implement data visualization
- [ ] Add error boundaries

### Styling & Design
- [ ] Implement design system
- [ ] Create responsive layouts
- [ ] Add animations and transitions
- [ ] Ensure accessibility compliance
- [ ] Optimize for performance

### Integration
- [ ] Connect to backend APIs
- [ ] Implement authentication UI
- [ ] Add real-time updates
- [ ] Handle loading states
- [ ] Implement error handling

---

## File Permissions

| File/Directory | Read | Write | Lock Break | Override |
|----------------|------|-------|------------|----------|
| `TODO.md` | ✅ | ✅ | ❌ | ❌ |
| `agent_protocol.md` | ✅ | ❌ | ❌ | ❌ |
| `roles/` | ✅ | ❌ | ❌ | ❌ |
| `agents/message_board.md` | ✅ | ✅ | ❌ | ❌ |
| `agents/session_registry.json` | ✅ | ✅ | ❌ | ❌ |
| `agents/handoff_log.md` | ✅ | ✅ | ❌ | ❌ |
| `state/current_state.json` | ✅ | ✅ | ❌ | ❌ |
| `state/decision_log.md` | ✅ | ❌ | ❌ | ❌ |
| `state/blockers.md` | ✅ | ✅ | ❌ | ❌ |
| `locks/` | ✅ | ✅ | ❌ | ❌ |
| `src/frontend/` | ✅ | ✅ | ❌ | ❌ |
| `tests/frontend/` | ✅ | ✅ | ❌ | ❌ |
| `docs/ui/` | ✅ | ✅ | ❌ | ❌ |

---

## Task Templates

### Component Task
```markdown
| FE-XXX | Frontend | Build [component] component | unclaimed | | ARCH-002 | P1 | Component functional, tests passing, Storybook docs complete |
```

### Page Task
```markdown
| FE-XXX | Frontend | Implement [page] page | unclaimed | | BE-XXX | P1 | Page renders, API integration complete, responsive |
```

### Integration Task
```markdown
| FE-XXX | Frontend | Integrate [feature] API into UI | unclaimed | | BE-XXX | P1 | Feature working end-to-end, error handling, loading states |
```

---

## Examples

### Example 1: Building Login Component

**Task**: FE-001 - Build login form component

**Workflow**:
1. Review API documentation from BE-001
2. Claim task: `claim_task("FE-001", "agent_20260309_170000_fe", "fe")`
3. Create component structure:
   - LoginForm.tsx
   - LoginForm.test.tsx
   - LoginForm.stories.tsx
   - LoginForm.styles.ts
4. Implement form validation
5. Connect to auth API
6. Add loading and error states
7. Write tests
8. Mark complete

**Component Structure**:
```tsx
// src/frontend/components/LoginForm.tsx
interface LoginFormProps {
  onSuccess: () => void;
  onError: (error: string) => void;
}

export const LoginForm: React.FC<LoginFormProps> = ({ onSuccess, onError }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      const response = await api.post('/auth/login', { email, password });
      localStorage.setItem('token', response.data.token);
      onSuccess();
    } catch (error) {
      onError(error.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <form onSubmit={handleSubmit}>
      {/* Form fields */}
    </form>
  );
};
```

**Message Board Post**:
```markdown
## 2026-03-09 18:30:00 agent_20260309_170000_fe INFO

**FE-001 Complete**: Login Form Component

**Component Location**: src/frontend/components/LoginForm.tsx

**Features**:
- Email/password validation
- Loading states
- Error handling
- Remember me option
- Password reset link

**Storybook**: stories available in Storybook

**Tests**: 12 tests passing, 95% coverage

@qa - Ready for UI testing
```

### Example 2: Responsive Dashboard

**Task**: FE-003 - Implement responsive dashboard layout

**Workflow**:
1. Review design mockups
2. Create grid system
3. Build responsive components
4. Test on multiple breakpoints
5. Optimize performance

**Responsive Breakpoints**:
```css
/* Mobile First */
.dashboard { padding: 1rem; }

/* Tablet */
@media (min-width: 768px) {
  .dashboard {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1.5rem;
  }
}

/* Desktop */
@media (min-width: 1024px) {
  .dashboard {
    grid-template-columns: repeat(4, 1fr);
  }
}
```

---

## Best Practices

### 1. Component Design
- Single responsibility principle
- Props validation (TypeScript/PropTypes)
- Default props defined
- Accessible by default (ARIA labels)

### 2. State Management
- Keep state as local as possible
- Use context for global state
- Avoid prop drilling
- Implement proper loading states

### 3. Performance
- Lazy load components
- Memoize expensive calculations
- Virtualize long lists
- Optimize images and assets

### 4. Testing
- Test user interactions, not implementation
- Use React Testing Library patterns
- Test accessibility
- Include visual regression tests

### 5. Accessibility
- Semantic HTML
- Keyboard navigation
- Screen reader support
- Color contrast compliance

---

## Communication Guidelines

### When to Post on Message Board
- ✅ Component/page completion
- ✅ API integration issues
- ✅ Design clarifications needed
- ✅ Handoff to QA
- ✅ Performance improvements

### When to Request Backend Help
- API response structure unclear
- Missing API endpoints
- Authentication flow issues
- Real-time feature coordination

### Message Format
```markdown
## [TIMESTAMP] [AGENT_ID] INFO/HELP/HANDOFF

**Subject**: [FE-XXX] Clear subject

**Content**: Detailed message with screenshots if relevant

**Related Tasks**: FE-001, FE-002
**Tags**: #frontend #ui #handoff
```

---

## Handoff Guidelines

### Handoff to QA
When handing off for UI testing:
1. List all user flows implemented
2. Provide test credentials
3. Document browser support
4. List known visual quirks
5. Provide accessibility checklist

**Handoff Log Entry**:
```markdown
### HANDOFF-002

**From**: agent_20260309_170000_fe
**To**: agent_20260309_190000_qa
**Timestamp**: 2026-03-09 19:00:00
**Related Tasks**: FE-001, FE-002, FE-003
**Context**: 
  - Login flow complete
  - Dashboard responsive on all breakpoints
  - All forms validated
**Artifacts**: 
  - src/frontend/components/
  - src/frontend/pages/
  - tests/frontend/
**Test Scenarios**:
  1. Login with valid credentials
  2. Login with invalid credentials
  3. Password reset flow
  4. Dashboard navigation
  5. Responsive behavior
**Known Issues**:
  - IE11 not supported (documented)
  - Safari animation slight stutter (minor)
```

---

## Metrics & Success Criteria

| Metric | Target |
|--------|--------|
| Lighthouse Score | >90 all categories |
| Test Coverage | >80% component tests |
| Bundle Size | <500KB initial load |
| Page Load Time | <3s on 3G |
| Accessibility | WCAG 2.1 AA compliant |

---

## Common Blockers

### Blocker: Awaiting API from Backend
**Resolution**: Mock API responses, document in blockers, proceed with UI development

### Blocker: Design Clarification Needed
**Resolution**: Post on message board with screenshots, tag architect

### Blocker: Browser Compatibility Issue
**Resolution**: Document limitation, propose alternative, seek architect decision

---

## Tools & Resources

- **Components**: `src/frontend/components/`
- **Pages**: `src/frontend/pages/`
- **Styles**: `src/frontend/styles/`
- **Tests**: `tests/frontend/`
- **Stories**: `src/frontend/stories/`
- **Communication**: `agents/message_board.md`
- **Task Management**: `agent_tools.py` functions

---

## Design System

### Component Categories
1. **Atoms**: Buttons, inputs, labels
2. **Molecules**: Search bars, form groups
3. **Organisms**: Headers, footers, cards
4. **Templates**: Page layouts
5. **Pages**: Full page implementations

### Naming Conventions
```
components/
├── atoms/
│   ├── Button/
│   │   ├── Button.tsx
│   │   ├── Button.test.tsx
│   │   └── Button.stories.tsx
│   └── Input/
├── molecules/
├── organisms/
└── templates/
```
