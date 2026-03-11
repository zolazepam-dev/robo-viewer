# 🤖 Autonomous Build Agent - Quick Reference

## 🚀 One-Liners

```bash
# Run diagnostic pass
python3 scripts/autonomous_build_agent.py

# Start continuous monitoring
./scripts/watch_build.sh

# Interactive session
./scripts/launch_agent.sh

# Check agent logs
tail -f agent_log.txt
```

---

## 📋 Common Commands

### Quick Build Check
```bash
bazel build //:train --compilation_mode=opt
```

### Run Agent with Custom Settings
```bash
python3 scripts/autonomous_build_agent.py --max-iterations 10
```

### Background Monitoring (tmux)
```bash
tmux new -s build-agent
./scripts/watch_build.sh
# Ctrl+B, D to detach
```

### View Recent Errors
```bash
grep "ERROR\|DIAGNOSIS" agent_log.txt | tail -20
```

---

## 🎯 Error Patterns Cheat Sheet

| Error | Pattern | Quick Fix |
|-------|---------|-----------|
| AVX2 Tanh | `_mm256_tanh_ps` | Use exponential identity or Pade approx |
| Jolt Include | `Jolt/Jolt.h` | Move to first include |
| Linker | `undefined reference` | Check BUILD deps |
| Signature | `no matching function` | Verify declaration |
| Undeclared | `use of undeclared` | Add header |
| Sum-Tree | `idx = 0` | Start at `idx = 1` |
| Priority | `static_cast<int>` | Use float |

---

## 📊 Log Files

| File | Purpose |
|------|---------|
| `agent_log.txt` | Main agent log |
| `watch_agent.log` | Watch script log |
| `build_output.txt` | Last build output |
| `test_output.txt` | Last test output |

---

## 🔧 Configuration

```bash
# Set max iterations
export MAX_ITERATIONS=5

# Run tests on success
export RUN_TESTS_ON_SUCCESS=true

# Custom build target
# Edit autonomous_build_agent.py, line ~100
```

---

## 🆘 Emergency Procedures

### Build Completely Broken
```bash
# Clean everything
bazel clean --expunge

# Reset agent state
rm -f build_status.txt agent_log.txt

# Fresh start
./scripts/launch_agent.sh
```

### Agent Stuck
```bash
# Kill agent
pkill -f autonomous_build_agent

# Check last diagnosis
cat agent_log.txt | grep -A 5 "DIAGNOSIS"

# Manual fix required
```

### Watch Script Not Responding
```bash
# Restart watch
pkill -f watch_build
./scripts/watch_build.sh &
```

---

## 📈 Monitoring Dashboard

```bash
# Real-time agent activity
tail -f agent_log.txt | grep -E "INFO|ERROR|DIAGNOSIS"

# Build status
watch -n 5 'ls -la build_status.txt'

# File changes
watch -n 2 'find src/ -name "*.cpp" -newer build_status.txt | head -5'
```

---

## 🎓 Agent Modes

| Mode | Use Case | Command |
|------|----------|---------|
| Watch | Continuous monitoring | `./watch_build.sh` |
| Single | One-time diagnostic | `python3 agent.py` |
| Interactive | Step-by-step | `./launch_agent.sh` → 3 |
| Quick | Fast check | `./launch_agent.sh` → 4 |

---

## ✅ Success Indicators

- ✓ Build succeeds
- ✓ Tests pass
- ✓ No new errors in log
- ✓ SPS > 6,000

---

## 📞 Related Docs

- `AUTONOMOUS_AGENT_DEPLOYMENT.md` - Full deployment guide
- `AUTONOMOUS_AGENT_NOTES.md` - Session notes
- `AGENTS.md` - Agentic development guidelines
- `scripts/README.md` - Detailed usage

---

**Quick Start**: `./scripts/launch_agent.sh`  
**Help**: See `scripts/README.md`  
**Logs**: `agent_log.txt`
