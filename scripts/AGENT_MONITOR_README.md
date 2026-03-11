# JOLTrl Agent Monitoring Dashboard 🎯

Real-time monitoring and SPS performance optimization for the JOLTrl training system.

---

## 🚀 Quick Start

```bash
# Launch interactive dashboard
./scripts/watch_agent_dashboard.sh

# Or use direct mode
./scripts/watch_agent_dashboard.sh --live      # Live monitor
./scripts/watch_agent_dashboard.sh --sps       # SPS optimizer
./scripts/watch_agent_dashboard.sh --benchmark # Quick SPS test
```

---

## 📊 Dashboard Features

### 1. Live Agent Monitor (Option 1)
**Real-time dashboard showing:**
- Agent iteration status
- Build/test progress
- SPS performance gauge
- Performance metrics
- Live agent log tail
- Interactive controls

**Controls:**
- `q` - Quit
- `r` - Restart agent
- `b` - Manual build
- `t` - Run tests
- `p` - Profile SPS

### 2. SPS Optimization Agent (Option 2)
**Automatically improves training performance:**

| Mode | Description |
|------|-------------|
| **Scan Only** | Analyzes code, generates report (safe) |
| **Auto-Apply** | Applies optimizations automatically |
| **Custom** | Configure envs, target SPS, auto-apply |

**Detects these performance anti-patterns:**
- ❌ Dynamic allocation in hot paths
- ❌ Unnecessary object copying
- ❌ Missing inline keywords
- ❌ Lock contention in hot loops
- ❌ Inefficient loop patterns
- ❌ Branch misprediction risks

**Expected gains:** 10-50% SPS improvement

### 3. Quick SPS Benchmark (Option 3)
**Fast SPS measurement:**
- Configurable environments
- Short test duration (30s)
- Instant results

### 4. Classic Build Agent (Option 4)
**Original autonomous build agent:**
- Monitors build failures
- Auto-diagnoses errors
- Applies targeted fixes
- Validates with tests

---

## 🎯 SPS Optimization Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                  SPS Optimization Cycle                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Measure Baseline                                             │
│     └─► Run training benchmark (30-60s)                         │
│     └─► Extract SPS from output                                 │
│                                                                  │
│  2. Scan for Anti-Patterns                                       │
│     └─► grep source code for performance issues                 │
│     └─► Categorize by severity (CRITICAL/HIGH/MEDIUM/LOW)       │
│                                                                  │
│  3. Analyze Build Config                                         │
│     └─► Check optimization flags (-O3, -march=native, etc)      │
│     └─► Verify LTO enabled                                      │
│                                                                  │
│  4. Generate Report                                              │
│     └─► Save to sps_optimization_report.txt                     │
│     └─► Include findings, recommendations, code locations       │
│                                                                  │
│  5. Apply Optimizations (Auto mode only)                         │
│     └─► Prioritize CRITICAL/HIGH severity                       │
│     └─► Apply targeted fixes                                    │
│     └─► Verify build still succeeds                             │
│                                                                  │
│  6. Measure Improvement                                          │
│     └─► Re-run benchmark                                        │
│     └─► Calculate SPS gain                                      │
│     └─► Report results                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Generated Files

| File | Description |
|------|-------------|
| `sps_optimization_report.txt` | Detailed optimization findings |
| `sps_optimizer_log.txt` | SPS optimizer activity log |
| `agent_log.txt` | Autonomous agent activity log |
| `watch_agent.log` | Dashboard activity log |
| `build_output.txt` | Last build output |
| `test_output.txt` | Last test output |

---

## 🎛 Command Line Options

### SPS Optimizer
```bash
python3 scripts/sps_optimizer_agent.py [OPTIONS]

Options:
  --target-sps INT    Target SPS (default: 100000)
  --auto-apply        Automatically apply optimizations
  --measure-only      Only measure SPS, no scanning
  --scan-only         Only scan for issues, no fixes
  --envs INT          Number of parallel environments (default: 128)
  --steps INT         Steps for SPS measurement (default: 5000)
```

### Live Monitor
```bash
python3 scripts/live_agent_monitor.py

No options required. Interactive dashboard.
```

### Dashboard
```bash
./scripts/watch_agent_dashboard.sh [MODE]

Modes:
  --live       Launch live monitor
  --sps        Launch SPS optimizer
  --benchmark  Quick SPS benchmark
  --build      Classic build agent
  --report     View optimization report
  --log        View agent log
```

---

## 🔍 Understanding SPS Metrics

| SPS Range | Status | Recommendation |
|-----------|--------|----------------|
| < 10,000 | 🔴 Critical | Run SPS optimizer immediately |
| 10,000 - 50,000 | 🟡 Suboptimal | Apply recommended optimizations |
| 50,000 - 100,000 | 🟢 Good | Minor optimizations possible |
| > 100,000 | 🔵 Excellent | Target achieved |

---

## 🛠 Troubleshooting

### Live Monitor Not Updating
**Symptom:** Dashboard frozen

**Solution:**
```bash
# Check if agent is running
ps aux | grep autonomous_build_agent

# Restart monitor
./scripts/watch_agent_dashboard.sh --live
```

### SPS Measurement Fails
**Symptom:** "Could not extract SPS from output"

**Solution:**
1. Verify training runs: `bazel run //:train --config=opt -- --envs 64 --steps 100`
2. Check output format in `src/main_train.cpp`
3. Ensure SPS logging is enabled

### Optimization Report Empty
**Symptom:** No findings in report

**Solution:**
- Code may already be optimized ✓
- Increase grep search depth
- Check if source files exist: `ls src/*.cpp`

---

## 📈 Performance Tips

### For Maximum SPS:
1. **Use maximum optimizations:**
   ```bash
   bazel build //:train \
       --compilation_mode=opt \
       --copt=-march=native \
       --copt=-O3 \
       --copt=-flto \
       --copt=-ffast-math
   ```

2. **Increase parallel environments:**
   ```bash
   bazel run //:train --config=opt -- --envs 256
   ```

3. **Run SPS optimizer regularly:**
   ```bash
   ./scripts/watch_agent_dashboard.sh --sps
   ```

4. **Pin CPU cores:**
   ```bash
   taskset -c 0-11 bazel run //:train --config=opt
   ```

---

## 🎓 Best Practices

1. **Run SPS benchmark before/after major changes**
2. **Review optimization report before auto-applying**
3. **Keep live monitor running during training sessions**
4. **Save optimization reports for tracking progress**
5. **Combine with autonomous build agent for continuous improvement**

---

## 📚 Related Documentation

- `AUTONOMOUS_AGENT_DEPLOYMENT.md` - Original build agent guide
- `AGENTS.md` - Agentic development standards
- `DOCS.md` - Complete project documentation
- `bug_report.md` - Known issues

---

**Created:** March 10, 2026
**Version:** 1.0.0
