#!/usr/bin/env python3
"""
Live Agent Monitor with SPS Performance Focus
Real-time dashboard showing autonomous agent activity and SPS metrics
"""

import subprocess
import sys
import os
import time
import threading
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import json

# ANSI Colors
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'
    BG_RED = '\033[41m'

class LiveMonitor:
    """Real-time monitoring dashboard for autonomous agent."""

    def __init__(self):
        self.running = True
        self.agent_process: Optional[subprocess.Popen] = None
        self.build_process: Optional[subprocess.Popen] = None
        self.current_iteration = 0
        self.max_iterations = 5
        self.current_phase = "IDLE"
        self.sps_history: List[float] = []
        self.target_sps = 100000  # Target: 100K SPS
        self.current_sps = 0
        self.build_status = "UNKNOWN"
        self.error_message = ""
        self.start_time = datetime.now()
        
        # Log tail buffer
        self.log_lines: List[str] = []
        self.max_log_lines = 10
        
        # Performance metrics
        self.metrics = {
            'build_time': 0,
            'test_time': 0,
            'fixes_applied': 0,
            'errors_detected': 0
        }

    def clear_screen(self):
        """Clear terminal screen."""
        print('\033[2J\033[H', end='')
        sys.stdout.flush()

    def draw_header(self):
        """Draw dashboard header."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = datetime.now() - self.start_time
        
        print(f"{Colors.BOLD}{Colors.BG_BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BG_BLUE}  JOLTrl Autonomous Agent - Live Monitor{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BG_BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.CYAN}Time:{Colors.RESET} {now:20} {Colors.CYAN}Elapsed:{Colors.RESET} {str(elapsed).split('.')[0]:15} {Colors.CYAN}Iteration:{Colors.RESET} {self.current_iteration}/{self.max_iterations}")
        print()

    def draw_status_panel(self):
        """Draw main status panel."""
        status_color = Colors.GREEN if self.build_status == "SUCCESS" else (Colors.RED if self.build_status == "FAILED" else Colors.YELLOW)
        
        print(f"{Colors.BOLD}┌{'─'*78}┐{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}  {Colors.BOLD}AGENT STATUS{Colors.RESET}{' '*64}{Colors.BOLD}│{Colors.RESET}")
        print(f"{Colors.BOLD}├{'─'*78}┤{Colors.RESET}")
        
        # Phase indicator
        phase_str = f"{Colors.BOLD}{Colors.CYAN}Current Phase:{Colors.RESET} {status_color}{self.current_phase:40}{Colors.RESET}"
        print(f"{Colors.BOLD}│{Colors.RESET}  {phase_str:76}{Colors.BOLD}│{Colors.RESET}")
        
        # Build status
        build_str = f"{Colors.BOLD}{Colors.CYAN}Build Status:{Colors.RESET} {status_color}{self.build_status:41}{Colors.RESET}"
        print(f"{Colors.BOLD}│{Colors.RESET}  {build_str:76}{Colors.BOLD}│{Colors.RESET}")
        
        # Error message (if any)
        if self.error_message:
            error_truncated = self.error_message[:60] + "..." if len(self.error_message) > 60 else self.error_message
            error_str = f"{Colors.BOLD}{Colors.RED}Error:{Colors.RESET} {Colors.RED}{error_truncated:67}{Colors.RESET}"
            print(f"{Colors.BOLD}│{Colors.RESET}  {error_str:76}{Colors.BOLD}│{Colors.RESET}")
        
        print(f"{Colors.BOLD}└{'─'*78}┘{Colors.RESET}")
        print()

    def draw_sps_gauge(self):
        """Draw SPS performance gauge."""
        print(f"{Colors.BOLD}┌{'─'*78}┐{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}  {Colors.BOLD}SPS PERFORMANCE (Target: {self.target_sps:,}){Colors.RESET}{' '*36}{Colors.BOLD}│{Colors.RESET}")
        print(f"{Colors.BOLD}├{'─'*78}┤{Colors.RESET}")
        
        # Calculate percentage
        sps_pct = min(100, (self.current_sps / self.target_sps) * 100)
        bar_width = 50
        filled = int(bar_width * sps_pct / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        # Color based on performance
        if sps_pct >= 80:
            color = Colors.GREEN
        elif sps_pct >= 50:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        
        sps_display = f"{self.current_sps:>10,}" if self.current_sps > 0 else "     N/A"
        print(f"{Colors.BOLD}│{Colors.RESET}  [{color}{bar}{Colors.RESET}] {sps_display} SPS ({sps_pct:5.1f}%)")
        
        # Show history sparkline
        if self.sps_history:
            history_str = " → ".join([f"{s:>6,.0f}" for s in self.sps_history[-5:]])
            print(f"{Colors.BOLD}│{Colors.RESET}  {Colors.CYAN}History:{Colors.RESET} {history_str:60}")
        
        print(f"{Colors.BOLD}└{'─'*78}┘{Colors.RESET}")
        print()

    def draw_metrics_panel(self):
        """Draw metrics panel."""
        print(f"{Colors.BOLD}┌{'─'*78}┐{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}  {Colors.BOLD}PERFORMANCE METRICS{Colors.RESET}{' '*55}{Colors.BOLD}│{Colors.RESET}")
        print(f"{Colors.BOLD}├{'─'*78}┤{Colors.RESET}")
        
        metrics = [
            ('Build Time', f"{self.metrics['build_time']:.1f}s"),
            ('Test Time', f"{self.metrics['test_time']:.1f}s"),
            ('Fixes Applied', str(self.metrics['fixes_applied'])),
            ('Errors Detected', str(self.metrics['errors_detected'])),
        ]
        
        for label, value in metrics:
            line = f"{Colors.CYAN}{label:20}{Colors.RESET} {value:50}"
            print(f"{Colors.BOLD}│{Colors.RESET}  {line:76}{Colors.BOLD}│{Colors.RESET}")
        
        print(f"{Colors.BOLD}└{'─'*78}┘{Colors.RESET}")
        print()

    def draw_log_tail(self):
        """Draw last N log lines."""
        print(f"{Colors.BOLD}┌{'─'*78}┐{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}  {Colors.BOLD}AGENT LOG (Last {self.max_log_lines} lines){Colors.RESET}{' '*47}{Colors.BOLD}│{Colors.RESET}")
        print(f"{Colors.BOLD}├{'─'*78}┤{Colors.RESET}")
        
        for line in self.log_lines[-self.max_log_lines:]:
            # Color-code log lines
            if '[ERROR]' in line or '[FAILED]' in line:
                colored_line = f"{Colors.RED}{line}{Colors.RESET}"
            elif '[SUCCESS]' in line:
                colored_line = f"{Colors.GREEN}{line}{Colors.RESET}"
            elif '[BUILD]' in line or '[TEST]' in line:
                colored_line = f"{Colors.YELLOW}{line}{Colors.RESET}"
            elif '[FIX]' in line:
                colored_line = f"{Colors.MAGENTA}{line}{Colors.RESET}"
            else:
                colored_line = f"{Colors.WHITE}{line}{Colors.RESET}"
            
            truncated = colored_line[:76]
            print(f"{Colors.BOLD}│{Colors.RESET}  {truncated:76}{Colors.BOLD}│{Colors.RESET}")
        
        print(f"{Colors.BOLD}└{'─'*78}┘{Colors.RESET}")
        print()

    def draw_controls(self):
        """Draw control instructions."""
        print(f"{Colors.CYAN}Controls:{Colors.RESET} [q] Quit  [r] Restart Agent  [b] Manual Build  [t] Run Tests  [p] Profile SPS")
        print()

    def draw(self):
        """Draw complete dashboard."""
        self.clear_screen()
        self.draw_header()
        self.draw_status_panel()
        self.draw_sps_gauge()
        self.draw_metrics_panel()
        self.draw_log_tail()
        self.draw_controls()

    def update_log(self, line: str):
        """Add line to log buffer."""
        self.log_lines.append(line)
        if len(self.log_lines) > self.max_log_lines * 2:
            self.log_lines = self.log_lines[-self.max_log_lines:]

    def parse_agent_output(self, line: str):
        """Parse agent output to update dashboard state."""
        self.update_log(line)
        
        # Parse iteration
        if 'Iteration' in line and '/' in line:
            match = re.search(r'Iteration (\d+)/(\d+)', line)
            if match:
                self.current_iteration = int(match.group(1))
                self.max_iterations = int(match.group(2))
        
        # Parse phase
        if '[BUILD]' in line:
            self.current_phase = "BUILDING"
            self.build_status = "IN_PROGRESS"
        elif '[TEST]' in line:
            self.current_phase = "TESTING"
        elif '[DIAGNOSE]' in line:
            self.current_phase = "DIAGNOSING"
        elif '[FIX]' in line:
            self.current_phase = "APPLYING FIX"
            self.metrics['fixes_applied'] += 1
        elif '[SUCCESS]' in line and 'Build succeeded' in line:
            self.build_status = "SUCCESS"
            self.current_phase = "BUILD SUCCESS"
        elif '[FAILED]' in line or ('[ERROR]' in line and 'failed' in line.lower()):
            self.build_status = "FAILED"
            self.error_message = line.split(']')[-1].strip()
            self.metrics['errors_detected'] += 1
        elif 'SPS' in line or 'steps/sec' in line.lower():
            # Extract SPS value
            match = re.search(r'([\d,]+)\s*(?:SPS|steps/sec)', line, re.IGNORECASE)
            if match:
                sps = float(match.group(1).replace(',', ''))
                self.current_sps = sps
                self.sps_history.append(sps)
                if len(self.sps_history) > 10:
                    self.sps_history = self.sps_history[-10:]

    def run_agent(self):
        """Run autonomous agent with output capture."""
        self.log("Starting autonomous agent...", "INFO")
        self.current_phase = "STARTING AGENT"
        
        cmd = "python3 scripts/autonomous_build_agent.py"
        self.agent_process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Read output line by line
        for line in self.agent_process.stdout:
            if not self.running:
                break
            line = line.strip()
            self.parse_agent_output(line)
            self.draw()
            time.sleep(0.1)
        
        self.agent_process.wait()
        self.current_phase = "AGENT FINISHED"
        self.draw()

    def measure_sps(self):
        """Run training benchmark to measure SPS."""
        self.log("Measuring SPS performance...", "INFO")
        self.current_phase = "BENCHMARKING SPS"
        self.draw()
        
        # Run short training benchmark
        cmd = "timeout 30 bazel run //:train --config=opt -- --envs 128 --steps 1000 2>&1"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=35)
            output = result.stdout + result.stderr
            
            # Look for SPS in output
            for line in output.split('\n'):
                if 'SPS' in line or 'steps/sec' in line:
                    match = re.search(r'([\d,]+)\s*(?:SPS|steps/sec)', line, re.IGNORECASE)
                    if match:
                        sps = float(match.group(1).replace(',', ''))
                        self.current_sps = sps
                        self.sps_history.append(sps)
                        self.log(f"Measured SPS: {sps:,.0f}", "INFO")
                        break
        except subprocess.TimeoutExpired:
            self.log("SPS benchmark timed out", "WARNING")
        except Exception as e:
            self.log(f"SPS benchmark error: {e}", "ERROR")
        
        self.draw()

    def log(self, message: str, level: str = "INFO"):
        """Log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{Colors.CYAN}[{timestamp}]{Colors.RESET} {message}")
        self.update_log(f"[{timestamp}] [{level}] {message}")

    def handle_input(self):
        """Handle user input."""
        import select
        import tty
        import termios
        
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                
                if ch == 'q':
                    self.running = False
                    self.log("Quit requested", "INFO")
                elif ch == 'r':
                    self.log("Restarting agent...", "INFO")
                    self.current_iteration = 0
                    self.run_agent()
                elif ch == 'b':
                    self.log("Running manual build...", "INFO")
                    self.current_phase = "MANUAL BUILD"
                    self.draw()
                    subprocess.run("bazel build //:train 2>&1 | tail -20", shell=True)
                elif ch == 't':
                    self.log("Running tests...", "INFO")
                    self.current_phase = "MANUAL TEST"
                    self.draw()
                    subprocess.run("bazel run //:system_test 2>&1 | tail -20", shell=True)
                elif ch == 'p':
                    self.measure_sps()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def run(self):
        """Main monitoring loop."""
        self.log("Starting Live Agent Monitor...", "INFO")
        self.draw()
        
        # Start agent in background thread
        agent_thread = threading.Thread(target=self.run_agent, daemon=True)
        agent_thread.start()
        
        # Main loop
        while self.running:
            self.handle_input()
            time.sleep(0.1)
        
        self.log("Monitor shutting down...", "INFO")
        if self.agent_process:
            self.agent_process.terminate()


if __name__ == "__main__":
    import re
    monitor = LiveMonitor()
    try:
        monitor.run()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Monitor interrupted{Colors.RESET}")
        sys.exit(0)
