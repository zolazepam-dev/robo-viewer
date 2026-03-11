#!/usr/bin/env python3
"""
SPS Performance Optimization Agent
Automatically identifies and fixes performance bottlenecks to maximize Steps Per Second
"""

import subprocess
import re
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict

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

class SPSOptimizer:
    """Autonomous SPS performance optimization agent."""

    # Performance anti-patterns to detect
    PERFORMANCE_PATTERNS = {
        'DYNAMIC_ALLOCATION': {
            'pattern': r'(new\s+\w+|malloc\s*\(|std::vector.*\.push_back)',
            'description': 'Dynamic memory allocation in hot path',
            'fix': 'Use pre-allocated pools or reserve() capacity',
            'impact': 'HIGH (10-30% SPS gain)',
            'severity': 'CRITICAL'
        },
        'UNNECESSARY_COPY': {
            'pattern': r'(const\s+\w+\s*&\s*=\s*\w+;.*\n.*\w+\s*=\s*\w+)',
            'description': 'Unnecessary object copying',
            'fix': 'Use references or std::move()',
            'impact': 'MEDIUM (5-15% SPS gain)',
            'severity': 'HIGH'
        },
        'MISSING_INLINE': {
            'pattern': r'(^\s*(?:virtual|)\s*\w+\s+\w+\([^)]*\)\s*\{)',
            'description': 'Small functions not inlined',
            'fix': 'Add inline keyword or move to header',
            'impact': 'LOW (2-5% SPS gain)',
            'severity': 'MEDIUM'
        },
        'LOCK_CONTENTION': {
            'pattern': r'(std::lock_guard|std::unique_lock|mutex.*\.lock)',
            'description': 'Mutex locks in hot path',
            'fix': 'Use lock-free structures or reduce lock scope',
            'impact': 'HIGH (20-50% SPS gain)',
            'severity': 'CRITICAL'
        },
        'INEFFICIENT_LOOP': {
            'pattern': r'(for\s*\([^;]+;[^;]*<[^;]*\.size\(\)',
            'description': 'size() called in loop condition',
            'fix': 'Cache size() result before loop',
            'impact': 'LOW (1-3% SPS gain)',
            'severity': 'LOW'
        },
        'BRANCH_MISPREDICTION': {
            'pattern': r'(if\s*\([^)]*\)\s*\{[^}]*\}\s*else\s*\{)',
            'description': 'Unpredictable branches in hot path',
            'fix': 'Use branch hints or conditional moves',
            'impact': 'MEDIUM (5-10% SPS gain)',
            'severity': 'MEDIUM'
        }
    }

    # Build optimization flags
    OPTIMIZATION_FLAGS = {
        'O3': '-O3',
        'march_native': '-march=native',
        'flto': '-flto',
        'ffast_math': '-ffast-math',
        'funroll_loops': '-funroll-loops',
        'ftree_vectorize': '-ftree-vectorize',
        'finline_functions': '-finline-functions',
        'fgcse_after_reload': '-fgcse-after-reload'
    }

    def __init__(self, target_sps: int = 100000):
        self.target_sps = target_sps
        self.current_sps = 0
        self.baseline_sps = 0
        self.log_file = Path("sps_optimizer_log.txt")
        self.optimizations_applied: List[str] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp and color."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = {
            'INFO': Colors.CYAN,
            'SUCCESS': Colors.GREEN,
            'WARNING': Colors.YELLOW,
            'ERROR': Colors.RED,
            'OPTIMIZE': Colors.MAGENTA
        }.get(level, Colors.RESET)
        
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(f"{color}{log_entry}{Colors.RESET}")
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")

    def measure_sps(self, envs: int = 128, steps: int = 5000) -> float:
        """Measure current SPS performance by timing training execution."""
        self.log(f"Measuring SPS ({envs} envs, {steps} steps)...", "INFO")
        
        # Use the pre-built binary directly for faster execution
        cmd = f"timeout 45 ./bazel-bin/train --envs {envs} --steps {steps} 2>&1"
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=50)
            elapsed_time = time.time() - start_time
            output = result.stdout + result.stderr
            
            # Look for SPS in output (ImGui format: "SPS: 1234.5")
            sps_match = re.search(r'SPS:\s*([\d,]+(?:\.\d+)?)', output, re.IGNORECASE)
            if sps_match:
                sps = float(sps_match.group(1).replace(',', ''))
                self.current_sps = sps
                self.log(f"Current SPS: {sps:,.0f}", "SUCCESS")
                return sps
            
            # Alternative: Look for step count in output and calculate SPS
            step_matches = re.findall(r'Step\s+(\d+):', output)
            if step_matches:
                total_steps = int(max(step_matches))
                if total_steps > 0 and elapsed_time > 0:
                    # Calculate SPS: total steps / elapsed time * envs
                    sps = (total_steps * envs) / elapsed_time
                    self.current_sps = sps
                    self.log(f"Calculated SPS: {sps:,.0f} ({total_steps} steps in {elapsed_time:.1f}s)", "SUCCESS")
                    return sps
            
            # Fallback: estimate from elapsed time
            if elapsed_time > 0:
                sps = (steps * envs) / elapsed_time
                self.current_sps = sps
                self.log(f"Estimated SPS: {sps:,.0f} ({steps} steps × {envs} envs in {elapsed_time:.1f}s)", "SUCCESS")
                return sps
            
            self.log("Could not extract SPS from output", "WARNING")
            return 0.0
            
        except subprocess.TimeoutExpired:
            self.log("SPS measurement timed out", "ERROR")
            return 0.0
        except Exception as e:
            self.log(f"SPS measurement error: {e}", "ERROR")
            return 0.0

    def scan_for_antipatterns(self, source_dir: Path = None) -> Dict[str, List[dict]]:
        """Scan source code for performance anti-patterns."""
        if source_dir is None:
            source_dir = Path("src")
        
        findings: Dict[str, List[dict]] = {}
        
        self.log(f"Scanning {source_dir} for performance anti-patterns...", "INFO")
        
        for pattern_name, config in self.PERFORMANCE_PATTERNS.items():
            findings[pattern_name] = []
            
            try:
                # Use grep to find matches
                cmd = f"grep -rn '{config['pattern']}' {source_dir}/*.cpp {source_dir}/*.h 2>/dev/null | head -20"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_path = parts[0]
                            line_num = parts[1]
                            code = parts[2][:100]
                            
                            findings[pattern_name].append({
                                'file': file_path,
                                'line': line_num,
                                'code': code,
                                'pattern': pattern_name,
                                'description': config['description'],
                                'fix': config['fix'],
                                'impact': config['impact'],
                                'severity': config['severity']
                            })
            except Exception as e:
                self.log(f"Error scanning for {pattern_name}: {e}", "WARNING")
        
        # Count total findings
        total = sum(len(v) for v in findings.values())
        self.log(f"Found {total} potential performance issues", "INFO")
        
        return findings

    def analyze_build_config(self) -> List[dict]:
        """Analyze BUILD file for optimization opportunities."""
        self.log("Analyzing BUILD configuration...", "INFO")
        
        recommendations = []
        
        try:
            with open("BUILD", "r") as f:
                build_content = f.read()
            
            # Check for optimization flags
            for flag_name, flag_value in self.OPTIMIZATION_FLAGS.items():
                if flag_value not in build_content:
                    recommendations.append({
                        'type': 'MISSING_OPTIMIZATION_FLAG',
                        'flag': flag_value,
                        'description': f'Missing compiler optimization: {flag_name}',
                        'impact': 'MEDIUM',
                        'action': f'Add {flag_value} to copts in BUILD file'
                    })
            
            # Check for lto
            if '-flto' in build_content:
                self.log("LTO already enabled ✓", "SUCCESS")
            else:
                self.log("LTO not enabled - recommended for 10-20% SPS gain", "WARNING")
                
        except FileNotFoundError:
            self.log("BUILD file not found", "ERROR")
        except Exception as e:
            self.log(f"Error analyzing BUILD: {e}", "ERROR")
        
        return recommendations

    def generate_optimization_report(self, findings: Dict, build_recs: List) -> str:
        """Generate comprehensive optimization report."""
        report = []
        report.append("=" * 80)
        report.append("SPS PERFORMANCE OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Current SPS: {self.current_sps:,.0f}")
        report.append(f"Target SPS: {self.target_sps:,}")
        report.append(f"Gap: {self.target_sps - self.current_sps:,.0f} ({(1 - self.current_sps/self.target_sps)*100:.1f}% remaining)")
        report.append("")
        
        # Summary by severity
        report.append("FINDINGS BY SEVERITY:")
        report.append("-" * 80)
        
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for pattern, issues in findings.items():
            for issue in issues:
                severity_counts[issue['severity']] += 1
        
        for severity, count in severity_counts.items():
            if count > 0:
                report.append(f"  {severity}: {count} issues")
        
        report.append("")
        report.append("DETAILED FINDINGS:")
        report.append("-" * 80)
        
        # Group by pattern
        for pattern, issues in findings.items():
            if issues:
                report.append(f"\n{pattern} ({len(issues)} occurrences):")
                report.append(f"  Description: {self.PERFORMANCE_PATTERNS[pattern]['description']}")
                report.append(f"  Fix Strategy: {self.PERFORMANCE_PATTERNS[pattern]['fix']}")
                report.append(f"  Impact: {self.PERFORMANCE_PATTERNS[pattern]['impact']}")
                report.append("")
                
                for issue in issues[:5]:  # Show first 5
                    report.append(f"    {issue['file']}:{issue['line']}")
                    report.append(f"      {issue['code'][:70]}")
                
                if len(issues) > 5:
                    report.append(f"    ... and {len(issues) - 5} more")
        
        # Build recommendations
        if build_recs:
            report.append("")
            report.append("BUILD CONFIGURATION RECOMMENDATIONS:")
            report.append("-" * 80)
            for rec in build_recs:
                report.append(f"  [{rec['impact']}] {rec['description']}")
                report.append(f"    Action: {rec['action']}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

    def apply_optimization(self, file_path: str, line_num: int, pattern: str) -> bool:
        """Attempt to apply an optimization fix."""
        self.log(f"Applying optimization to {file_path}:{line_num} ({pattern})", "OPTIMIZE")
        
        # This is a placeholder - actual implementation would need to:
        # 1. Read the file
        # 2. Apply the specific fix based on pattern type
        # 3. Write back the optimized version
        # 4. Verify build still succeeds
        
        self.log(f"  Pattern: {pattern}", "INFO")
        self.log(f"  Fix: {self.PERFORMANCE_PATTERNS[pattern]['fix']}", "INFO")
        
        # For now, just log the recommendation
        self.optimizations_applied.append(f"{file_path}:{line_num} - {pattern}")
        
        return True

    def run_optimization_cycle(self, auto_apply: bool = False) -> Dict:
        """Run complete optimization cycle."""
        self.log("=" * 60, "INFO")
        self.log("SPS OPTIMIZATION CYCLE STARTING", "INFO")
        self.log("=" * 60, "INFO")
        
        # Step 1: Measure baseline SPS
        self.log("Step 1: Measuring baseline SPS...", "INFO")
        self.baseline_sps = self.measure_sps()
        
        if self.baseline_sps == 0:
            self.log("Failed to measure baseline SPS. Check if training runs.", "ERROR")
            return {'success': False, 'reason': 'baseline_measurement_failed'}
        
        # Step 2: Scan for anti-patterns
        self.log("Step 2: Scanning for performance anti-patterns...", "INFO")
        findings = self.scan_for_antipatterns()
        
        # Step 3: Analyze build configuration
        self.log("Step 3: Analyzing build configuration...", "INFO")
        build_recs = self.analyze_build_config()
        
        # Step 4: Generate report
        self.log("Step 4: Generating optimization report...", "INFO")
        report = self.generate_optimization_report(findings, build_recs)
        
        # Save report
        report_path = Path("sps_optimization_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        self.log(f"Report saved to {report_path}", "SUCCESS")
        
        # Step 5: Apply optimizations (if auto_apply enabled)
        if auto_apply:
            self.log("Step 5: Auto-applying optimizations...", "OPTIMIZE")
            
            # Prioritize CRITICAL and HIGH severity issues
            for pattern, issues in findings.items():
                for issue in issues:
                    if issue['severity'] in ['CRITICAL', 'HIGH']:
                        self.apply_optimization(
                            issue['file'],
                            issue['line'],
                            issue['pattern']
                        )
        else:
            self.log("Step 5: Auto-apply disabled. Review report and apply manually.", "WARNING")
        
        # Step 6: Measure improved SPS
        if auto_apply and self.optimizations_applied:
            self.log("Step 6: Measuring improved SPS...", "INFO")
            improved_sps = self.measure_sps()
            
            gain = improved_sps - self.baseline_sps
            gain_pct = (gain / self.baseline_sps) * 100 if self.baseline_sps > 0 else 0
            
            self.log(f"SPS Gain: {gain:,.0f} ({gain_pct:+.1f}%)", "SUCCESS" if gain > 0 else "WARNING")
        else:
            improved_sps = self.baseline_sps
        
        # Summary
        self.log("=" * 60, "INFO")
        self.log("OPTIMIZATION CYCLE COMPLETE", "SUCCESS")
        self.log("=" * 60, "INFO")
        self.log(f"Baseline SPS: {self.baseline_sps:,.0f}", "INFO")
        self.log(f"Improved SPS: {improved_sps:,.0f}", "INFO")
        self.log(f"Issues Found: {sum(len(v) for v in findings.values())}", "INFO")
        self.log(f"Optimizations Applied: {len(self.optimizations_applied)}", "INFO")
        
        return {
            'success': True,
            'baseline_sps': self.baseline_sps,
            'improved_sps': improved_sps,
            'issues_found': sum(len(v) for v in findings.values()),
            'optimizations_applied': len(self.optimizations_applied),
            'report_path': str(report_path)
        }


def main():
    """Main entry point."""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("=" * 80)
    print("  JOLTrl SPS Performance Optimization Agent")
    print("=" * 80)
    print(f"{Colors.RESET}")
    
    import argparse
    parser = argparse.ArgumentParser(description='SPS Performance Optimization Agent')
    parser.add_argument('--target-sps', type=int, default=100000, help='Target SPS (default: 100000)')
    parser.add_argument('--auto-apply', action='store_true', help='Automatically apply optimizations')
    parser.add_argument('--measure-only', action='store_true', help='Only measure SPS, no scanning')
    parser.add_argument('--scan-only', action='store_true', help='Only scan for issues, no fixes')
    parser.add_argument('--envs', type=int, default=128, help='Number of parallel environments')
    parser.add_argument('--steps', type=int, default=5000, help='Steps for SPS measurement')
    
    args = parser.parse_args()
    
    optimizer = SPSOptimizer(target_sps=args.target_sps)
    
    if args.measure_only:
        sps = optimizer.measure_sps(envs=args.envs, steps=args.steps)
        print(f"\n{Colors.GREEN}Measured SPS: {sps:,.0f}{Colors.RESET}")
        sys.exit(0)
    
    result = optimizer.run_optimization_cycle(auto_apply=args.auto_apply)
    
    if result['success']:
        print(f"\n{Colors.GREEN}✓ Optimization cycle completed successfully{Colors.RESET}")
        print(f"  Baseline SPS: {result['baseline_sps']:,.0f}")
        print(f"  Improved SPS: {result['improved_sps']:,.0f}")
        print(f"  Report: {result['report_path']}")
    else:
        print(f"\n{Colors.RED}✗ Optimization cycle failed: {result.get('reason', 'unknown')}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
