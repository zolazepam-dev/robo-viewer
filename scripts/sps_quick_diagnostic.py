#!/usr/bin/env python3
"""
Quick SPS Diagnostic Tool
Analyzes training output for performance bottlenecks
"""

import subprocess
import re
import sys
from pathlib import Path
from datetime import datetime

# Colors
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'

def run_diagnostic(envs=128, steps=500):
    """Run training and analyze performance."""
    print(f"{Colors.CYAN}{Colors.BOLD}╔════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}║  JOLTrl SPS Quick Diagnostic                           ║{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}╚════════════════════════════════════════════════════════╝{Colors.RESET}")
    print()
    
    cmd = f"timeout 35 ./bazel-bin/train --envs {envs} --steps {steps} 2>&1"
    print(f"{Colors.YELLOW}Running training benchmark ({envs} envs, {steps} steps)...{Colors.RESET}")
    print()
    
    try:
        import time
        start = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=40)
        elapsed = time.time() - start
        output = result.stdout + result.stderr
        
        # Calculate basic SPS
        step_matches = re.findall(r'Step\s+(\d+):', output)
        if step_matches:
            max_step = int(max(step_matches))
            sps = (max_step * envs) / elapsed if elapsed > 0 else 0
        else:
            sps = (steps * envs) / elapsed if elapsed > 0 else 0
        
        # Analyze stutter patterns
        stutter_matches = re.findall(r'\[STUTTER DETECTED\].*?:\s*([\d.]+)ms', output)
        if stutter_matches:
            stutter_times = [float(x) for x in stutter_matches]
            avg_stutter = sum(stutter_times) / len(stutter_times)
            max_stutter = max(stutter_times)
            stutter_count = len(stutter_matches)
        else:
            avg_stutter = 0
            max_stutter = 0
            stutter_count = 0
        
        # Analyze training step times
        training_matches = re.findall(r'Background:\s*TrainingStep:\s*([\d.]+)ms', output)
        if training_matches:
            training_times = [float(x) for x in training_matches]
            avg_training = sum(training_times) / len(training_times)
            max_training = max(training_times)
        else:
            avg_training = 0
            max_training = 0
        
        # Analyze action selection times
        action_matches = re.findall(r'ActionSelection:\s*([\d.]+)ms', output)
        if action_matches:
            action_times = [float(x) for x in action_matches]
            avg_action = sum(action_times) / len(action_times)
            max_action = max(action_times)
        else:
            avg_action = 0
            max_action = 0
        
        # Print report
        print(f"{Colors.BOLD}┌{'─'*56}┐{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}  {Colors.GREEN}PERFORMANCE ANALYSIS REPORT{Colors.RESET}{' '*26}{Colors.BOLD}│{Colors.RESET}")
        print(f"{Colors.BOLD}├{'─'*56}┤{Colors.RESET}")
        
        # Basic metrics
        print(f"{Colors.BOLD}│{Colors.RESET}  {Colors.CYAN}Basic Metrics:{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}    Steps completed: {max_step if step_matches else steps}")
        print(f"{Colors.BOLD}│{Colors.RESET}    Elapsed time:    {elapsed:.1f}s")
        print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.GREEN}Calculated SPS:  {sps:,.0f}{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}")
        
        # Stutter analysis
        print(f"{Colors.BOLD}│{Colors.RESET}  {Colors.RED}Stutter Analysis:{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}    Stutter events:  {stutter_count}")
        if stutter_count > 0:
            print(f"{Colors.BOLD}│{Colors.RESET}    Average stutter: {avg_stutter:.1f}ms")
            print(f"{Colors.BOLD}│{Colors.RESET}    Max stutter:     {max_stutter:.1f}ms")
            
            # Severity assessment
            if max_stutter > 500:
                print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.RED}Severity: CRITICAL - Severe frame drops{Colors.RESET}")
            elif max_stutter > 100:
                print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.YELLOW}Severity: HIGH - Noticeable stuttering{Colors.RESET}")
            elif max_stutter > 50:
                print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.BLUE}Severity: MEDIUM - Minor hitches{Colors.RESET}")
            else:
                print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.GREEN}Severity: LOW - Acceptable{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}")
        
        # Training step analysis
        print(f"{Colors.BOLD}│{Colors.RESET}  {Colors.MAGENTA}Training Step Times:{Colors.RESET}")
        if avg_training > 0:
            print(f"{Colors.BOLD}│{Colors.RESET}    Average: {avg_training:.1f}ms")
            print(f"{Colors.BOLD}│{Colors.RESET}    Max:     {max_training:.1f}ms")
            
            # Target: <50ms per training step
            if avg_training > 200:
                print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.RED}Status: CRITICAL - Training too slow{Colors.RESET}")
            elif avg_training > 100:
                print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.YELLOW}Status: HIGH - Needs optimization{Colors.RESET}")
            elif avg_training > 50:
                print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.BLUE}Status: OK - Could be faster{Colors.RESET}")
            else:
                print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.GREEN}Status: GOOD - Training fast{Colors.RESET}")
        else:
            print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.YELLOW}No training step data captured{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}")
        
        # Action selection analysis
        print(f"{Colors.BOLD}│{Colors.RESET}  {Colors.CYAN}Action Selection Times:{Colors.RESET}")
        if avg_action > 0:
            print(f"{Colors.BOLD}│{Colors.RESET}    Average: {avg_action:.1f}ms")
            print(f"{Colors.BOLD}│{Colors.RESET}    Max:     {max_action:.1f}ms")
            
            # Target: <20ms per action selection
            if avg_action > 200:
                print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.RED}Status: CRITICAL - Action selection bottleneck{Colors.RESET}")
            elif avg_action > 100:
                print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.YELLOW}Status: HIGH - Needs optimization{Colors.RESET}")
            else:
                print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.GREEN}Status: OK{Colors.RESET}")
        else:
            print(f"{Colors.BOLD}│{Colors.RESET}    {Colors.YELLOW}No action selection data captured{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}")
        
        # Recommendations
        print(f"{Colors.BOLD}│{Colors.RESET}  {Colors.BOLD}Recommendations:{Colors.RESET}")
        print(f"{Colors.BOLD}│{Colors.RESET}")
        
        recommendations = []
        
        if stutter_count > 10 and max_stutter > 200:
            recommendations.append("1. {RED}CRITICAL: Reduce lock contention on gSimMutex{RESET}")
            recommendations.append("   - Use triple-buffering for visual state")
            recommendations.append("   - Minimize mutex hold time in hot path")
        
        if avg_training > 100:
            recommendations.append("2. {YELLOW}HIGH: Optimize training step{RESET}")
            recommendations.append("   - Enable batched neural network forward pass")
            recommendations.append("   - Use SIMD-optimized kernels (AVX2/FMA)")
            recommendations.append("   - Consider async training thread")
        
        if avg_action > 100:
            recommendations.append("3. {YELLOW}HIGH: Optimize action selection{RESET}")
            recommendations.append("   - Pre-allocate action buffers")
            recommendations.append("   - Use SoA (Structure of Arrays) layout")
        
        if sps < 1000:
            recommendations.append("4. {RED}CRITICAL: Overall SPS too low{RESET}")
            recommendations.append("   - Target: 6,000+ SPS (current: {:.0f})".format(sps))
            recommendations.append("   - Check physics step optimization")
            recommendations.append("   - Verify parallel environment scaling")
        
        if not recommendations:
            recommendations.append("{GREEN}No critical issues detected{RESET}")
        
        for rec in recommendations:
            # Process color codes
            rec_formatted = rec.format(RED=Colors.RED, YELLOW=Colors.YELLOW, 
                                       GREEN=Colors.GREEN, RESET=Colors.RESET)
            print(f"{Colors.BOLD}│{Colors.RESET}  {rec_formatted}")
        
        print(f"{Colors.BOLD}│{Colors.RESET}")
        print(f"{Colors.BOLD}└{'─'*56}┘{Colors.RESET}")
        print()
        
        # Save detailed output
        output_file = Path("diagnostic_output.txt")
        with open(output_file, "w") as f:
            f.write(output)
        print(f"{Colors.CYAN}Detailed output saved to: {output_file}{Colors.RESET}")
        
        return {
            'sps': sps,
            'stutter_count': stutter_count,
            'max_stutter': max_stutter,
            'avg_training': avg_training,
            'avg_action': avg_action
        }
        
    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}Diagnostic timed out{Colors.RESET}")
        return None
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SPS Quick Diagnostic')
    parser.add_argument('--envs', type=int, default=128, help='Number of environments')
    parser.add_argument('--steps', type=int, default=500, help='Number of steps')
    
    args = parser.parse_args()
    
    result = run_diagnostic(envs=args.envs, steps=args.steps)
    
    if result:
        print(f"\n{Colors.GREEN}Diagnostic complete. Review recommendations above.{Colors.RESET}")
        if result['sps'] < 1000:
            sys.exit(1)  # Indicate performance issue
    else:
        sys.exit(1)
