#!/usr/bin/env python3
"""
Autonomous Build-Test-Debug Agent
Continuously monitors build failures, diagnoses root causes,
applies targeted fixes, and validates through test execution.
"""

import subprocess
import re
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

class BuildDiagnostician:
    """Analyzes build errors and proposes fixes."""
    
    ERROR_PATTERNS = {
        'AVX2_TANH_ERROR': {
            'pattern': r'_mm256_tanh_ps',
            'description': '_mm256_tanh_ps is non-standard in AVX2',
            'fix_strategy': 'Implement custom tanh using polynomial approximation',
            'severity': 'CRITICAL'
        },
        'JOLT_INCLUDE_ERROR': {
            'pattern': r'Jolt/Jolt\.h.*must be first',
            'description': 'Jolt headers not included first',
            'fix_strategy': 'Add #include <Jolt/Jolt.h> as first include in file',
            'severity': 'CRITICAL'
        },
        'LINKER_ERROR': {
            'pattern': r'undefined reference to',
            'description': 'Missing symbol definitions',
            'fix_strategy': 'Check BUILD file dependencies and ensure all symbols are defined',
            'severity': 'HIGH'
        },
        'SIGNATURE_MISMATCH': {
            'pattern': r'no matching function for call',
            'description': 'Function signature mismatch',
            'fix_strategy': 'Verify function declaration matches definition',
            'severity': 'HIGH'
        },
        'UNDECLARED_IDENTIFIER': {
            'pattern': r'use of undeclared identifier',
            'description': 'Missing include or forward declaration',
            'fix_strategy': 'Add required header or forward declaration',
            'severity': 'MEDIUM'
        },
        'SUM_TREE_INDEX_ERROR': {
            'pattern': r'idx\s*=\s*0.*2\s*\*\s*idx',
            'description': 'Sum-tree indexing starts at 0, causing infinite loop',
            'fix_strategy': 'Start tree traversal at idx = 1',
            'severity': 'CRITICAL'
        },
        'PRIORITY_TRUNCATION': {
            'pattern': r'static_cast<int>\(priority\)',
            'description': 'Float priorities truncated to integers',
            'fix_strategy': 'Use float priorities throughout sum-tree',
            'severity': 'HIGH'
        }
    }
    
    def diagnose(self, error_output: str) -> Optional[dict]:
        """Analyze error output and return diagnosis."""
        for error_type, config in self.ERROR_PATTERNS.items():
            if re.search(config['pattern'], error_output, re.IGNORECASE):
                return {
                    'type': error_type,
                    'description': config['description'],
                    'fix_strategy': config['fix_strategy'],
                    'severity': config['severity'],
                    'matched_text': re.search(config['pattern'], error_output, re.IGNORECASE).group(0)
                }
        return None


class AutonomousBuildAgent:
    """Main agent that orchestrates build-test-debug loop."""
    
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.diagnostician = BuildDiagnostician()
        self.log_file = Path("agent_log.txt")
        self.iteration = 0
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
    
    def run_command(self, cmd: str, capture: bool = True) -> Tuple[bool, str]:
        """Execute shell command and return success status + output."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=capture,
                text=True,
                timeout=300  # 5 minute timeout
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "Command timed out after 5 minutes"
        except Exception as e:
            return False, str(e)
    
    def attempt_build(self) -> Tuple[bool, str]:
        """Attempt to build the project."""
        self.log("Building project...", "BUILD")
        
        cmd = """
        bazel build //:train \\
            --compilation_mode=opt \\
            --copt=-march=native \\
            --copt=-O3 \\
            --copt=-flto \\
            --copt=-ffast-math
        """
        
        success, output = self.run_command(cmd)
        
        if success:
            self.log("Build succeeded!", "SUCCESS")
        else:
            self.log(f"Build failed", "ERROR")
            
        return success, output
    
    def run_tests(self) -> Tuple[bool, str]:
        """Run system tests."""
        self.log("Running system tests...", "TEST")
        
        cmd = "bazel run //:system_test"
        success, output = self.run_command(cmd)
        
        if success:
            self.log("All tests passed!", "SUCCESS")
        else:
            self.log("Tests failed", "ERROR")
            
        return success, output
    
    def locate_error_source(self, error_type: str, matched_text: str) -> List[str]:
        """Search codebase for error locations."""
        self.log(f"Searching for error source: {matched_text}", "SEARCH")
        
        # Use grep to find occurrences
        cmd = f"grep -rn '{matched_text}' src/ 2>/dev/null || true"
        success, output = self.run_command(cmd)
        
        if success and output.strip():
            locations = output.strip().split('\n')
            self.log(f"Found {len(locations)} potential locations", "SEARCH")
            return locations
        return []
    
    def propose_fix(self, diagnosis: dict, locations: List[str]) -> dict:
        """Propose a fix based on diagnosis."""
        fix_proposal = {
            'diagnosis': diagnosis,
            'affected_files': [],
            'fix_description': diagnosis['fix_strategy'],
            'confidence': 'MEDIUM'
        }
        
        # Parse locations to get file paths
        for loc in locations:
            if ':' in loc:
                file_path = loc.split(':')[0]
                if file_path not in fix_proposal['affected_files']:
                    fix_proposal['affected_files'].append(file_path)
        
        # Adjust confidence based on pattern matching
        if len(locations) == 1:
            fix_proposal['confidence'] = 'HIGH'
        elif len(locations) > 5:
            fix_proposal['confidence'] = 'LOW'
            
        return fix_proposal
    
    def run_loop(self):
        """Main agent loop."""
        self.log("=" * 60)
        self.log("Autonomous Build-Test-Debug Agent Started")
        self.log(f"Max iterations: {self.max_iterations}")
        self.log("=" * 60)
        
        while self.iteration < self.max_iterations:
            self.iteration += 1
            self.log(f"\n{'='*40}", "LOOP")
            self.log(f"Iteration {self.iteration}/{self.max_iterations}", "LOOP")
            self.log(f"{'='*40}", "LOOP")
            
            # Step 1: Attempt build
            build_success, build_output = self.attempt_build()
            
            if build_success:
                # Step 2: Run tests
                test_success, test_output = self.run_tests()
                
                if test_success:
                    self.log("\n" + "="*60, "SUCCESS")
                    self.log("Agent completed successfully - all tests passing!", "SUCCESS")
                    self.log("="*60)
                    return True
                else:
                    # Analyze test failures
                    error_output = test_output
                    error_source = "tests"
            else:
                # Analyze build errors
                error_output = build_output
                error_source = "build"
            
            # Step 3: Diagnose root cause
            self.log(f"Diagnosing {error_source} errors...", "DIAGNOSIS")
            diagnosis = self.diagnostician.diagnose(error_output)
            
            if diagnosis:
                self.log(f"✓ Identified: {diagnosis['type']}", "DIAGNOSIS")
                self.log(f"  Description: {diagnosis['description']}", "DIAGNOSIS")
                self.log(f"  Severity: {diagnosis['severity']}", "DIAGNOSIS")
                self.log(f"  Fix Strategy: {diagnosis['fix_strategy']}", "DIAGNOSIS")
                
                # Step 4: Locate error source
                locations = self.locate_error_source(
                    diagnosis['type'],
                    diagnosis['matched_text']
                )
                
                # Step 5: Propose fix
                fix_proposal = self.propose_fix(diagnosis, locations)
                
                self.log("\n" + "-"*40, "FIX_PROPOSAL")
                self.log("Fix Proposal:", "FIX_PROPOSAL")
                self.log(f"  Type: {diagnosis['type']}", "FIX_PROPOSAL")
                self.log(f"  Affected Files: {', '.join(fix_proposal['affected_files'])}", "FIX_PROPOSAL")
                self.log(f"  Confidence: {fix_proposal['confidence']}", "FIX_PROPOSAL")
                self.log(f"  Action: {diagnosis['fix_strategy']}", "FIX_PROPOSAL")
                self.log("-"*40, "FIX_PROPOSAL")
                
                # In autonomous mode, we would apply fixes here
                # For now, we report and wait for manual intervention
                self.log("\nManual intervention required to apply fix.", "ACTION")
                self.log("After applying fix, restart agent to continue.", "ACTION")
                
            else:
                self.log("Unable to automatically diagnose error.", "ERROR")
                self.log(f"Error output:\n{error_output[:2000]}", "ERROR")
                break
        
        self.log("\n" + "="*60, "COMPLETE")
        self.log(f"Agent stopped after {self.max_iterations} iterations", "COMPLETE")
        self.log(f"Review {self.log_file} for detailed history", "COMPLETE")
        self.log("="*60)
        
        return False


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Autonomous Build-Test-Debug Agent for JOLTrl"
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum number of fix attempts'
    )
    parser.add_argument(
        '--auto-apply',
        action='store_true',
        help='Automatically apply fixes (experimental)'
    )
    
    args = parser.parse_args()
    
    agent = AutonomousBuildAgent(max_iterations=args.max_iterations)
    success = agent.run_loop()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
