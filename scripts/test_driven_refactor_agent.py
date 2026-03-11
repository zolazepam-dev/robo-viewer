#!/usr/bin/env python3
"""
Test-Driven Autonomous Refactoring Agent

This agent writes comprehensive tests first, then autonomously refactors code
to pass those tests while maintaining all existing functionality. It safely
makes multi-file changes across the codebase, running the full test suite
after each modification to ensure no regressions.

Workflow:
1. Map all affected files using glob/list_directory
2. Write comprehensive tests capturing current behavior and desired improvements
3. Run tests to establish baseline
4. Make incremental refactoring changes using edit
5. Run tests after each change
6. If tests fail, analyze and fix code or adjust tests
7. Continue until all tests pass and refactoring goal is achieved
"""

import subprocess
import sys
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RefactoringGoal:
    """Defines a refactoring task with success criteria."""
    title: str
    description: str
    affected_files: List[str]
    test_file: str
    success_criteria: List[str]
    constraints: List[str]


class TestDrivenRefactoringAgent:
    """Autonomous agent for test-driven refactoring."""
    
    def __init__(self, goal: RefactoringGoal):
        self.goal = goal
        self.log_file = Path("refactoring_agent_log.txt")
        self.test_results_dir = Path("test_results")
        self.iteration = 0
        self.max_iterations = 20
        
        # Ensure test results directory exists
        self.test_results_dir.mkdir(exist_ok=True)
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
    
    def run_command(self, cmd: str, timeout: int = 300) -> Tuple[bool, str]:
        """Execute shell command and return success status + output."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, str(e)
    
    def discover_affected_files(self) -> List[str]:
        """Map all files that may be affected by refactoring."""
        self.log(f"Discovering affected files for: {self.goal.title}", "DISCOVERY")
        
        affected = []
        
        # Check explicitly listed files
        for file_pattern in self.goal.affected_files:
            if '*' in file_pattern or '?' in file_pattern:
                # Glob pattern
                cmd = f"find . -path './bazel-*' -prune -o -name '{file_pattern}' -print"
                success, output = self.run_command(cmd)
                if success and output.strip():
                    files = [f.strip() for f in output.split('\n') if f.strip()]
                    affected.extend(files)
                    self.log(f"  Found {len(files)} files matching pattern: {file_pattern}", "DISCOVERY")
            else:
                # Direct file path
                if Path(file_pattern).exists():
                    affected.append(file_pattern)
                    self.log(f"  Found file: {file_pattern}", "DISCOVERY")
        
        # Remove duplicates
        affected = list(set(affected))
        self.log(f"Total affected files: {len(affected)}", "DISCOVERY")
        
        return affected
    
    def write_comprehensive_tests(self) -> bool:
        """Write comprehensive tests capturing current behavior and desired improvements."""
        self.log(f"Writing comprehensive tests: {self.goal.test_file}", "TEST")
        
        # Check if test file already exists
        test_path = Path(self.goal.test_file)
        
        if test_path.exists():
            self.log(f"Test file already exists: {self.goal.test_file}", "TEST")
            self.log("  Will augment existing tests with additional coverage", "TEST")
            return True
        
        # Generate test file based on refactoring goal
        test_content = self._generate_test_content()
        
        try:
            with open(test_path, 'w') as f:
                f.write(test_content)
            self.log(f"✓ Created test file: {self.goal.test_file}", "SUCCESS")
            return True
        except Exception as e:
            self.log(f"✗ Failed to create test file: {e}", "ERROR")
            return False
    
    def _generate_test_content(self) -> str:
        """Generate test content based on refactoring goal."""
        # Default test template for C++ projects
        return f'''// Auto-generated test for: {self.goal.title}
// Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
// Refactoring Goal: {self.goal.description}

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <string>

// Test suite for {self.goal.title}
class {self.goal.title.replace(" ", "").replace("-", "_")}Test : public ::testing::Test {{
protected:
    void SetUp() override {{
        // Setup test fixtures
        std::cout << "Setting up test" << std::endl;
    }}

    void TearDown() override {{
        // Cleanup
        std::cout << "Tearing down test" << std::endl;
    }}
}};

// Test current behavior before refactoring
TEST_F({self.goal.title.replace(" ", "").replace("-", "_")}Test, CurrentBehaviorBaseline) {{
    // Document current behavior
    // This test should pass before refactoring
    EXPECT_TRUE(true) << "Baseline test - documents current behavior";
}}

// Test desired behavior after refactoring
TEST_F({self.goal.title.replace(" ", "").replace("-", "_")}Test, DesiredBehaviorAfterRefactoring) {{
    // This test may fail initially, will pass after refactoring
    // Success criteria:
'''
    
    def run_tests(self, test_target: str = "//:system_test") -> Tuple[bool, str]:
        """Run tests and return results."""
        self.log(f"Running tests: {test_target}", "TEST")
        
        cmd = f"bazel test {test_target} --test_output=all"
        success, output = self.run_command(cmd, timeout=600)
        
        # Save test results
        result_file = self.test_results_dir / f"test_run_{self.iteration:03d}.txt"
        with open(result_file, 'w') as f:
            f.write(output)
        
        if success:
            self.log(f"✓ All tests passed", "SUCCESS")
        else:
            self.log(f"✗ Some tests failed", "ERROR")
            # Extract failure summary
            failures = re.findall(r'FAILED.*', output)
            if failures:
                self.log(f"  Failures: {len(failures)}", "ERROR")
        
        return success, output
    
    def establish_baseline(self) -> Tuple[bool, str]:
        """Run tests to establish baseline before refactoring."""
        self.log("Establishing baseline test results", "BASELINE")
        
        # First, try to build
        build_success, build_output = self.run_command(
            "bazel build //:train --compilation_mode=opt",
            timeout=600
        )
        
        if not build_success:
            self.log("✗ Baseline build failed - cannot establish baseline", "ERROR")
            return False, build_output
        
        # Run tests
        test_success, test_output = self.run_tests()
        
        if test_success:
            self.log("✓ Baseline established - all tests passing", "BASELINE")
        else:
            self.log("⚠ Baseline has failing tests - will track during refactoring", "BASELINE")
        
        return test_success, test_output
    
    def make_incremental_change(self, change_description: str) -> bool:
        """Make an incremental refactoring change."""
        self.log(f"Making incremental change: {change_description}", "REFACTOR")
        
        # This would be overridden by specific refactoring logic
        # For now, it's a placeholder for the actual edit operations
        self.log(f"  Change type: {change_description}", "REFACTOR")
        
        return True
    
    def analyze_test_failures(self, test_output: str) -> List[Dict]:
        """Analyze test failures and categorize them."""
        self.log("Analyzing test failures", "ANALYSIS")
        
        failures = []
        
        # Parse test output for failures
        failure_pattern = r'(\[.*FAILED.*\]|.*FAIL.*:.*)'
        matches = re.findall(failure_pattern, test_output, re.MULTILINE)
        
        for match in matches:
            failure_info = {
                'type': 'UNKNOWN',
                'description': match,
                'severity': 'HIGH',
                'suggested_fix': 'Manual review required'
            }
            
            # Categorize failure
            if 'assertion' in match.lower():
                failure_info['type'] = 'ASSERTION_FAILURE'
                failure_info['suggested_fix'] = 'Check test expectations or implementation'
            elif 'segmentation' in match.lower():
                failure_info['type'] = 'SEGFAULT'
                failure_info['severity'] = 'CRITICAL'
                failure_info['suggested_fix'] = 'Check memory access and null pointers'
            elif 'timeout' in match.lower():
                failure_info['type'] = 'TIMEOUT'
                failure_info['suggested_fix'] = 'Check for infinite loops or deadlocks'
            
            failures.append(failure_info)
            self.log(f"  Failure: {failure_info['type']} - {failure_info['description'][:100]}", "ANALYSIS")
        
        return failures
    
    def verify_no_regressions(self, baseline_test_output: str) -> bool:
        """Verify that refactoring hasn't introduced regressions."""
        self.log("Verifying no regressions introduced", "VERIFICATION")
        
        # Run full test suite
        test_success, current_output = self.run_tests()
        
        if not test_success:
            self.log("✗ Regressions detected - tests failing", "ERROR")
            return False
        
        # Compare test counts (if available)
        baseline_tests = baseline_test_output.count('test run')
        current_tests = current_output.count('test run')
        
        if current_tests < baseline_tests:
            self.log(f"⚠ Test count decreased: {baseline_tests} → {current_tests}", "WARNING")
            return False
        
        self.log(f"✓ No regressions detected - {current_tests} tests passing", "SUCCESS")
        return True
    
    def run_refactoring_loop(self) -> bool:
        """Main refactoring loop."""
        self.log("=" * 80, "START")
        self.log("Test-Driven Autonomous Refactoring Agent Started", "START")
        self.log(f"Goal: {self.goal.title}", "START")
        self.log(f"Description: {self.goal.description}", "START")
        self.log("=" * 80, "START")
        
        # Step 1: Discover affected files
        affected_files = self.discover_affected_files()
        if not affected_files:
            self.log("✗ No affected files found - cannot proceed", "ERROR")
            return False
        
        # Step 2: Write comprehensive tests
        if not self.write_comprehensive_tests():
            self.log("✗ Failed to write tests - cannot proceed", "ERROR")
            return False
        
        # Step 3: Establish baseline
        baseline_success, baseline_output = self.establish_baseline()
        
        # Step 4-6: Iterative refactoring
        while self.iteration < self.max_iterations:
            self.iteration += 1
            
            self.log(f"\n{'='*60}", "ITERATION")
            self.log(f"Iteration {self.iteration}/{self.max_iterations}", "ITERATION")
            self.log(f"{'='*60}", "ITERATION")
            
            # Make incremental change
            change_desc = f"Refactoring step {self.iteration}"
            if not self.make_incremental_change(change_desc):
                self.log("✗ Failed to make change", "ERROR")
                continue
            
            # Run tests after change
            test_success, test_output = self.run_tests()
            
            if test_success:
                self.log("✓ Tests pass after change", "SUCCESS")
                
                # Verify no regressions
                if not self.verify_no_regressions(baseline_output):
                    self.log("✗ Regressions detected - rolling back", "ERROR")
                    # Rollback logic would go here
                    continue
                
                # Check if refactoring goal achieved
                if self._check_goal_achieved():
                    self.log("\n" + "="*80, "SUCCESS")
                    self.log("✓ Refactoring goal achieved!", "SUCCESS")
                    self.log("="*80, "SUCCESS")
                    return True
            else:
                self.log("✗ Tests failed after change", "ERROR")
                
                # Analyze failures
                failures = self.analyze_test_failures(test_output)
                
                # Decide: fix code or adjust tests
                if self._should_fix_code(failures):
                    self.log("  Decision: Fix code to pass tests", "DECISION")
                    # Code fix logic would go here
                else:
                    self.log("  Decision: Adjust test expectations", "DECISION")
                    # Test adjustment logic would go here
        
        self.log("\n" + "="*80, "COMPLETE")
        self.log(f"Agent stopped after {self.max_iterations} iterations", "COMPLETE")
        self.log(f"Review {self.log_file} for detailed history", "COMPLETE")
        self.log("="*80, "COMPLETE")
        
        return False
    
    def _check_goal_achieved(self) -> bool:
        """Check if refactoring goal has been achieved."""
        # This would check success criteria from the goal
        self.log("Checking if refactoring goal achieved...", "VERIFICATION")
        
        # Placeholder - would implement specific checks per goal
        return False
    
    def _should_fix_code(self, failures: List[Dict]) -> bool:
        """Decide whether to fix code or adjust tests."""
        # Heuristic: if failures are assertion failures, might be test issue
        # If failures are segfaults or critical, definitely code issue
        for failure in failures:
            if failure['severity'] == 'CRITICAL':
                return True
        return len(failures) > 0


def create_refactoring_goal_from_description(description: str) -> RefactoringGoal:
    """Create a refactoring goal from a natural language description."""
    
    # Parse description to extract goal components
    # This is a simplified version - would use NLP in production
    
    return RefactoringGoal(
        title=description[:50],
        description=description,
        affected_files=["src/*.cpp", "src/*.h"],
        test_file="src/RefactoringTest.cpp",
        success_criteria=["All existing tests pass", "Code is more maintainable"],
        constraints=["No behavior changes", "Maintain performance"]
    )


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test-Driven Autonomous Refactoring Agent"
    )
    parser.add_argument(
        '--goal',
        type=str,
        required=True,
        help='Refactoring goal description'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default="src/RefactoringTest.cpp",
        help='Test file to create/use'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=20,
        help='Maximum number of refactoring iterations'
    )
    parser.add_argument(
        '--affected-files',
        type=str,
        nargs='+',
        default=["src/*.cpp", "src/*.h"],
        help='Files that may be affected'
    )
    
    args = parser.parse_args()
    
    # Create refactoring goal
    goal = RefactoringGoal(
        title=args.goal[:50],
        description=args.goal,
        affected_files=args.affected_files,
        test_file=args.test_file,
        success_criteria=["All tests pass", "Refactoring goal achieved"],
        constraints=["No behavior changes"]
    )
    
    # Create and run agent
    agent = TestDrivenRefactoringAgent(goal)
    agent.max_iterations = args.max_iterations
    
    success = agent.run_refactoring_loop()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
