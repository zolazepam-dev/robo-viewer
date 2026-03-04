#!/usr/bin/env python3
"""
JOLTrl Parallel Environments Stress Test Script

Tests the maximum number of stable parallel environments supported by the JOLTrl system.
The script runs the training executable with increasing numbers of environments,
measures performance (steps per second - SPS), and stops when the system becomes unstable.

Requirements:
- Python 3.6+
- psutil library (install with: pip install psutil)
"""

import subprocess
import time
import psutil
import os
import sys
from typing import Optional


class StressTestConfig:
    """Configuration for the stress test"""

    MIN_ENVS = 1
    MAX_ENVS = 4
    PERFORMANCE_THRESHOLD = 0.1  # Stop if performance drops below 10% of peak SPS
    TIMEOUT_PER_RUN = 90  # Seconds per test run before timeout
    WARMUP_DURATION = 1  # Seconds to wait for system to stabilize before measuring
    EXECUTABLE_PATH = "bazel-bin/train_headless"
    BASE_COMMAND = ["bazel-bin/train_headless"]


def check_system_resources() -> bool:
    """Check if system has enough resources to run the test."""
    cpu_count = psutil.cpu_count(logical=False)
    mem_total = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB

    print(f"System Information:")
    print(f"  CPU Cores (physical): {cpu_count}")
    print(f"  Memory Total: {mem_total:.1f} GB")

    if cpu_count < 4:
        print(
            "Warning: System has fewer than 4 physical CPU cores, test may be limited"
        )

    if mem_total < 8:
        print("Warning: System has less than 8 GB RAM, test may fail early")

    return True


def run_training_envs(num_envs: int, timeout: int) -> Optional[float]:
    """Run training with specific number of environments and measure SPS."""
    print(f"\n{'=' * 60}")
    print(f"Testing {num_envs} parallel environments...")
    print(f"{'=' * 60}")

    # Check if training executable exists
    if not os.path.exists(StressTestConfig.EXECUTABLE_PATH):
        print(f"Compiling training executable first...")
        try:
            compile_result = subprocess.run(
                [
                    "bazel",
                    "build",
                    "//:train",
                    "--copt=-march=native",
                    "--copt=-O3",
                    "--copt=-flto",
                    "--copt=-ffast-math",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print("Compilation successful")
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e}\n{e.stderr}")
            return None

    # Command to run training (valid args for train_headless)
    command = StressTestConfig.BASE_COMMAND + [
        "--envs",
        str(num_envs),
        "--max-steps",
        str(1500),  # Run for enough steps to measure performance
    ]

    try:
        start_time = time.time()
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        sps_measurements = []

        # Monitor process
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                print(f"Process exited prematurely with code: {process.returncode}")
                break

            # Check system resources during run
            cpu_percent = psutil.cpu_percent()
            mem_percent = psutil.virtual_memory().percent

            # Read output
            while True:
                output = process.stdout.readline()
                if not output:
                    break

                output = output.strip()

                # Look for SPS measurements in log output (train_headless format)
                if "SPS" in output and "[JOLTrl]" in output:
                    try:
                        sections = output.strip().split("|")
                        for section in sections:
                            section = section.strip()
                            if "SPS:" in section:
                                sps_str = section.split(":")[1].strip().replace(",", "")
                                sps = float(sps_str)
                                if (
                                    time.time() - start_time
                                    > StressTestConfig.WARMUP_DURATION
                                ):
                                    if sps > 100:
                                        sps_measurements.append(sps)
                                        print(f"  SPS: {sps:.1f}")
                                    else:
                                        print(f"  Ignoring invalid SPS: {sps:.1f}")
                                else:
                                    print(f"  Pre-warmup SPS: {sps:.1f}")
                                break
                    except (IndexError, ValueError):
                        continue

                if "ERROR" in output or "error" in output or "segfault" in output:
                    print(f"Error in training output: {output}")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return None

            # Check for high system resource usage
            if cpu_percent > 99 or mem_percent > 98:
                print(
                    f"System resources exhausted: CPU {cpu_percent:.1f}%, RAM {mem_percent:.1f}%"
                )
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return None

            time.sleep(0.5)

        # If process is still running, terminate it
        if process.poll() is None:
            print("Test completed (timeout reached)")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        # Calculate average SPS from measurements
        if sps_measurements:
            avg_sps = sum(sps_measurements) / len(sps_measurements)
            print(
                f"\nAverage SPS after {len(sps_measurements)} measurements: {avg_sps:.1f}"
            )
            return avg_sps
        else:
            print("No valid SPS measurements collected")
            return None

    except FileNotFoundError:
        print("Error: bazel command not found")
        return None
    except Exception as e:
        print(f"Error running test: {e}")
        return None


def main():
    print("JOLTrl Parallel Environments Stress Test")
    print("=" * 50)

    # Check system resources
    if not check_system_resources():
        return 1

    print("\nStarting stress test...")

    test_results = []
    current_envs = StressTestConfig.MIN_ENVS
    peak_sps = 0.0
    last_stable_envs = 0

    while current_envs <= StressTestConfig.MAX_ENVS:
        print(
            f"Config: {StressTestConfig.TIMEOUT_PER_RUN} sec timeout, {StressTestConfig.WARMUP_DURATION} sec warmup"
        )
        avg_sps = run_training_envs(current_envs, StressTestConfig.TIMEOUT_PER_RUN)

        if avg_sps is not None:
            print(
                f"Test passed for {current_envs} environments, avg SPS: {avg_sps:.1f}"
            )
            test_results.append((current_envs, avg_sps))

            if avg_sps > peak_sps:
                peak_sps = avg_sps
                last_stable_envs = current_envs
                print(f"New peak performance: {avg_sps:.1f} SPS at {current_envs} envs")

            # Check performance threshold
            if avg_sps < StressTestConfig.PERFORMANCE_THRESHOLD * peak_sps:
                print(
                    f"\nPerformance dropped below {StressTestConfig.PERFORMANCE_THRESHOLD * 100:.0f}% of peak SPS"
                )
                print(f"Current SPS: {avg_sps:.1f}, Peak SPS: {peak_sps:.1f}")
                break

            # Double environments for next test
            next_envs = current_envs * 2
            # Prevent going beyond max environments
            if next_envs > StressTestConfig.MAX_ENVS:
                break

            # Wait for system to cool down
            print(f"\nSystem cooldown: Waiting 10 seconds...")
            time.sleep(10)

            current_envs = next_envs

        else:
            print(f"\nTest failed for {current_envs} environments")
            break

    # Generate summary
    print("\n" + "=" * 60)
    print("STRESS TEST COMPLETED")
    print("=" * 60)
    print(f"\nTest Results:")
    print(f"{'Environments':<15} {'SPS':<10} {'Status':<10}")
    print("-" * 35)

    for envs, sps in test_results:
        status = "Stable"
        if sps < StressTestConfig.PERFORMANCE_THRESHOLD * peak_sps:
            status = "Degraded"
        print(f"{envs:<15} {sps:<10.1f} {status:<10}")

    if test_results:
        print(f"\nSummary Statistics:")
        print(f"  Maximum stable environments: {last_stable_envs}")
        print(f"  Peak performance: {peak_sps:.1f} SPS")
        print(
            f"  Environment range tested: {StressTestConfig.MIN_ENVS} - {current_envs}"
        )

        # Calculate scalability
        if len(test_results) > 1:
            first_sps = test_results[0][1]
            best_sps = test_results[-1][1]
            best_envs = test_results[-1][0]
            scalability = (
                (best_sps / first_sps) / (best_envs / StressTestConfig.MIN_ENVS) * 100
            )

            print(f"  Scalability: {scalability:.1f}%")
            print(f"  (performance scaling relative to linear scaling)")

    else:
        print("\nNo valid test results collected")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
