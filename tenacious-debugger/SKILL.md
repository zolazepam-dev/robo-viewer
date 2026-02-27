---
name: tenacious-debugger
description: A rigorous debugging protocol. Use when a user reports a persistent bug or explicitly requests a fix without workarounds. This skill enforces strict root-cause analysis, forbids neutering functionality to bypass issues, and requires exhaustive verification until the exact bug is resolved.
---

# Tenacious Debugging Protocol

When activated, you must adopt a zero-compromise approach to resolving the user's reported bug. Your primary directive is to fix the problem at its source, preserving all intended functionality.

## Core Mandates

1. **No Workarounds**: Never comment out, disable, or neuter a feature to make an error go away. The goal is a functional system, not a quiet compiler or a suppressed exception.
2. **Root Cause Analysis (RCA)**: Before writing any code, you must investigate the stack trace, surrounding logic, and project architecture to identify the exact mechanism of failure. State your RCA explicitly.
3. **Preserve Architectural Intent**: Your fix must align with the existing design patterns. Do not introduce entirely new libraries or rewrite unrelated systems just to bypass a localized bug.
4. **Empirical Verification**: Do not assume a fix works just because the syntax is correct. You must plan and execute a verification strategy (e.g., running the specific test, building the specific target) to confirm the exact bug is resolved.

## Workflow

1. **Investigate**: Use `grep_search`, `glob`, and `read_file` to trace the execution path leading to the bug. Look at the immediate failure point and the logic that feeds data into it.
2. **Diagnose**: Explicitly state: "The root cause of this failure is..." and explain *why* the current logic is flawed.
3. **Plan**: Propose a precise, surgical fix that directly addresses the root cause while maintaining all existing capabilities. Wait for user confirmation if the fix requires significant structural changes.
4. **Execute**: Apply the fix.
5. **Validate**: Run the build or test command associated with the failing component. If it fails again, return to step 1. Do not stop until the validation passes.
