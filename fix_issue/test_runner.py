"""
test_runner.py — Run the target repo's test suite and capture the result.

Designed to do one thing: execute the detected test command, capture output,
and return a structured result. The retry loop is in cli.py.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestResult:
    passed: bool
    output: str          # combined stdout + stderr
    duration_seconds: float
    command: list[str]


class NoTestCommandError(Exception):
    """Raised when no test command can be detected."""


def run_tests(
    command: list[str],
    cwd: Path,
    timeout_seconds: int = 300,
) -> TestResult:
    """
    Execute `command` in `cwd` and return a TestResult.

    Parameters
    ----------
    command:          The test command to run (e.g. ["python", "-m", "pytest", "-q"]).
    cwd:              Directory to run the command in (repo root).
    timeout_seconds:  Hard kill timeout (default 5 minutes).
    """
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        duration = time.monotonic() - start
        output = _combine(proc.stdout, proc.stderr)
        passed = proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        output = (
            f"Test suite timed out after {timeout_seconds}s.\n"
            f"Partial stdout:\n{exc.stdout or ''}\n"
            f"Partial stderr:\n{exc.stderr or ''}"
        )
        passed = False
    except FileNotFoundError:
        duration = time.monotonic() - start
        output = (
            f"Test command not found: {command[0]}. "
            f"Make sure the test runner is installed in the repo's environment."
        )
        passed = False

    return TestResult(
        passed=passed,
        output=output[-8000:],  # keep the tail — most relevant for failures
        duration_seconds=duration,
        command=command,
    )


def _combine(stdout: str, stderr: str) -> str:
    parts = []
    if stdout.strip():
        parts.append(stdout)
    if stderr.strip():
        parts.append(stderr)
    return "\n".join(parts)
