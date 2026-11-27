"""
test_runner.py — Run the target repo's test suite and capture the result.

Designed to do one thing: execute the detected test command, capture output,
and return a structured result. The retry loop lives in the orchestrator.

Commands run through ``sandbox.run_command`` (the only way this module starts a process):
the child gets an allowlist environment (no tokens or API keys), no shell is involved, the
timeout is wall-clock and kills the whole process tree, output is captured in temporary files
(a lingering grandchild cannot block us), and a ``.bat``/``.cmd`` command is refused when an
argument contains characters cmd.exe would interpret.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from fix_issue.sandbox import run_command


@dataclass
class TestResult:
    passed: bool
    output: str          # combined stdout + stderr
    duration_seconds: float
    command: list[str]
    failed_tests: list[str] = field(default_factory=list)   # pytest node ids, when parseable
    summary: str = ""                                       # e.g. "3 failed, 41 passed"
    timed_out: bool = False

    __test__ = False  # not a pytest test class


class NoTestCommandError(Exception):
    """Raised when no test command can be detected."""


_FAILED_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)
_SUMMARY_RE = re.compile(r"=+\s*(.*?(?:passed|failed|error|errors|skipped|no tests ran).*?)\s+in\s+[\d.]+s", re.IGNORECASE)


def parse_pytest_failures(output: str) -> list[str]:
    ids = []
    for m in _FAILED_RE.finditer(output):
        node = m.group(1).split(" - ")[0]
        if node not in ids:
            ids.append(node)
    return ids


def parse_summary(output: str) -> str:
    matches = _SUMMARY_RE.findall(output)
    return matches[-1].strip("= ") if matches else ""


def run_tests(
    command: list[str],
    cwd: Path,
    timeout_seconds: int = 300,
    extra_args: list[str] | None = None,
    prefix: list[str] | None = None,
) -> TestResult:
    """
    Execute `command` in `cwd` and return a TestResult.

    Parameters
    ----------
    command:          The test command to run (e.g. ["python", "-m", "pytest", "-q"]).
    cwd:              Directory to run the command in (repo root).
    timeout_seconds:  Hard kill timeout (default 5 minutes).
    extra_args:       Appended to the command (e.g. specific test paths).
    prefix:           Optional sandbox wrapper (e.g. a docker run invocation).
    """
    full = [*command, *(extra_args or [])]
    res = run_command(full, cwd, timeout=timeout_seconds, prefix=prefix)
    if res.timed_out:
        output = f"Test suite timed out after {timeout_seconds}s.\n{res.output}"
    elif res.refused:
        output = res.output
    elif res.not_found:
        output = (
            f"Test command not found: {full[0]}. "
            f"Make sure the test runner is installed in the repo's environment."
        )
    else:
        output = res.output
    # pytest exit code 5 = "no tests collected"; treat as not passed.
    passed = res.returncode == 0
    return TestResult(
        passed=passed,
        output=output[-8000:],  # keep the tail — most relevant for failures
        duration_seconds=res.duration_seconds,
        command=full,
        failed_tests=parse_pytest_failures(output),
        summary=parse_summary(output),
        timed_out=res.timed_out,
    )
