"""
bench/validate.py — Prove every benchmark case is well-formed before trusting results.

For each fix case we check, in fresh temporary copies:

  clean fixture  : visible suite passes, hidden tests pass   (the oracle is satisfiable)
  buggy baseline : hidden tests FAIL                          (the bug is real)
                   visible suite passes, or fails when the case declares
                   ``visible_tests_fail_before = true``       (matches the scenario)

Triage cases only need to materialise cleanly.  Nothing here calls an LLM.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from fix_issue.bench.cases import (
    HIDDEN_TEST_DIR,
    Case,
    CaseError,
    copy_fixture,
    inject_bug,
    install_hidden_tests,
)
from fix_issue.sandbox import safe_env


@dataclass
class ValidationReport:
    case_id: str
    ok: bool
    problems: list[str] = field(default_factory=list)


def run_pytest(root: Path, targets: list[str] | None = None, timeout: int = 180) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short"]
    if targets:
        cmd += targets
    else:
        cmd += ["--ignore", HIDDEN_TEST_DIR]
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout, env=safe_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]


def validate_case(case: Case) -> ValidationReport:
    rep = ValidationReport(case.id, ok=True)

    def problem(msg: str) -> None:
        rep.ok = False
        rep.problems.append(msg)

    with tempfile.TemporaryDirectory(prefix=f"bench-clean-{case.id[:20]}-") as tmp:
        clean = Path(tmp)
        copy_fixture(case, clean)
        ok, out = run_pytest(clean)
        if not ok:
            problem(f"clean fixture: visible tests fail\n{out}")
        if case.is_fix:
            hidden = install_hidden_tests(case, clean)
            ok, out = run_pytest(clean, hidden)
            if not ok:
                problem(f"clean fixture: hidden tests fail (oracle unsatisfiable)\n{out}")

    with tempfile.TemporaryDirectory(prefix=f"bench-bug-{case.id[:20]}-") as tmp:
        buggy = Path(tmp)
        copy_fixture(case, buggy)
        try:
            inject_bug(case, buggy)
        except CaseError as exc:
            problem(str(exc))
            return rep
        ok, out = run_pytest(buggy)
        if case.visible_tests_fail_before and ok:
            problem("buggy baseline: visible tests pass but case declares visible_tests_fail_before")
        if not case.visible_tests_fail_before and not ok:
            problem(f"buggy baseline: visible tests fail unexpectedly\n{out}")
        if case.is_fix:
            hidden = install_hidden_tests(case, buggy)
            ok, out = run_pytest(buggy, hidden)
            if ok:
                problem("buggy baseline: hidden tests PASS — the injected bug is not detected")
        for path in case.expected_files:
            if not (buggy / path).exists():
                problem(f"expected file '{path}' does not exist in the fixture")
    return rep
