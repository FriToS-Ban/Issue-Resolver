"""
scope_guard.py — Enforce scope limits BEFORE running any tests or opening a PR.

Checks two things:
  1. Does the diff touch any path listed in the target repo's .fix-issue-ignore?
  2. Does the diff exceed MAX_DIFF_LINES?

Both checks raise ScopeViolation, which the CLI catches and converts to a
"needs review — too large / out of scope" outcome without burning a test run.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path


class ScopeViolation(Exception):
    """Raised when a diff violates scope limits. Message is user-readable."""


def check(
    diff_text: str,
    repo_root: Path,
    max_diff_lines: int = 150,
) -> None:
    """
    Run all scope checks against `diff_text`.

    Parameters
    ----------
    diff_text:      Unified diff string to validate.
    repo_root:      Root of the temp clone (used to find .fix-issue-ignore).
    max_diff_lines: Maximum number of ±-prefixed lines allowed in the diff.

    Raises
    ------
    ScopeViolation  — with a human-readable explanation.
    """
    _check_diff_size(diff_text, max_diff_lines)
    _check_ignored_paths(diff_text, repo_root)


# ---------------------------------------------------------------------------
# Size check
# ---------------------------------------------------------------------------

def _check_diff_size(diff_text: str, max_diff_lines: int) -> None:
    """Count lines that add or remove code (+ or - prefix, excluding --- and +++ headers)."""
    change_lines = [
        line for line in diff_text.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("---")
        and not line.startswith("+++")
    ]
    actual = len(change_lines)
    if actual > max_diff_lines:
        raise ScopeViolation(
            f"Diff is too large: {actual} changed lines (limit is {max_diff_lines}). "
            f"This issue likely requires a larger refactor — needs human review. "
            f"Increase FIX_ISSUE_MAX_DIFF_LINES in ~/.fix-issue.toml to override."
        )


# ---------------------------------------------------------------------------
# Ignored paths check
# ---------------------------------------------------------------------------

_DIFF_FILE_RE = re.compile(r"^(?:---|\+\+\+) (?:a/|b/)(.+)$", re.MULTILINE)


def _parse_changed_paths(diff_text: str) -> list[str]:
    """Extract file paths from `--- a/...` and `+++ b/...` diff headers."""
    paths: set[str] = set()
    for m in _DIFF_FILE_RE.finditer(diff_text):
        p = m.group(1).strip()
        if p != "/dev/null":
            paths.add(p)
    return sorted(paths)


def _load_ignore_patterns(repo_root: Path) -> list[str]:
    """
    Read .fix-issue-ignore from the repo root.
    Lines starting with # are comments. Blank lines are skipped.
    """
    ignore_file = repo_root / ".fix-issue-ignore"
    if not ignore_file.exists():
        return []
    lines = ignore_file.read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def _check_ignored_paths(diff_text: str, repo_root: Path) -> None:
    patterns = _load_ignore_patterns(repo_root)
    if not patterns:
        return

    changed_paths = _parse_changed_paths(diff_text)
    for path in changed_paths:
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path.split("/")[-1], pattern):
                raise ScopeViolation(
                    f"Diff touches '{path}' which matches the ignored pattern '{pattern}' "
                    f"in .fix-issue-ignore. This path is off-limits for the agent."
                )
