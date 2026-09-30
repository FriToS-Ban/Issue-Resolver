"""Tests for scope_guard.py"""
import textwrap
from pathlib import Path

import pytest

from fix_issue.scope_guard import ScopeViolation, check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL_DIFF = textwrap.dedent("""\
    diff --git a/foo.py b/foo.py
    --- a/foo.py
    +++ b/foo.py
    @@ -1,3 +1,3 @@
    -x = 1
    +x = 2
     y = 3
""")

def _make_ignore(tmp_path: Path, patterns: list[str]) -> None:
    ignore = tmp_path / ".fix-issue-ignore"
    ignore.write_text("\n".join(patterns) + "\n")


# ---------------------------------------------------------------------------
# Size checks
# ---------------------------------------------------------------------------

def test_small_diff_passes(tmp_path):
    check(SMALL_DIFF, tmp_path, max_diff_lines=150)  # should not raise


def test_oversized_diff_raises(tmp_path):
    big_diff = "\n".join(["+line"] * 200)
    with pytest.raises(ScopeViolation, match="too large"):
        check(big_diff, tmp_path, max_diff_lines=150)


def test_exactly_at_limit_passes(tmp_path):
    diff = "\n".join(["+line"] * 150)
    check(diff, tmp_path, max_diff_lines=150)  # boundary: should pass


def test_one_over_limit_raises(tmp_path):
    diff = "\n".join(["+line"] * 151)
    with pytest.raises(ScopeViolation):
        check(diff, tmp_path, max_diff_lines=150)


def test_header_lines_not_counted(tmp_path):
    """--- and +++ header lines must not count toward the size limit."""
    diff = textwrap.dedent("""\
        --- a/file.py
        +++ b/file.py
        +actual change
    """)
    # Only 1 real changed line; should pass even with a tight limit
    check(diff, tmp_path, max_diff_lines=1)


# ---------------------------------------------------------------------------
# Ignored paths
# ---------------------------------------------------------------------------

def test_no_ignore_file_passes(tmp_path):
    check(SMALL_DIFF, tmp_path, max_diff_lines=150)


def test_unmatched_pattern_passes(tmp_path):
    _make_ignore(tmp_path, ["auth/**", "payments/**"])
    check(SMALL_DIFF, tmp_path, max_diff_lines=150)  # touches foo.py → passes


def test_matching_glob_raises(tmp_path):
    _make_ignore(tmp_path, ["auth/**"])
    diff = textwrap.dedent("""\
        diff --git a/auth/login.py b/auth/login.py
        --- a/auth/login.py
        +++ b/auth/login.py
        @@ -1 +1 @@
        -old
        +new
    """)
    with pytest.raises(ScopeViolation, match="auth/login.py"):
        check(diff, tmp_path, max_diff_lines=150)


def test_comment_lines_in_ignore_ignored(tmp_path):
    _make_ignore(tmp_path, ["# this is a comment", "foo.py"])
    with pytest.raises(ScopeViolation):
        check(SMALL_DIFF, tmp_path, max_diff_lines=150)


def test_blank_lines_in_ignore_ignored(tmp_path):
    _make_ignore(tmp_path, ["", "  ", "auth/**"])
    # SMALL_DIFF touches foo.py — should still pass
    check(SMALL_DIFF, tmp_path, max_diff_lines=150)


def test_wildcard_extension_pattern(tmp_path):
    _make_ignore(tmp_path, ["*.env"])
    diff = textwrap.dedent("""\
        --- a/.env
        +++ b/.env
        @@ -1 +1 @@
        -SECRET=old
        +SECRET=new
    """)
    with pytest.raises(ScopeViolation, match=r"\.env"):
        check(diff, tmp_path, max_diff_lines=150)
