"""Tests for pr_builder.py — confidence score formula"""
import pytest

from fix_issue.pr_builder import PRContent, _calculate_score, _count_diff_lines, build_pr


# ---------------------------------------------------------------------------
# _count_diff_lines
# ---------------------------------------------------------------------------

def test_count_ignores_headers():
    diff = "--- a/foo.py\n+++ b/foo.py\n+added\n-removed\n unchanged"
    assert _count_diff_lines(diff) == 2  # only +added and -removed


def test_count_empty_diff():
    assert _count_diff_lines("") == 0


# ---------------------------------------------------------------------------
# _calculate_score — formula verification
# ---------------------------------------------------------------------------

def test_perfect_score():
    # Small diff, tests passed, high triage confidence → 10
    assert _calculate_score(0, True, 0.95) == 10


def test_size_penalty_one_point_per_30_lines():
    # 30 lines → −1
    assert _calculate_score(30, True, 0.95) == 9
    # 60 lines → −2
    assert _calculate_score(60, True, 0.95) == 8
    # 150 lines → −5 (max size penalty)
    assert _calculate_score(150, True, 0.95) == 5


def test_size_penalty_capped_at_5():
    # 300 lines → still only −5
    assert _calculate_score(300, True, 0.95) == 5


def test_failed_tests_penalty():
    # No tests, no other penalties → 10 − 3 = 7
    assert _calculate_score(0, False, 0.95) == 7


def test_low_triage_confidence_penalty():
    # Low confidence, tests pass → 10 − 2 = 8
    assert _calculate_score(0, True, 0.5) == 8


def test_all_penalties_combined():
    # 150 lines (−5) + failed tests (−3) + low confidence (−2) = 10 − 10 = floor at 1
    assert _calculate_score(150, False, 0.3) == 1


def test_score_never_below_1():
    assert _calculate_score(9999, False, 0.0) == 1


def test_score_boundary_confidence():
    # exactly 0.7 → no penalty
    assert _calculate_score(0, True, 0.7) == 10
    # just below 0.7 → penalty
    assert _calculate_score(0, True, 0.699) == 8


# ---------------------------------------------------------------------------
# build_pr — smoke test
# ---------------------------------------------------------------------------

_SAMPLE_ISSUE = {
    "number": 42,
    "title": "Fix the login bug",
    "body": "Users cannot log in when the session expires.",
    "labels": [],
}

_SAMPLE_DIFF = "--- a/auth.py\n+++ b/auth.py\n-old\n+new\n"


def test_build_pr_returns_prcontent():
    result = build_pr(
        issue=_SAMPLE_ISSUE,
        diff_text=_SAMPLE_DIFF,
        triage_reason="Small bug in auth.py",
        triage_confidence=0.85,
        llm_summary="Fixed the session expiry check.",
        tests_passed=True,
        test_command=["pytest"],
        attempt_count=1,
        repo="owner/repo",
    )
    assert isinstance(result, PRContent)
    assert "42" in result.title
    assert result.confidence >= 1
    assert result.confidence <= 10


def test_build_pr_body_contains_formula_table():
    result = build_pr(
        issue=_SAMPLE_ISSUE,
        diff_text=_SAMPLE_DIFF,
        triage_reason="Bug.",
        triage_confidence=0.9,
        llm_summary="Summary.",
        tests_passed=True,
        test_command=["pytest"],
        attempt_count=1,
        repo="owner/repo",
    )
    # The formula table must be present so reviewers can verify the score
    assert "Penalty Applied" in result.body
    assert "Confidence Score" in result.body


def test_build_pr_closes_issue():
    result = build_pr(
        issue=_SAMPLE_ISSUE,
        diff_text=_SAMPLE_DIFF,
        triage_reason="Bug.",
        triage_confidence=0.9,
        llm_summary="Summary.",
        tests_passed=True,
        test_command=None,
        attempt_count=1,
        repo="owner/repo",
    )
    assert "Resolves #42" in result.body or "closes #42" in result.title.lower()
