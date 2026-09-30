"""
pr_builder.py — Compose the PR title and body with an honestly-defined confidence score.

Confidence formula (defined, not opaque):
  score = 10
  score -= min(5, diff_lines // 30)      # −1 per 30 lines, max penalty −5
  score -= 3  if not tests_passed        # −3 if test suite didn't go green
  score -= 2  if triage_confidence < 0.7 # −2 if triage was uncertain
  score = max(1, score)                  # floor at 1 — never claim zero confidence

The formula and each component are shown explicitly in the PR body so reviewers
can see exactly how the score was derived.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class PRContent:
    title: str
    body: str
    confidence: int  # 1–10


def build_pr(
    *,
    issue: dict[str, Any],
    diff_text: str,
    triage_reason: str,
    triage_confidence: float,
    llm_summary: str,
    tests_passed: bool,
    test_command: list[str] | None,
    attempt_count: int,
    repo: str,
) -> PRContent:
    """
    Compose the PR title and body.

    Parameters
    ----------
    issue:              Raw GitHub issue dict.
    diff_text:          The unified diff that was applied.
    triage_reason:      One-sentence triage reason.
    triage_confidence:  0.0–1.0 float from triage step.
    llm_summary:        One-paragraph plain-English fix explanation from the LLM.
    tests_passed:       Whether the test suite was green after the patch.
    test_command:       The command that was run (for display).
    attempt_count:      How many patch-generate/apply/test cycles were needed.
    repo:               "owner/repo" string.
    """
    diff_lines = _count_diff_lines(diff_text)
    score = _calculate_score(diff_lines, tests_passed, triage_confidence)
    issue_number = issue.get("number", "?")
    issue_title = issue.get("title", "Unknown issue")

    title = f"fix: {issue_title} (closes #{issue_number})"

    test_status = "✅ Passing" if tests_passed else "⚠️ Not verified (no test command found or tests failed)"
    test_cmd_str = " ".join(test_command) if test_command else "N/A"

    size_penalty = min(5, diff_lines // 30)
    test_penalty = 0 if tests_passed else 3
    conf_penalty = 0 if triage_confidence >= 0.7 else 2
    computed = 10 - size_penalty - test_penalty - conf_penalty

    body = f"""\
## Root Cause
{triage_reason}

## Fix Summary
{llm_summary}

## Changes
- **Diff size:** {diff_lines} changed lines across the patch
- **Attempts needed:** {attempt_count}
- **Test suite ({test_cmd_str}):** {test_status}

## Confidence Score: {score}/10

| Component | Value | Penalty Applied |
|---|---|---|
| Diff size ({diff_lines} lines) | 1 point deducted per 30 lines, max −5 | −{size_penalty} |
| Tests green | −3 if test suite did not pass | −{test_penalty} |
| Triage confidence ({triage_confidence:.0%}) | −2 if below 70% | −{conf_penalty} |
| **Total** | | **{score}/10** |

*Formula: `score = max(1, 10 − size_penalty − test_penalty − conf_penalty)`.
This is honest math, not a heuristic black box.*

---

Resolves #{issue_number}

---
*This PR was opened automatically by [fix-issue](https://github.com/your-org/fix-issue).*
*Always review before merging. The confidence score is a guide, not a guarantee.*
"""

    return PRContent(title=title, body=body, confidence=score)


def generate_llm_summary(llm: Any, issue: dict[str, Any], diff_text: str) -> str:
    """
    Ask the LLM for a one-paragraph plain-English explanation of what the fix does.
    Uses the same LLMProvider interface.
    """
    prompt = f"""\
In one short paragraph (3–5 sentences), explain in plain English what the following patch does
and why it fixes the reported issue. Write as if explaining to a code reviewer.

Issue title: {issue.get('title', '')}
Issue body: {(issue.get('body') or '')[:1000]}

Patch:
```diff
{diff_text[:3000]}
```

Output only the paragraph — no headings, no bullet points.
"""
    return llm.generate(prompt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_diff_lines(diff_text: str) -> int:
    """Count added/removed lines (excluding file headers)."""
    return sum(
        1
        for line in diff_text.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("---")
        and not line.startswith("+++")
    )


def _calculate_score(diff_lines: int, tests_passed: bool, triage_confidence: float) -> int:
    score = 10
    score -= min(5, diff_lines // 30)
    if not tests_passed:
        score -= 3
    if triage_confidence < 0.7:
        score -= 2
    return max(1, score)
