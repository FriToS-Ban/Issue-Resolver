"""
pr_builder.py — Compose the PR title and body with an honestly-defined confidence score.

Confidence formula (defined, not opaque):
  score = 10
  score -= min(5, diff_lines // 30)      # −1 per 30 lines, max penalty −5
  score -= 3  if not tests_passed        # −3 if test suite didn't go green
  score -= 2  if triage_confidence < 0.7 # −2 if triage was uncertain
  score -= 2  if review == HUMAN_REVIEW  # −2 if the independent reviewer could not approve
  score -= 1  if scope needs review      # −1 if sensitive paths were touched
  score = max(1, score)                  # floor at 1 — never claim zero confidence

The last two terms only apply when the new pipeline supplies a ``PRExplanation``, so legacy
callers get exactly the original formula. The formula and each component are shown
explicitly in the PR body so reviewers can see exactly how the score was derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PRContent:
    title: str
    body: str
    confidence: int  # 1–10
    draft: bool = False


@dataclass
class PRExplanation:
    """Structured explanation produced by the new pipeline (all fields optional)."""
    root_cause: str = ""
    files_modified: list[tuple[str, int]] = field(default_factory=list)   # (path, changed lines)
    why_it_works: list[str] = field(default_factory=list)
    tests_executed: list[str] = field(default_factory=list)               # human-readable lines
    validators: list[str] = field(default_factory=list)
    review_verdict: str | None = None
    review_summary: str = ""
    review_gates: list[str] = field(default_factory=list)
    scope_notes: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    run_id: str | None = None


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
    explanation: PRExplanation | None = None,
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
    explanation:        Optional structured explanation (root cause, files, validation, review).
    """
    diff_lines = _count_diff_lines(diff_text)
    verdict = explanation.review_verdict if explanation else None
    scope_review = bool(explanation and explanation.scope_notes)
    score = _calculate_score(diff_lines, tests_passed, triage_confidence,
                             critique_verdict=verdict, scope_review=scope_review)
    issue_number = issue.get("number", "?")
    issue_title = issue.get("title", "Unknown issue")

    title = f"fix: {issue_title} (closes #{issue_number})"

    test_status = "✅ Passing" if tests_passed else "⚠️ Not verified (no test command found or tests failed)"
    test_cmd_str = " ".join(test_command) if test_command else "N/A"

    size_penalty = min(5, diff_lines // 30)
    test_penalty = 0 if tests_passed else 3
    conf_penalty = 0 if triage_confidence >= 0.7 else 2
    review_penalty = 2 if verdict == "HUMAN_REVIEW" else 0
    scope_penalty = 1 if scope_review else 0

    extra_rows = ""
    formula_extra = ""
    if explanation is not None:
        extra_rows = (
            f"| Independent review ({verdict or 'n/a'}) | −2 if HUMAN_REVIEW | −{review_penalty} |\n"
            f"| Sensitive paths touched | −1 if the scope guard requires review | −{scope_penalty} |\n"
        )
        formula_extra = " − review_penalty − scope_penalty"

    root_cause = explanation.root_cause if explanation and explanation.root_cause else triage_reason
    explanation_md = _render_explanation(explanation) if explanation else ""
    run_note = f" Run id: `{explanation.run_id}`." if explanation and explanation.run_id else ""

    body = f"""\
{_review_banner(verdict)}## Root Cause
{root_cause}

## Fix Summary
{llm_summary}
{explanation_md}
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
{extra_rows}| **Total** | | **{score}/10** |

*Formula: `score = max(1, 10 − size_penalty − test_penalty − conf_penalty{formula_extra})`.
This is honest math, not a heuristic black box.*

---

Resolves #{issue_number}

---
*This PR was opened automatically by [fix-issue](https://github.com/FriToS-Ban/Issue-Resolver).*{run_note}
*Always review before merging. The confidence score is a guide, not a guarantee.*
"""

    return PRContent(title=title, body=body, confidence=score, draft=verdict == "HUMAN_REVIEW")


def _review_banner(verdict: str | None) -> str:
    if verdict == "HUMAN_REVIEW":
        return (
            "> [!WARNING]\n> The independent review could not approve this change automatically "
            "(see *Independent Review*). It was opened as a draft: a human must verify it.\n\n"
        )
    return ""


def _render_explanation(e: PRExplanation) -> str:
    out: list[str] = []
    if e.files_modified:
        out.append("\n## Files Modified")
        out += [f"- `{p}` ({n} changed lines)" for p, n in e.files_modified]
    if e.why_it_works:
        out.append("\n## Why This Should Work")
        out += [f"- {w}" for w in e.why_it_works]
    if e.tests_executed or e.validators:
        out.append("\n## Validation Executed")
        out += [f"- {t}" for t in e.tests_executed]
        out += [f"- {v}" for v in e.validators]
    if e.review_verdict:
        out.append("\n## Independent Review")
        summary = f" — {e.review_summary}" if e.review_summary else ""
        out.append(f"**Verdict:** `{e.review_verdict}`{summary}")
        out += [f"- gate: {g}" for g in e.review_gates]
    if e.scope_notes:
        out.append("\n## Scope Notes")
        out += [f"- {n}" for n in e.scope_notes]
    if e.limitations:
        out.append("\n## Remaining Limitations")
        out += [f"- {lim}" for lim in e.limitations]
    return "\n".join(out) + "\n" if out else ""


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


def _calculate_score(
    diff_lines: int,
    tests_passed: bool,
    triage_confidence: float,
    critique_verdict: str | None = None,
    scope_review: bool = False,
) -> int:
    score = 10
    score -= min(5, diff_lines // 30)
    if not tests_passed:
        score -= 3
    if triage_confidence < 0.7:
        score -= 2
    if critique_verdict == "HUMAN_REVIEW":
        score -= 2
    if scope_review:
        score -= 1
    return max(1, score)
