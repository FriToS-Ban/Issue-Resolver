"""
recovery.py — Turn a failed attempt into precise, current-state feedback.

Blind retries ("try again") waste attempts. Every failure is analysed deterministically:

  1. capture the exact error (edit matcher message, syntax error, scope finding, failing
     test ids + output tail, new linter findings),
  2. RE-READ the affected files from disk (the model's view may be stale),
  3. detect partial modifications already present in the working tree (vs. the base commit),
  4. hand the model the current file content around the failure point,
  5. flag evidence that contradicts the plan (e.g. a failing traceback in a file the plan
     never considered) so the orchestrator can revise the plan instead of re-patching.

The resulting ``FailureAnalysis.feedback()`` replaces the previous patch in the next prompt;
the model is asked for a corrected edit against the current state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fix_issue.retrieval.mentions import extract_mentions

FAILURE_CATEGORIES = (
    "llm_error",            # provider/transport failure
    "format_error",         # response not in the required edit format
    "cannot_fix",           # model declined
    "missing_search",       # SEARCH text not found in current file
    "ambiguous_match",      # SEARCH matched several places
    "edit_error",           # other edit errors (unsafe path, file exists, no change...)
    "syntax_error",         # resulting file does not parse
    "scope_violation",      # scope guard blocked the change
    "tests_failed",         # tests fail after the change
    "regression",           # tests that passed at baseline now fail
    "lint_failed",          # new linter / type-checker findings
    "critique_revision",    # reviewer asked for changes
)

MAX_FILE_CHARS = 9000


@dataclass
class FailureAnalysis:
    category: str
    message: str
    files: list[str] = field(default_factory=list)
    partial_changes: list[str] = field(default_factory=list)
    current_excerpts: dict[str, str] = field(default_factory=dict)
    plan_contradiction: str | None = None
    retry_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message[:4000],
            "files": self.files,
            "partial_changes": self.partial_changes,
            "plan_contradiction": self.plan_contradiction,
            "retry_reason": self.retry_reason,
            "details": self.details,
        }

    def feedback(self) -> str:
        parts = [
            f"### Previous attempt failed: {self.category}",
            "Exact error:",
            "```",
            self.message[:4000],
            "```",
        ]
        if self.partial_changes:
            parts.append(
                "Files ALREADY modified in the working tree by earlier steps/attempts (these changes are "
                "in place — do not re-apply them; build on the current content): "
                + ", ".join(self.partial_changes)
            )
        for path, excerpt in self.current_excerpts.items():
            parts.append(f"CURRENT content of {path} (re-read from disk after the failure):\n```\n{excerpt}\n```")
        parts.append(self.retry_reason or "Produce a corrected edit against the CURRENT file content.")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(root: Path, rel: str) -> str | None:
    p = root / rel
    try:
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
    except OSError:
        return None


def _excerpt(text: str, center: int | None = None, radius: int = 25) -> str:
    if len(text) <= MAX_FILE_CHARS or center is None:
        if len(text) <= MAX_FILE_CHARS:
            return text
        center = 1
    lines = text.splitlines()
    a = max(1, center - radius)
    b = min(len(lines), center + radius)
    return f"# (lines {a}-{b} of {len(lines)})\n" + "\n".join(lines[a - 1:b])


def _partial(workspace) -> list[str]:
    try:
        return workspace.changed_files()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Analysers
# ---------------------------------------------------------------------------

def analyze_llm_error(exc: Exception) -> FailureAnalysis:
    return FailureAnalysis("llm_error", f"{type(exc).__name__}: {exc}",
                           retry_reason="The model call failed; retry the same step.")


def analyze_format_failure(parsed, workspace) -> FailureAnalysis:
    msg = "; ".join(parsed.format_errors) or "No SEARCH/REPLACE edit blocks found in the response."
    return FailureAnalysis(
        "format_error", msg + f"\nResponse began with: {parsed.raw[:400]!r}",
        partial_changes=_partial(workspace),
        retry_reason="Respond with INTENT: followed by FILE/SEARCH/REPLACE blocks exactly in the required format.",
    )


def analyze_edit_failure(batch, workspace) -> FailureAnalysis:
    """Edits were rejected before anything was written (plan_edits is atomic)."""
    root = Path(workspace.root)
    failed = [o for o in batch.outcomes if o.status == "failed"]
    syntax = {p: e for p, e in (batch.validation or {}).items() if e}
    kinds = [o.error_kind for o in failed]
    if syntax and not failed:
        category = "syntax_error"
    elif "missing_search" in kinds:
        category = "missing_search"
    elif "ambiguous_match" in kinds:
        category = "ambiguous_match"
    else:
        category = "edit_error"
    messages: list[str] = []
    excerpts: dict[str, str] = {}
    files: list[str] = []
    for o in failed:
        messages.append(f"[{o.error_kind}] {o.edit.file}: {o.error}")
        files.append(o.edit.file)
        current = _read(root, o.edit.file)
        if current is not None and o.edit.file not in excerpts:
            center = (o.best_match or {}).get("start_line") or o.line
            excerpts[o.edit.file] = _excerpt(current, center)
    for path, err in syntax.items():
        messages.append(f"[syntax_error] {path}: {err} (the edits were NOT written)")
        files.append(path)
        current = _read(root, path)
        if current is not None and path not in excerpts:
            excerpts[path] = _excerpt(current)
    applied = [o.edit.file for o in batch.outcomes if o.status in ("applied", "already_applied")]
    reason = (
        "None of the edits in that response were written (edits are applied atomically). "
        "Re-issue ALL edits for this step, copying SEARCH text exactly from the CURRENT content shown above."
    )
    if applied:
        reason += f" (Edits to {', '.join(sorted(set(applied)))} matched but were discarded with the batch.)"
    return FailureAnalysis(category, "\n".join(messages), sorted(set(files)), _partial(workspace),
                           excerpts, retry_reason=reason,
                           details={"outcomes": batch.to_dict().get("outcomes", []) if hasattr(batch, "to_dict") else []})


def analyze_scope_failure(report) -> FailureAnalysis:
    msg = report.blocking_message() if hasattr(report, "blocking_message") else str(report)
    return FailureAnalysis(
        "scope_violation", msg,
        files=sorted({f.path for f in report.findings if f.path and f.severity == "block"}),
        retry_reason=(
            "The scope guard blocked that change; it was NOT written. Find a fix that stays within the "
            "allowed files and the size limit (do not touch blocked paths). If that is impossible, answer CANNOT_FIX."
        ),
        details={"scope": report.to_dict() if hasattr(report, "to_dict") else {}},
    )


def analyze_test_failure(workspace, test_result, baseline_failed: set[str], plan_files: list[str],
                         repo_files: list[str]) -> FailureAnalysis:
    root = Path(workspace.root)
    failed_now = set(test_result.failed_tests or [])
    # New failures = tests that did not fail before the patch (the patch broke them).
    # Baseline failures that persist mean the fix is incomplete, not a regression.
    new_failures = sorted(failed_now - baseline_failed)
    category = "regression" if new_failures else "tests_failed"
    out = test_result.output or ""
    mentions = extract_mentions(out[-6000:], repo_files)
    frames = [f for f in mentions.frames if not f.path.startswith(("tests/", "test/"))]
    excerpts: dict[str, str] = {}
    for fr in frames[-2:]:
        text = _read(root, fr.path)
        if text is not None and fr.path not in excerpts:
            excerpts[fr.path] = _excerpt(text, fr.line)
    outside = [f.path for f in frames if f.path not in plan_files]
    contradiction = None
    if outside:
        contradiction = (
            f"The failing traceback runs through {', '.join(sorted(set(outside)))}, which the plan did not "
            "consider — the root-cause hypothesis may be wrong or incomplete."
        )
    summary = f"{test_result.summary or 'tests failed'}; failing: {', '.join(sorted(failed_now)[:15]) or 'see output'}"
    if new_failures:
        summary += f"\nNEW failures vs. baseline (broken by the patch): {', '.join(new_failures[:15])}"
    persisting = sorted(failed_now & baseline_failed)
    if persisting:
        summary += f"\nStill failing since before the patch: {', '.join(persisting[:15])}"
    return FailureAnalysis(
        category, f"{summary}\n\n{out[-3500:]}", sorted(set(f.path for f in frames)), _partial(workspace),
        excerpts, contradiction,
        retry_reason=(
            "Your previous edits ARE applied in the working tree (see the cumulative diff). Diagnose the failure "
            "from the output above and produce corrective SEARCH/REPLACE edits against the CURRENT files. "
            "You may revert or adjust earlier edits."
        ),
        details={"failed_tests": sorted(failed_now), "new_failures": new_failures},
    )


def analyze_validator_failure(results, workspace) -> FailureAnalysis:
    lines = []
    files: list[str] = []
    for r in results:
        for f in getattr(r, "new_findings", []) or []:
            lines.append(f"{r.tool}: {f.path}:{f.line}: {f.code} {f.message}")
            files.append(f.path)
    return FailureAnalysis(
        "lint_failed", "\n".join(lines[:40]) or "new findings", sorted(set(files)), _partial(workspace),
        retry_reason="Fix the NEW linter/type-checker findings introduced by your change (pre-existing ones are ignored).",
    )


def analyze_critique(critique, workspace) -> FailureAnalysis:
    return FailureAnalysis(
        "critique_revision", critique.feedback(), [], _partial(workspace),
        retry_reason=(
            "An independent reviewer requested changes. Your current edits are applied; make the requested "
            "corrections with new SEARCH/REPLACE edits against the CURRENT files."
        ),
    )
