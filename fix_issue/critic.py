"""
critic.py — Independent review of a validated patch before any PR is opened.

Independence measures:
  * a separate system prompt (and optionally a separate model via the [critic] config),
  * the reviewer sees artifacts only — issue, acceptance criteria, plan, full diff,
    post-change surrounding code, test/linter results and the scope report — and NEVER the
    implementer's own claims ("INTENT"/summaries) about why the patch works,
  * the LLM verdict is then passed through deterministic gates that can only make the
    outcome MORE conservative.

Questions the reviewer must answer (structured):
  1 addresses_issue   2 all_relevant_files_updated   3 regressions
  4 missing_edge_cases   5 tests_meaningful   6 change_minimal

Verdicts: APPROVED | NEEDS_REVISION | HUMAN_REVIEW | REJECTED

Deterministic gates (applied in order, each recorded in ``gates``):
  G1 tests did not run              -> at most HUMAN_REVIEW
  G2 tests failed                   -> at most NEEDS_REVISION
  G3 scope requires human review    -> at most HUMAN_REVIEW
  G4 scope blocked                  -> REJECTED
  G5 new linter/type findings       -> at most NEEDS_REVISION
  G6 APPROVED but reviewer says the patch does not (fully) address the issue
                                    -> NEEDS_REVISION
  G7 APPROVED but reviewer says tests do not verify the fix -> HUMAN_REVIEW
  G8 APPROVED but likely regressions -> NEEDS_REVISION
  G9 strict mode (high-risk triage): APPROVED requires every answer positive, else HUMAN_REVIEW
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fix_issue.llm_provider import LLMProvider
from fix_issue.structured import Field, StructuredOutputError, generate_structured

VERDICTS = ("APPROVED", "NEEDS_REVISION", "HUMAN_REVIEW", "REJECTED")
_RANK = {"APPROVED": 0, "HUMAN_REVIEW": 1, "NEEDS_REVISION": 2, "REJECTED": 3}

_ANSWER = {"answer": Field(str, lower=True, default="unknown"), "details": Field(str, default="")}
CRITIQUE_SCHEMA = {
    "verdict": Field(str, required=True, upper=True, enum=VERDICTS),
    "addresses_issue": Field(dict, schema=_ANSWER, default=dict),
    "all_relevant_files_updated": Field(dict, schema=_ANSWER, default=dict),
    "regressions": Field(dict, schema=_ANSWER, default=dict),
    "missing_edge_cases": Field(list, items=str, default=list, max_items=10),
    "tests_meaningful": Field(dict, schema=_ANSWER, default=dict),
    "change_minimal": Field(dict, schema=_ANSWER, default=dict),
    "required_changes": Field(list, items=str, default=list, max_items=10),
    "summary": Field(str, default=""),
}

_SYSTEM = """\
You are an independent, skeptical code reviewer. Another engineer wrote the patch below; you did
not. Your job is to decide whether it should become a pull request. Passing tests are NOT proof of
correctness: the tests may not exercise the reported scenario at all. Judge the code itself against
the issue and the acceptance criteria. Look for: the symptom being patched instead of the root cause,
missed call sites or sibling files, changed behaviour for other callers, missing edge cases implied by
the issue, and changes larger than necessary.

Answer with ONLY one JSON object:
{
  "verdict": "APPROVED" | "NEEDS_REVISION" | "HUMAN_REVIEW" | "REJECTED",
  "addresses_issue":            {"answer": "yes" | "partial" | "no", "details": "..."},
  "all_relevant_files_updated": {"answer": "yes" | "no" | "unknown", "details": "..."},
  "regressions":                {"answer": "none_found" | "possible" | "likely", "details": "..."},
  "missing_edge_cases":         ["..."],
  "tests_meaningful":           {"answer": "yes" | "partial" | "no", "details": "do the executed tests exercise the fixed behaviour?"},
  "change_minimal":             {"answer": "yes" | "no", "details": "..."},
  "required_changes":           ["concrete changes needed if NEEDS_REVISION"],
  "summary":                    "2-4 sentences for the PR reviewer"
}
APPROVED: correct, complete, minimal, no likely regressions. NEEDS_REVISION: fixable problems you can
name concretely (list them in required_changes). HUMAN_REVIEW: plausible but you cannot verify it
(ambiguous requirements, sensitive area, tests do not cover it). REJECTED: wrong approach or harmful.
"""


@dataclass
class Critique:
    verdict: str
    llm_verdict: str
    answers: dict[str, Any]
    missing_edge_cases: list[str]
    required_changes: list[str]
    summary: str
    gates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "llm_verdict": self.llm_verdict, "answers": self.answers,
            "missing_edge_cases": self.missing_edge_cases, "required_changes": self.required_changes,
            "summary": self.summary, "gates": self.gates, "warnings": self.warnings, "error": self.error,
        }

    def feedback(self) -> str:
        """Text handed back to the implementer for a NEEDS_REVISION round."""
        parts = [f"Independent review verdict: {self.verdict}. {self.summary}"]
        if self.required_changes:
            parts.append("Required changes:\n" + "\n".join(f"- {c}" for c in self.required_changes))
        if self.missing_edge_cases:
            parts.append("Missing edge cases:\n" + "\n".join(f"- {c}" for c in self.missing_edge_cases))
        for key in ("addresses_issue", "regressions", "all_relevant_files_updated"):
            a = self.answers.get(key) or {}
            if a.get("details"):
                parts.append(f"{key}: {a.get('answer')} — {a['details']}")
        parts += [f"Gate: {g}" for g in self.gates]
        return "\n".join(parts)


def _cap(current: str, ceiling: str) -> str:
    """Return the more conservative of two verdicts."""
    return current if _RANK[current] >= _RANK[ceiling] else ceiling


def apply_gates(
    llm_verdict: str,
    answers: dict[str, Any],
    *,
    tests_ran: bool,
    tests_passed: bool | None,
    scope_ok: bool,
    scope_requires_review: bool,
    new_validator_findings: int,
    strict: bool = False,
) -> tuple[str, list[str]]:
    v = llm_verdict if llm_verdict in VERDICTS else "HUMAN_REVIEW"
    gates: list[str] = []

    def ans(key: str) -> str:
        return str((answers.get(key) or {}).get("answer", "unknown")).lower()

    if not scope_ok:
        v = "REJECTED"
        gates.append("G4 scope guard blocked the change")
    if not tests_ran:
        nv = _cap(v, "HUMAN_REVIEW")
        if nv != v:
            gates.append("G1 no test suite ran — cannot approve automatically")
        v = nv
    elif tests_passed is False:
        nv = _cap(v, "NEEDS_REVISION")
        if nv != v:
            gates.append("G2 tests failed")
        v = nv
    if scope_requires_review:
        nv = _cap(v, "HUMAN_REVIEW")
        if nv != v:
            gates.append("G3 change touches paths that require human review")
        v = nv
    if new_validator_findings > 0:
        nv = _cap(v, "NEEDS_REVISION")
        if nv != v:
            gates.append(f"G5 {new_validator_findings} new linter/type-checker finding(s)")
        v = nv
    if v == "APPROVED" and ans("addresses_issue") != "yes":
        v = "NEEDS_REVISION"
        gates.append(f"G6 reviewer answered addresses_issue={ans('addresses_issue')} but approved")
    if v == "APPROVED" and ans("tests_meaningful") == "no":
        v = "HUMAN_REVIEW"
        gates.append("G7 reviewer says the tests do not verify the fix")
    if v == "APPROVED" and ans("regressions") == "likely":
        v = "NEEDS_REVISION"
        gates.append("G8 reviewer expects regressions but approved")
    if strict and v == "APPROVED":
        positives = (ans("addresses_issue") == "yes" and ans("regressions") == "none_found"
                     and ans("tests_meaningful") in ("yes",) and ans("all_relevant_files_updated") in ("yes",))
        if not positives:
            v = "HUMAN_REVIEW"
            gates.append("G9 strict review (high-risk issue) requires every check to be positive")
    return v, gates


def critique_patch(
    llm: LLMProvider,
    *,
    issue: dict,
    acceptance_criteria: list[str],
    plan_text: str,
    diff: str,
    surrounding_code: str,
    test_report: dict,
    validator_report: list[dict],
    scope_report: dict,
    strict: bool = False,
) -> Critique:
    tests_ran = bool(test_report.get("ran"))
    tests_passed = test_report.get("passed") if tests_ran else None
    new_findings = sum(len(v.get("new_findings", [])) for v in validator_report if v.get("ran"))
    criteria = "\n".join(f"- {c}" for c in acceptance_criteria) or "(none stated — infer from the issue)"
    validators = "\n".join(
        f"- {v.get('tool')} ({v.get('kind')}): "
        + ("skipped: " + str(v.get("skipped_reason")) if not v.get("ran") else
           ("ok" if v.get("ok") else f"{len(v.get('new_findings', []))} NEW finding(s)"))
        for v in validator_report
    ) or "(no linters/type checkers configured)"
    prompt = f"""\
### Issue #{issue.get('number')} — {issue.get('title', '')}
{(issue.get('body') or '').strip()[:4000]}

### Acceptance criteria
{criteria}

### Implementation plan that was followed
{plan_text[:3000]}

### Complete diff
```diff
{diff[:14000]}
```

### Code around the changes (after the patch)
{surrounding_code[:14000]}

### Test results
command: {test_report.get('command')}
ran: {tests_ran}  passed: {tests_passed}  summary: {test_report.get('summary')}
failures before the patch (baseline): {test_report.get('baseline_failures')}
failures after the patch: {test_report.get('failed_tests')}
tests changed/added by the patch: {test_report.get('tests_touched')}
output tail:
```
{str(test_report.get('output_tail', ''))[-2500:]}
```

### Linters / type checkers (only NEW findings relative to the base commit count)
{validators}

### Scope guard
{scope_report.get('summary', scope_report)}

Review the patch and output ONLY the JSON object.
"""
    try:
        value, warnings, _raw = generate_structured(llm, prompt, _SYSTEM, CRITIQUE_SCHEMA)
    except (StructuredOutputError, Exception) as exc:  # reviewer unavailable → never auto-approve
        verdict, gates = apply_gates("HUMAN_REVIEW", {}, tests_ran=tests_ran, tests_passed=tests_passed,
                                     scope_ok=scope_report.get("ok", True),
                                     scope_requires_review=scope_report.get("requires_review", False),
                                     new_validator_findings=new_findings, strict=strict)
        return Critique(verdict, "UNAVAILABLE", {}, [], [], "Reviewer unavailable; human review required.",
                        gates + ["reviewer error → HUMAN_REVIEW"], [], f"{type(exc).__name__}: {exc}"[:500])
    answers = {k: value.get(k) or {} for k in
               ("addresses_issue", "all_relevant_files_updated", "regressions", "tests_meaningful", "change_minimal")}
    verdict, gates = apply_gates(
        value["verdict"], answers, tests_ran=tests_ran, tests_passed=tests_passed,
        scope_ok=scope_report.get("ok", True), scope_requires_review=scope_report.get("requires_review", False),
        new_validator_findings=new_findings, strict=strict,
    )
    return Critique(verdict, value["verdict"], answers, value.get("missing_edge_cases", []),
                    value.get("required_changes", []), value.get("summary", ""), gates, warnings)
