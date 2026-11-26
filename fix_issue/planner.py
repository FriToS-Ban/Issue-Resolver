"""
planner.py — Explicit, validated implementation plans (and revisions).

After triage and retrieval the model must commit to a plan BEFORE writing code:

    root cause · files to inspect · relevant symbols · expected changes per file ·
    ordered steps with dependencies (task decomposition) · tests that should validate
    the fix · regression risks · acceptance criteria · extra context it still needs

The LLM output is parsed through a schema (structured.py) and then validated
deterministically against the real repository:

  * unknown files are dropped unless the step explicitly creates them (warning recorded),
  * step ids are unique, ``depends_on`` references exist, the step graph is acyclic
    (cycles are broken and reported), steps are topologically ordered,
  * a plan with no concrete change is rejected (one repair round, then failure).

``revise_plan`` produces a new plan version when execution evidence contradicts the
current one (e.g. the failing test points at a file the plan never considered).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fix_issue.llm_provider import LLMProvider
from fix_issue.structured import Field, generate_structured

_STEP = {
    "id": Field(str, required=True),
    "description": Field(str, required=True),
    "files": Field(list, items=str, default=list),
    "symbols": Field(list, items=str, default=list),
    "depends_on": Field(list, items=str, default=list),
    "creates_files": Field(list, items=str, default=list),
}
_CHANGE = {"file": Field(str, required=True), "change": Field(str, required=True)}
PLAN_SCHEMA = {
    "root_cause": Field(str, required=True),
    "root_cause_confidence": Field(str, enum=("low", "medium", "high"), default="medium", lower=True),
    "files_to_inspect": Field(list, items=str, default=list, max_items=15),
    "relevant_symbols": Field(list, items=str, default=list, max_items=20),
    "expected_changes": Field(list, items=_CHANGE, default=list, max_items=12),
    "steps": Field(list, items=_STEP, default=list, max_items=8),
    "tests_to_validate": Field(list, items=str, default=list, max_items=12),
    "regression_risks": Field(list, items=str, default=list, max_items=10),
    "acceptance_criteria": Field(list, items=str, default=list, max_items=10),
    "needs_more_context": Field(list, items=str, default=list, max_items=8),
    "revision_reason": Field(str, default=""),
}

_SYSTEM = """\
You are a senior software engineer planning a minimal, correct fix for a GitHub issue.
You have NOT written any code yet. Study the issue and the provided source files, identify the
real root cause (which may be in a different file than where the symptom appears, e.g. a
helper two calls away from the traceback), and produce an implementation plan.

Output ONLY one JSON object with these keys:
  root_cause            string — precise: which function does what wrong, and why that causes the symptom
  root_cause_confidence "low" | "medium" | "high"
  files_to_inspect      [repo-relative paths you consider relevant]
  relevant_symbols      [function/class/method names involved, e.g. "Inventory.release"]
  expected_changes      [{"file": path, "change": what will change there}]  (only files that must change)
  steps                 [{"id": "S1", "description": ..., "files": [paths], "symbols": [...],
                          "depends_on": [step ids], "creates_files": [new paths, usually empty]}]
  tests_to_validate     [test files or test names that should pass after the fix; may include a new regression test]
  regression_risks      [callers/behaviours that could break]
  acceptance_criteria   [short testable statements of the required behaviour]
  needs_more_context    [paths or symbols you need to see before you can be sure; [] if none]

Rules: keep the change as small as possible; fix the root cause, not the symptom; do not plan
edits to files you have not seen unless you list them in needs_more_context; never plan changes
to CI config, lock files, secrets or unrelated refactors. Use 1 step for a localized fix; use
several dependent steps only when the change genuinely spans multiple files or concerns.
"""

_REVISE_SYSTEM = _SYSTEM + """
You are REVISING an existing plan because execution produced evidence that contradicts it.
Explain the contradiction in "revision_reason", keep what is still valid, and correct the rest.
"""


@dataclass
class PlanStep:
    id: str
    description: str
    files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    creates_files: list[str] = field(default_factory=list)


@dataclass
class Plan:
    root_cause: str
    root_cause_confidence: str
    files_to_inspect: list[str]
    relevant_symbols: list[str]
    expected_changes: list[dict]
    steps: list[PlanStep]
    tests_to_validate: list[str]
    regression_risks: list[str]
    acceptance_criteria: list[str]
    needs_more_context: list[str]
    version: int = 1
    revision_reason: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def files_to_change(self) -> list[str]:
        out: list[str] = []
        for c in self.expected_changes:
            if c["file"] not in out:
                out.append(c["file"])
        for s in self.steps:
            for f in s.files + s.creates_files:
                if f not in out:
                    out.append(f)
        return out

    def is_multi_step(self) -> bool:
        return len(self.steps) > 1

    def ordered_steps(self) -> list[PlanStep]:
        """Steps in dependency order (validate_plan already sorted them topologically)."""
        return list(self.steps)

    def render(self, current_step: str | None = None) -> str:
        lines = [f"Root cause ({self.root_cause_confidence} confidence): {self.root_cause}"]
        if self.expected_changes:
            lines.append("Expected changes:")
            lines += [f"  - {c['file']}: {c['change']}" for c in self.expected_changes]
        if self.steps:
            lines.append("Steps:")
            for s in self.steps:
                marker = "  >>" if s.id == current_step else "   -"
                dep = f" (after {', '.join(s.depends_on)})" if s.depends_on else ""
                files = ", ".join(s.files + s.creates_files)
                lines.append(f"{marker} {s.id}{dep}: {s.description} [{files}]")
        if self.acceptance_criteria:
            lines.append("Acceptance criteria:")
            lines += [f"  - {a}" for a in self.acceptance_criteria]
        if self.regression_risks:
            lines.append("Regression risks: " + "; ".join(self.regression_risks))
        if self.tests_to_validate:
            lines.append("Tests to validate: " + ", ".join(self.tests_to_validate))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "root_cause": self.root_cause,
            "root_cause_confidence": self.root_cause_confidence,
            "files_to_inspect": self.files_to_inspect,
            "relevant_symbols": self.relevant_symbols,
            "expected_changes": self.expected_changes,
            "steps": [s.__dict__ for s in self.steps],
            "tests_to_validate": self.tests_to_validate,
            "regression_risks": self.regression_risks,
            "acceptance_criteria": self.acceptance_criteria,
            "needs_more_context": self.needs_more_context,
            "revision_reason": self.revision_reason,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Deterministic validation
# ---------------------------------------------------------------------------

def _norm(path: str) -> str:
    p = path.strip().strip("`'\"").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def validate_plan(data: dict, repo_files: list[str], version: int = 1) -> Plan:
    files_set = set(repo_files)
    warnings: list[str] = []

    def known(path: str, allow_new: bool = False) -> str | None:
        p = _norm(path)
        if not p:
            return None
        if p in files_set:
            return p
        # tolerate a path given relative to a different root (e.g. without src/)
        suffix = [f for f in repo_files if f.endswith("/" + p)]
        if len(suffix) == 1:
            return suffix[0]
        if allow_new:
            if p.startswith("/") or ".." in p.split("/") or p.startswith(".git/"):
                warnings.append(f"rejected unsafe new path '{path}'")
                return None
            return p
        warnings.append(f"dropped unknown file '{path}'")
        return None

    inspect = [k for k in (known(p) for p in data.get("files_to_inspect", [])) if k]
    changes = []
    for c in data.get("expected_changes", []):
        k = known(c["file"], allow_new=False)
        if k is None and any(_norm(c["file"]) in s.get("creates_files", []) for s in data.get("steps", [])):
            k = _norm(c["file"])
        if k:
            changes.append({"file": k, "change": c["change"]})

    steps: list[PlanStep] = []
    seen_ids: set[str] = set()
    for i, s in enumerate(data.get("steps", []), 1):
        sid = s["id"] or f"S{i}"
        if sid in seen_ids:
            sid = f"{sid}_{i}"
            warnings.append(f"duplicate step id renamed to {sid}")
        seen_ids.add(sid)
        creates = [k for k in (known(p, allow_new=True) for p in s.get("creates_files", [])) if k]
        files = [k for k in (known(p) for p in s.get("files", [])) if k]
        steps.append(PlanStep(sid, s["description"], files, s.get("symbols", []),
                              [d for d in s.get("depends_on", [])], creates))
    ids = {s.id for s in steps}
    for s in steps:
        bad = [d for d in s.depends_on if d not in ids or d == s.id]
        if bad:
            warnings.append(f"step {s.id}: removed invalid dependencies {bad}")
            s.depends_on = [d for d in s.depends_on if d in ids and d != s.id]

    steps = _toposort(steps, warnings)
    # A step with no files inherits the expected-change files (keeps execution concrete).
    for s in steps:
        if not s.files and not s.creates_files and changes and len(steps) == 1:
            s.files = [c["file"] for c in changes]
    if not steps and changes:
        steps = [PlanStep("S1", "Apply the expected changes", [c["file"] for c in changes])]
        warnings.append("plan had no steps; synthesised one from expected_changes")
    for f in [f for s in steps for f in s.files]:
        if f not in [c["file"] for c in changes]:
            changes.append({"file": f, "change": "(see step description)"})

    return Plan(
        root_cause=data["root_cause"],
        root_cause_confidence=data.get("root_cause_confidence") or "medium",
        files_to_inspect=inspect,
        relevant_symbols=data.get("relevant_symbols", []),
        expected_changes=changes,
        steps=steps,
        tests_to_validate=data.get("tests_to_validate", []),
        regression_risks=data.get("regression_risks", []),
        acceptance_criteria=data.get("acceptance_criteria", []),
        needs_more_context=data.get("needs_more_context", []),
        version=version,
        revision_reason=data.get("revision_reason", ""),
        warnings=warnings,
    )


def _toposort(steps: list[PlanStep], warnings: list[str]) -> list[PlanStep]:
    by_id = {s.id: s for s in steps}
    order: list[PlanStep] = []
    state: dict[str, int] = {}

    def visit(s: PlanStep) -> None:
        state[s.id] = 1
        for d in list(s.depends_on):
            if state.get(d) == 1:
                warnings.append(f"dependency cycle {s.id} -> {d} broken")
                s.depends_on.remove(d)
                continue
            if state.get(d) is None:
                visit(by_id[d])
        state[s.id] = 2
        order.append(s)

    for s in steps:
        if state.get(s.id) is None:
            visit(s)
    return order


def _structural_problems(value: dict) -> list[str]:
    if not value.get("expected_changes") and not value.get("steps"):
        return ["plan must contain at least one expected change or step"]
    return []


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _prompt(issue: dict, triage: dict, context: str, repo_files: list[str], decompose: bool) -> str:
    listing = "\n".join(repo_files[:300])
    tri = "\n".join(f"  {k}: {v}" for k, v in triage.items() if v not in (None, "", [], {}))
    guidance = (
        "This issue looks multi-faceted: decompose it into ordered, dependent steps (one concern per step)."
        if decompose else
        "This issue looks localized: prefer a single step unless the code clearly needs more."
    )
    return f"""\
### Issue #{issue.get('number')} — {issue.get('title', '')}
{(issue.get('body') or '').strip()[:5000] or '(no body)'}

### Triage signals
{tri or '  (none)'}

### Retrieved source files (with why each was selected)
{context}

### Repository file list (truncated)
{listing}

{guidance}
Output ONLY the JSON plan object.
"""


def make_plan(llm: LLMProvider, issue: dict, triage: dict, context: str, repo_files: list[str],
              decompose: bool = False) -> tuple[Plan, str]:
    value, warnings, raw = generate_structured(
        llm, _prompt(issue, triage, context, repo_files, decompose), _SYSTEM, PLAN_SCHEMA,
        post_validate=_structural_problems,
    )
    plan = validate_plan(value, repo_files, version=1)
    plan.warnings = warnings + plan.warnings
    return plan, raw


def revise_plan(llm: LLMProvider, issue: dict, plan: Plan, evidence: str, context: str,
                repo_files: list[str]) -> tuple[Plan, str]:
    prompt = (
        _prompt(issue, {}, context, repo_files, plan.is_multi_step())
        + f"\n### Current plan (version {plan.version})\n{plan.render()}\n\n"
        f"### Evidence from execution that contradicts or refines the plan\n{evidence[:5000]}\n"
    )
    value, warnings, raw = generate_structured(llm, prompt, _REVISE_SYSTEM, PLAN_SCHEMA,
                                               post_validate=_structural_problems)
    new = validate_plan(value, repo_files, version=plan.version + 1)
    new.warnings = warnings + new.warnings
    return new, raw


def plan_from_retrieval(issue: dict, files: list[str]) -> Plan:
    """Fallback plan when planning is disabled or the planner fails twice."""
    return Plan(
        root_cause="(planning skipped) see issue description",
        root_cause_confidence="low",
        files_to_inspect=files,
        relevant_symbols=[],
        expected_changes=[],
        steps=[PlanStep("S1", "Fix the issue described above", [])],
        tests_to_validate=[],
        regression_risks=[],
        acceptance_criteria=[],
        needs_more_context=[],
        warnings=["fallback plan (planner unavailable)"],
    )
