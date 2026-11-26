"""
triage.py — Classify a GitHub issue and turn the result into pipeline control signals.

Takes an LLMProvider instance — no direct API imports, fully provider-neutral.

Two layers:

  triage_issue()      asks the model for one JSON object (classification + structured
                      estimates: complexity, risk, reproduction quality, scope, budgets,
                      acceptance criteria, duplicate_of) and validates it with
                      ``structured.Field`` schemas. Invalid values fall back to safe
                      defaults and are recorded in ``TriageResult.warnings``. Old-style
                      JSON (only the four original keys) is still accepted.
                      ``legacy=True`` sends the ORIGINAL prompt verbatim (benchmark baseline).

  derive_controls()   deterministic rules mapping a TriageResult + configured limits to
                      ControlSignals (retry budget, diff budget, decomposition, planning
                      depth, critique strictness, human review, scope, context size).
                      Triage can only TIGHTEN configured limits, never loosen them, and the
                      model's confidence is a hint only — it never skips validation.

  apply_duplicate_evidence()
                      a "duplicate" verdict is only kept when a listed similar issue clears
                      the configured similarity threshold; otherwise it is downgraded.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from fix_issue.config import LimitsConfig, PipelineConfig
from fix_issue.llm_provider import LLMProvider
from fix_issue.structured import Field, StructuredOutputError, extract_json, validate


CLASSIFICATIONS = {"fixable", "needs_info", "out_of_scope", "duplicate"}

COMPLEXITIES = ("trivial", "simple", "moderate", "complex")
REPRODUCTIONS = ("complete", "partial", "missing")
DIFFICULTIES = ("easy", "medium", "hard")
RISK_LEVELS = ("low", "medium", "high")
ROOT_CAUSES = (
    "off_by_one", "null_handling", "logic_error", "type_error", "formatting", "ordering",
    "serialization", "config", "arithmetic", "concurrency", "performance", "documentation",
    "missing_feature", "unclear", "other",
)

# Original system prompt — kept verbatim: it is the benchmark baseline (legacy=True).
_LEGACY_SYSTEM = """\
You are a senior software engineer acting as a triage assistant for a GitHub issue tracker.
Your job is to classify an issue so an autonomous agent can decide whether to attempt a fix.

Output ONLY a valid JSON object — no prose, no markdown fences, no explanation outside the JSON.
The JSON must have exactly these keys:
  - "classification": one of "fixable" | "needs_info" | "out_of_scope" | "duplicate"
  - "reason": a single sentence explaining the classification
  - "confidence": a float between 0.0 and 1.0 (your certainty)
  - "suggested_question": if classification is "needs_info", the one clarifying question to ask;
                          otherwise null

Classification rules:
  fixable        — Small, well-scoped bug or doc fix. Can be addressed by changing ≤ 5 files
                   with a patch under ~150 lines. No ambiguity about the expected behavior.
  needs_info     — Repro steps missing, ambiguous expected behavior, or not enough detail.
  out_of_scope   — Large refactor, new feature, security-sensitive (auth/payments/migrations),
                   or touches CI/CD config.
  duplicate      — Clearly the same root cause as another open/closed issue or PR.
"""

_SYSTEM = """\
You are a senior software engineer acting as the triage stage of an autonomous issue-fixing agent.
Your answer decides whether the agent attempts a fix, and how much budget, scope and review it gets.

Output ONLY one valid JSON object — no prose, no markdown fences, no text outside the JSON.
The object must have these keys:
  "classification":       "fixable" | "needs_info" | "out_of_scope" | "duplicate"
  "reason":               one sentence explaining the classification
  "confidence":           float 0.0–1.0, your honest certainty in the classification
  "suggested_question":   for "needs_info", the single most useful clarifying question; otherwise null
  "complexity":           "trivial" | "simple" | "moderate" | "complex"
  "expected_files":       integer 1–20, how many source files a correct fix most likely changes
  "reproduction":         "complete" (exact inputs/steps AND expected vs actual behaviour)
                          | "partial" | "missing"
  "root_cause_category":  one of "off_by_one", "null_handling", "logic_error", "type_error",
                          "formatting", "ordering", "serialization", "config", "arithmetic",
                          "concurrency", "performance", "documentation", "missing_feature",
                          "unclear", "other"
  "difficulty":           "easy" | "medium" | "hard"
  "risk_level":           "low" | "medium" | "high" — "high" whenever the fix touches
                          authentication/authorization, payments, database migrations,
                          a public API contract, or persisted user data
  "recommended_retry_limit":    integer 1–5, fix attempts worth spending on this issue
  "recommended_max_diff_lines": integer 5–500, the largest diff (added + removed lines)
                                a correct fix should need
  "suggested_scope":      {"allowed_paths": [glob, ...], "forbidden_paths": [glob, ...]}
                          paths the fix should stay within / must never touch ([] when unsure)
  "acceptance_criteria":  [short, testable statements of the correct behaviour after the fix,
                          e.g. "parse_date('2024-02-29') returns a date instead of raising"]
  "duplicate_of":         the number of a listed recently resolved issue with the SAME root cause,
                          otherwise null

Classification rules:
  fixable        — A well-scoped bug or documentation fix with unambiguous expected behaviour,
                   addressable by a focused patch (typically ≤ 5 files).
  needs_info     — Reproduction steps missing, expected behaviour ambiguous, or not enough detail
                   to act on. Ask exactly one clarifying question.
  out_of_scope   — Feature requests, new subsystems/integrations, large refactors,
                   security-sensitive redesigns (auth, permissions, crypto, payments, migrations),
                   or CI/CD configuration changes.
  duplicate      — Only when one of the issues under "Recently resolved issues with similar text"
                   has the same root cause. A confident duplicate REQUIRES a listed similar issue:
                   if no list is given or nothing listed matches, do not answer "duplicate" and
                   set "duplicate_of" to null. Similar wording alone is not the same root cause.

Calibration:
  - Budgets should fit the issue: a one-line off-by-one is trivial (1 file, ~10 lines, 1–2 retries);
    a cross-module logic change is moderate or complex.
  - Confidence is a hint, not a gate; below 0.6 means you are guessing.
"""


@dataclass
class TriageResult:
    classification: str          # one of CLASSIFICATIONS
    reason: str
    confidence: float            # 0.0–1.0
    suggested_question: str | None  # only set when classification == "needs_info"
    raw: dict[str, Any]          # full parsed JSON from LLM
    # --- structured estimates (defaults = "model did not say"; all safe) -------------
    complexity: str = "moderate"                 # COMPLEXITIES
    expected_files: int = 2                      # 1–20
    reproduction: str = "partial"                # REPRODUCTIONS
    root_cause_category: str = "other"           # ROOT_CAUSES
    difficulty: str = "medium"                   # DIFFICULTIES
    risk_level: str = "medium"                   # RISK_LEVELS
    recommended_retry_limit: int | None = None   # 1–5; None = no recommendation
    recommended_max_diff_lines: int | None = None  # 5–500; None = no recommendation
    suggested_scope: dict[str, list[str]] = field(
        default_factory=lambda: {"allowed_paths": [], "forbidden_paths": []}
    )
    acceptance_criteria: list[str] = field(default_factory=list)
    duplicate_of: int | None = None
    warnings: list[str] = field(default_factory=list)  # validation / evidence warnings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def triage_issue(
    llm: LLMProvider,
    issue: dict[str, Any],
    readme_snippet: str = "",
    *,
    similar_issues: list[dict] | None = None,
    legacy: bool = False,
) -> TriageResult:
    """
    Classify `issue` using `llm`.

    Parameters
    ----------
    llm:            Any LLMProvider implementation.
    issue:          Raw GitHub issue dict (title, body, labels, comments_url …).
    readme_snippet: First 2000 chars of the repo README for additional context.
    similar_issues: Recently resolved issues with similar text (dicts or SimilarIssue
                    objects: number, title, similarity, body/body_excerpt). Shown to the
                    model as duplicate evidence. Callers should still run
                    ``apply_duplicate_evidence`` with the configured threshold.
    legacy:         Use the original prompt and system prompt verbatim (benchmark baseline);
                    ``similar_issues`` is ignored in this mode.
    """
    if legacy:
        prompt = _build_legacy_prompt(issue, readme_snippet)
        raw_text = llm.generate(prompt, system=_LEGACY_SYSTEM)
        return _parse_response(raw_text)

    prompt = _build_prompt(issue, readme_snippet, similar_issues)
    raw_text = llm.generate(prompt, system=_SYSTEM)
    try:
        return _parse_response(raw_text)
    except ValueError as exc:
        # One bounded repair round when the response has no parseable JSON object.
        repair = (
            f"{prompt}\n\n### Your previous response was invalid\n- {str(exc).splitlines()[0]}\n\n"
            f"Previous response (truncated):\n{(raw_text or '')[:1500]}\n\n"
            "Respond again with ONLY the JSON object described in the system prompt."
        )
        result = _parse_response(llm.generate(repair, system=_SYSTEM))
        result.warnings.insert(0, "first triage response had no valid JSON; repaired on retry")
        return result


# ---------------------------------------------------------------------------
# Control signals
# ---------------------------------------------------------------------------

@dataclass
class ControlSignals:
    max_retries: int
    max_diff_lines: int
    decompose: bool                 # multi-step plan execution
    planning_depth: str             # "light" | "full"
    critique_strictness: str        # "standard" | "strict"
    require_human_review: bool
    scope_forbidden: list[str]
    scope_allowed: list[str]
    context_files: int              # how many files retrieval should return
    rationale: list[str]            # one line per adjustment

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LOW_CONFIDENCE = 0.6
MIN_DIFF_LINES = 10


def derive_controls(
    triage: TriageResult,
    limits: LimitsConfig,
    pipeline: PipelineConfig | None = None,
) -> ControlSignals:
    """
    Deterministically map triage estimates onto pipeline controls.

    Rules (triage can only tighten configured limits — never loosen them):

      max_retries          min(config.max_retries, recommended_retry_limit), at least 1.
                           complex/hard issues keep the full configured budget (a lower
                           recommendation is ignored) but never exceed it.
      max_diff_lines       min(config.max_diff_lines, recommended_max_diff_lines), never below
                           10 and never above config.
      decompose            complexity in (moderate, complex) or expected_files >= 3.
      planning_depth       "light" only for trivial/simple issues with complete reproduction,
                           otherwise "full".
      critique_strictness  "strict" when risk_level is high, triage confidence < 0.6, or
                           reproduction is missing; otherwise "standard".
      require_human_review risk_level high.
      scope_*              triage suggested_scope (already sanitised by the parser).
      context_files        clamp(expected_files + 3, 4, 10), but at least config.relevant_files.

    PipelineConfig (optional): profile "baseline" disables every triage-driven adjustment
    (config values pass through); planning disabled forces decompose off.
    The model's confidence is only ever used to make review STRICTER.
    """
    rationale: list[str] = []
    cfg_retries = max(1, int(limits.max_retries))
    cfg_diff = int(limits.max_diff_lines)
    cfg_files = int(limits.relevant_files)

    if pipeline is not None and pipeline.profile == "baseline":
        return ControlSignals(
            max_retries=cfg_retries, max_diff_lines=cfg_diff, decompose=False,
            planning_depth="full", critique_strictness="standard",
            require_human_review=False, scope_forbidden=[], scope_allowed=[],
            context_files=cfg_files,
            rationale=["baseline profile: triage-driven controls not applied"],
        )

    # --- retry budget ------------------------------------------------------------
    rec = triage.recommended_retry_limit
    hard = triage.complexity == "complex" or triage.difficulty == "hard"
    max_retries = cfg_retries
    if hard:
        why = f"complexity {triage.complexity}" if triage.complexity == "complex" else "difficulty hard"
        if rec is not None and rec < cfg_retries:
            rationale.append(
                f"max_retries kept at {cfg_retries} (triage: {why} uses the full configured budget; "
                f"recommended {rec} ignored)"
            )
        elif rec is not None and rec > cfg_retries:
            rationale.append(
                f"max_retries kept at {cfg_retries} (triage recommended {rec}, capped at config limit)"
            )
    elif rec is not None:
        max_retries = max(1, min(cfg_retries, int(rec)))
        if max_retries != cfg_retries:
            rationale.append(
                f"max_retries {cfg_retries} -> {max_retries} "
                f"(triage recommended {rec}; complexity {triage.complexity})"
            )
        elif rec > cfg_retries:
            rationale.append(
                f"max_retries kept at {cfg_retries} (triage recommended {rec}, capped at config limit)"
            )

    # --- diff budget ---------------------------------------------------------------
    rec_diff = triage.recommended_max_diff_lines
    max_diff_lines = cfg_diff
    if rec_diff is not None:
        max_diff_lines = min(cfg_diff, max(MIN_DIFF_LINES, int(rec_diff)))
        if max_diff_lines != cfg_diff:
            note = f"triage recommended {rec_diff}"
            if rec_diff < MIN_DIFF_LINES and max_diff_lines == MIN_DIFF_LINES:
                note += f", raised to floor {MIN_DIFF_LINES}"
            rationale.append(f"max_diff_lines {cfg_diff} -> {max_diff_lines} ({note})")
        elif rec_diff > cfg_diff:
            rationale.append(
                f"max_diff_lines kept at {cfg_diff} (triage recommended {rec_diff}, capped at config limit)"
            )

    # --- decomposition / planning ----------------------------------------------------
    decompose_reasons = []
    if triage.complexity in ("moderate", "complex"):
        decompose_reasons.append(f"complexity {triage.complexity}")
    if triage.expected_files >= 3:
        decompose_reasons.append(f"expected_files {triage.expected_files} >= 3")
    decompose = bool(decompose_reasons)

    if triage.complexity in ("trivial", "simple") and triage.reproduction == "complete":
        planning_depth = "light"
        rationale.append(
            f"planning_depth full -> light (complexity {triage.complexity}, reproduction complete)"
        )
    else:
        planning_depth = "full"

    if pipeline is not None and not pipeline.planning:
        # Multi-step execution needs a plan; planning_depth is recorded but unused.
        if decompose:
            rationale.append(
                f"decompose off ({'; '.join(decompose_reasons)}, but planning disabled in config)"
            )
        decompose = False
    elif decompose:
        rationale.append(f"decompose on ({'; '.join(decompose_reasons)})")

    # --- review strictness ---------------------------------------------------------------
    strict_reasons = []
    if triage.risk_level == "high":
        strict_reasons.append("risk high")
    if triage.confidence < LOW_CONFIDENCE:
        strict_reasons.append(f"triage confidence {triage.confidence:.2f} < {LOW_CONFIDENCE}")
    if triage.reproduction == "missing":
        strict_reasons.append("reproduction missing")
    critique_strictness = "strict" if strict_reasons else "standard"
    if strict_reasons:
        line = f"critique standard -> strict ({'; '.join(strict_reasons)})"
        if pipeline is not None and not pipeline.critique:
            line += " [critique disabled in config: recorded, not enforced]"
        rationale.append(line)

    require_human_review = triage.risk_level == "high"
    if require_human_review:
        rationale.append("require_human_review on (risk high)")

    # --- scope ---------------------------------------------------------------------------
    scope = triage.suggested_scope or {}
    scope_forbidden = list(scope.get("forbidden_paths") or [])
    scope_allowed = list(scope.get("allowed_paths") or [])
    if scope_forbidden:
        rationale.append(f"scope: {len(scope_forbidden)} forbidden glob(s) from triage")
    if scope_allowed:
        rationale.append(f"scope: {len(scope_allowed)} allowed glob(s) from triage")

    # --- retrieval breadth ----------------------------------------------------------------
    context_files = max(min(max(triage.expected_files + 3, 4), 10), cfg_files)
    if context_files != cfg_files:
        rationale.append(
            f"context_files {cfg_files} -> {context_files} (expected_files {triage.expected_files} + 3)"
        )

    return ControlSignals(
        max_retries=max_retries,
        max_diff_lines=max_diff_lines,
        decompose=decompose,
        planning_depth=planning_depth,
        critique_strictness=critique_strictness,
        require_human_review=require_human_review,
        scope_forbidden=scope_forbidden,
        scope_allowed=scope_allowed,
        context_files=context_files,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Duplicate evidence
# ---------------------------------------------------------------------------

def apply_duplicate_evidence(
    triage: TriageResult,
    similar_issues: list[Any] | None,
    threshold: float,
) -> TriageResult:
    """
    Keep a "duplicate" verdict only when it is backed by a listed similar issue whose
    similarity >= ``threshold``. Returns a new TriageResult (input is not mutated).

      - duplicate_of given and among the qualifying issues     -> kept
      - duplicate_of missing, some issue qualifies             -> kept, duplicate_of = best match
      - otherwise                                              -> downgraded to "fixable"
        ("needs_info" when reproduction is missing), confidence capped at 0.5, warning added.

    Non-duplicate verdicts pass through unchanged.
    """
    if triage.classification != "duplicate":
        return triage

    sims: dict[int, float] = {}
    for s in similar_issues or []:
        d = s.to_dict() if hasattr(s, "to_dict") else (s if isinstance(s, dict) else {})
        num = _as_int(d.get("number"))
        if num is None:
            continue
        try:
            sim = float(d.get("similarity") or 0.0)
        except (TypeError, ValueError):
            sim = 0.0
        sims[num] = max(sim, sims.get(num, 0.0))
    qualifying = {n: s for n, s in sims.items() if s >= threshold}

    warnings = list(triage.warnings)
    if triage.duplicate_of is not None and triage.duplicate_of in qualifying:
        return replace(triage, warnings=warnings)
    if triage.duplicate_of is None and qualifying:
        best = max(qualifying, key=lambda n: qualifying[n])
        warnings.append(
            f"duplicate_of not given by triage; set to #{best} (similarity {qualifying[best]:.2f})"
        )
        return replace(triage, duplicate_of=best, warnings=warnings)

    if triage.duplicate_of is not None:
        if triage.duplicate_of in sims:
            why = (f"#{triage.duplicate_of} similarity {sims[triage.duplicate_of]:.2f} "
                   f"is below threshold {threshold:.2f}")
        else:
            why = f"#{triage.duplicate_of} is not among the listed similar issues"
    elif sims:
        why = f"no listed similar issue reaches similarity {threshold:.2f}"
    else:
        why = "no similar resolved issues were found"
    new_cls = "needs_info" if triage.reproduction == "missing" else "fixable"
    warnings.append(f"duplicate claim rejected ({why}); downgraded to {new_cls}")
    question = triage.suggested_question
    if new_cls == "needs_info" and not question:
        question = "Could you provide a minimal reproduction (inputs, expected and actual behaviour)?"
    return replace(
        triage,
        classification=new_cls,
        confidence=min(triage.confidence, 0.5),
        suggested_question=question,
        duplicate_of=None,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _labels(issue: dict[str, Any]) -> str:
    names = []
    for lbl in issue.get("labels") or []:
        names.append(lbl.get("name", "") if isinstance(lbl, dict) else str(lbl))
    return ", ".join(n for n in names if n) or "none"


def _build_legacy_prompt(issue: dict[str, Any], readme_snippet: str) -> str:
    """The original triage prompt (benchmark baseline) — do not change."""
    labels = _labels(issue)
    body = (issue.get("body") or "").strip()[:4000]

    readme_section = ""
    if readme_snippet:
        readme_section = f"\n\n### README (first 2000 chars)\n{readme_snippet[:2000]}"

    return f"""\
Classify the following GitHub issue.

### Issue #{issue.get('number')}
**Title:** {issue.get('title', '')}
**Labels:** {labels}

**Body:**
{body or '(no body)'}
{readme_section}

Remember: output ONLY the JSON object described in the system prompt.
"""


def _similar_section(similar_issues: list[Any] | None) -> str:
    if similar_issues is None:
        return ""
    lines = ["", "", "### Recently resolved issues with similar text"]
    if not similar_issues:
        lines.append("None found — do not classify this issue as a duplicate.")
        return "\n".join(lines)
    lines.append(
        "(similarity is 0–1 text similarity; similar wording is not proof of the same root cause)"
    )
    for s in similar_issues:
        d = s.to_dict() if hasattr(s, "to_dict") else (s if isinstance(s, dict) else {})
        sim = d.get("similarity")
        sim_txt = f"{float(sim):.2f}" if isinstance(sim, (int, float)) else "n/a"
        body = d.get("body_excerpt") or d.get("body") or ""
        body = re.sub(r"\s+", " ", str(body)).strip()[:300].rstrip()
        resolved = f" — resolved by {d['resolved_by']}" if d.get("resolved_by") else ""
        lines.append(f"- #{d.get('number')} (similarity {sim_txt}): {d.get('title', '')}{resolved}")
        if body:
            lines.append(f"  {body}")
    return "\n".join(lines)


def _build_prompt(
    issue: dict[str, Any],
    readme_snippet: str,
    similar_issues: list[Any] | None = None,
) -> str:
    labels = _labels(issue)
    body = (issue.get("body") or "").strip()[:4000]

    readme_section = ""
    if readme_snippet:
        readme_section = f"\n\n### README (first 2000 chars)\n{readme_snippet[:2000]}"

    return f"""\
Triage the following GitHub issue.

### Issue #{issue.get('number')}
**Title:** {issue.get('title', '')}
**Labels:** {labels}

**Body:**
{body or '(no body)'}
{readme_section}{_similar_section(similar_issues)}

Remember: output ONLY the JSON object described in the system prompt.
"""


def _scope_default() -> dict[str, list[str]]:
    return {"allowed_paths": [], "forbidden_paths": []}


_SCHEMA: dict[str, Field] = {
    "classification": Field(str, required=True, lower=True, enum=tuple(sorted(CLASSIFICATIONS)),
                            default="needs_info"),
    "reason": Field(str, default="No reason provided."),
    "confidence": Field(float, minimum=0.0, maximum=1.0, default=0.5),
    "suggested_question": Field(str, default=None),
    "complexity": Field(str, lower=True, enum=COMPLEXITIES, default="moderate"),
    "expected_files": Field(int, minimum=1, maximum=20, default=2),
    "reproduction": Field(str, lower=True, enum=REPRODUCTIONS, default="partial"),
    "root_cause_category": Field(str, lower=True, enum=ROOT_CAUSES, default="other"),
    "difficulty": Field(str, lower=True, enum=DIFFICULTIES, default="medium"),
    "risk_level": Field(str, lower=True, enum=RISK_LEVELS, default="medium"),
    "recommended_retry_limit": Field(int, minimum=1, maximum=5, default=None),
    "recommended_max_diff_lines": Field(int, minimum=5, maximum=500, default=None),
    "suggested_scope": Field(dict, default=_scope_default, schema={
        "allowed_paths": Field(list, items=str, default=list, max_items=30),
        "forbidden_paths": Field(list, items=str, default=list, max_items=50),
    }),
    "acceptance_criteria": Field(list, items=str, default=list, max_items=10),
    "duplicate_of": Field(int, default=None),
}

_ENUM_KEYS = ("classification", "complexity", "reproduction", "root_cause_category",
              "difficulty", "risk_level")

# Paths used to recognise catch-all globs (a forbidden "**" would block every change).
_PROBE_PATHS = ("src/app.py", "README.md", "tests/test_x.py", "lib/a/b.js", "setup.cfg")


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip().lstrip("#")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalise(data: dict[str, Any]) -> dict[str, Any]:
    """Lenient pre-normalisation before schema validation (copy; raw stays untouched)."""
    d = dict(data)
    for key in _ENUM_KEYS:
        v = d.get(key)
        if isinstance(v, str):
            d[key] = re.sub(r"[\s\-]+", "_", v.strip().lower())
    dup = d.get("duplicate_of")
    if isinstance(dup, str):
        s = dup.strip().lstrip("#").strip()
        d["duplicate_of"] = None if s.lower() in ("", "null", "none", "n/a") else s
    if _as_int(d.get("duplicate_of")) is not None and _as_int(d.get("duplicate_of")) <= 0:
        d["duplicate_of"] = None  # 0 / negative = "no duplicate"
    sq = d.get("suggested_question")
    if isinstance(sq, str) and sq.strip().lower() in ("", "null", "none"):
        d["suggested_question"] = None
    return d


def _clean_globs(patterns: list[str], kind: str, warnings: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        g = str(p).strip().replace("\\", "/")
        while g.startswith("./"):
            g = g[2:]
        g = g.lstrip("/")
        if not g:
            continue
        if kind == "forbidden_paths" and all(fnmatch.fnmatchcase(x, g) for x in _PROBE_PATHS):
            warnings.append(f"suggested_scope.forbidden_paths: catch-all glob {p!r} dropped")
            continue
        if g not in out:
            out.append(g)
    return out


def _load_json(raw_text: str) -> Any:
    """Robust extraction first; fall back to the original regex for its error messages."""
    try:
        data = extract_json(raw_text)
        if isinstance(data, dict):
            return data
    except StructuredOutputError:
        pass
    # Strip markdown fences if the model added them despite instructions
    cleaned = re.sub(r"```(?:json)?", "", raw_text or "").strip()
    # Find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"LLM triage response contained no JSON object:\n{(raw_text or '')[:500]}")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in triage response: {exc}\nRaw:\n{raw_text[:500]}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"LLM triage response contained no JSON object:\n{raw_text[:500]}")
    return data


def _parse_response(raw_text: str) -> TriageResult:
    """Extract and validate JSON from the model's response."""
    data: dict[str, Any] = _load_json(raw_text)
    value, errors = validate(_normalise(data), _SCHEMA)
    warnings = list(errors)

    scope = value.get("suggested_scope") or _scope_default()
    scope = {
        "allowed_paths": _clean_globs(scope.get("allowed_paths") or [], "allowed_paths", warnings),
        "forbidden_paths": _clean_globs(scope.get("forbidden_paths") or [], "forbidden_paths", warnings),
    }
    criteria = [c[:300] for c in (value.get("acceptance_criteria") or []) if isinstance(c, str) and c]
    confidence = float(value["confidence"])
    if not math.isfinite(confidence):  # json.loads accepts NaN/Infinity
        warnings.append(f"confidence: non-finite value {confidence!r} (defaulted to 0.5)")
        confidence = 0.5

    return TriageResult(
        classification=value["classification"],
        reason=str(value["reason"]),
        confidence=confidence,
        suggested_question=value.get("suggested_question"),
        raw=data,
        complexity=value["complexity"],
        expected_files=int(value["expected_files"]),
        reproduction=value["reproduction"],
        root_cause_category=value["root_cause_category"],
        difficulty=value["difficulty"],
        risk_level=value["risk_level"],
        recommended_retry_limit=value.get("recommended_retry_limit"),
        recommended_max_diff_lines=value.get("recommended_max_diff_lines"),
        suggested_scope=scope,
        acceptance_criteria=criteria,
        duplicate_of=value.get("duplicate_of"),
        warnings=warnings,
    )
