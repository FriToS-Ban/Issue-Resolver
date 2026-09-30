"""
triage.py — Classify a GitHub issue as fixable / needs_info / out_of_scope / duplicate.

Takes an LLMProvider instance — no direct API imports, fully provider-neutral.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from fix_issue.llm_provider import LLMProvider


CLASSIFICATIONS = {"fixable", "needs_info", "out_of_scope", "duplicate"}

_SYSTEM = """\
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


@dataclass
class TriageResult:
    classification: str          # one of CLASSIFICATIONS
    reason: str
    confidence: float            # 0.0–1.0
    suggested_question: str | None  # only set when classification == "needs_info"
    raw: dict[str, Any]          # full parsed JSON from LLM


def triage_issue(
    llm: LLMProvider,
    issue: dict[str, Any],
    readme_snippet: str = "",
) -> TriageResult:
    """
    Classify `issue` using `llm`.

    Parameters
    ----------
    llm:            Any LLMProvider implementation.
    issue:          Raw GitHub issue dict (title, body, labels, comments_url …).
    readme_snippet: First 2000 chars of the repo README for additional context.
    """
    prompt = _build_prompt(issue, readme_snippet)
    raw_text = llm.generate(prompt, system=_SYSTEM)
    return _parse_response(raw_text)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_prompt(issue: dict[str, Any], readme_snippet: str) -> str:
    labels = ", ".join(lbl["name"] for lbl in issue.get("labels", [])) or "none"
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


def _parse_response(raw_text: str) -> TriageResult:
    """Extract and validate JSON from the model's response."""
    # Strip markdown fences if the model added them despite instructions
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    # Find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"LLM triage response contained no JSON object:\n{raw_text[:500]}")

    try:
        data: dict[str, Any] = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in triage response: {exc}\nRaw:\n{raw_text[:500]}") from exc

    classification = data.get("classification", "").lower()
    if classification not in CLASSIFICATIONS:
        # Lenient fallback: default to needs_info rather than crashing
        classification = "needs_info"

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return TriageResult(
        classification=classification,
        reason=str(data.get("reason", "No reason provided.")),
        confidence=confidence,
        suggested_question=data.get("suggested_question"),
        raw=data,
    )
