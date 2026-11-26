"""
bench/manual.py — Human verdicts where automated verification is insufficient.

Hidden tests are strong evidence but not proof (ambiguous requirements, behaviour the tests
don't pin down). ``fix-issue bench review results.json`` walks through cases that need a
human look — those flagged ``requires_manual_review`` plus every "tests green" outcome
that the hidden tests disagree with, or vice versa — shows the issue, the diff and the
automated grades, and records a verdict in ``<results>.manual.json``. ``bench report``
applies those verdicts (they override ``issue_fixed`` for the affected cases) and reports
how many reviews are still pending. Verdicts can also be set non-interactively.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

VERDICTS = ("correct", "incorrect", "unsure")


def manual_path(results_path: Path) -> Path:
    return results_path.with_name(results_path.stem + ".manual.json")


def load_manual(results_path: Path) -> dict[str, dict]:
    p = manual_path(results_path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_verdict(results_path: Path, case_id: str, verdict: str, note: str = "", reviewer: str = "human") -> dict:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    data = load_manual(results_path)
    data[case_id] = {"verdict": verdict, "note": note, "reviewer": reviewer,
                     "at": datetime.now(tz=timezone.utc).isoformat()}
    manual_path(results_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data[case_id]


def cases_needing_review(results: dict[str, Any], include_all: bool = False) -> list[dict]:
    out = []
    for r in results["cases"]:
        if r.get("kind") != "fix":
            continue
        disagreement = bool(r.get("visible_tests_passed")) != bool(r.get("issue_fixed"))
        if include_all or r.get("requires_manual_review") or (r.get("patch_applied") and disagreement):
            out.append(r)
    return out


def interactive_review(results_path: Path, ask: Callable[[str], str], show: Callable[[str], None],
                       diff_loader: Callable[[dict], str], include_all: bool = False) -> int:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    existing = load_manual(results_path)
    count = 0
    for r in cases_needing_review(results, include_all):
        if r["case_id"] in existing:
            continue
        show(f"\n=== {r['case_id']} ({r['difficulty']}) ===")
        show(f"automated: visible tests={r.get('visible_tests_passed')} hidden tests={r.get('issue_fixed')} "
             f"review={r.get('critique_verdict')} failure={r.get('failure_category')}")
        show(diff_loader(r) or "(no diff recorded)")
        answer = ask("Is the underlying issue correctly fixed? [c]orrect / [i]ncorrect / [u]nsure / [s]kip: ").strip().lower()
        mapping = {"c": "correct", "i": "incorrect", "u": "unsure"}
        if answer[:1] in mapping:
            note = ask("note (optional): ")
            save_verdict(results_path, r["case_id"], mapping[answer[:1]], note)
            count += 1
    return count
