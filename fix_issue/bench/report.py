"""
bench/report.py — Aggregate benchmark metrics, render Markdown, compare runs.

Headline definitions (all computed from harness-graded case results, never from the
pipeline's own claims):

  success                 fix case: classification correct, patch present, visible suite
                          passes, HIDDEN tests pass, scope respected, pipeline delivered;
                          triage case: classification correct and no patch produced
  issue_fixed_rate        hidden tests pass / fix cases
  test_pass_rate          visible suite passes after the run / fix cases
  tests_pass_but_not_fixed visible suite green while hidden tests fail — the gap between
                          "tests pass" and "the issue is fixed"
  patch_application_rate  fix cases that ended with an applied patch / fix cases triaged fixable
  edit_round_apply_rate   generate→apply rounds that applied cleanly / all rounds
  first_attempt_success   successes needing exactly one attempt / fix cases

Manual verdicts (bench review) override ``issue_fixed`` for cases flagged as needing
human judgement; pending manual reviews are counted, not guessed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

DIFFS = ("easy", "medium", "hard")


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def apply_manual(results: list[dict], manual: dict[str, dict]) -> list[dict]:
    out = []
    for r in results:
        r = dict(r)
        m = manual.get(r.get("case_id"))
        if m:
            r["manual_verdict"] = m.get("verdict")
            if r.get("kind") == "fix" and m.get("verdict") in ("correct", "incorrect"):
                r["issue_fixed_automated"] = r.get("issue_fixed")
                r["issue_fixed"] = m["verdict"] == "correct"
                r["success"] = bool(r["issue_fixed"] and r.get("visible_tests_passed") and r.get("scope_respected")
                                    and r.get("classification_correct") and r.get("patch_applied"))
        out.append(r)
    return out


def compute_metrics(results: list[dict], llm_mode: str = "real") -> dict[str, Any]:
    n = len(results)
    fix = [r for r in results if r.get("kind") == "fix"]
    tri = [r for r in results if r.get("kind") == "triage"]
    fixable_triaged = [r for r in fix if r.get("classification") == "fixable"]
    rounds = sum(r.get("edit_rounds", 0) or 0 for r in fix)
    applied_rounds = sum(r.get("edit_rounds_applied", 0) or 0 for r in fix)
    by_diff = {}
    for d in DIFFS:
        rs = [r for r in results if r.get("difficulty") == d]
        by_diff[d] = {"n": len(rs), "success": sum(bool(r.get("success")) for r in rs),
                      "success_rate": _rate(sum(bool(r.get("success")) for r in rs), len(rs))}
    recalls = [r["retrieval_recall"] for r in fix if r.get("retrieval_recall") is not None]
    verdicts = Counter(r.get("critique_verdict") or "none" for r in fix)
    approved = [r for r in fix if r.get("critique_verdict") == "APPROVED"]
    metrics = {
        "llm_mode": llm_mode,
        "headline_valid": llm_mode == "real",
        "n_cases": n,
        "n_fix_cases": len(fix),
        "n_triage_cases": len(tri),
        "successes": sum(bool(r.get("success")) for r in results),
        "success_rate": _rate(sum(bool(r.get("success")) for r in results), n),
        "fix_success_rate": _rate(sum(bool(r.get("success")) for r in fix), len(fix)),
        "triage_case_success_rate": _rate(sum(bool(r.get("success")) for r in tri), len(tri)),
        "by_difficulty": by_diff,
        "classification_accuracy": _rate(sum(bool(r.get("classification_correct")) for r in results), n),
        "retrieval_mean_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "retrieval_all_expected_found": _rate(sum(1 for x in recalls if x == 1.0), len(recalls)),
        "patch_application_rate": _rate(sum(bool(r.get("patch_applied")) for r in fixable_triaged), len(fixable_triaged)),
        "edit_round_apply_rate": _rate(applied_rounds, rounds),
        "test_pass_rate": _rate(sum(bool(r.get("visible_tests_passed")) for r in fix), len(fix)),
        "issue_fixed_rate": _rate(sum(bool(r.get("issue_fixed")) for r in fix), len(fix)),
        "tests_pass_but_not_fixed": sum(bool(r.get("tests_pass_but_not_fixed")) for r in fix),
        "first_attempt_success_rate": _rate(sum(bool(r.get("first_attempt_success")) for r in fix), len(fix)),
        "scope_respected_rate": _rate(sum(bool(r.get("scope_respected")) for r in fix if "scope_respected" in r), len(fix)),
        "mean_attempts": round(sum(r.get("attempts", 0) or 0 for r in fix) / len(fix), 2) if fix else None,
        "critique_verdicts": dict(verdicts),
        "critique_false_approvals": sum(1 for r in approved if not r.get("issue_fixed")),
        "critique_approval_precision": _rate(sum(1 for r in approved if r.get("issue_fixed")), len(approved)),
        "failure_categories": dict(Counter(r.get("failure_category") for r in results if r.get("failure_category"))),
        "manual_review_pending": sum(1 for r in results if r.get("requires_manual_review") and not r.get("manual_verdict")),
        "harness_errors": sum(1 for r in results if r.get("failure_category") == "harness_error"),
        "total_llm_calls": sum((r.get("llm_calls") or 0) + (r.get("critic_calls") or 0) for r in results),
        "total_duration_seconds": round(sum(r.get("duration_seconds") or 0 for r in results), 1),
    }
    return metrics


def load_results(path: Path, manual_path: Path | None = None) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    manual_path = manual_path or Path(path).with_name(Path(path).stem + ".manual.json")
    if manual_path.exists():
        manual = json.loads(manual_path.read_text(encoding="utf-8"))
        data["cases"] = apply_manual(data["cases"], manual)
        data["metrics"] = compute_metrics(data["cases"], data["meta"].get("llm_mode", "real"))
        data["metrics"]["manual_reviews_applied"] = len(manual)
    return data


def render_markdown(data: dict[str, Any], title: str | None = None) -> str:
    m, meta = data["metrics"], data["meta"]
    lines = [f"# {title or 'Benchmark results'} — profile `{meta.get('profile')}`", ""]
    if not m.get("headline_valid"):
        lines += ["> **NOT A REAL RESULT** — this run used a scripted/mock LLM "
                  f"(`llm_mode={m.get('llm_mode')}`). It only exercises the plumbing.", ""]
    lines += [
        f"- Model: `{meta.get('model')}` · critic: `{meta.get('critic_model')}` · temperature {meta.get('temperature')}",
        f"- Embeddings: `{meta.get('embedding_backend')}` {meta.get('embedding_model') or ''}",
        f"- Cases: {m['n_cases']} ({m['n_fix_cases']} fix, {m['n_triage_cases']} triage) · tool commit `{meta.get('tool_commit')}`",
        f"- Started {meta.get('started_at', '')[:19]} · total {m['total_duration_seconds']}s · {m['total_llm_calls']} LLM calls",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Overall success | {m['successes']}/{m['n_cases']} ({_pct(m['success_rate'])}) |",
        f"| Fix-case success | {_pct(m['fix_success_rate'])} |",
        f"| Triage-case success | {_pct(m['triage_case_success_rate'])} |",
        f"| Classification accuracy | {_pct(m['classification_accuracy'])} |",
        f"| Retrieval recall (expected files retrieved) | {_pct(m['retrieval_mean_recall'])} |",
        f"| Patch application rate | {_pct(m['patch_application_rate'])} |",
        f"| Edit-round apply rate | {_pct(m['edit_round_apply_rate'])} |",
        f"| Visible test pass rate | {_pct(m['test_pass_rate'])} |",
        f"| **Issue actually fixed (hidden tests)** | {_pct(m['issue_fixed_rate'])} |",
        f"| Tests pass but issue NOT fixed | {m['tests_pass_but_not_fixed']} |",
        f"| First-attempt success | {_pct(m['first_attempt_success_rate'])} |",
        f"| Scope respected | {_pct(m['scope_respected_rate'])} |",
        f"| Mean attempts (fix cases) | {m['mean_attempts']} |",
        f"| Critique verdicts | {m['critique_verdicts']} |",
        f"| Critique false approvals | {m['critique_false_approvals']} |",
        f"| Manual reviews pending | {m['manual_review_pending']} |",
        "",
        "## By difficulty", "", "| Difficulty | Success |", "|---|---|",
    ]
    for d, v in m["by_difficulty"].items():
        lines.append(f"| {d} | {v['success']}/{v['n']} ({_pct(v['success_rate'])}) |")
    lines += ["", "## Failure categories", ""]
    if m["failure_categories"]:
        lines += [f"- `{k}`: {v}" for k, v in sorted(m["failure_categories"].items(), key=lambda kv: -kv[1])]
    else:
        lines.append("- none")
    lines += ["", "## Per case", "",
              "| Case | Diff. | Class. | Retrieval | Patch | Visible | Hidden | Scope | Att. | Review | Result |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]

    def b(x: Any) -> str:
        return "—" if x is None else ("✅" if x else "❌")

    for r in sorted(data["cases"], key=lambda r: (DIFFS.index(r.get("difficulty", "easy")), r["case_id"])):
        rec = r.get("retrieval_recall")
        lines.append(
            f"| `{r['case_id']}` | {r.get('difficulty')} | {r.get('classification') or '—'}"
            f"{'' if r.get('classification_correct') else ' ⚠'} | {_pct(rec)} | {b(r.get('patch_applied'))} | "
            f"{b(r.get('visible_tests_passed'))} | {b(r.get('issue_fixed'))} | {b(r.get('scope_respected'))} | "
            f"{r.get('attempts', '—')} | {r.get('critique_verdict') or '—'} | "
            f"{'✅' if r.get('success') else '❌ ' + str(r.get('failure_category'))} |"
        )
    return "\n".join(lines) + "\n"


def render_comparison(a: dict[str, Any], b: dict[str, Any]) -> str:
    ma, mb = a["metrics"], b["metrics"]
    keys = [
        ("success_rate", "Overall success"), ("fix_success_rate", "Fix-case success"),
        ("classification_accuracy", "Classification accuracy"), ("retrieval_mean_recall", "Retrieval recall"),
        ("patch_application_rate", "Patch application"), ("edit_round_apply_rate", "Edit-round apply rate"),
        ("test_pass_rate", "Visible tests pass"), ("issue_fixed_rate", "Issue fixed (hidden)"),
        ("first_attempt_success_rate", "First-attempt success"), ("scope_respected_rate", "Scope respected"),
    ]
    la = a["meta"].get("profile")
    lb = b["meta"].get("profile")
    lines = [f"| Metric | {la} | {lb} | Δ |", "|---|---|---|---|"]
    for k, label in keys:
        va, vb = ma.get(k), mb.get(k)
        delta = "—" if va is None or vb is None else f"{(vb - va) * 100:+.0f} pp"
        lines.append(f"| {label} | {_pct(va)} | {_pct(vb)} | {delta} |")
    for d in DIFFS:
        va = ma["by_difficulty"][d]["success_rate"]
        vb = mb["by_difficulty"][d]["success_rate"]
        delta = "—" if va is None or vb is None else f"{(vb - va) * 100:+.0f} pp"
        lines.append(f"| Success ({d}) | {_pct(va)} | {_pct(vb)} | {delta} |")
    lines.append(f"| Tests pass but not fixed | {ma['tests_pass_but_not_fixed']} | {mb['tests_pass_but_not_fixed']} | |")
    lines.append(f"| Mean attempts | {ma['mean_attempts']} | {mb['mean_attempts']} | |")
    lines.append(f"| LLM calls | {ma['total_llm_calls']} | {mb['total_llm_calls']} | |")
    return "\n".join(lines) + "\n"
