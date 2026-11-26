"""
bench/runner.py — Execute benchmark cases through the REAL orchestrator and grade them.

For each case:
  1. materialise the buggy baseline repo in a fresh temp dir (git-initialised),
  2. run ``Orchestrator`` with a local workspace, the case's issue as the issue source and a
     recording publisher (nothing reaches GitHub) — the exact pipeline the CLI runs,
  3. grade independently of what the pipeline claims:
       * visible test suite re-run by the harness on the final working tree,
       * hidden tests copied in only now → "issue actually fixed",
       * scope checked from the real diff against the case's forbidden paths / size cap,
       * retrieval checked against the case's ground-truth files,
       * failure category derived from the run record.

Results are machine-readable JSON (one object per case + aggregate metrics). A run whose
LLM is not a real model (``llm_mode != "real"``) is labelled and excluded from headline
metrics — scripted test doubles are never reported as benchmark success.
"""

from __future__ import annotations

import concurrent.futures as cf
import fnmatch
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fix_issue.adapters import RecordingPublisher
from fix_issue.bench.cases import HIDDEN_TEST_DIR, Case, install_hidden_tests, materialise
from fix_issue.bench.validate import run_pytest
from fix_issue.config import Config
from fix_issue.llm_provider import LLMProvider
from fix_issue.orchestrator import Orchestrator, Reporter
from fix_issue.repo_manager import LocalWorkspace
from fix_issue.run_history import RunRecorder

LLMFactory = Callable[[], tuple[LLMProvider, LLMProvider | None]]


class BenchIssueSource:
    def __init__(self, case: Case) -> None:
        self.case = case
        self.repo = f"bench/{case.fixture}"
        self.issue_number = int(case.issue["number"])

    def get_issue(self) -> dict[str, Any]:
        issue = dict(self.case.issue)
        issue.setdefault("state", "open")
        issue.setdefault("html_url", f"bench://{self.case.id}")
        return issue

    def readme(self) -> str:
        p = self.case.fixture_dir / "README.md"
        return p.read_text(encoding="utf-8")[:2000] if p.exists() else ""

    def recent_resolved(self, limit: int = 50) -> list[dict[str, Any]]:
        out = []
        for r in self.case.recent_resolved[:limit]:
            out.append({"number": r.get("number"), "title": r.get("title", ""), "body": r.get("body", ""),
                        "state": r.get("state", "closed"), "resolved_by": r.get("resolved_by"),
                        "source": "provided"})
        return out


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def _changed_files(root: Path, base: str) -> list[str]:
    _git(root, "add", "--intent-to-add", ".")
    names = _git(root, "diff", "--name-only", base).splitlines()
    return sorted(n.strip() for n in names if n.strip() and "__pycache__" not in n)


def _changed_lines(root: Path, base: str) -> int:
    diff = _git(root, "diff", "--no-color", base)
    return sum(1 for ln in diff.splitlines() if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")))


def grade_case(case: Case, root: Path, base_sha: str, record: dict[str, Any], result: str) -> dict[str, Any]:
    """Independent grading of the final working tree + the run record."""
    triage = record.get("triage") or {}
    classification = triage.get("classification")
    retrieval = record.get("retrieval") or {}
    retrieved = [f["path"] for f in retrieval.get("files", [])]
    attempts = record.get("attempts") or []
    critiques = record.get("critiques") or []

    changed = _changed_files(root, base_sha)
    lines = _changed_lines(root, base_sha)
    patch_present = bool(changed)

    exp = case.expected_files
    recall = (sum(1 for e in exp if e in retrieved) / len(exp)) if exp else None
    ranks = {e: (retrieved.index(e) + 1 if e in retrieved else None) for e in exp}

    # Edit-round application statistics (structured editing) / legacy apply results.
    rounds = applied_rounds = 0
    for a in attempts:
        for e in a.get("edits", []) or []:
            rounds += 1
            if e.get("summary"):
                applied_rounds += 1
        if "edits" not in a:  # legacy attempt: one generate→apply round per attempt
            rounds += 1
            category = (a.get("failure") or {}).get("category")
            if a.get("diff") is not None and category not in ("apply_error", "edit_error"):
                applied_rounds += 1

    visible_ok = hidden_ok = None
    visible_out = hidden_out = ""
    if case.is_fix:
        visible_ok, visible_out = run_pytest(root)
        hidden = install_hidden_tests(case, root)
        hidden_ok, hidden_out = run_pytest(root, hidden)
        shutil.rmtree(root / HIDDEN_TEST_DIR, ignore_errors=True)

    forbidden_hits = [p for p in changed for pat in case.forbidden_paths if fnmatch.fnmatch(p, pat)]
    over_size = case.max_diff_lines is not None and lines > int(case.max_diff_lines)
    scope_respected = not forbidden_hits and not over_size

    classification_correct = classification == case.expected_classification
    if case.is_fix:
        success = bool(classification_correct and patch_present and visible_ok and hidden_ok and scope_respected
                       and result in ("dry_run", "pr_opened"))
    else:
        success = bool(classification_correct and not patch_present)

    first_attempt = bool(success and case.is_fix and len(attempts) == 1)
    verdict = (record.get("outcome") or {}).get("verdict") or (critiques[-1]["verdict"] if critiques else None)

    failure = None
    if not success:
        last_fail = next((a.get("failure") for a in reversed(attempts) if a.get("failure")), None) or {}
        if not classification_correct:
            failure = "triage_misclassified"
        elif not case.is_fix:
            failure = "patched_triage_case"
        elif result == "error":
            failure = "pipeline_error"
        elif result == "cannot_fix":
            failure = "cannot_fix"
        elif result == "scope_violation":
            failure = "scope_violation"
        elif result == "rejected":
            failure = "critique_rejected"
        elif result == "failed":
            failure = f"exhausted:{last_fail.get('category', 'unknown')}"
        elif not scope_respected:
            failure = "scope_not_respected"
        elif patch_present and visible_ok and not hidden_ok:
            failure = "tests_pass_but_not_fixed"
        elif patch_present and not visible_ok:
            failure = "visible_tests_failed"
        else:
            failure = "no_patch"
        if exp and recall == 0 and failure not in ("triage_misclassified",):
            failure += "+retrieval_miss"

    return {
        "case_id": case.id,
        "fixture": case.fixture,
        "difficulty": case.difficulty,
        "kind": case.kind,
        "expected_classification": case.expected_classification,
        "classification": classification,
        "classification_correct": classification_correct,
        "pipeline_result": result,
        "retrieved_files": retrieved,
        "expected_files": exp,
        "retrieval_recall": recall,
        "retrieval_ranks": ranks,
        "edit_rounds": rounds,
        "edit_rounds_applied": applied_rounds,
        "patch_applied": patch_present,
        "changed_files": changed,
        "changed_lines": lines,
        "visible_tests_passed": visible_ok,
        "issue_fixed": hidden_ok,
        "tests_pass_but_not_fixed": bool(visible_ok and hidden_ok is False and patch_present),
        "scope_respected": scope_respected,
        "forbidden_paths_touched": sorted(set(forbidden_hits)),
        "attempts": len(attempts),
        "first_attempt_success": first_attempt,
        "critique_verdict": verdict,
        "critique_rounds": len(critiques),
        "success": success,
        "failure_category": failure,
        "requires_manual_review": case.requires_manual_review,
        "manual_verdict": None,
        "visible_output_tail": visible_out[-800:] if visible_ok is False else "",
        "hidden_output_tail": hidden_out[-800:] if hidden_ok is False else "",
    }


def run_case(case: Case, cfg: Config, llm_factory: LLMFactory, runs_dir: Path, *, llm_mode: str = "real",
             keep_workspace: bool = False, reporter: Reporter | None = None) -> dict[str, Any]:
    start = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix=f"bench-{case.id[:24]}-"))
    record: dict[str, Any] = {}
    try:
        base_sha = materialise(case, tmp)
        llm, critic = llm_factory()
        recorder = RunRecorder(
            repo=f"bench/{case.fixture}", issue_number=int(case.issue["number"]), params={},
            source={"type": "bench", "case_id": case.id}, directory=runs_dir, write_summary_log=False,
        )
        orch = Orchestrator(cfg, llm, BenchIssueSource(case), lambda: LocalWorkspace(tmp, "fix-issue-bench"),
                            RecordingPublisher(), critic_llm=critic, reporter=reporter, recorder=recorder,
                            dry_run=True, source_meta={"type": "bench", "case_id": case.id})
        recorder.record["params"] = orch.params() | {"llm_mode": llm_mode}
        res = orch.run()
        record = res.record
        graded = grade_case(case, tmp, base_sha, record, res.result)
        llm_summary = (record.get("llm") or {}).get("implementer") or {}
        graded.update({
            "run_id": res.run_id,
            "llm_calls": llm_summary.get("calls"),
            "critic_calls": ((record.get("llm") or {}).get("critic") or {}).get("calls"),
            "llm_seconds": llm_summary.get("seconds"),
            "duration_seconds": round(time.monotonic() - start, 1),
            "error": res.error,
        })
        return graded
    except Exception as exc:  # harness failure is recorded, never silently counted
        return {
            "case_id": case.id, "fixture": case.fixture, "difficulty": case.difficulty, "kind": case.kind,
            "success": False, "failure_category": "harness_error",
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:],
            "duration_seconds": round(time.monotonic() - start, 1),
        }
    finally:
        if not keep_workspace:
            shutil.rmtree(tmp, ignore_errors=True)


def run_benchmark(cases: list[Case], cfg: Config, llm_factory: LLMFactory, out_path: Path, *,
                  llm_mode: str = "real", parallel: int = 2, reporter_factory: Callable[[Case], Reporter] | None = None,
                  progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    from fix_issue.bench.report import compute_metrics

    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs_dir = out_path.parent / "runs" / out_path.stem
    started = datetime.now(tz=timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    meta = {
        "profile": cfg.pipeline.profile,
        "llm_mode": llm_mode,
        "model": cfg.llm.model,
        "critic_model": cfg.critic.model or cfg.llm.model,
        "provider": cfg.llm.provider,
        "temperature": cfg.llm.temperature,
        "embedding_backend": cfg.retrieval.embedding_backend,
        "embedding_model": cfg.retrieval.embedding_model or None,
        "max_retries": cfg.limits.max_retries,
        "planning": cfg.pipeline.planning,
        "critique": cfg.pipeline.critique,
        "n_cases": len(cases),
        "started_at": started,
        "tool_commit": _git(Path(__file__).resolve().parents[2], "rev-parse", "--short", "HEAD").strip() or None,
        "python": sys.version.split()[0],
        "runs_dir": runs_dir.as_posix(),
    }

    def save() -> None:
        data = {"meta": meta, "cases": sorted(results, key=lambda r: r["case_id"]),
                "metrics": compute_metrics(results, llm_mode)}
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(out_path)

    with cf.ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {
            pool.submit(run_case, c, cfg, llm_factory, runs_dir, llm_mode=llm_mode,
                        reporter=reporter_factory(c) if reporter_factory else None): c
            for c in cases
        }
        for fut in cf.as_completed(futures):
            r = fut.result()
            results.append(r)
            save()
            if progress:
                progress(r)
    meta["finished_at"] = datetime.now(tz=timezone.utc).isoformat()
    save()
    return json.loads(out_path.read_text(encoding="utf-8"))
