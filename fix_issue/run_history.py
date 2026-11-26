"""
run_history.py — Structured, inspectable record of every pipeline run.

Each run gets an id and a JSON document at ``~/.fix-issue/runs/<run_id>.json`` (override the
root with FIX_ISSUE_HOME) containing everything needed to understand — and re-run — it:

    params        exact parameters/limits/model used (enables `fix-issue rerun <id>`)
    issue         issue metadata
    triage        validated triage output + derived control signals
    retrieval     ranked files with per-signal scores and human-readable reasons
    plans         every plan version (initial + revisions) with validation warnings
    attempts      per attempt: intent, structured edits + validation, diff, scope report,
                  tests/linters, failure analysis and the retry reason
    critiques     every review round with deterministic gates applied
    llm           per-call log (purpose, sizes, latency, truncated prompt tail/response)
    outcome       final result, PR url, confidence

The legacy one-line summary in ``~/.fix-issue-runs.jsonl`` is still written (logger.py).
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fix_issue import logger
from fix_issue.config import fix_issue_home


def runs_dir() -> Path:
    return fix_issue_home() / "runs"


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict())
    return str(obj)


class RunRecorder:
    """Accumulates a run's structured record and persists it after every major event."""

    def __init__(
        self,
        *,
        repo: str,
        issue_number: int,
        params: dict[str, Any],
        source: dict[str, Any] | None = None,
        directory: Path | None = None,
        persist: bool = True,
        write_summary_log: bool = True,
    ) -> None:
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        slug = repo.replace("/", "-")[:40]
        self.run_id = f"{stamp}-{slug}-{issue_number}-{secrets.token_hex(2)}"
        self.directory = directory or runs_dir()
        self.persist = persist
        self.write_summary_log = write_summary_log
        self._t0 = time.monotonic()
        self.record: dict[str, Any] = {
            "run_id": self.run_id,
            "schema_version": 2,
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "repo": repo,
            "issue_number": issue_number,
            "params": _jsonable(params),
            "source": source or {"type": "github"},
            "issue": {},
            "triage": None,
            "controls": None,
            "similar_issues": [],
            "tooling": None,
            "baseline": None,
            "retrieval": None,
            "plans": [],
            "attempts": [],
            "critiques": [],
            "events": [],
            "llm": None,
            "outcome": None,
        }

    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.directory / f"{self.run_id}.json"

    def set(self, key: str, value: Any) -> None:
        self.record[key] = _jsonable(value)
        self.save()

    def append(self, key: str, value: Any) -> None:
        self.record.setdefault(key, []).append(_jsonable(value))
        self.save()

    def event(self, stage: str, message: str, **data: Any) -> None:
        self.record["events"].append({
            "t": round(time.monotonic() - self._t0, 2),
            "stage": stage,
            "message": message,
            **({"data": _jsonable(data)} if data else {}),
        })

    def finish(self, result: str, **fields: Any) -> dict[str, Any]:
        outcome = {"result": result, **_jsonable(fields)}
        outcome["duration_seconds"] = round(time.monotonic() - self._t0, 1)
        self.record["outcome"] = outcome
        self.record["finished_at"] = datetime.now(tz=timezone.utc).isoformat()
        self.save()
        if self.write_summary_log:
            triage = self.record.get("triage") or {}
            logger.log_run(
                repo=self.record["repo"],
                issue_number=self.record["issue_number"],
                result=result,
                pr_url=fields.get("pr_url"),
                confidence=fields.get("confidence"),
                classification=triage.get("classification"),
                triage_confidence=triage.get("confidence"),
                attempts=len(self.record.get("attempts", [])),
                error_message=fields.get("error"),
                extra={"run_id": self.run_id, "verdict": fields.get("verdict")},
            )
        return outcome

    def save(self) -> None:
        if not self.persist:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.record, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # history must never crash the pipeline


# ---------------------------------------------------------------------------
# Reading history
# ---------------------------------------------------------------------------

def list_runs(directory: Path | None = None) -> list[dict[str, Any]]:
    directory = directory or runs_dir()
    if not directory.is_dir():
        return []
    out = []
    for p in sorted(directory.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def load_run(run_id: str, directory: Path | None = None) -> dict[str, Any]:
    directory = directory or runs_dir()
    exact = directory / f"{run_id}.json"
    if exact.exists():
        return json.loads(exact.read_text(encoding="utf-8"))
    matches = sorted(directory.glob(f"*{run_id}*.json")) if directory.is_dir() else []
    if len(matches) == 1:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    if not matches:
        raise FileNotFoundError(f"No run matching '{run_id}' in {directory}")
    raise FileNotFoundError(
        f"'{run_id}' is ambiguous: " + ", ".join(m.stem for m in matches[:5])
    )


def summarize(record: dict[str, Any]) -> dict[str, Any]:
    """Compact one-row view of a run for listings."""
    outcome = record.get("outcome") or {}
    triage = record.get("triage") or {}
    retrieval = record.get("retrieval") or {}
    crit = (record.get("critiques") or [{}])[-1] if record.get("critiques") else {}
    return {
        "run_id": record.get("run_id"),
        "started_at": record.get("started_at", "")[:19],
        "repo": record.get("repo"),
        "issue": record.get("issue_number"),
        "profile": (record.get("params") or {}).get("profile"),
        "classification": triage.get("classification"),
        "files_retrieved": len(retrieval.get("files", []) or []),
        "attempts": len(record.get("attempts") or []),
        "verdict": crit.get("verdict"),
        "result": outcome.get("result", "incomplete"),
        "duration": outcome.get("duration_seconds"),
        "pr_url": outcome.get("pr_url"),
    }
