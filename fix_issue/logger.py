"""
logger.py — Append a structured JSON record to ~/.fix-issue-runs.jsonl for each run.

One line per run. Easy to query with `jq`, `grep`, or a spreadsheet.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_FILE = Path.home() / ".fix-issue-runs.jsonl"


def log_run(
    *,
    repo: str,
    issue_number: int,
    result: str,               # pr_opened | needs_info | out_of_scope | duplicate | error | scope_violation | cannot_fix
    pr_url: str | None = None,
    confidence: int | None = None,
    classification: str | None = None,
    triage_confidence: float | None = None,
    attempts: int = 0,
    error_message: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a JSON line to ~/.fix-issue-runs.jsonl."""
    record: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "repo": repo,
        "issue": issue_number,
        "result": result,
        "pr_url": pr_url,
        "confidence": confidence,
        "classification": classification,
        "triage_confidence": triage_confidence,
        "attempts": attempts,
        "error_message": error_message,
    }
    if extra:
        record.update(extra)

    # Remove None values for cleaner logs
    record = {k: v for k, v in record.items() if v is not None}

    try:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # logging failures must never crash the main pipeline
