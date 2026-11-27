"""`fix-issue runs` reads benchmark run directories (--runs-dir), not only FIX_ISSUE_HOME/runs."""

from __future__ import annotations

import json

from click.testing import CliRunner

from fix_issue.cli import main


def _record(run_id: str) -> dict:
    return {"run_id": run_id, "repo": "bench/textkit", "issue_number": 7, "started_at": "2026-11-28T00:00:00",
            "params": {"profile": "full", "model": "m"}, "source": {"type": "bench", "case_id": "x"},
            "issue": {"title": "Slug keeps trailing separators"}, "triage": {"classification": "fixable"},
            "attempts": [], "outcome": {"result": "dry_run", "duration_seconds": 1.0}}


def test_runs_show_and_list_with_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FIX_ISSUE_HOME", str(tmp_path / "home"))   # empty home: must not be needed
    runs = tmp_path / "bench-runs"
    runs.mkdir()
    (runs / "20261128T000000-bench-textkit-7-abcd.json").write_text(
        json.dumps(_record("20261128T000000-bench-textkit-7-abcd")), encoding="utf-8")

    r = CliRunner().invoke(main, ["runs", "show", "textkit-7-abcd", "--runs-dir", str(runs), "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["run_id"] == "20261128T000000-bench-textkit-7-abcd"

    r = CliRunner().invoke(main, ["runs", "list", "--runs-dir", str(runs), "--json"])
    assert r.exit_code == 0, r.output
    assert [row["run_id"] for row in json.loads(r.output)] == ["20261128T000000-bench-textkit-7-abcd"]


def test_runs_show_missing_run_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("FIX_ISSUE_HOME", str(tmp_path / "home"))
    (tmp_path / "empty").mkdir()
    r = CliRunner().invoke(main, ["runs", "show", "nope", "--runs-dir", str(tmp_path / "empty")])
    assert r.exit_code == 1 and "No run matching" in r.output
