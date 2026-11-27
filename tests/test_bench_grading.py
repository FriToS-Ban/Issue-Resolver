"""Benchmark grading (``bench/runner.grade_case``) on a real fixture, with no LLM involved.

The "patches" here are written by the test itself (the case's reference fix, reversed from its
injection), so this checks the grader, not the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fix_issue.bench.cases import DEFAULT_BENCH_DIR, load_case, materialise
from fix_issue.bench.runner import grade_case

CASE = "shopcart-e2-negative-money-format"


@pytest.fixture
def case():
    return load_case(DEFAULT_BENCH_DIR / "cases" / CASE)


def _record(classification: str = "fixable") -> dict:
    return {"triage": {"classification": classification}, "retrieval": {"files": [{"path": "shopcart/utils/money.py"}]},
            "attempts": [{"edits": [{"summary": "1 edit(s) applied"}]}], "critiques": []}


def _write(fp: Path, text: str) -> None:
    fp.write_bytes(text.encode("utf-8"))          # keep LF on Windows (write_text would emit CRLF)


def _apply_reference_fix(case, root: Path) -> None:
    for inj in case.inject:
        fp = root / inj.file
        text = fp.read_text(encoding="utf-8")
        assert inj.replace in text
        _write(fp, text.replace(inj.replace, inj.search, 1))


def test_reference_fix_with_regression_test_respects_scope(case, tmp_path):
    base = materialise(case, tmp_path)
    _apply_reference_fix(case, tmp_path)
    extra = "\n".join(f"def test_extra_{i}():\n    assert True\n" for i in range(10))   # 30 test lines
    _write(tmp_path / "tests" / "test_extra_regression.py", extra)

    g = grade_case(case, tmp_path, base, _record(), "dry_run")

    assert g["issue_fixed"] is True and g["visible_tests_passed"] is True
    assert g["changed_test_lines"] >= 20
    assert g["changed_source_lines"] <= case.max_diff_lines < g["changed_lines"]
    assert g["scope_respected"] is True          # tests do not count against the fix-size cap
    assert g["success"] is True and g["failure_category"] is None


def test_oversized_source_change_is_not_scope_respected(case, tmp_path):
    base = materialise(case, tmp_path)
    _apply_reference_fix(case, tmp_path)
    money = tmp_path / "shopcart" / "utils" / "money.py"
    _write(money, money.read_text(encoding="utf-8")
           + "".join(f"\n_PAD_{i} = {i}" for i in range(case.max_diff_lines + 5)))

    g = grade_case(case, tmp_path, base, _record(), "dry_run")

    assert g["changed_source_lines"] > case.max_diff_lines
    assert g["scope_respected"] is False and g["success"] is False
    assert g["failure_category"] == "scope_not_respected"


def test_unfixed_tree_is_tests_pass_but_not_fixed_or_failed(case, tmp_path):
    base = materialise(case, tmp_path)
    readme = tmp_path / "README.md"
    _write(readme, readme.read_text(encoding="utf-8") + "\ntouched\n")

    g = grade_case(case, tmp_path, base, _record(), "dry_run")

    assert g["issue_fixed"] is False and g["success"] is False
    assert g["failure_category"] in ("tests_pass_but_not_fixed", "visible_tests_failed")
