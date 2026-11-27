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


# ---------------------------------------------------------------------------
# Manual verdicts and harness robustness
# ---------------------------------------------------------------------------

def _graded(**kw) -> dict:
    base = {"case_id": "c", "kind": "fix", "difficulty": "easy", "classification_correct": True,
            "patch_applied": True, "visible_tests_passed": True, "issue_fixed": True, "scope_respected": True,
            "pipeline_result": "dry_run", "attempts": 1, "success": True, "first_attempt_success": True,
            "tests_pass_but_not_fixed": False, "failure_category": None}
    return base | kw


def test_manual_correct_does_not_rescue_a_pipeline_error():
    from fix_issue.bench.report import apply_manual

    [r] = apply_manual([_graded(pipeline_result="error", success=False, failure_category="pipeline_error")],
                       {"c": {"verdict": "correct"}})
    assert r["success"] is False and r["failure_category"] == "pipeline_error"


def test_manual_incorrect_rederives_dependent_fields():
    from fix_issue.bench.report import apply_manual, compute_metrics

    [r] = apply_manual([_graded()], {"c": {"verdict": "incorrect"}})
    assert r["success"] is False and r["first_attempt_success"] is False
    assert r["tests_pass_but_not_fixed"] is True and r["failure_category"] == "manual_verdict_incorrect"
    assert r["issue_fixed_automated"] is True
    m = compute_metrics([r])
    assert m["first_attempt_success_rate"] == 0 and m["tests_pass_but_not_fixed"] == 1


def test_manual_correct_clears_tests_pass_but_not_fixed():
    from fix_issue.bench.report import apply_manual

    [r] = apply_manual([_graded(issue_fixed=False, success=False, first_attempt_success=False,
                                tests_pass_but_not_fixed=True, failure_category="tests_pass_but_not_fixed")],
                       {"c": {"verdict": "correct"}})
    assert r["success"] is True and r["failure_category"] is None and r["tests_pass_but_not_fixed"] is False
    assert r["failure_category_automated"] == "tests_pass_but_not_fixed"


def test_run_pytest_survives_non_ascii_output(tmp_path):
    from fix_issue.bench.validate import run_pytest

    _write(tmp_path / "test_quotes.py", 'def test_q():\n    print("\u201dquoted\u201d \u274c")\n    assert False\n')
    ok, out = run_pytest(tmp_path, ["test_quotes.py"])
    assert ok is False and "test_q" in out
