"""Unit tests for structured outputs, planner validation, critic gates, recovery, history, reporting."""
import json
from unittest.mock import MagicMock

import pytest

from fix_issue.bench.report import compute_metrics, render_comparison, render_markdown
from fix_issue.critic import apply_gates, critique_patch
from fix_issue.planner import validate_plan
from fix_issue.pr_builder import PRExplanation, build_pr
from fix_issue.run_history import RunRecorder, list_runs, load_run, summarize
from fix_issue.structured import Field, StructuredOutputError, extract_json, generate_structured, validate


# --- structured ----------------------------------------------------------------------

def test_extract_json_handles_prose_fences_think_and_trailing_commas():
    raw = '<think>hmm</think>Sure!\n```json\n{"a": 1, "b": [1, 2,],}\n```\nthanks'
    assert extract_json(raw) == {"a": 1, "b": [1, 2]}


def test_extract_json_raises_without_object():
    with pytest.raises(StructuredOutputError):
        extract_json("no json here")


def test_validate_enum_clamp_and_required():
    schema = {"v": Field(str, required=True, enum=("A", "B"), upper=True),
              "n": Field(int, minimum=1, maximum=5, default=3), "xs": Field(list, items=str, default=list)}
    value, errors = validate({"v": "a", "n": 99, "xs": "single"}, schema)
    assert value == {"v": "A", "n": 5, "xs": ["single"]}
    assert any("above maximum" in e for e in errors)
    value, errors = validate({"v": "zzz"}, schema)
    assert value["v"] is None and any("required" in e for e in errors)


def test_generate_structured_repairs_once():
    llm = MagicMock()
    llm.generate.side_effect = ["not json", '{"v": "A"}']
    value, _, _ = generate_structured(llm, "p", "s", {"v": Field(str, required=True)})
    assert value["v"] == "A" and llm.generate.call_count == 2
    assert "previous response was invalid" in llm.generate.call_args_list[1][0][0]


def test_generate_structured_gives_up_after_bounded_repairs():
    llm = MagicMock()
    llm.generate.return_value = "still not json"
    with pytest.raises(StructuredOutputError):
        generate_structured(llm, "p", "s", {"v": Field(str, required=True)}, repair_attempts=1)
    assert llm.generate.call_count == 2


# --- planner validation -------------------------------------------------------------------

_FILES = ["pkg/a.py", "pkg/b.py", "tests/test_a.py"]


def _plan(**kw):
    base = {"root_cause": "x", "root_cause_confidence": "high", "files_to_inspect": [], "relevant_symbols": [],
            "expected_changes": [], "steps": [], "tests_to_validate": [], "regression_risks": [],
            "acceptance_criteria": [], "needs_more_context": [], "revision_reason": ""}
    base.update(kw)
    return base


def test_plan_drops_unknown_files_and_orders_steps():
    p = validate_plan(_plan(
        expected_changes=[{"file": "pkg/a.py", "change": "fix"}, {"file": "nope.py", "change": "?"}],
        steps=[{"id": "S2", "description": "second", "files": ["pkg/b.py"], "symbols": [], "depends_on": ["S1"],
                "creates_files": []},
               {"id": "S1", "description": "first", "files": ["./pkg/a.py"], "symbols": [], "depends_on": [],
                "creates_files": []}]), _FILES)
    assert [s.id for s in p.ordered_steps()] == ["S1", "S2"]
    assert any("nope.py" in w for w in p.warnings)
    assert p.files_to_change == ["pkg/a.py", "pkg/b.py"]


def test_plan_breaks_cycles_and_bad_dependencies():
    p = validate_plan(_plan(steps=[
        {"id": "S1", "description": "a", "files": ["pkg/a.py"], "symbols": [], "depends_on": ["S2"], "creates_files": []},
        {"id": "S2", "description": "b", "files": ["pkg/b.py"], "symbols": [], "depends_on": ["S1", "S9"], "creates_files": []},
    ]), _FILES)
    assert any("cycle" in w for w in p.warnings) and any("S9" in w for w in p.warnings)
    assert len(p.steps) == 2


def test_plan_allows_new_files_only_when_declared_and_safe():
    p = validate_plan(_plan(steps=[{"id": "S1", "description": "add test", "files": [], "symbols": [],
                                     "depends_on": [], "creates_files": ["tests/test_new.py", "../escape.py"]}]),
                      _FILES)
    assert p.steps[0].creates_files == ["tests/test_new.py"]
    assert any("unsafe" in w for w in p.warnings)


def test_plan_synthesises_step_from_expected_changes():
    p = validate_plan(_plan(expected_changes=[{"file": "pkg/a.py", "change": "fix"}]), _FILES)
    assert p.steps and p.steps[0].files == ["pkg/a.py"]


# --- critic gates -------------------------------------------------------------------------

_YES = {"addresses_issue": {"answer": "yes"}, "regressions": {"answer": "none_found"},
        "tests_meaningful": {"answer": "yes"}, "all_relevant_files_updated": {"answer": "yes"}}


def _gate(verdict="APPROVED", answers=None, **kw):
    args = dict(tests_ran=True, tests_passed=True, scope_ok=True, scope_requires_review=False,
                new_validator_findings=0, strict=False)
    args.update(kw)
    return apply_gates(verdict, answers if answers is not None else _YES, **args)


def test_gates_approve_only_when_everything_holds():
    assert _gate()[0] == "APPROVED"


def test_gate_no_tests_cannot_approve():
    v, gates = _gate(tests_ran=False)
    assert v == "HUMAN_REVIEW" and gates[0].startswith("G1")


def test_gate_failed_tests_and_lint():
    assert _gate(tests_passed=False)[0] == "NEEDS_REVISION"
    assert _gate(new_validator_findings=2)[0] == "NEEDS_REVISION"


def test_gate_scope():
    assert _gate(scope_requires_review=True)[0] == "HUMAN_REVIEW"
    assert _gate(scope_ok=False)[0] == "REJECTED"


def test_gate_inconsistent_approval():
    assert _gate(answers={**_YES, "addresses_issue": {"answer": "partial"}})[0] == "NEEDS_REVISION"
    assert _gate(answers={**_YES, "tests_meaningful": {"answer": "no"}})[0] == "HUMAN_REVIEW"
    assert _gate(answers={**_YES, "regressions": {"answer": "likely"}})[0] == "NEEDS_REVISION"


def test_gate_strict_mode():
    assert _gate(answers={**_YES, "tests_meaningful": {"answer": "partial"}}, strict=True)[0] == "HUMAN_REVIEW"


def test_gates_never_loosen():
    assert _gate("REJECTED")[0] == "REJECTED"
    assert _gate("HUMAN_REVIEW")[0] == "HUMAN_REVIEW"


def test_critique_prompt_excludes_implementer_claims_and_reviewer_error_is_safe():
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("down")
    c = critique_patch(llm, issue={"number": 1, "title": "t", "body": "b"}, acceptance_criteria=[],
                       plan_text="p", diff="d", surrounding_code="", test_report={"ran": True, "passed": True},
                       validator_report=[], scope_report={"ok": True})
    assert c.verdict == "HUMAN_REVIEW" and c.error


def test_critique_parses_and_gates():
    llm = MagicMock()
    llm.generate.return_value = json.dumps({"verdict": "approved", **_YES, "summary": "fine"})
    c = critique_patch(llm, issue={"number": 1, "title": "t", "body": "b"}, acceptance_criteria=["x"],
                       plan_text="p", diff="d", surrounding_code="", test_report={"ran": False},
                       validator_report=[], scope_report={"ok": True})
    assert c.llm_verdict == "APPROVED" and c.verdict == "HUMAN_REVIEW"   # G1: no tests ran
    prompt = llm.generate.call_args[0][0]
    assert "INTENT" not in prompt


# --- run history --------------------------------------------------------------------------

def test_run_recorder_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FIX_ISSUE_HOME", str(tmp_path))
    rec = RunRecorder(repo="o/r", issue_number=7, params={"profile": "full"}, write_summary_log=False)
    rec.set("triage", {"classification": "fixable", "confidence": 0.9})
    rec.append("attempts", {"attempt": 1, "result": "accepted"})
    rec.finish("dry_run", verdict="APPROVED")
    loaded = load_run(rec.run_id[-12:])
    assert loaded["outcome"]["result"] == "dry_run"
    row = summarize(list_runs()[0])
    assert row["attempts"] == 1 and row["classification"] == "fixable"


# --- PR body ---------------------------------------------------------------------------------

def test_pr_explanation_sections_and_draft():
    pr = build_pr(issue={"number": 3, "title": "Bug"}, diff_text="--- a/x\n+++ b/x\n-a\n+b\n",
                  triage_reason="r", triage_confidence=0.9, llm_summary="s", tests_passed=True,
                  test_command=["pytest"], attempt_count=1, repo="o/r",
                  explanation=PRExplanation(root_cause="off by one", files_modified=[("x", 2)],
                                            why_it_works=["w"], tests_executed=["pytest passed"],
                                            review_verdict="HUMAN_REVIEW", review_summary="unsure",
                                            limitations=["no isolation"], run_id="rid"))
    for section in ("Root Cause", "Files Modified", "Why This Should Work", "Validation Executed",
                    "Independent Review", "Remaining Limitations", "Penalty Applied"):
        assert section in pr.body
    assert pr.draft and pr.confidence == 8 and "off by one" in pr.body


# --- benchmark metrics ------------------------------------------------------------------------

def _case(cid, diff="easy", kind="fix", **kw):
    base = dict(case_id=cid, difficulty=diff, kind=kind, classification="fixable", classification_correct=True,
                patch_applied=True, visible_tests_passed=True, issue_fixed=True, scope_respected=True,
                attempts=1, first_attempt_success=True, success=True, edit_rounds=1, edit_rounds_applied=1,
                retrieval_recall=1.0, critique_verdict="APPROVED", tests_pass_but_not_fixed=False)
    base.update(kw)
    return base


def test_metrics_distinguish_tests_passing_from_fixed():
    rs = [_case("a"),
          _case("b", "medium", issue_fixed=False, success=False, tests_pass_but_not_fixed=True,
                first_attempt_success=False, failure_category="tests_pass_but_not_fixed"),
          _case("c", "hard", kind="triage", classification="needs_info", patch_applied=False, issue_fixed=None,
                visible_tests_passed=None, critique_verdict=None)]
    m = compute_metrics(rs)
    assert m["test_pass_rate"] == 1.0 and m["issue_fixed_rate"] == 0.5
    assert m["tests_pass_but_not_fixed"] == 1
    assert m["critique_false_approvals"] == 1
    assert m["by_difficulty"]["medium"]["success_rate"] == 0.0
    assert m["failure_categories"] == {"tests_pass_but_not_fixed": 1}


def test_mock_runs_are_labelled_not_real():
    data = {"meta": {"profile": "full"}, "cases": [_case("a")], "metrics": compute_metrics([_case("a")], "mock")}
    assert not data["metrics"]["headline_valid"]
    assert "NOT A REAL RESULT" in render_markdown(data)


def test_comparison_table():
    a = {"meta": {"profile": "baseline"}, "metrics": compute_metrics([_case("a", success=False, issue_fixed=False)])}
    b = {"meta": {"profile": "full"}, "metrics": compute_metrics([_case("a")])}
    md = render_comparison(a, b)
    assert "+100 pp" in md


# --- config ---------------------------------------------------------------------------------------

def test_dotenv_and_nvidia_key(tmp_path, monkeypatch):
    from fix_issue import config as c

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    (tmp_path / ".env").write_text("NVIDIA_API_KEY=nv-test-123\n", encoding="utf-8")
    toml = tmp_path / "c.toml"
    toml.write_text('[llm]\nprovider="openai_compatible"\nbase_url="https://integrate.api.nvidia.com/v1"\n'
                    'api_key="stale"\n[pipeline]\nplanning=false\n', encoding="utf-8")
    cfg = c.load(toml)
    assert cfg.llm.api_key == "nv-test-123"
    assert cfg.pipeline.planning is False
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
