"""Tests for triage.py — focuses on response parsing, not LLM output."""
import json
from unittest.mock import MagicMock

import pytest

from fix_issue.triage import CLASSIFICATIONS, TriageResult, _parse_response, triage_issue


# ---------------------------------------------------------------------------
# _parse_response — unit tests (no LLM calls)
# ---------------------------------------------------------------------------

def _make_json(**kwargs) -> str:
    return json.dumps({
        "classification": "fixable",
        "reason": "Small bug.",
        "confidence": 0.9,
        "suggested_question": None,
        **kwargs,
    })


def test_parse_valid_fixable():
    result = _parse_response(_make_json(classification="fixable"))
    assert result.classification == "fixable"
    assert result.confidence == 0.9
    assert result.suggested_question is None


def test_parse_needs_info_with_question():
    raw = _make_json(
        classification="needs_info",
        confidence=0.6,
        suggested_question="Can you provide a minimal reproduction?",
    )
    result = _parse_response(raw)
    assert result.classification == "needs_info"
    assert "reproduction" in result.suggested_question


def test_parse_strips_markdown_fences():
    raw = "```json\n" + _make_json(classification="out_of_scope") + "\n```"
    result = _parse_response(raw)
    assert result.classification == "out_of_scope"


def test_parse_clamps_confidence_above_1():
    result = _parse_response(_make_json(confidence=2.5))
    assert result.confidence == 1.0


def test_parse_clamps_confidence_below_0():
    result = _parse_response(_make_json(confidence=-0.5))
    assert result.confidence == 0.0


def test_parse_unknown_classification_defaults_to_needs_info():
    result = _parse_response(_make_json(classification="totally_unknown"))
    assert result.classification == "needs_info"


def test_parse_no_json_raises():
    with pytest.raises(ValueError, match="no JSON object"):
        _parse_response("This is just a sentence with no JSON.")


def test_parse_invalid_json_raises():
    # A string that looks like JSON (matches the regex) but can't be parsed
    with pytest.raises(ValueError, match="Invalid JSON"):
        _parse_response('{"key": unquoted_value}')


def test_parse_no_json_object_raises():
    # A string with no JSON-like structure at all
    with pytest.raises(ValueError, match="no JSON object"):
        _parse_response("{broken json: true")


def test_all_valid_classifications():
    for cls in CLASSIFICATIONS:
        result = _parse_response(_make_json(classification=cls))
        assert result.classification == cls


# ---------------------------------------------------------------------------
# triage_issue — integration with mock LLM
# ---------------------------------------------------------------------------

def _mock_llm(response: str) -> MagicMock:
    mock = MagicMock()
    mock.generate.return_value = response
    return mock


def test_triage_issue_calls_llm():
    llm = _mock_llm(_make_json(classification="fixable"))
    issue = {"number": 42, "title": "Fix the thing", "body": "It's broken", "labels": []}
    result = triage_issue(llm, issue)
    assert result.classification == "fixable"
    llm.generate.assert_called_once()


def test_triage_issue_passes_system_prompt():
    llm = _mock_llm(_make_json())
    issue = {"number": 1, "title": "T", "body": "B", "labels": []}
    triage_issue(llm, issue)
    # system arg should be passed
    call_kwargs = llm.generate.call_args
    assert call_kwargs[1].get("system") or call_kwargs[0][1]  # positional or keyword


def test_triage_issue_includes_readme_in_prompt():
    llm = _mock_llm(_make_json())
    issue = {"number": 1, "title": "T", "body": "B", "labels": []}
    triage_issue(llm, issue, readme_snippet="This project does XYZ")
    prompt_arg = llm.generate.call_args[0][0]
    assert "XYZ" in prompt_arg


# ---------------------------------------------------------------------------
# Structured triage fields (v2 schema)
# ---------------------------------------------------------------------------

from fix_issue.config import LimitsConfig, PipelineConfig  # noqa: E402
from fix_issue.triage import (  # noqa: E402
    _LEGACY_SYSTEM,
    _SYSTEM,
    ControlSignals,
    apply_duplicate_evidence,
    derive_controls,
)


def _full_json(**kwargs) -> str:
    data = {
        "classification": "fixable",
        "reason": "Off-by-one in pagination.",
        "confidence": 0.85,
        "suggested_question": None,
        "complexity": "trivial",
        "expected_files": 1,
        "reproduction": "complete",
        "root_cause_category": "off_by_one",
        "difficulty": "easy",
        "risk_level": "low",
        "recommended_retry_limit": 2,
        "recommended_max_diff_lines": 20,
        "suggested_scope": {"allowed_paths": ["src/pager/*.py"], "forbidden_paths": ["migrations/**"]},
        "acceptance_criteria": ["page(2, size=10) returns items 10-19"],
        "duplicate_of": None,
    }
    data.update(kwargs)
    return json.dumps(data)


def test_parse_new_fields():
    r = _parse_response(_full_json())
    assert r.complexity == "trivial"
    assert r.expected_files == 1
    assert r.reproduction == "complete"
    assert r.root_cause_category == "off_by_one"
    assert r.difficulty == "easy"
    assert r.risk_level == "low"
    assert r.recommended_retry_limit == 2
    assert r.recommended_max_diff_lines == 20
    assert r.suggested_scope == {"allowed_paths": ["src/pager/*.py"], "forbidden_paths": ["migrations/**"]}
    assert r.acceptance_criteria == ["page(2, size=10) returns items 10-19"]
    assert r.duplicate_of is None
    assert r.warnings == []
    assert r.raw["complexity"] == "trivial"


def test_parse_old_style_json_uses_safe_defaults():
    r = _parse_response(_make_json())
    assert r.classification == "fixable"
    assert (r.complexity, r.reproduction, r.difficulty, r.risk_level, r.root_cause_category) == (
        "moderate", "partial", "medium", "medium", "other")
    assert r.expected_files >= 1
    assert r.recommended_retry_limit is None
    assert r.recommended_max_diff_lines is None
    assert r.suggested_scope == {"allowed_paths": [], "forbidden_paths": []}
    assert r.acceptance_criteria == []
    assert r.duplicate_of is None
    assert r.warnings == []
    # Old-style results still produce usable controls (config limits unchanged).
    c = derive_controls(r, LimitsConfig())
    assert (c.max_retries, c.max_diff_lines) == (3, 150)


def test_positional_construction_still_works():
    r = TriageResult("fixable", "why", 0.9, None, {})
    assert r.complexity == "moderate" and r.warnings == []
    assert r.to_dict()["classification"] == "fixable"


def test_parse_invalid_enums_fall_back_with_warnings():
    r = _parse_response(_full_json(
        complexity="gigantic", reproduction="sort of", difficulty="insane",
        risk_level="extreme", root_cause_category="cosmic_rays",
    ))
    assert r.complexity == "moderate"
    assert r.reproduction == "partial"
    assert r.difficulty == "medium"
    assert r.risk_level == "medium"
    assert r.root_cause_category == "other"
    joined = " ".join(r.warnings)
    for key in ("complexity", "reproduction", "difficulty", "risk_level", "root_cause_category"):
        assert key in joined


def test_parse_enum_normalisation():
    r = _parse_response(_full_json(root_cause_category="Off-By-One", risk_level="HIGH",
                                   classification="Out of scope"))
    assert r.root_cause_category == "off_by_one"
    assert r.risk_level == "high"
    assert r.classification == "out_of_scope"
    assert r.warnings == []


def test_parse_unknown_classification_records_warning():
    r = _parse_response(_make_json(classification="totally_unknown"))
    assert r.classification == "needs_info"
    assert any("classification" in w for w in r.warnings)


def test_parse_clamps_numeric_fields():
    r = _parse_response(_full_json(expected_files=99, recommended_retry_limit=0,
                                   recommended_max_diff_lines=10_000))
    assert r.expected_files == 20
    assert r.recommended_retry_limit == 1
    assert r.recommended_max_diff_lines == 500
    assert len([w for w in r.warnings if "clamped" in w]) == 3
    r = _parse_response(_full_json(expected_files=0, recommended_max_diff_lines=1, recommended_retry_limit=9))
    assert (r.expected_files, r.recommended_max_diff_lines, r.recommended_retry_limit) == (1, 5, 5)


def test_parse_non_numeric_values_default():
    r = _parse_response(_full_json(confidence="very", expected_files="a few"))
    assert r.confidence == 0.5
    assert r.expected_files == 2
    assert any("confidence" in w for w in r.warnings)
    r = _parse_response('{"classification": "fixable", "confidence": NaN}')
    assert r.confidence == 0.5 and any("non-finite" in w for w in r.warnings)


def test_parse_duplicate_of_variants():
    assert _parse_response(_full_json(classification="duplicate", duplicate_of="#17")).duplicate_of == 17
    assert _parse_response(_full_json(duplicate_of="null")).duplicate_of is None
    assert _parse_response(_full_json(duplicate_of=0)).duplicate_of is None


def test_parse_scope_sanitised():
    r = _parse_response(_full_json(suggested_scope={
        "allowed_paths": ["./src/a.py", "src\\b.py", "", "/lib/*"],
        "forbidden_paths": ["**", "*", "migrations/**"],
    }))
    assert r.suggested_scope["allowed_paths"] == ["src/a.py", "src/b.py", "lib/*"]
    assert r.suggested_scope["forbidden_paths"] == ["migrations/**"]
    assert sum("catch-all" in w for w in r.warnings) == 2


def test_parse_scope_wrong_type_defaults():
    r = _parse_response(_full_json(suggested_scope="src/", acceptance_criteria="it works"))
    assert r.suggested_scope == {"allowed_paths": [], "forbidden_paths": []}
    assert r.acceptance_criteria == ["it works"]  # single string promoted to a list
    assert any("suggested_scope" in w for w in r.warnings)


def test_parse_json_with_prose_and_trailing_comma():
    raw = 'Sure! Here is the triage:\n{"classification": "fixable", "reason": "r", "confidence": 0.7,}\nThanks'
    r = _parse_response(raw)
    assert r.classification == "fixable"
    assert r.confidence == 0.7


# ---------------------------------------------------------------------------
# triage_issue — prompts
# ---------------------------------------------------------------------------

_ISSUE = {"number": 7, "title": "Pager skips first item", "body": "page(1) starts at 1",
          "labels": [{"name": "bug"}]}


def test_triage_issue_default_uses_new_system_prompt():
    llm = _mock_llm(_full_json())
    r = triage_issue(llm, _ISSUE)
    assert llm.generate.call_args[1]["system"] == _SYSTEM
    assert "recommended_max_diff_lines" in _SYSTEM and "acceptance_criteria" in _SYSTEM
    assert r.complexity == "trivial"


def test_triage_issue_legacy_uses_original_prompt():
    llm = _mock_llm(_make_json())
    r = triage_issue(llm, _ISSUE, readme_snippet="README XYZ",
                     similar_issues=[{"number": 3, "title": "Old pager bug", "similarity": 0.9, "body": "b"}],
                     legacy=True)
    system = llm.generate.call_args[1]["system"]
    assert system == _LEGACY_SYSTEM
    assert system != _SYSTEM
    assert system.startswith("You are a senior software engineer acting as a triage assistant")
    prompt = llm.generate.call_args[0][0]
    assert prompt.startswith("Classify the following GitHub issue.")
    assert "Recently resolved issues" not in prompt
    assert "XYZ" in prompt
    assert r.classification == "fixable"


def test_triage_issue_includes_similar_issues_in_prompt():
    llm = _mock_llm(_full_json())
    similar = [
        {"number": 31, "title": "Pager off by one on first page", "similarity": 0.87,
         "body": "The pager drops the first element. " * 30},
        {"number": 32, "title": "Pager crash", "similarity": 0.41, "body": ""},
    ]
    triage_issue(llm, _ISSUE, similar_issues=similar)
    prompt = llm.generate.call_args[0][0]
    assert "Recently resolved issues with similar text" in prompt
    assert "#31" in prompt and "0.87" in prompt and "Pager off by one on first page" in prompt
    assert "#32" in prompt
    # body excerpt is truncated to ~300 chars
    excerpt_line = next(line for line in prompt.splitlines() if line.strip().startswith("The pager drops"))
    assert len(excerpt_line.strip()) <= 300
    assert "**Labels:** bug" in prompt


def test_triage_issue_similar_issue_objects_accepted():
    from fix_issue.duplicates import SimilarIssue

    llm = _mock_llm(_full_json())
    triage_issue(llm, _ISSUE, similar_issues=[
        SimilarIssue(5, "Old title", 0.9, "closed", "github", body_excerpt="excerpt text"),
    ])
    prompt = llm.generate.call_args[0][0]
    assert "#5" in prompt and "excerpt text" in prompt


def test_triage_issue_empty_similar_list_says_none_found():
    llm = _mock_llm(_full_json())
    triage_issue(llm, _ISSUE, similar_issues=[])
    assert "None found" in llm.generate.call_args[0][0]


def test_triage_issue_no_similar_section_when_not_given():
    llm = _mock_llm(_full_json())
    triage_issue(llm, _ISSUE)
    assert "Recently resolved issues" not in llm.generate.call_args[0][0]


def test_triage_issue_repairs_unparseable_response_once():
    llm = MagicMock()
    llm.generate.side_effect = ["I think it is fixable.", _full_json()]
    r = triage_issue(llm, _ISSUE)
    assert r.classification == "fixable"
    assert llm.generate.call_count == 2
    assert "previous response was invalid" in llm.generate.call_args[0][0]
    assert any("repaired" in w for w in r.warnings)


def test_triage_issue_raises_after_failed_repair():
    llm = MagicMock()
    llm.generate.side_effect = ["nope", "still nope"]
    with pytest.raises(ValueError, match="no JSON object"):
        triage_issue(llm, _ISSUE)
    assert llm.generate.call_count == 2


def test_triage_issue_legacy_does_not_repair():
    llm = MagicMock()
    llm.generate.side_effect = ["nope", _make_json()]
    with pytest.raises(ValueError, match="no JSON object"):
        triage_issue(llm, _ISSUE, legacy=True)
    assert llm.generate.call_count == 1


# ---------------------------------------------------------------------------
# derive_controls
# ---------------------------------------------------------------------------

def _tr(**kw) -> TriageResult:
    base = dict(classification="fixable", reason="r", confidence=0.9, suggested_question=None, raw={})
    base.update(kw)
    return TriageResult(**base)


def test_controls_trivial_tightens_limits():
    t = _tr(complexity="trivial", expected_files=1, reproduction="complete", risk_level="low",
            difficulty="easy", recommended_retry_limit=2, recommended_max_diff_lines=20)
    c = derive_controls(t, LimitsConfig(max_retries=3, max_diff_lines=150, relevant_files=5))
    assert isinstance(c, ControlSignals)
    assert c.max_retries == 2
    assert c.max_diff_lines == 20
    assert c.decompose is False
    assert c.planning_depth == "light"
    assert c.critique_strictness == "standard"
    assert c.require_human_review is False
    assert c.context_files == 5
    assert "max_retries 3 -> 2 (triage recommended 2; complexity trivial)" in c.rationale
    assert any(line.startswith("max_diff_lines 150 -> 20") for line in c.rationale)
    assert any("planning_depth full -> light" in line for line in c.rationale)
    assert c.to_dict()["max_retries"] == 2


def test_controls_triage_cannot_loosen_limits():
    t = _tr(complexity="simple", expected_files=2, recommended_retry_limit=5,
            recommended_max_diff_lines=500)
    c = derive_controls(t, LimitsConfig(max_retries=2, max_diff_lines=80))
    assert c.max_retries == 2
    assert c.max_diff_lines == 80
    assert any("capped at config limit" in line and "max_retries" in line for line in c.rationale)
    assert any("capped at config limit" in line and "max_diff_lines" in line for line in c.rationale)


def test_controls_complex_keeps_full_retry_budget_but_never_exceeds():
    t = _tr(complexity="complex", difficulty="hard", expected_files=6, recommended_retry_limit=1)
    c = derive_controls(t, LimitsConfig(max_retries=4))
    assert c.max_retries == 4
    assert any("full configured budget" in line for line in c.rationale)
    t2 = _tr(complexity="moderate", difficulty="hard", recommended_retry_limit=5)
    assert derive_controls(t2, LimitsConfig(max_retries=3)).max_retries == 3


def test_controls_retries_at_least_one():
    c = derive_controls(_tr(complexity="trivial", recommended_retry_limit=1), LimitsConfig(max_retries=0))
    assert c.max_retries == 1


def test_controls_diff_floor_and_config_ceiling():
    c = derive_controls(_tr(recommended_max_diff_lines=5), LimitsConfig(max_diff_lines=150))
    assert c.max_diff_lines == 10
    assert any("floor 10" in line for line in c.rationale)
    # the floor never lifts the budget above the configured limit
    c = derive_controls(_tr(recommended_max_diff_lines=5), LimitsConfig(max_diff_lines=8))
    assert c.max_diff_lines == 8


def test_controls_decompose_rules():
    assert derive_controls(_tr(complexity="moderate", expected_files=1), LimitsConfig()).decompose
    assert derive_controls(_tr(complexity="complex", expected_files=1), LimitsConfig()).decompose
    c = derive_controls(_tr(complexity="simple", expected_files=3), LimitsConfig())
    assert c.decompose and any("expected_files 3 >= 3" in line for line in c.rationale)
    assert not derive_controls(_tr(complexity="simple", expected_files=2), LimitsConfig()).decompose
    # planning disabled in config -> no multi-step execution
    c = derive_controls(_tr(complexity="complex"), LimitsConfig(), PipelineConfig(planning=False))
    assert c.decompose is False
    assert any("planning disabled" in line for line in c.rationale)


def test_controls_planning_depth_light_needs_complete_reproduction():
    lim = LimitsConfig()
    assert derive_controls(_tr(complexity="simple", reproduction="partial"), lim).planning_depth == "full"
    assert derive_controls(_tr(complexity="moderate", reproduction="complete"), lim).planning_depth == "full"
    assert derive_controls(_tr(complexity="simple", reproduction="complete"), lim).planning_depth == "light"


def test_controls_high_risk_strict_and_human_review():
    c = derive_controls(_tr(risk_level="high"), LimitsConfig())
    assert c.critique_strictness == "strict"
    assert c.require_human_review is True
    assert any("risk high" in line for line in c.rationale)
    assert "require_human_review on (risk high)" in c.rationale


def test_controls_low_confidence_and_missing_repro_strict():
    c = derive_controls(_tr(confidence=0.4), LimitsConfig())
    assert c.critique_strictness == "strict"
    assert c.require_human_review is False
    assert any("confidence 0.40 < 0.6" in line for line in c.rationale)
    c = derive_controls(_tr(reproduction="missing"), LimitsConfig())
    assert c.critique_strictness == "strict"
    assert any("reproduction missing" in line for line in c.rationale)


def test_controls_high_confidence_never_relaxes_review():
    c = derive_controls(_tr(confidence=1.0, risk_level="high"), LimitsConfig())
    assert c.critique_strictness == "strict" and c.require_human_review


def test_controls_context_files():
    assert derive_controls(_tr(expected_files=1), LimitsConfig(relevant_files=2)).context_files == 4
    assert derive_controls(_tr(expected_files=4), LimitsConfig(relevant_files=5)).context_files == 7
    assert derive_controls(_tr(expected_files=20), LimitsConfig(relevant_files=5)).context_files == 10
    assert derive_controls(_tr(expected_files=20), LimitsConfig(relevant_files=12)).context_files == 12
    c = derive_controls(_tr(expected_files=4), LimitsConfig(relevant_files=5))
    assert "context_files 5 -> 7 (expected_files 4 + 3)" in c.rationale


def test_controls_scope_passthrough():
    t = _tr(suggested_scope={"allowed_paths": ["src/**"], "forbidden_paths": ["migrations/**"]})
    c = derive_controls(t, LimitsConfig())
    assert c.scope_allowed == ["src/**"]
    assert c.scope_forbidden == ["migrations/**"]


def test_controls_baseline_profile_passes_config_through():
    t = _tr(complexity="trivial", risk_level="high", recommended_retry_limit=1, recommended_max_diff_lines=10)
    c = derive_controls(t, LimitsConfig(max_retries=3, max_diff_lines=150), PipelineConfig(profile="baseline"))
    assert (c.max_retries, c.max_diff_lines, c.require_human_review) == (3, 150, False)


# ---------------------------------------------------------------------------
# apply_duplicate_evidence
# ---------------------------------------------------------------------------

def test_duplicate_kept_with_evidence():
    t = _tr(classification="duplicate", duplicate_of=12)
    sims = [{"number": 12, "similarity": 0.9}, {"number": 13, "similarity": 0.5}]
    out = apply_duplicate_evidence(t, sims, 0.82)
    assert out.classification == "duplicate"
    assert out.duplicate_of == 12


def test_duplicate_of_inferred_from_best_qualifying_issue():
    t = _tr(classification="duplicate", duplicate_of=None)
    sims = [{"number": 12, "similarity": 0.85}, {"number": 14, "similarity": 0.95}]
    out = apply_duplicate_evidence(t, sims, 0.82)
    assert out.classification == "duplicate"
    assert out.duplicate_of == 14
    assert out.warnings and "#14" in out.warnings[-1]


def test_duplicate_downgraded_without_evidence():
    t = _tr(classification="duplicate", duplicate_of=12, confidence=0.95)
    out = apply_duplicate_evidence(t, [{"number": 12, "similarity": 0.5}], 0.82)
    assert out.classification == "fixable"
    assert out.duplicate_of is None
    assert out.confidence <= 0.5
    assert "below threshold" in out.warnings[-1]
    assert t.classification == "duplicate"  # input not mutated


def test_duplicate_downgraded_when_not_listed():
    t = _tr(classification="duplicate", duplicate_of=99)
    out = apply_duplicate_evidence(t, [{"number": 12, "similarity": 0.95}], 0.82)
    assert out.classification == "fixable"
    assert "not among" in out.warnings[-1]


def test_duplicate_downgraded_to_needs_info_when_repro_missing():
    t = _tr(classification="duplicate", duplicate_of=None, reproduction="missing")
    out = apply_duplicate_evidence(t, [], 0.82)
    assert out.classification == "needs_info"
    assert out.suggested_question
    assert "no similar resolved issues" in out.warnings[-1]


def test_duplicate_evidence_accepts_similar_issue_objects():
    from fix_issue.duplicates import SimilarIssue

    t = _tr(classification="duplicate", duplicate_of=8)
    out = apply_duplicate_evidence(t, [SimilarIssue(8, "t", 0.9, "closed", "github")], 0.82)
    assert out.classification == "duplicate" and out.duplicate_of == 8


def test_non_duplicate_passes_through():
    t = _tr(classification="fixable")
    assert apply_duplicate_evidence(t, [{"number": 1, "similarity": 0.99}], 0.82) is t
