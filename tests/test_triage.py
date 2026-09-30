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
