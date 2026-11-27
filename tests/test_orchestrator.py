"""
Integration tests: drive ``Orchestrator.run()`` end to end on a real benchmark fixture.

THE LLM IN THIS FILE IS A MOCK. ``ScriptedLLM`` is a scripted test double, not a language
model: every response is canned in the test and picked by the purpose tag the orchestrator
puts on each call ("triage", "plan", "edit:all", "critique", "pr_summary", "legacy_patch").
Passing these tests says the pipeline's plumbing works; it says nothing about how well a real
model resolves issues (that is what ``fix-issue bench run`` measures).

Everything else is real and wired exactly like ``fix_issue.bench.runner.run_case``: the case
``taskq-e1-interval-seconds-unit`` (``_UNIT_SECONDS["s"]`` injected as 60 instead of 1) is
materialised once per module and each test gets its own copy of that git repository in
``tmp_path``; the orchestrator gets a ``LocalWorkspace``, a ``RecordingPublisher`` and a
``RunRecorder`` writing into ``tmp_path``, runs with ``dry_run=True``, executes the fixture's
own pytest suite for the baseline and for validation, and applies edits through the real
SEARCH/REPLACE engine, scope guard, recovery analysis and critic gates. Local n-gram
embeddings only; FIX_ISSUE_HOME, the embedding cache and ``tempfile`` all point into
``tmp_path``; any HTTP request fails the test. Nothing under ``benchmark/`` is written.

Runtime is dominated by the fixture's pytest subprocesses (17 runs in total).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator, Union

import pytest
import requests

from fix_issue.adapters import RecordingPublisher
from fix_issue.bench.cases import DEFAULT_BENCH_DIR, Case, load_case, materialise
from fix_issue.bench.runner import BenchIssueSource
from fix_issue.config import Config
from fix_issue.llm_provider import LLMProvider, RecordingLLM
from fix_issue.orchestrator import Orchestrator, PipelineResult
from fix_issue.repo_manager import LocalWorkspace
from fix_issue.run_history import RunRecorder, summarize

CASE_ID = "taskq-e1-interval-seconds-unit"
SCHEDULE = "taskq/schedule.py"

# ---------------------------------------------------------------------------
# Scripted LLM (MOCK / test double)
# ---------------------------------------------------------------------------

Response = Union[str, BaseException, Callable[[str, "str | None"], str]]


@dataclass
class Call:
    tag: str
    prompt: str
    system: str | None


class ScriptedLLM(LLMProvider):
    """
    MOCK LLM — a scripted test double, NOT a real model.

    Responses are queued per purpose tag with ``script(tag, *responses)``. A call is served
    from the queue of its exact tag ("edit:all"), else of the tag's prefix ("edit"). A
    response may be a string, an exception to raise, or ``fn(prompt, system) -> str``.
    Every call is recorded. A call nobody scripted raises AssertionError AND is remembered
    in ``unexpected``, because the planner and critic deliberately swallow exceptions —
    ``assert_exhausted()`` makes such calls (and unused responses) fail the test.
    """

    def __init__(self, name: str = "implementer") -> None:
        self.name = name
        self.current_tag = "untagged"
        self.scripts: dict[str, list[Response]] = {}
        self.calls: list[Call] = []
        self.unexpected: list[str] = []

    def script(self, tag: str, *responses: Response) -> "ScriptedLLM":
        self.scripts.setdefault(tag, []).extend(responses)
        return self

    def generate(self, prompt: str, system: str | None = None) -> str:
        tag = self.current_tag
        self.calls.append(Call(tag, prompt, system))
        queue = self.scripts.get(tag)
        if queue is None:
            queue = self.scripts.get(tag.split(":", 1)[0])
        if not queue:
            msg = f"{self.name}: unscripted LLM call #{len(self.calls)} tagged {tag!r}"
            self.unexpected.append(msg)
            raise AssertionError(msg)
        response = queue.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response(prompt, system)
        return response

    def tags(self) -> list[str]:
        return [c.tag for c in self.calls]

    def prompts(self, tag: str) -> list[str]:
        return [c.prompt for c in self.calls if c.tag == tag]

    def assert_exhausted(self) -> None:
        assert not self.unexpected, self.unexpected
        left = {t: len(q) for t, q in self.scripts.items() if q}
        assert not left, f"{self.name}: scripted responses never requested: {left}"


class TaggingRecorder(RecordingLLM):
    """The production ``RecordingLLM`` plus one thing: it tells the scripted double which
    purpose tag the orchestrator set (via the public ``tag()`` context manager)."""

    @contextmanager
    def tag(self, purpose: str) -> Iterator[None]:
        inner = self.inner
        previous = inner.current_tag
        inner.current_tag = purpose
        try:
            with super().tag(purpose):
                yield
        finally:
            inner.current_tag = previous


# ---------------------------------------------------------------------------
# Canned responses (shaped after what triage.py / planner.py / critic.py / editing.py ask for)
# ---------------------------------------------------------------------------

S_BUG = '    "s": 60,\n    "sec": 1,\n'
S_FIX = '    "s": 1,\n    "sec": 1,\n'
FIX_INTENT = "Map the short 's' suffix in _UNIT_SECONDS to 1 second instead of 60."
SUMMARY = "The interval unit table mapped the 's' suffix to 60 seconds; it now maps it to 1 second."


def triage_json(**over) -> str:
    data = {
        "classification": "fixable",
        "reason": "The 's' unit is mapped to 60 seconds in the interval unit table.",
        "confidence": 0.9,
        "suggested_question": None,
        "complexity": "trivial",
        "expected_files": 1,
        "reproduction": "complete",
        "root_cause_category": "logic_error",
        "difficulty": "easy",
        "risk_level": "low",
        "recommended_retry_limit": 3,
        "recommended_max_diff_lines": 20,
        "suggested_scope": {"allowed_paths": [], "forbidden_paths": []},
        "acceptance_criteria": ['parse_interval("90s") == timedelta(seconds=90)',
                                'parse_interval("1m30s") == timedelta(seconds=90)'],
        "duplicate_of": None,
    }
    data.update(over)
    return json.dumps(data)


def legacy_triage_json() -> str:
    return json.dumps({"classification": "fixable", "reason": "Wrong unit factor for 's'.",
                       "confidence": 0.9, "suggested_question": None})


def plan_json(**over) -> str:
    data = {
        "root_cause": "_UNIT_SECONDS in taskq/schedule.py maps 's' to 60, so '90s' parses as 90 minutes.",
        "root_cause_confidence": "high",
        "files_to_inspect": [SCHEDULE],
        "relevant_symbols": ["parse_interval", "_parse_text", "_UNIT_SECONDS"],
        "expected_changes": [{"file": SCHEDULE, "change": "map 's' to 1 second"}],
        "steps": [{"id": "S1", "description": "Fix the 's' entry of _UNIT_SECONDS", "files": [SCHEDULE],
                   "symbols": ["_UNIT_SECONDS"], "depends_on": [], "creates_files": []}],
        "tests_to_validate": ["tests/test_schedule.py"],
        "regression_risks": ["other unit suffixes share the table"],
        "acceptance_criteria": ['parse_interval("90s") == timedelta(seconds=90)'],
        "needs_more_context": [],
    }
    data.update(over)
    return json.dumps(data)


def critique_json(verdict: str = "APPROVED", **over) -> str:
    data = {
        "verdict": verdict,
        "addresses_issue": {"answer": "yes", "details": "fixes the unit factor at its single definition"},
        "all_relevant_files_updated": {"answer": "yes", "details": "only the unit table is involved"},
        "regressions": {"answer": "none_found", "details": ""},
        "missing_edge_cases": [],
        "tests_meaningful": {"answer": "yes", "details": "suite exercises parse_interval"},
        "change_minimal": {"answer": "yes", "details": "one line"},
        "required_changes": [],
        "summary": "Correct one-line fix of the 's' unit factor.",
    }
    data.update(over)
    return json.dumps(data)


def edits(intent: str, *blocks: tuple[str, str, str]) -> str:
    """An answer in editing.EDIT_FORMAT_INSTRUCTIONS format: INTENT + FILE/SEARCH/REPLACE blocks."""
    out = ["INTENT:", intent, ""]
    for path, search, replace in blocks:
        out += [f"FILE: {path}", "<<<<<<< SEARCH", *search.splitlines(), "=======",
                *replace.splitlines(), ">>>>>>> REPLACE", ""]
    return "\n".join(out)


FIX_RESPONSE = edits(FIX_INTENT, (SCHEDULE, S_BUG, S_FIX))


# ---------------------------------------------------------------------------
# Harness (mirrors bench/runner.py::run_case)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case() -> Case:
    return load_case(DEFAULT_BENCH_DIR / "cases" / CASE_ID)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No home directory, no system temp dir, no network. The fixture's own test suite needs
    no pytest plugins, so the child pytest runs skip plugin autoloading (pure speed-up)."""
    monkeypatch.setenv("FIX_ISSUE_HOME", str(tmp_path / "fix-issue-home"))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    def no_network(self, method, url, *args, **kwargs):
        raise AssertionError(f"network access attempted: {method} {url}")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)


@dataclass
class Run:
    result: PipelineResult
    root: Path
    base_sha: str
    publisher: RecordingPublisher
    workspace_entries: int
    runs_dir: Path

    @property
    def record(self) -> dict:
        return self.result.record


@pytest.fixture(scope="module")
def template(case, tmp_path_factory) -> tuple[Path, str]:
    """The buggy baseline repo, materialised once (``materialise`` = copy fixture, inject the
    bug, git init + commit — about 8 git calls); every test then works on its own copy."""
    root = tmp_path_factory.mktemp("taskq-template") / "repo"
    return root, materialise(case, root)


@pytest.fixture
def pipeline(case, template, tmp_path) -> Callable[..., Run]:
    def run(llm: ScriptedLLM, **kwargs) -> Run:
        return run_pipeline(case, template, tmp_path, llm, **kwargs)
    return run


def run_pipeline(case: Case, template: tuple[Path, str], tmp_path: Path, llm: ScriptedLLM, *,
                 critic: ScriptedLLM | None = None, profile: str = "full",
                 configure: Callable[[Config], None] | None = None) -> Run:
    """Build and run the orchestrator exactly like ``bench.runner.run_case`` does."""
    root = tmp_path / "repo"
    shutil.copytree(template[0], root)
    base_sha = template[1]
    cfg = Config()
    cfg.pipeline.profile = profile
    cfg.retrieval.embedding_backend = "local"
    cfg.retrieval.cache_dir = str(tmp_path / "cache")
    cfg.limits.test_timeout_seconds = 120
    if configure:
        configure(cfg)
    publisher = RecordingPublisher()
    runs_dir = tmp_path / "runs"
    recorder = RunRecorder(repo=f"bench/{case.fixture}", issue_number=int(case.issue["number"]), params={},
                           source={"type": "bench", "case_id": case.id}, directory=runs_dir,
                           write_summary_log=False)
    entries: list[Path] = []

    def workspace_factory() -> LocalWorkspace:
        entries.append(root)
        return LocalWorkspace(root, "fix-issue-bench")

    orch = Orchestrator(cfg, TaggingRecorder(llm), BenchIssueSource(case), workspace_factory, publisher,
                        critic_llm=TaggingRecorder(critic) if critic else None, recorder=recorder,
                        dry_run=True, source_meta={"type": "bench", "case_id": case.id})
    recorder.record["params"] = orch.params() | {"llm_mode": "scripted"}
    result = orch.run()
    return Run(result, root, base_sha, publisher, len(entries), runs_dir)


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout


def read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def changed_lines(diff: str) -> list[str]:
    return [ln for ln in diff.splitlines() if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]


def load_workspace_module(path: Path, monkeypatch) -> ModuleType:
    """Import one fixture module from the workspace under a private name (no sys.path changes)."""
    name = f"_orchestrator_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)   # dataclasses look the module up while executing
    spec.loader.exec_module(module)
    return module


def assert_tree_clean(run: Run) -> None:
    assert git(run.root, "status", "--porcelain", "--untracked-files=all") == ""
    assert git(run.root, "diff", run.base_sha) == ""
    assert '    "s": 60,\n' in read(run.root, SCHEDULE)


# ---------------------------------------------------------------------------
# Full profile
# ---------------------------------------------------------------------------

def test_full_profile_happy_path(case, pipeline, monkeypatch):
    llm = (ScriptedLLM("implementer").script("triage", triage_json()).script("plan", plan_json())
           .script("edit:all", FIX_RESPONSE).script("pr_summary", SUMMARY))
    critic = ScriptedLLM("critic").script("critique", critique_json("APPROVED"))

    run = pipeline(llm, critic=critic)
    res, rec = run.result, run.record

    assert res.result == "dry_run", res.error
    assert (res.verdict, res.tests_passed, res.attempts) == ("APPROVED", True, 1)
    assert changed_lines(res.final_diff) == ['-    "s": 60,', '+    "s": 1,']
    assert '    "s": 1,\n' in read(run.root, SCHEDULE)
    assert res.pr_title == f'fix: {case.issue["title"]} (closes #107)'
    assert SUMMARY in res.pr_body and "Confidence Score" in res.pr_body
    assert run.publisher.actions == []          # dry run: no PR, no comment

    # Call sequence, and the reviewer is independent: it never sees the implementer's INTENT.
    assert llm.tags() == ["triage", "plan", "edit:all", "pr_summary"]
    assert critic.tags() == ["critique"]
    review_prompt = critic.calls[0].prompt
    assert FIX_INTENT not in review_prompt and '+    "s": 1,' in review_prompt
    assert "passed: True" in review_prompt
    # The edit prompt carries the edit-format contract and the current file content.
    edit_call = llm.calls[2]
    assert "<<<<<<< SEARCH" in edit_call.system and "CANNOT_FIX" in edit_call.system
    assert '"s": 60,' in edit_call.prompt

    # Run record
    assert rec["triage"]["classification"] == "fixable"
    assert rec["controls"]["max_diff_lines"] == 20 and rec["controls"]["decompose"] is False
    assert rec["baseline"]["passed"] is True
    assert SCHEDULE in [f["path"] for f in rec["retrieval"]["files"]]
    assert [p["version"] for p in rec["plans"]] == [1]
    assert rec["plans"][0]["expected_changes"][0]["file"] == SCHEDULE
    [attempt] = rec["attempts"]
    assert attempt["result"] == "accepted" and attempt["tests"]["passed"] is True
    assert attempt["edits"][0]["intent"] == FIX_INTENT
    assert attempt["edits"][0]["batch"]["ok"] is True and attempt["edits"][0]["scope"]["ok"] is True
    [crit] = rec["critiques"]
    assert crit["verdict"] == crit["llm_verdict"] == "APPROVED" and crit["gates"] == []
    assert rec["outcome"]["result"] == "dry_run" and rec["outcome"]["diff_lines"] == 2
    assert rec["llm"]["implementer"]["by_purpose"] == {"triage": 1, "plan": 1, "edit:all": 1, "pr_summary": 1}
    assert rec["llm"]["critic"]["by_purpose"] == {"critique": 1}
    persisted = json.loads((run.runs_dir / f"{res.run_id}.json").read_text(encoding="utf-8"))
    assert persisted["outcome"]["result"] == "dry_run" and len(persisted["attempts"]) == 1

    # The final tree really fixes the issue: the hidden tests' expectations hold (checked
    # in-process; a hidden-test pytest run would cost two more subprocess runs).
    schedule = load_workspace_module(run.root / SCHEDULE, monkeypatch)
    assert schedule.parse_interval("90s") == timedelta(seconds=90)
    assert schedule.parse_interval("1m30s") == timedelta(seconds=90)
    assert schedule.parse_interval("2h15m10s") == timedelta(hours=2, minutes=15, seconds=10)
    for seconds in (5, 59, 61, 3601, 90061):
        delta = timedelta(seconds=seconds)
        assert schedule.parse_interval(schedule.format_interval(delta)) == delta

    llm.assert_exhausted()
    critic.assert_exhausted()


def test_edit_failure_feeds_recovery_into_next_edit_prompt(pipeline):
    stale = edits("Fix the unit.", (SCHEDULE, '    "s": 60,  # seconds\n', '    "s": 1,  # seconds\n'))
    llm = (ScriptedLLM().script("triage", triage_json()).script("plan", plan_json())
           .script("edit:all", stale, FIX_RESPONSE).script("critique", critique_json())
           .script("pr_summary", SUMMARY))

    run = pipeline(llm)
    res, rec = run.result, run.record

    assert res.result == "dry_run", res.error
    assert res.attempts == 2
    first, second = rec["attempts"]
    assert first["result"] == "failed" and first["failure"]["category"] == "missing_search"
    assert first["diff"] == "" and "tests" not in first          # nothing written, no test run
    assert first["edits"][0]["batch"]["edits"][0]["error_kind"] == "missing_search"
    assert second["result"] == "accepted"

    prompt1, prompt2 = llm.prompts("edit:all")
    assert "Previous attempt failed" not in prompt1
    assert "### Previous attempt failed: missing_search" in prompt2
    assert "SEARCH text was not found in the current file" in prompt2
    assert f"CURRENT content of {SCHEDULE} (re-read from disk after the failure)" in prompt2
    assert "None of the edits in that response were written" in prompt2
    assert changed_lines(res.final_diff) == ['-    "s": 60,', '+    "s": 1,']
    assert llm.tags() == ["triage", "plan", "edit:all", "edit:all", "critique", "pr_summary"]
    llm.assert_exhausted()


def test_test_failure_retries_with_test_feedback(pipeline):
    wrong = edits("The minutes factor is what inflates '90s'.",
                  (SCHEDULE, '    "m": 60,\n    "min": 60,\n', '    "m": 1,\n    "min": 60,\n'))
    corrected = edits("Revert the minutes change and fix the seconds factor instead.",
                      (SCHEDULE, '    "m": 1,\n    "min": 60,\n', '    "m": 60,\n    "min": 60,\n'),
                      (SCHEDULE, S_BUG, S_FIX))
    llm = (ScriptedLLM().script("triage", triage_json()).script("plan", plan_json())
           .script("edit:all", wrong, corrected).script("critique", critique_json())
           .script("pr_summary", SUMMARY))

    run = pipeline(llm)
    res, rec = run.result, run.record

    assert res.result == "dry_run", res.error
    assert (res.attempts, res.tests_passed) == (2, True)
    first, second = rec["attempts"]
    assert first["result"] == "failed" and first["tests"]["passed"] is False
    # Visible tests passed at baseline, so the broken ones are NEW failures → "regression".
    assert first["failure"]["category"] == "regression"
    assert first["tests"]["new_failures"] and all("test_schedule" in t for t in first["tests"]["new_failures"])
    assert '+    "m": 1,' in first["diff"]
    assert second["result"] == "accepted" and second["tests"]["passed"] is True

    _, prompt2 = llm.prompts("edit:all")
    assert "### Previous attempt failed: regression" in prompt2
    assert "NEW failures vs. baseline" in prompt2 and "test_parse_interval" in prompt2
    assert "Your previous edits ARE applied in the working tree" in prompt2
    assert "### Changes already applied in this run" in prompt2 and '+    "m": 1,' in prompt2
    # The bad edit was undone by the repair round; only the real fix remains.
    assert changed_lines(res.final_diff) == ['-    "s": 60,', '+    "s": 1,']
    assert "plan_revision" not in llm.tags()       # traceback stays in tests/: no plan contradiction
    assert [p["version"] for p in rec["plans"]] == [1]
    llm.assert_exhausted()


def test_traceback_outside_plan_revises_plan_and_resets_tree(pipeline):
    # Plan v1 blames the config default; the failing traceback then runs through schedule.py.
    config_plan = plan_json(
        root_cause="The default poll interval '5s' in taskq/config.py is misread.",
        expected_changes=[{"file": "taskq/config.py", "change": "spell the default as 5sec"}],
        steps=[{"id": "S1", "description": "Change the default poll interval", "files": ["taskq/config.py"],
                "symbols": ["DEFAULTS"], "depends_on": [], "creates_files": []}])
    crash = edits("Normalise unit aliases before the table lookup.",
                  (SCHEDULE, "        number, unit = match.groups()\n",
                   "        number, unit = match.groups()\n        unit = _normalise_unit(unit)\n"))
    revised = plan_json(revision_reason="The traceback shows the unit lookup in taskq/schedule.py.")
    llm = (ScriptedLLM().script("triage", triage_json()).script("plan", config_plan)
           .script("edit:all", crash, FIX_RESPONSE).script("plan_revision", revised)
           .script("critique", critique_json()).script("pr_summary", SUMMARY))

    run = pipeline(llm)
    res, rec = run.result, run.record

    assert res.result == "dry_run", res.error
    assert llm.tags() == ["triage", "plan", "edit:all", "plan_revision", "edit:all", "critique", "pr_summary"]
    first, second = rec["attempts"]
    assert first["failure"]["category"] == "regression"
    assert SCHEDULE in first["failure"]["files"]                 # pytest --tb=short frames are parsed
    assert SCHEDULE in (first["failure"]["plan_contradiction"] or "")
    assert [p["version"] for p in rec["plans"]] == [1, 2]
    assert rec["plans"][1]["revision_reason"].startswith("The traceback shows")
    assert (first["plan_version"], second["plan_version"]) == (1, 2)

    # The reviser is told the contradiction itself, plus the failure and the failing file.
    [revision_prompt] = llm.prompts("plan_revision")
    assert "The failing traceback runs through taskq/schedule.py" in revision_prompt
    assert "_normalise_unit" in revision_prompt and f"CURRENT content of {SCHEDULE}" in revision_prompt

    # After the reset the next edit prompt must not claim the old edits are still applied.
    _, prompt2 = llm.prompts("edit:all")
    assert "(Plan revised to version 2; the working tree was reset.)" in prompt2
    assert "### Implementation plan (v2)" in prompt2
    assert "### Changes already applied in this run" not in prompt2
    assert "Your previous edits ARE applied" not in prompt2
    assert "Files ALREADY modified" not in prompt2
    assert changed_lines(res.final_diff) == ['-    "s": 60,', '+    "s": 1,']
    llm.assert_exhausted()


REGRESSION_TEST = (
    "def test_parse_interval_seconds_suffix():\n"
    '    assert parse_interval("90s") == timedelta(seconds=90)\n'
    '    assert parse_interval("1m30s") == timedelta(seconds=90)\n'
    "\n"
    "\n"
    "def test_format_interval():\n"
)


def test_critique_needs_revision_gets_one_more_edit_round(pipeline):
    needs_revision = critique_json(
        "NEEDS_REVISION",
        tests_meaningful={"answer": "no", "details": "no test exercises the 's' suffix"},
        required_changes=["Add a regression test for parse_interval('90s') in tests/test_schedule.py"],
        summary="Fix is right but unguarded by tests.")
    add_test = edits("Add the regression test the reviewer asked for.",
                     ("tests/test_schedule.py", "def test_format_interval():\n", REGRESSION_TEST))
    llm = (ScriptedLLM().script("triage", triage_json()).script("plan", plan_json())
           .script("edit:all", FIX_RESPONSE, add_test)
           .script("critique", needs_revision, critique_json("APPROVED"))
           .script("pr_summary", SUMMARY))

    run = pipeline(llm)
    res, rec = run.result, run.record

    assert res.result == "dry_run", res.error
    assert (res.verdict, res.attempts) == ("APPROVED", 2)
    assert [c["verdict"] for c in rec["critiques"]] == ["NEEDS_REVISION", "APPROVED"]
    assert [c["attempt"] for c in rec["critiques"]] == [1, 2]
    assert [a["result"] for a in rec["attempts"]] == ["needs_revision", "accepted"]
    assert rec["attempts"][0]["failure"]["category"] == "critique_revision"

    _, prompt2 = llm.prompts("edit:all")
    assert "### Previous attempt failed: critique_revision" in prompt2
    assert "Independent review verdict: NEEDS_REVISION" in prompt2
    assert "Add a regression test for parse_interval('90s')" in prompt2
    # Round 2 builds on round 1 (cumulative diff) and the reviewer sees the new test.
    assert "### Changes already applied in this run" in prompt2 and '+    "s": 1,' in prompt2
    review2 = llm.prompts("critique")[1]
    assert "tests changed/added by the patch: ['tests/test_schedule.py']" in review2
    assert "+def test_parse_interval_seconds_suffix():" in res.final_diff
    assert '+    "s": 1,' in res.final_diff
    assert llm.tags() == ["triage", "plan", "edit:all", "critique", "edit:all", "critique", "pr_summary"]
    llm.assert_exhausted()


def test_critique_needs_revision_without_budget_becomes_human_review_draft(pipeline):
    needs_revision = critique_json("NEEDS_REVISION", required_changes=["Also accept 'S' (upper case)"])
    llm = (ScriptedLLM().script("triage", triage_json()).script("plan", plan_json())
           .script("edit:all", FIX_RESPONSE).script("critique", needs_revision)
           .script("pr_summary", SUMMARY))

    def no_review_rounds(cfg: Config) -> None:
        cfg.pipeline.max_critique_rounds = 0

    run = pipeline(llm, configure=no_review_rounds)
    res, rec = run.result, run.record

    # Revision budget exhausted: NEEDS_REVISION is escalated to a draft PR, not looped on.
    assert res.result == "dry_run", res.error
    assert res.verdict == "HUMAN_REVIEW"
    assert rec["pr"]["draft"] is True and "[!WARNING]" in res.pr_body
    assert [a["result"] for a in rec["attempts"]] == ["accepted"]
    # The run record must show the verdict that was acted on, not the pre-escalation one.
    [crit] = rec["critiques"]
    assert crit["verdict"] == "HUMAN_REVIEW" and crit["llm_verdict"] == "NEEDS_REVISION"
    assert "revision budget exhausted → HUMAN_REVIEW" in crit["gates"]
    assert summarize(rec)["verdict"] == "HUMAN_REVIEW"
    assert llm.tags() == ["triage", "plan", "edit:all", "critique", "pr_summary"]
    llm.assert_exhausted()


def test_critique_rejected_resets_tree_and_comments(pipeline):
    rejected = critique_json("REJECTED", addresses_issue={"answer": "no", "details": "wrong approach"},
                             required_changes=["Do not change the unit table; fix the parser"],
                             summary="Changes shared data instead of the parser.")
    llm = (ScriptedLLM().script("triage", triage_json()).script("plan", plan_json())
           .script("edit:all", FIX_RESPONSE).script("critique", rejected))

    run = pipeline(llm)
    res, rec = run.result, run.record

    assert res.result == "rejected"
    assert (res.verdict, res.tests_passed, res.attempts) == ("REJECTED", True, 1)
    assert '+    "s": 1,' in res.final_diff                     # the rejected candidate is reported
    assert_tree_clean(run)                                      # ...but the working tree is reset
    assert [a["result"] for a in rec["attempts"]] == ["rejected"]
    assert rec["critiques"][0]["verdict"] == "REJECTED"
    [action] = run.publisher.actions
    assert action["type"] == "comment" and action["kind"] == "rejected"
    assert "Changes shared data instead of the parser." in action["body"]
    assert "- Do not change the unit table; fix the parser" in action["body"]
    assert "pr_summary" not in llm.tags() and "pr" not in rec
    llm.assert_exhausted()


def test_scope_forbidden_path_blocks_before_writing(pipeline):
    touches_config = edits(
        "Fix the unit and stop relying on 's' in the default poll interval.",
        (SCHEDULE, S_BUG, S_FIX),
        ("taskq/config.py",
         '    "worker": {"name": "worker", "batch_size": 10, "poll_interval": "5s"},\n',
         '    "worker": {"name": "worker", "batch_size": 10, "poll_interval": "5sec"},\n'))
    triage = triage_json(recommended_retry_limit=1,
                         suggested_scope={"allowed_paths": [], "forbidden_paths": ["taskq/config.py"]})
    llm = (ScriptedLLM().script("triage", triage).script("plan", plan_json())
           .script("edit:all", touches_config))

    run = pipeline(llm)
    res, rec = run.result, run.record

    assert res.result == "scope_violation"
    assert res.attempts == 1
    assert "taskq/config.py" in res.error and "off-limits" in res.error
    [attempt] = rec["attempts"]
    assert attempt["failure"]["category"] == "scope_violation"
    assert attempt["failure"]["files"] == ["taskq/config.py"]
    assert attempt["edits"][0]["batch"]["ok"] is True           # the edits themselves were valid...
    assert attempt["edits"][0]["scope"]["ok"] is False          # ...but blocked before writing
    assert attempt["diff"] == "" and "tests" not in attempt     # atomic: neither file was written
    assert rec["outcome"]["failure_category"] == "scope_violation"
    assert_tree_clean(run)
    [action] = run.publisher.actions
    assert action["kind"] == "scope_violation" and "taskq/config.py" in action["body"]
    assert "critique" not in llm.tags() and "pr_summary" not in llm.tags()
    llm.assert_exhausted()


def test_scope_oversized_edit_is_blocked_with_feedback(pipeline):
    padding = "".join(f"    # note {i}: the short suffix counts seconds\n" for i in range(12))
    oversized = edits("Fix the unit and document every suffix.",
                      (SCHEDULE, S_BUG, padding + S_FIX))
    triage = triage_json(recommended_retry_limit=2, recommended_max_diff_lines=10)
    llm = (ScriptedLLM().script("triage", triage).script("plan", plan_json())
           .script("edit:all", oversized, oversized))

    run = pipeline(llm)
    res, rec = run.result, run.record

    assert rec["controls"]["max_diff_lines"] == 10
    assert res.result == "scope_violation"
    assert res.attempts == 2
    assert "Diff is too large" in res.error
    for attempt in rec["attempts"]:
        assert attempt["failure"]["category"] == "scope_violation"
        assert attempt["diff"] == ""
        assert attempt["edits"][0]["scope"]["raw_lines"] == 14
    _, prompt2 = llm.prompts("edit:all")
    assert "### Previous attempt failed: scope_violation" in prompt2
    assert "Diff is too large" in prompt2 and "it was NOT written" in prompt2
    assert_tree_clean(run)
    assert [a["kind"] for a in run.publisher.actions] == ["scope_violation"]
    llm.assert_exhausted()


# ---------------------------------------------------------------------------
# Baseline (legacy) profile
# ---------------------------------------------------------------------------

def test_baseline_profile_happy_path_applies_unified_diff(pipeline):
    legacy_answer = "\n".join(["FILE: taskq/schedule.py", "<<<<<<< SEARCH", *S_BUG.splitlines(), "=======",
                               *S_FIX.splitlines(), ">>>>>>> REPLACE"])
    llm = (ScriptedLLM().script("triage", legacy_triage_json()).script("legacy_patch", legacy_answer)
           .script("pr_summary", SUMMARY))

    run = pipeline(llm, profile="baseline")
    res, rec = run.result, run.record

    assert res.result == "dry_run", res.error
    assert (res.tests_passed, res.attempts, res.verdict) == (True, 1, None)
    # generate_patch turned the blocks into a unified diff that `git apply` accepted.
    assert "--- a/taskq/schedule.py" in res.final_diff and "+++ b/taskq/schedule.py" in res.final_diff
    assert "\n@@ " in "\n" + res.final_diff
    assert changed_lines(res.final_diff) == ['-    "s": 60,', '+    "s": 1,']
    assert '    "s": 1,\n' in read(run.root, SCHEDULE)
    assert not (run.root / "_fix_issue.patch").exists()

    assert llm.tags() == ["triage", "legacy_patch", "pr_summary"]
    assert "triage assistant" in llm.calls[0].system                  # the original legacy prompt
    assert rec["controls"] is None and rec["plans"] == [] and rec["critiques"] == []
    assert rec["retrieval"]["method"] == "keyword"
    [attempt] = rec["attempts"]
    assert attempt["result"] == "accepted" and attempt["tests"]["passed"] is True
    assert run.publisher.actions == []
    llm.assert_exhausted()


# ---------------------------------------------------------------------------
# Triage short-circuits
# ---------------------------------------------------------------------------

def _assert_workspace_untouched(run: Run) -> None:
    assert run.workspace_entries == 0
    assert git(run.root, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert git(run.root, "rev-parse", "HEAD").strip() == run.base_sha
    assert_tree_clean(run)
    rec = run.record
    assert rec["tooling"] is None and rec["baseline"] is None and rec["retrieval"] is None
    assert rec["attempts"] == [] and rec["plans"] == []


def test_triage_needs_info_short_circuits(pipeline):
    question = "Which taskq version and which exact interval strings do you use?"
    llm = ScriptedLLM().script("triage", triage_json(classification="needs_info", confidence=0.7,
                                                     reproduction="missing", suggested_question=question))

    run = pipeline(llm)

    assert run.result.result == "needs_info"
    _assert_workspace_untouched(run)
    [action] = run.publisher.actions
    assert action["kind"] == "needs_info" and question in action["body"]
    assert llm.tags() == ["triage"]
    llm.assert_exhausted()


def test_triage_out_of_scope_short_circuits(pipeline):
    llm = ScriptedLLM().script("triage", triage_json(classification="out_of_scope",
                                                     reason="Asks for a new scheduling subsystem."))

    run = pipeline(llm)

    assert run.result.result == "out_of_scope"
    assert run.record["outcome"]["result"] == "out_of_scope"
    _assert_workspace_untouched(run)
    assert run.publisher.actions == []           # only needs_info / duplicate post a comment
    assert llm.tags() == ["triage"]
    llm.assert_exhausted()


# ---------------------------------------------------------------------------
# Fallbacks, error handling and public comments
# ---------------------------------------------------------------------------

STALE = edits("Fix the unit.", (SCHEDULE, '    "s": 60,  # seconds\n', '    "s": 1,  # seconds\n'))


def test_failed_revision_round_falls_back_to_reviewed_candidate(pipeline):
    """A validated patch that got NEEDS_REVISION is not thrown away when the revision fails."""
    needs_revision = critique_json("NEEDS_REVISION", required_changes=["Also accept 'S' (upper case)"])
    bad_revision = edits("Accept upper case too.", (SCHEDULE, "    NOT IN THE FILE\n", "    x\n"))
    llm = (ScriptedLLM().script("triage", triage_json()).script("plan", plan_json())
           .script("edit:all", FIX_RESPONSE, bad_revision).script("critique", needs_revision)
           .script("pr_summary", SUMMARY))

    def one_attempt(cfg: Config) -> None:
        cfg.limits.max_retries = 1          # + 1 review round = 2 attempts

    run = pipeline(llm, configure=one_attempt)
    res, rec = run.result, run.record

    assert res.result == "dry_run", res.error
    assert res.verdict == "HUMAN_REVIEW" and rec["pr"]["draft"] is True
    assert [a["result"] for a in rec["attempts"]] == ["needs_revision", "failed"]
    assert rec["final_from_attempt"] == 1
    assert changed_lines(res.final_diff) == ['-    "s": 60,', '+    "s": 1,']
    assert changed_lines(git(run.root, "diff", run.base_sha)) == ['-    "s": 60,', '+    "s": 1,']
    assert not [a for a in run.publisher.actions if a["type"] == "comment"]
    llm.assert_exhausted()


def test_all_attempts_failing_posts_a_comment_without_raw_output(pipeline):
    llm = ScriptedLLM().script("triage", triage_json()).script("plan", plan_json()).script("edit:all", STALE)

    def one_attempt(cfg: Config) -> None:
        cfg.limits.max_retries = 1

    run = pipeline(llm, configure=one_attempt)
    assert run.result.result == "failed"
    [comment] = [a for a in run.publisher.actions if a["type"] == "comment"]
    assert comment["kind"] == "failed" and "missing_search" in comment["body"]
    assert run.result.run_id in comment["body"]
    assert "# seconds" not in comment["body"] and "```" not in comment["body"]   # no model/tool output
    assert_tree_clean(run)
    llm.assert_exhausted()


def test_unexpected_exception_is_a_recorded_error_with_secrets_redacted(case, tmp_path):
    from fix_issue.redaction import register_secret

    token = "ghp_" + "Z9y8X7w6V5" * 4
    register_secret(token)

    def broken_workspace():
        raise subprocess.CalledProcessError(128, ["git", "clone", f"https://x-access-token:{token}@github.com/o/r"])

    cfg = Config()
    cfg.retrieval.embedding_backend = "local"
    cfg.retrieval.cache_dir = str(tmp_path / "cache")
    recorder = RunRecorder(repo="bench/x", issue_number=1, params={}, source={"type": "bench"},
                           directory=tmp_path / "runs", write_summary_log=False)
    llm = ScriptedLLM().script("triage", triage_json())
    orch = Orchestrator(cfg, TaggingRecorder(llm), BenchIssueSource(case), broken_workspace, RecordingPublisher(),
                        recorder=recorder, dry_run=True)
    res = orch.run()

    assert res.result == "error" and "CalledProcessError" in res.error
    assert token not in json.dumps(res.record) and token not in res.error
    assert any(e["stage"] == "traceback" for e in res.record["events"])
