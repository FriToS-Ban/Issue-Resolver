"""
bench/cases.py — Load benchmark cases and materialise their buggy workspaces.

A case lives in ``benchmark/cases/<id>/case.toml`` and points at a clean fixture
repository in ``benchmark/fixtures/<fixture>/``.  The buggy baseline the agent sees
is produced by applying the case's ``[[inject]]`` replacements (and ``[[add_files]]``)
to a fresh copy of the fixture.  Hidden tests in ``benchmark/cases/<id>/hidden_tests/``
are copied in only AFTER the agent has finished, so they verify that the underlying
issue is fixed rather than merely that the visible suite is green.

Injection deliberately uses plain exact string replacement, NOT the agent's own
editing engine, so benchmark integrity never depends on the code under test.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

DIFFICULTIES = ("easy", "medium", "hard")
KINDS = ("fix", "triage")
CLASSIFICATIONS = ("fixable", "needs_info", "out_of_scope", "duplicate")

DEFAULT_BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmark"
HIDDEN_TEST_DIR = "tests/_hidden"


class CaseError(Exception):
    """Raised when a case definition is malformed or cannot be materialised."""


@dataclass
class Injection:
    file: str
    search: str
    replace: str


@dataclass
class AddedFile:
    path: str
    content: str


@dataclass
class Case:
    id: str
    fixture: str
    difficulty: str
    kind: str
    expected_classification: str
    issue: dict[str, Any]
    inject: list[Injection]
    add_files: list[AddedFile]
    expected_files: list[str]
    expected_symbols: list[str]
    forbidden_paths: list[str]
    max_diff_lines: int | None
    root_cause_category: str
    requires_manual_review: bool
    visible_tests_fail_before: bool
    recent_resolved: list[dict[str, Any]]
    notes: str
    case_dir: Path
    fixture_dir: Path
    hidden_tests: list[Path] = field(default_factory=list)

    @property
    def is_fix(self) -> bool:
        return self.kind == "fix"


def load_case(case_dir: Path, bench_dir: Path = DEFAULT_BENCH_DIR) -> Case:
    toml_path = case_dir / "case.toml"
    if not toml_path.exists():
        raise CaseError(f"{case_dir}: missing case.toml")
    try:
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CaseError(f"{toml_path}: invalid TOML: {exc}") from exc

    def req(key: str) -> Any:
        if key not in raw:
            raise CaseError(f"{toml_path}: missing required key '{key}'")
        return raw[key]

    case_id = str(req("id"))
    if case_id != case_dir.name:
        raise CaseError(f"{toml_path}: id '{case_id}' must equal directory name '{case_dir.name}'")
    difficulty = str(req("difficulty"))
    if difficulty not in DIFFICULTIES:
        raise CaseError(f"{case_id}: difficulty must be one of {DIFFICULTIES}")
    kind = str(raw.get("kind", "fix"))
    if kind not in KINDS:
        raise CaseError(f"{case_id}: kind must be one of {KINDS}")
    expected_cls = str(raw.get("expected_classification", "fixable"))
    if expected_cls not in CLASSIFICATIONS:
        raise CaseError(f"{case_id}: expected_classification must be one of {CLASSIFICATIONS}")

    fixture = str(req("fixture"))
    fixture_dir = bench_dir / "fixtures" / fixture
    if not fixture_dir.is_dir():
        raise CaseError(f"{case_id}: fixture '{fixture}' not found at {fixture_dir}")

    issue = dict(req("issue"))
    for key in ("number", "title", "body"):
        if key not in issue:
            raise CaseError(f"{case_id}: [issue] missing '{key}'")
    issue.setdefault("labels", [])
    # Mimic the GitHub API shape that the pipeline consumes.
    issue["labels"] = [{"name": lbl} if isinstance(lbl, str) else lbl for lbl in issue["labels"]]

    inject = [Injection(str(i["file"]), str(i["search"]), str(i["replace"])) for i in raw.get("inject", [])]
    add_files = [AddedFile(str(a["path"]), str(a["content"])) for a in raw.get("add_files", [])]
    expected = raw.get("expected", {})

    hidden_dir = case_dir / "hidden_tests"
    hidden = sorted(hidden_dir.glob("*.py")) if hidden_dir.is_dir() else []
    if kind == "fix" and not hidden:
        raise CaseError(f"{case_id}: fix cases need at least one hidden test in hidden_tests/")
    if kind == "fix" and not inject:
        raise CaseError(f"{case_id}: fix cases need at least one [[inject]] block")

    return Case(
        id=case_id,
        fixture=fixture,
        difficulty=difficulty,
        kind=kind,
        expected_classification=expected_cls,
        issue=issue,
        inject=inject,
        add_files=add_files,
        expected_files=[str(p) for p in expected.get("files", [])],
        expected_symbols=[str(s) for s in expected.get("symbols", [])],
        forbidden_paths=[str(p) for p in expected.get("forbidden_paths", [])],
        max_diff_lines=expected.get("max_diff_lines"),
        root_cause_category=str(raw.get("root_cause_category", "other")),
        requires_manual_review=bool(raw.get("requires_manual_review", False)),
        visible_tests_fail_before=bool(raw.get("visible_tests_fail_before", False)),
        recent_resolved=list(raw.get("recent_resolved", [])),
        notes=str(raw.get("notes", "")),
        case_dir=case_dir,
        fixture_dir=fixture_dir,
        hidden_tests=hidden,
    )


def load_cases(bench_dir: Path = DEFAULT_BENCH_DIR, only: list[str] | None = None) -> list[Case]:
    cases_root = bench_dir / "cases"
    if not cases_root.is_dir():
        raise CaseError(f"No cases directory at {cases_root}")
    cases = [load_case(d, bench_dir) for d in sorted(cases_root.iterdir()) if (d / "case.toml").exists()]
    if only:
        wanted = set(only)
        cases = [c for c in cases if c.id in wanted or c.difficulty in wanted or c.fixture in wanted]
    return cases


# ---------------------------------------------------------------------------
# Workspace materialisation
# ---------------------------------------------------------------------------

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".git")


def copy_fixture(case: Case, dest: Path) -> None:
    shutil.copytree(case.fixture_dir, dest, ignore=_IGNORE, dirs_exist_ok=True)


def inject_bug(case: Case, root: Path) -> None:
    """Apply the case's injections with exact, unique string replacement."""
    for inj in case.inject:
        target = root / inj.file
        if not target.is_file():
            raise CaseError(f"{case.id}: inject target '{inj.file}' does not exist")
        text = target.read_text(encoding="utf-8")
        count = text.count(inj.search)
        if count != 1:
            raise CaseError(
                f"{case.id}: inject search text for '{inj.file}' matched {count} times (need exactly 1)"
            )
        target.write_text(text.replace(inj.search, inj.replace, 1), encoding="utf-8", newline="\n")
    for added in case.add_files:
        target = root / added.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(added.content, encoding="utf-8", newline="\n")


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(root), check=True, capture_output=True, text=True,
    )


def init_git_repo(root: Path, message: str) -> str:
    """Initialise a deterministic git repo at ``root`` and commit everything. Returns HEAD sha."""
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "bench@fix-issue.local")
    git(root, "config", "user.name", "fix-issue bench")
    git(root, "config", "core.autocrlf", "false")
    git(root, "config", "commit.gpgsign", "false")
    exclude = root / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text("__pycache__/\n*.pyc\n.pytest_cache/\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD").stdout.strip()


def materialise(case: Case, dest: Path) -> str:
    """Create the buggy baseline repo the agent will work on. Returns the base commit sha."""
    copy_fixture(case, dest)
    inject_bug(case, dest)
    return init_git_repo(dest, f"buggy baseline for {case.id}")


def install_hidden_tests(case: Case, root: Path) -> list[str]:
    """Copy hidden tests into the workspace (after the agent finished). Returns relative paths."""
    target_dir = root / HIDDEN_TEST_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "__init__.py").touch()
    out = []
    for src in case.hidden_tests:
        shutil.copy2(src, target_dir / src.name)
        out.append(f"{HIDDEN_TEST_DIR}/{src.name}")
    return out
