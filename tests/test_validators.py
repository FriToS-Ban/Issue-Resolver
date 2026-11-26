"""Tests for validators.py — discovery, parsing, and diff-aware runs in a real git worktree."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fix_issue import validators
from fix_issue.config import ValidationConfig
from fix_issue.validators import Finding, ToolSpec, discover_tools, parse_output, run_validators


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@pytest.fixture
def installed(monkeypatch):
    """Control which tools count as installed: python tools via _resolve_python_tool, others via _which."""
    state = {"python": set(), "path": set()}

    def resolve(module, exe):
        return [f"/fake/{exe}"] if exe in state["python"] else None

    def which(exe):
        return f"/fake/bin/{exe}" if exe in state["path"] else None

    monkeypatch.setattr(validators, "_resolve_python_tool", resolve)
    monkeypatch.setattr(validators, "_which", which)
    return state


def _names(tools):
    return [t.name for t in tools]


def _skipped(skipped, tool):
    return [s["reason"] for s in skipped if s["tool"] == tool]


def test_py_syntax_always_discovered(tmp_path, installed):
    tools, skipped = discover_tools(tmp_path)
    assert _names(tools) == ["py-syntax"]
    assert tools[0].kind == "static" and tools[0].command == []
    assert skipped == []


def test_configured_and_installed_vs_not(tmp_path, installed):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n\n[tool.mypy]\nstrict = true\n")
    installed["python"] |= {"ruff"}
    tools, skipped = discover_tools(tmp_path)
    by_name = {t.name: t for t in tools}
    assert "ruff" in by_name and by_name["ruff"].reason == "[tool.ruff] in pyproject.toml"
    assert by_name["ruff"].command[0] == "/fake/ruff" and by_name["ruff"].per_file
    assert "mypy" not in by_name
    assert "not installed" in _skipped(skipped, "mypy")[0]
    # not configured -> neither runnable nor listed as skipped
    assert "flake8" not in by_name and not _skipped(skipped, "flake8")
    assert "black" not in by_name and not _skipped(skipped, "black")


@pytest.mark.parametrize("files,tool,reason", [
    ({"ruff.toml": "line-length = 88\n"}, "ruff", "ruff.toml present"),
    ({".ruff.toml": "[format]\nquote-style = 'single'\n"}, "ruff-format", "[format] in .ruff.toml"),
    ({"pyproject.toml": "[tool.ruff.format]\nquote-style = 'single'\n"}, "ruff-format",
     "[tool.ruff.format] in pyproject.toml"),
    ({".flake8": "[flake8]\nmax-line-length = 100\n"}, "flake8", ".flake8 present"),
    ({"setup.cfg": "[metadata]\nname = x\n\n[flake8]\nmax-line-length = 100\n"}, "flake8", "[flake8] in setup.cfg"),
    ({"tox.ini": "[flake8]\nignore = E501\n"}, "flake8", "[flake8] in tox.ini"),
    ({"mypy.ini": "[mypy]\nstrict = True\n"}, "mypy", "mypy.ini present"),
    ({"setup.cfg": "[mypy]\nstrict = True\n"}, "mypy", "[mypy] in setup.cfg"),
    ({"pyrightconfig.json": "{}"}, "pyright", "pyrightconfig.json present"),
    ({"pyproject.toml": "[tool.pyright]\ntypeCheckingMode = 'basic'\n"}, "pyright", "[tool.pyright] in pyproject.toml"),
    ({"pyproject.toml": "[tool.black]\nline-length = 100\n"}, "black", "[tool.black] in pyproject.toml"),
])
def test_python_tool_config_sources(tmp_path, installed, files, tool, reason):
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    installed["python"] |= {"ruff", "flake8", "mypy", "black"}
    installed["path"] |= {"pyright"}
    tools, _ = discover_tools(tmp_path)
    match = [t for t in tools if t.name == tool]
    assert match and match[0].reason == reason


def test_kind_toggles_respected(tmp_path, installed):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n[tool.mypy]\n[tool.black]\n")
    installed["python"] |= {"ruff", "mypy", "black"}
    cfg = ValidationConfig(run_linters=False, run_type_checkers=False, run_formatters=False)
    tools, skipped = discover_tools(tmp_path, cfg)
    assert _names(tools) == ["py-syntax"]
    assert "run_linters" in _skipped(skipped, "ruff")[0]
    assert "run_type_checkers" in _skipped(skipped, "mypy")[0]
    assert "run_formatters" in _skipped(skipped, "black")[0]


def test_node_and_go_tools(tmp_path, installed):
    (tmp_path / ".eslintrc.json").write_text("{}")
    (tmp_path / "tsconfig.json").write_text("{}")
    (tmp_path / "go.mod").write_text("module example.com/x\n")
    tools, skipped = discover_tools(tmp_path)
    assert {"eslint", "tsc", "go-vet"}.isdisjoint(_names(tools))
    assert "node_modules/.bin/eslint" in _skipped(skipped, "eslint")[0]
    assert "go not found" in _skipped(skipped, "go-vet")[0]

    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "eslint").write_text("")
    (bin_dir / "tsc").write_text("")
    installed["path"] |= {"go"}
    tools, _ = discover_tools(tmp_path)
    by_name = {t.name: t for t in tools}
    assert by_name["eslint"].per_file and by_name["eslint"].command[-2:] == ["-f", "json"]
    assert not by_name["tsc"].per_file and by_name["tsc"].kind == "typecheck"
    assert by_name["go-vet"].command == ["/fake/bin/go", "vet", "./..."]


def test_project_scripts_gated(tmp_path, installed):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"lint": "eslint ."}}))
    (tmp_path / "Makefile").write_text("lint:\n\techo lint\n")
    installed["path"] |= {"npm", "make"}
    tools, skipped = discover_tools(tmp_path)
    assert "npm-lint" not in _names(tools) and "make-lint" not in _names(tools)
    expected = "project script; set [validation] allow_project_scripts = true"
    assert _skipped(skipped, "npm-lint") == [expected] and _skipped(skipped, "make-lint") == [expected]

    tools, _ = discover_tools(tmp_path, ValidationConfig(allow_project_scripts=True))
    scripts = {t.name: t for t in tools if t.project_script}
    assert set(scripts) == {"npm-lint", "make-lint"}
    assert scripts["make-lint"].command == ["/fake/bin/make", "lint"] and not scripts["make-lint"].per_file


def test_real_availability_check_for_missing_module(tmp_path, monkeypatch):
    monkeypatch.setattr(validators, "_which", lambda exe: None)
    assert validators._resolve_python_tool("definitely_not_a_tool_xyz", "definitely-not-a-tool-xyz") is None


# ---------------------------------------------------------------------------
# Findings and parsing
# ---------------------------------------------------------------------------

def test_finding_key_ignores_line_numbers():
    a = Finding("mypy", "pkg/a.py", 10, "assignment", 'Name "x" already defined on line 3')
    b = Finding("mypy", "pkg/a.py", 42, "assignment", 'Name "x"  already defined on line 17')
    c = Finding("mypy", "pkg/a.py", 10, "assignment", 'Name "y" already defined on line 3')
    assert a.key() == b.key()
    assert a.key() != c.key()


def test_parse_outputs(tmp_path):
    ruff = json.dumps([{"code": "F401", "message": "`os` imported but unused",
                        "filename": str(tmp_path / "pkg" / "a.py"), "location": {"row": 1, "column": 8}}])
    assert parse_output("ruff", ruff, tmp_path) == [Finding("ruff", "pkg/a.py", 1, "F401", "`os` imported but unused")]

    flake8 = "pkg\\a.py:3:1: E302 expected 2 blank lines, found 1\n"
    assert parse_output("flake8", flake8, tmp_path)[0] == Finding("flake8", "pkg/a.py", 3, "E302",
                                                                  "expected 2 blank lines, found 1")

    mypy = ('pkg/a.py:5: error: Incompatible return value type (got "int", expected "str")  [return-value]\n'
            "pkg/a.py:5: note: See docs\n")
    found = parse_output("mypy", mypy, tmp_path)
    assert len(found) == 1 and found[0].code == "return-value" and found[0].line == 5

    black = f"would reformat {tmp_path / 'pkg' / 'a.py'}\nOh no!\n1 file would be reformatted.\n"
    assert parse_output("black", black, tmp_path) == [Finding("black", "pkg/a.py", None, "format", "would reformat")]

    tsc = "src/a.ts(3,7): error TS2322: Type 'number' is not assignable to type 'string'.\n"
    assert parse_output("tsc", tsc, tmp_path)[0].code == "TS2322"

    eslint = json.dumps([{"filePath": str(tmp_path / "src" / "a.js"),
                          "messages": [{"ruleId": "no-unused-vars", "severity": 2, "message": "'x' is unused", "line": 2}]}])
    assert parse_output("eslint", eslint, tmp_path) == [Finding("eslint", "src/a.js", 2, "no-unused-vars", "'x' is unused")]

    vet = "# example.com/x\nvet: ./main.go:10:2: unreachable code\n"
    assert parse_output("go-vet", vet, tmp_path)[0] == Finding("go-vet", "main.go", 10, "vet", "unreachable code")


# ---------------------------------------------------------------------------
# run_validators in a real git repo (exercises the baseline worktree)
# ---------------------------------------------------------------------------

FAKE_LINT = textwrap.dedent("""\
    import os, sys
    if os.path.exists("CRASH_MARKER"):
        sys.exit(2)
    found = False
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if "BAD" in line:
                    print(f"{path}:{n}: BAD001 {line.strip()}")
                    found = True
    sys.exit(1 if found else 0)
""")

FAKE_FORMATTER = textwrap.dedent("""\
    import sys
    bad = [p for p in sys.argv[2:] if "UGLY" in open(p, encoding="utf-8").read()]
    for p in bad:
        print(f"would reformat {p}", file=sys.stderr)
    sys.exit(1 if bad else 0)
""")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false",
         "-c", "core.autocrlf=false", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "a.py").write_text("x = 1  # BAD old\n")
    (root / "b.py").write_text("y = 2\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")

    made: list[Path] = []

    def make_temp_dir():
        d = tmp_path / f"wt-parent-{len(made)}"
        d.mkdir()
        made.append(d)
        return d

    monkeypatch.setattr(validators, "_make_temp_dir", make_temp_dir)
    return root, made


def _fake_tool(tmp_path: Path, name: str = "fake-lint", script: str = FAKE_LINT, kind: str = "lint",
               extra: list[str] | None = None) -> ToolSpec:
    path = tmp_path / f"{name}.py"
    path.write_text(script)
    return ToolSpec(name, kind, [sys.executable, str(path), *(extra or [])], True, "test tool")


def _assert_worktree_cleaned(root: Path, made: list[Path]) -> None:
    listing = _git(root, "worktree", "list", "--porcelain")
    assert listing.count("worktree ") == 1, listing
    assert made and all(not d.exists() for d in made)


def test_baseline_diffing_only_reports_new_findings(tmp_path, repo):
    root, made = repo
    (root / "a.py").write_text("import os\n\nx = 1  # BAD old\nz = 3  # BAD new\n")   # old one shifted down
    (root / "c.py").write_text("w = 1  # BAD fresh\n")                                  # file absent at base
    [res] = run_validators(root, [_fake_tool(tmp_path)], ["a.py", "c.py"])
    assert res.ran and not res.ok
    assert sorted((f.path, f.message) for f in res.new_findings) == [
        ("a.py", "z = 3  # BAD new"), ("c.py", "w = 1  # BAD fresh")]
    assert res.baseline_count == 1 and res.after_count == 3
    _assert_worktree_cleaned(root, made)


def test_preexisting_findings_are_not_new(tmp_path, repo):
    root, made = repo
    (root / "a.py").write_text("# header\n\n\nx = 1  # BAD old\n")
    [res] = run_validators(root, [_fake_tool(tmp_path)], ["a.py"])
    assert res.ran and res.ok and res.new_findings == []
    assert res.baseline_count == 1 and res.after_count == 1
    _assert_worktree_cleaned(root, made)


def test_formatter_only_counts_newly_unformatted_files(tmp_path, repo):
    root, made = repo
    (root / "a.py").write_text("x = 1  # UGLY\n")
    (root / "b.py").write_text("y = 2\n")
    _git(root, "commit", "-q", "-am", "a unformatted at base")
    (root / "a.py").write_text("x = 2  # UGLY\n")   # was already unformatted -> doesn't count
    (root / "b.py").write_text("y = 3  # UGLY\n")   # was formatted -> new
    tool = _fake_tool(tmp_path, name="black", script=FAKE_FORMATTER, kind="format", extra=["--check"])
    [res] = run_validators(root, [tool], ["a.py", "b.py"])
    assert res.ran and not res.ok
    assert [(f.path, f.code) for f in res.new_findings] == [("b.py", "format")]
    _assert_worktree_cleaned(root, made)


def test_py_syntax_finds_new_syntax_error(tmp_path, repo):
    root, made = repo
    (root / "b.py").write_text("def broken(:\n    pass\n")
    tools, _ = discover_tools(root)
    [res] = run_validators(root, tools, ["b.py"])
    assert res.tool == "py-syntax" and res.ran and not res.ok
    assert res.new_findings[0].path == "b.py" and res.new_findings[0].code == "SyntaxError"
    assert res.new_findings[0].line == 1
    _assert_worktree_cleaned(root, made)


def test_py_syntax_without_git_baseline(tmp_path):
    (tmp_path / "x.py").write_text("if True\n    pass\n")
    [res] = run_validators(tmp_path, [ToolSpec("py-syntax", "static", [], True, "built-in")], ["x.py"])
    assert res.ran and not res.ok and "baseline unavailable" in res.output_tail


def test_irrelevant_or_missing_files_skip(tmp_path, repo):
    root, made = repo
    (root / "notes.txt").write_text("hi\n")
    py = ToolSpec("py-syntax", "static", [], True, "built-in")
    [res] = run_validators(root, [py], ["notes.txt", "deleted.py"])
    assert not res.ran and res.ok and res.skipped_reason == "no relevant changed files"
    assert made == []   # no worktree needed


def test_tool_not_found_does_not_fail(tmp_path, repo):
    root, made = repo
    (root / "a.py").write_text("x = 1\n")
    tool = ToolSpec("ghost", "lint", [str(tmp_path / "no-such-binary-xyz")], True, "test")
    [res] = run_validators(root, [tool], ["a.py"])
    assert not res.ran and res.ok and "not found" in res.skipped_reason
    _assert_worktree_cleaned(root, made)


def test_baseline_crash_is_skipped_not_failed(tmp_path, repo):
    root, made = repo
    (root / "CRASH_MARKER").write_text("")
    _git(root, "add", "CRASH_MARKER")
    _git(root, "commit", "-q", "-m", "marker")
    (root / "CRASH_MARKER").unlink()                   # present at base only
    (root / "a.py").write_text("x = 1  # BAD old\nq = 1  # BAD new\n")
    [res] = run_validators(root, [_fake_tool(tmp_path)], ["a.py", "CRASH_MARKER"])
    assert not res.ran and res.ok and "baseline run failed" in res.skipped_reason
    _assert_worktree_cleaned(root, made)


def test_worktree_removed_even_when_a_tool_errors(tmp_path, repo, monkeypatch):
    root, made = repo
    (root / "a.py").write_text("x = 1  # BAD old\n")

    def boom(*args, **kwargs):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(validators, "parse_output", boom)
    [res, syntax] = run_validators(root, [_fake_tool(tmp_path), ToolSpec("py-syntax", "static", [], True, "b")],
                                   ["a.py"])
    assert not res.ran and res.ok and "parser exploded" in res.skipped_reason
    assert syntax.ran and syntax.ok    # other tools still run
    _assert_worktree_cleaned(root, made)


def test_result_to_dict_is_serializable(tmp_path, repo):
    root, _ = repo
    (root / "a.py").write_text("x = 1  # BAD old\ny = 2  # BAD new\n")
    [res] = run_validators(root, [_fake_tool(tmp_path)], ["a.py"])
    data = json.loads(json.dumps(res.to_dict()))
    assert data["tool"] == "fake-lint" and data["new_findings"][0]["code"] == "BAD001"
