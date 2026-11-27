"""Tests for editing.py — SEARCH/REPLACE parsing, planning, writing and diffing."""
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from fix_issue.editing import (
    EDIT_FORMAT_INSTRUCTIONS,
    Edit,
    EditBatchResult,
    EditWriteError,
    locate_best_match,
    parse_edit_response,
    plan_edits,
    resolve_repo_path,
    unified_diff,
    validate_syntax,
    write_edits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def _read(root: Path, rel: str) -> str:
    return (root / rel).read_bytes().decode("utf-8")


def _dedent(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


CALC = _dedent("""
    def add(a, b):
        return a + b


    def sub(a, b):
        return a - b
""")


# ---------------------------------------------------------------------------
# Exact matching, ordering, multi-file
# ---------------------------------------------------------------------------

def test_exact_edit_applies_and_writes(tmp_path):
    _write(tmp_path, "calc.py", CALC)
    result = plan_edits(tmp_path, [Edit("calc.py", "    return a - b\n", "    return a - b - 0\n")])

    assert result.ok, result.errors
    outcome = result.outcomes[0]
    assert (outcome.status, outcome.matched, outcome.line, outcome.occurrences_found) == ("applied", "exact", 6, 1)
    assert _read(tmp_path, "calc.py") == CALC, "plan_edits must never write"

    assert write_edits(tmp_path, result) == ["calc.py"]
    assert _read(tmp_path, "calc.py") == CALC.replace("return a - b\n", "return a - b - 0\n")


def test_multi_file_edits(tmp_path):
    _write(tmp_path, "pkg/a.py", "A = 1\n")
    _write(tmp_path, "pkg/b.py", "B = 2\n")
    result = plan_edits(tmp_path, [
        Edit("pkg/a.py", "A = 1\n", "A = 10\n"),
        Edit("pkg/b.py", "B = 2\n", "B = 20\n"),
    ])
    assert result.ok
    assert set(result.new_contents) == {"pkg/a.py", "pkg/b.py"}
    assert "--- a/pkg/a.py" in result.diff and "+++ b/pkg/b.py" in result.diff
    write_edits(tmp_path, result)
    assert _read(tmp_path, "pkg/a.py") == "A = 10\n"
    assert _read(tmp_path, "pkg/b.py") == "B = 20\n"


def test_several_edits_to_one_file_apply_in_order(tmp_path):
    _write(tmp_path, "calc.py", CALC)
    result = plan_edits(tmp_path, [
        Edit("calc.py", "def add(a, b):\n", "def plus(a, b):\n"),
        # second edit only matches the content produced by the first
        Edit("calc.py", "def plus(a, b):\n    return a + b\n", "def plus(a, b):\n    return b + a\n"),
    ])
    assert result.ok, result.errors
    assert result.new_contents["calc.py"].startswith("def plus(a, b):\n    return b + a\n")


def test_line_based_search_does_not_match_mid_line(tmp_path):
    _write(tmp_path, "m.py", "max = 1\n")
    result = plan_edits(tmp_path, [Edit("m.py", "x = 1\n", "x = 2\n")])
    assert not result.ok
    assert result.outcomes[0].error_kind == "missing_search"


# ---------------------------------------------------------------------------
# Ambiguity and occurrences
# ---------------------------------------------------------------------------

DUP = _dedent("""
    def a():
        value = compute()
        return value

    def b():
        value = compute()
        return value
""")


def test_ambiguous_match_lists_every_line(tmp_path):
    _write(tmp_path, "dup.py", DUP)
    result = plan_edits(tmp_path, [Edit("dup.py", "    value = compute()\n", "    value = compute(1)\n")])
    outcome = result.outcomes[0]
    assert not result.ok
    assert outcome.error_kind == "ambiguous_match"
    assert outcome.occurrences_found == 2
    assert "lines 2, 6" in outcome.error
    assert "surrounding" in outcome.error
    assert result.new_contents == {}


def test_occurrences_n_replaces_all(tmp_path):
    _write(tmp_path, "dup.py", DUP)
    result = plan_edits(tmp_path, [
        Edit("dup.py", "    value = compute()\n", "    value = compute(1)\n", expected_occurrences=2)
    ])
    assert result.ok, result.errors
    assert result.new_contents["dup.py"].count("compute(1)") == 2
    assert result.outcomes[0].occurrences_found == 2


def test_occurrence_mismatch_when_fewer_found(tmp_path):
    _write(tmp_path, "dup.py", DUP)
    result = plan_edits(tmp_path, [
        Edit("dup.py", "    value = compute()\n", "    value = compute(1)\n", expected_occurrences=3)
    ])
    outcome = result.outcomes[0]
    assert outcome.error_kind == "occurrence_mismatch"
    assert "expected 3" in outcome.error and "found 2" in outcome.error


def test_overlapping_matches_are_ambiguous(tmp_path):
    _write(tmp_path, "x.txt", "x\nx\nx\n")
    result = plan_edits(tmp_path, [Edit("x.txt", "x\nx\n", "y\n")])
    assert result.outcomes[0].error_kind == "ambiguous_match"


# ---------------------------------------------------------------------------
# Whitespace-tolerant matching
# ---------------------------------------------------------------------------

def test_trailing_whitespace_tolerant_match(tmp_path):
    _write(tmp_path, "t.py", "def f():   \n    return 1  \n")
    result = plan_edits(tmp_path, [Edit("t.py", "def f():\n    return 1\n", "def f():\n    return 2\n")])
    assert result.ok, result.errors
    assert result.outcomes[0].matched == "trailing_ws"
    assert result.new_contents["t.py"] == "def f():\n    return 2\n"


def test_indent_tolerant_match_reindents_replacement(tmp_path):
    source = _dedent("""
        class Service:
            def run(self):
                if self.ready:
                    return self.go()
                return None
    """)
    _write(tmp_path, "svc.py", source)
    # The model dropped the class-level indentation in both SEARCH and REPLACE.
    search = "def run(self):\n    if self.ready:\n        return self.go()\n"
    replace = "def run(self):\n    if not self.ready:\n        self.prepare()\n    return self.go()\n"
    result = plan_edits(tmp_path, [Edit("svc.py", search, replace)])

    assert result.ok, result.errors
    outcome = result.outcomes[0]
    assert (outcome.matched, outcome.line) == ("indent", 2)
    assert result.new_contents["svc.py"] == _dedent("""
        class Service:
            def run(self):
                if not self.ready:
                    self.prepare()
                return self.go()
                return None
    """)


def test_indent_tolerant_match_rejects_multiple_windows(tmp_path):
    source = "def a():\n    x = 1\n\nclass C:\n    def a():\n        x = 1\n"
    _write(tmp_path, "m.py", source)
    result = plan_edits(tmp_path, [Edit("m.py", "  def a():\n      x = 1\n", "  def a():\n      x = 2\n")])
    outcome = result.outcomes[0]
    assert outcome.error_kind == "ambiguous_match"
    assert outcome.matched == "indent"
    assert "1, 5" in outcome.error


def test_indent_mode_respects_relative_indentation(tmp_path):
    # Same stripped text but different nesting must not be treated as a match.
    _write(tmp_path, "n.py", "if a:\n    b()\n")
    result = plan_edits(tmp_path, [Edit("n.py", "if a:\nb()\n", "if a:\nc()\n")])
    assert result.outcomes[0].error_kind == "missing_search"


# ---------------------------------------------------------------------------
# Missing SEARCH, best match, already applied
# ---------------------------------------------------------------------------

def test_missing_search_reports_best_match(tmp_path):
    body = "".join(f"line_{i} = {i}\n" for i in range(1, 30))
    body += "def total(items):\n    result = sum(items)\n    return result\n"
    _write(tmp_path, "big.py", body)
    search = "def total(items):\n    result = sum(item)\n    return result\n"   # typo: item
    result = plan_edits(tmp_path, [Edit("big.py", search, "def total(items):\n    return sum(items)\n")])

    outcome = result.outcomes[0]
    assert not result.ok
    assert outcome.error_kind == "missing_search"
    best = outcome.best_match
    assert (best["start_line"], best["end_line"]) == (30, 32)
    assert best["similarity"] > 0.9
    assert "result = sum(items)" in best["text"]
    assert "   31 |     result = sum(items)" in outcome.error
    assert "SEARCH line 2" in outcome.error and "file line 31" in outcome.error


def test_locate_best_match_direct():
    content = "a = 1\nb = 2\nc = 3\nd = 4\n"
    best = locate_best_match(content, "b = 2\nc = 33\n")
    assert best["start_line"] == 2 and best["end_line"] == 3
    assert 0 < best["similarity"] <= 1
    assert locate_best_match("", "x\n") is None
    assert locate_best_match(content, "") is None


def test_already_applied_detected_when_replace_present(tmp_path):
    # A re-sent edit: REPLACE (with its context line) sits where the SEARCH best matches.
    _write(tmp_path, "calc.py", CALC.replace("return a - b", "return b - a"))
    edit = Edit("calc.py", "def sub(a, b):\n    return a - b\n", "def sub(a, b):\n    return b - a\n")
    result = plan_edits(tmp_path, [edit])
    assert result.ok
    assert result.outcomes[0].status == "already_applied"
    assert result.new_contents == {} and result.diff == ""


def test_replace_text_elsewhere_is_not_already_applied(tmp_path):
    # The REPLACE line exists in another function; the (wrong) SEARCH best-matches second().
    _write(tmp_path, "m.py", _dedent("""
        def first(x):
            return None


        def second(y):
            value = compute(y)
            return value
    """))
    result = plan_edits(tmp_path, [Edit("m.py", "    value = compute(y, 1)\n", "    return None\n")])
    outcome = result.outcomes[0]
    assert not result.ok
    assert outcome.status == "failed" and outcome.error_kind == "missing_search"
    assert outcome.best_match["start_line"] == 6


def test_already_applied_when_replace_contains_search(tmp_path):
    _write(tmp_path, "imp.py", "import os\n")
    edit = Edit("imp.py", "import os\n", "import os\nimport sys\n")
    first = plan_edits(tmp_path, [edit])
    write_edits(tmp_path, first)
    second = plan_edits(tmp_path, [edit])
    assert second.ok
    assert second.outcomes[0].status == "already_applied"
    assert second.new_contents == {}


def test_partial_batch_reports_failure_without_writing(tmp_path):
    _write(tmp_path, "calc.py", CALC)
    result = plan_edits(tmp_path, [
        Edit("calc.py", "    return a + b\n", "    return b + a\n"),
        Edit("calc.py", "    return nothing\n", "    return 0\n"),
    ])
    assert not result.ok
    assert [o.status for o in result.outcomes] == ["applied", "failed"]
    assert len(result.errors) == 1 and "Edit #2" in result.errors[0]
    with pytest.raises(EditWriteError):
        write_edits(tmp_path, result)
    assert _read(tmp_path, "calc.py") == CALC


def test_no_change_and_empty_edit_list(tmp_path):
    _write(tmp_path, "calc.py", CALC)
    same = plan_edits(tmp_path, [Edit("calc.py", "    return a + b\n", "    return a + b\n")])
    assert same.outcomes[0].error_kind == "no_change"
    empty = plan_edits(tmp_path, [])
    assert not empty.ok and empty.errors


# ---------------------------------------------------------------------------
# New files, missing files, unsafe paths
# ---------------------------------------------------------------------------

def test_new_file_creation(tmp_path):
    result = plan_edits(tmp_path, [Edit("pkg/sub/new_mod.py", "", "VALUE = 1\n", create=True)])
    assert result.ok, result.errors
    assert result.outcomes[0].matched == "new_file"
    assert result.original_contents == {"pkg/sub/new_mod.py": None}
    assert "--- /dev/null\n+++ b/pkg/sub/new_mod.py" in result.diff
    assert write_edits(tmp_path, result) == ["pkg/sub/new_mod.py"]
    assert _read(tmp_path, "pkg/sub/new_mod.py") == "VALUE = 1\n"


def test_new_file_then_edit_in_same_batch(tmp_path):
    result = plan_edits(tmp_path, [
        Edit("n.py", "", "A = 1\nB = 2\n", create=True),
        Edit("n.py", "B = 2\n", "B = 3\n"),
    ])
    assert result.ok, result.errors
    assert result.new_contents["n.py"] == "A = 1\nB = 3\n"


def test_create_existing_file_is_file_exists(tmp_path):
    _write(tmp_path, "calc.py", CALC)
    result = plan_edits(tmp_path, [Edit("calc.py", "", "x = 1\n", create=True)])
    assert result.outcomes[0].error_kind == "file_exists"


def test_empty_search_on_existing_file(tmp_path):
    _write(tmp_path, "calc.py", CALC)
    result = plan_edits(tmp_path, [Edit("calc.py", "", "x = 1\n")])
    assert result.outcomes[0].error_kind == "empty_search"


def test_file_not_found_suggests_close_names(tmp_path):
    _write(tmp_path, "pkg/handler.py", "x = 1\n")
    result = plan_edits(tmp_path, [Edit("pkg/handlers.py", "x = 1\n", "x = 2\n")])
    outcome = result.outcomes[0]
    assert outcome.error_kind == "file_not_found"
    assert "pkg/handler.py" in outcome.error


@pytest.mark.parametrize("path", [
    "/etc/passwd",
    "C:/Windows/win.ini",
    "C:\\Windows\\win.ini",
    "../outside.py",
    "pkg/../../outside.py",
    ".git/config",
    "sub/.git/hooks/pre-commit",
    "",
])
def test_unsafe_paths_rejected(tmp_path, path):
    result = plan_edits(tmp_path, [Edit(path, "", "boom\n", create=True)])
    assert not result.ok
    assert result.outcomes[0].error_kind == "unsafe_path"
    assert result.original_contents == {} and result.new_contents == {}


def test_backslash_and_dot_prefix_paths_normalised(tmp_path):
    _write(tmp_path, "pkg/a.py", "A = 1\n")
    result = plan_edits(tmp_path, [
        Edit(".\\pkg\\a.py", "A = 1\n", "A = 2\n"),
        Edit("./pkg/a.py", "A = 2\n", "A = 3\n"),
    ])
    assert result.ok, result.errors
    assert result.new_contents == {"pkg/a.py": "A = 3\n"}
    assert result.outcomes[0].edit.file == "pkg/a.py"


# ---------------------------------------------------------------------------
# Syntax validation
# ---------------------------------------------------------------------------

def test_python_syntax_error_blocks_batch(tmp_path):
    _write(tmp_path, "calc.py", CALC)
    result = plan_edits(tmp_path, [Edit("calc.py", "    return a + b\n", "    return (a + b\n")])
    assert not result.ok
    outcome = result.outcomes[0]
    assert outcome.status == "failed" and outcome.error_kind == "syntax_error"
    assert "line 2" in result.validation["calc.py"] or "line 1" in result.validation["calc.py"]
    assert any("Syntax check failed" in e for e in result.errors)


def test_preexisting_syntax_error_is_not_blamed(tmp_path):
    # JSONC-style config: already invalid JSON before the edit.
    _write(tmp_path, "tsconfig.json", '{\n  // comment\n  "strict": false\n}\n')
    result = plan_edits(tmp_path, [Edit("tsconfig.json", '  "strict": false\n', '  "strict": true\n')])
    assert result.ok, result.errors
    assert result.validation["tsconfig.json"] is None


def test_validate_syntax_by_type():
    assert validate_syntax("a.py", "x = 1\n") is None
    assert "line 1" in validate_syntax("a.py", "def f(:\n")
    assert validate_syntax("a.json", '{"a": 1}') is None
    assert "line 2" in validate_syntax("a.json", '{\n"a": }')
    assert validate_syntax("a.toml", 'a = 1\n') is None
    assert "line 1" in validate_syntax("a.toml", 'a = \n')
    assert validate_syntax("README.md", "def f(:\n") is None
    assert validate_syntax("a.py", "\ufeffx = 1\r\n") is None


# ---------------------------------------------------------------------------
# Line endings, trailing newline, BOM
# ---------------------------------------------------------------------------

def test_crlf_file_stays_crlf(tmp_path):
    _write(tmp_path, "win.py", "def f():\r\n    return 1\r\n\r\nX = 2\r\n")
    result = plan_edits(tmp_path, [Edit("win.py", "    return 1\n", "    value = 1\n    return value\n")])
    assert result.ok, result.errors
    assert "\r" not in result.diff
    write_edits(tmp_path, result)
    data = (tmp_path / "win.py").read_bytes()
    assert data == b"def f():\r\n    value = 1\r\n    return value\r\n\r\nX = 2\r\n"


def test_no_trailing_newline_preserved(tmp_path):
    _write(tmp_path, "nonl.txt", "first\nlast")
    result = plan_edits(tmp_path, [Edit("nonl.txt", "last\n", "LAST\nappended\n")])
    assert result.ok, result.errors
    assert result.new_contents["nonl.txt"] == "first\nLAST\nappended"
    assert "\\ No newline at end of file" in result.diff
    write_edits(tmp_path, result)
    assert _read(tmp_path, "nonl.txt") == "first\nLAST\nappended"


def test_bom_preserved(tmp_path):
    _write(tmp_path, "bom.py", "\ufeffX = 1\nY = 2\n")
    result = plan_edits(tmp_path, [Edit("bom.py", "X = 1\n", "X = 3\n")])
    assert result.ok, result.errors
    write_edits(tmp_path, result)
    assert (tmp_path / "bom.py").read_bytes() == b"\xef\xbb\xbfX = 3\nY = 2\n"


def test_write_edits_refuses_stale_file(tmp_path):
    _write(tmp_path, "calc.py", CALC)
    result = plan_edits(tmp_path, [Edit("calc.py", "    return a + b\n", "    return b + a\n")])
    _write(tmp_path, "calc.py", CALC + "# changed meanwhile\n")
    with pytest.raises(EditWriteError, match="changed on disk"):
        write_edits(tmp_path, result)


# ---------------------------------------------------------------------------
# unified_diff + git apply
# ---------------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=cwd, capture_output=True, text=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_unified_diff_applies_with_git(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    files = {
        "calc.py": CALC,
        "pkg/nonl.txt": "alpha\nbeta",
        "pkg/long.py": "".join(f"v{i} = {i}\n" for i in range(40)),
    }
    for rel, text in files.items():
        _write(repo, rel, text)
    assert _git(repo, "init", "-q").returncode == 0
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-qm", "init").returncode == 0

    result = plan_edits(repo, [
        Edit("calc.py", "    return a - b\n", "    return a - b  # subtract\n"),
        Edit("pkg/nonl.txt", "beta\n", "gamma\n"),
        Edit("pkg/long.py", "v3 = 3\n", "v3 = 33\n"),
        Edit("pkg/long.py", "v30 = 30\n", ""),
        Edit("pkg/new_mod.py", "", "NEW = True\n", create=True),
        Edit("pkg/__init__.py", "", "", create=True),        # empty new file
    ])
    assert result.ok, result.errors

    patch = tmp_path / "fix.patch"
    patch.write_bytes(result.diff.encode("utf-8"))
    check = _git(repo, "apply", "--check", str(patch))
    assert check.returncode == 0, check.stderr + "\n" + result.diff

    assert _git(repo, "apply", str(patch)).returncode == 0
    for rel, expected in result.new_contents.items():
        assert _read(repo, rel) == expected, rel
    assert (repo / "pkg/__init__.py").exists()


def test_unified_diff_headers_and_skips_unchanged():
    diff = unified_diff({"a.py": "x\n", "same.py": "s\n", "new.py": None},
                        {"a.py": "y\n", "same.py": "s\n", "new.py": "n\n"})
    assert "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n" in diff
    assert "new file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+n\n" in diff
    assert "same.py" not in diff
    assert "\r" not in unified_diff({"c.py": "a\r\nb\r\n"}, {"c.py": "a\nc\n"})


def test_to_dict_is_json_serialisable_without_contents(tmp_path):
    _write(tmp_path, "calc.py", CALC)
    result = plan_edits(tmp_path, [
        Edit("calc.py", "    return a + b\n", "    return b + a\n"),
        Edit("calc.py", "    return nope\n", "    return 0\n"),
    ])
    summary = result.to_dict()
    text = json.dumps(summary)
    assert summary["ok"] is False
    assert summary["lines_added"] == 1 and summary["lines_removed"] == 1
    assert [e["status"] for e in summary["edits"]] == ["applied", "failed"]
    assert summary["edits"][1]["error_kind"] == "missing_search"
    assert summary["edits"][1]["best_match"]["start_line"] >= 1
    assert "def sub(a, b)" not in text, "summary must not embed full file contents"
    assert "new_contents" not in summary and "original_contents" not in summary
    assert isinstance(EditBatchResult(ok=True).to_dict(), dict)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

BASIC = """\
INTENT:
Fix subtraction order. Also add a helper.

FILE: calc.py
<<<<<<< SEARCH
    return a - b
=======
    return b - a
>>>>>>> REPLACE

FILE: util/helpers.py (new)
<<<<<<< SEARCH
=======
def helper():
    return 1
>>>>>>> REPLACE
"""


def test_parse_basic_response():
    parsed = parse_edit_response(BASIC)
    assert parsed.intent == "Fix subtraction order. Also add a helper."
    assert parsed.cannot_fix is None and parsed.format_errors == []
    assert [e.file for e in parsed.edits] == ["calc.py", "util/helpers.py"]
    assert parsed.edits[0].search == "    return a - b\n"
    assert parsed.edits[0].replace == "    return b - a\n"
    assert parsed.edits[1].create and parsed.edits[1].search == ""
    assert parsed.edits[1].replace == "def helper():\n    return 1\n"
    assert parsed.raw == BASIC


def test_parse_fences_think_bold_and_backslashes():
    text = (
        "<think>Maybe FILE: wrong.py\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n</think>\n"
        "```\n"
        "**INTENT:** Rename the variable.\n\n"
        "**FILE:** `.\\pkg\\mod.py` (occurrences: 2)\n"
        "```python\n"
        "<<<<<<<  SEARCH  \n"
        "value = 1\n"
        "  =======\n"
        "count = 1\n"
        ">>>>>>>   REPLACE\n"
        "```\n"
        "```\n"
    ).replace("\n", "\r\n")
    parsed = parse_edit_response(text)
    assert parsed.format_errors == []
    assert parsed.intent == "Rename the variable."
    assert len(parsed.edits) == 1
    edit = parsed.edits[0]
    assert edit.file == "pkg/mod.py"
    assert edit.expected_occurrences == 2
    assert (edit.search, edit.replace) == ("value = 1\n", "count = 1\n")


def test_parse_cannot_fix():
    parsed = parse_edit_response("**CANNOT_FIX:** the issue needs a product decision.")
    assert parsed.cannot_fix == "the issue needs a product decision."
    assert parsed.edits == [] and parsed.format_errors == []


def test_parse_missing_intent_and_multiple_blocks_same_file():
    text = (
        "FILE: a.py\n<<<<<<< SEARCH\nA = 1\n=======\nA = 2\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\nB = 1\n=======\nB = 2\n>>>>>>> REPLACE\n"
    )
    parsed = parse_edit_response(text)
    assert parsed.intent == ""
    assert [e.file for e in parsed.edits] == ["a.py", "a.py"]
    assert parsed.format_errors == []


def test_parse_unterminated_block_is_reported_and_dropped():
    text = "FILE: a.py\n<<<<<<< SEARCH\nA = 1\n=======\nA = 2\n"
    parsed = parse_edit_response(text)
    assert parsed.edits == []
    assert any("not terminated" in e for e in parsed.format_errors)


def test_parse_recovers_after_missing_replace_marker():
    text = (
        "FILE: a.py\n<<<<<<< SEARCH\nA = 1\n=======\nA = 2\n"
        "FILE: b.py\n<<<<<<< SEARCH\nB = 1\n=======\nB = 2\n>>>>>>> REPLACE\n"
    )
    parsed = parse_edit_response(text)
    assert [e.file for e in parsed.edits] == ["b.py"]
    assert any("missing its '>>>>>>> REPLACE'" in e for e in parsed.format_errors)


def test_parse_missing_file_line_and_no_blocks():
    parsed = parse_edit_response("<<<<<<< SEARCH\nA\n=======\nB\n>>>>>>> REPLACE\n")
    assert parsed.edits == []
    assert any("no preceding 'FILE: <path>'" in e for e in parsed.format_errors)

    prose = parse_edit_response("I think you should change line 3.")
    assert prose.edits == [] and prose.cannot_fix is None
    assert any("No complete SEARCH/REPLACE blocks" in e for e in prose.format_errors)


def test_parse_file_line_without_block_is_reported():
    parsed = parse_edit_response("FILE: a.py\n```python\nprint('whole file')\n```\n")
    assert any("not followed by" in e for e in parsed.format_errors)


def test_parse_then_plan_end_to_end(tmp_path):
    _write(tmp_path, "calc.py", CALC)
    parsed = parse_edit_response(BASIC.replace("    return a - b\n=======", "    return a - b\n======="))
    result = plan_edits(tmp_path, parsed.edits)
    assert result.ok, result.errors
    write_edits(tmp_path, result)
    assert "return b - a" in _read(tmp_path, "calc.py")
    assert _read(tmp_path, "util/helpers.py") == "def helper():\n    return 1\n"


def test_format_instructions_describe_the_format():
    for marker in ("INTENT:", "FILE:", "<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE",
                   "(occurrences: N)", "(new)", "CANNOT_FIX:"):
        assert marker in EDIT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Path aliasing: NTFS trailing dots/spaces, 8.3 short names, streams, symlinks
# ---------------------------------------------------------------------------

def _git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)


@pytest.mark.parametrize("path", [
    ".git./hooks/pre-commit",          # Windows strips the trailing dot -> .git/hooks
    ".git /hooks/pre-commit",          # ... and trailing spaces
    ".git../hooks/pre-commit",
    "x/.git./config",
    ".GIT/hooks/pre-commit",
    "GIT~1/hooks/pre-commit",          # 8.3 short name of .git
    "git~2/config",
    "GITHUB~1/WORKFL~1/ci.yml",        # 8.3 short names of .github/workflows
    ".git::$INDEX_ALLOCATION/hooks/x",  # NTFS directory stream of .git
    "src/file.py:stream",
    ".g\u200cit/config",          # HFS+ ignores U+200C -> .git on macOS
    "src /mod.py",
    "src./mod.py",
])
def test_ntfs_and_hfs_alias_paths_rejected(tmp_path, path):
    _git_init(tmp_path)
    result = plan_edits(tmp_path, [Edit(path, "", "#!/bin/sh\necho pwned\n", create=True)])
    outcome = result.outcomes[0]
    assert not result.ok
    assert outcome.error_kind == "unsafe_path" and outcome.path_rejected and outcome.safe_path is None
    assert result.new_contents == {} and result.original_contents == {}
    with pytest.raises(EditWriteError):
        write_edits(tmp_path, result)
    assert not (tmp_path / ".git" / "hooks" / "pre-commit").exists()


def test_write_edits_rechecks_alias_paths(tmp_path):
    _git_init(tmp_path)
    forged = EditBatchResult(ok=True, original_contents={".git./hooks/pre-commit": None},
                             new_contents={".git./hooks/pre-commit": "echo pwned\n"})
    with pytest.raises(EditWriteError):
        write_edits(tmp_path, forged)
    assert not (tmp_path / ".git" / "hooks" / "pre-commit").exists()


def test_symlink_into_git_dir_is_rejected_after_resolving(tmp_path):
    _git_init(tmp_path)
    try:
        (tmp_path / "gitlink").symlink_to(tmp_path / ".git", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    result = plan_edits(tmp_path, [Edit("gitlink/hooks/pre-commit", "", "echo pwned\n", create=True)])
    assert result.outcomes[0].error_kind == "unsafe_path"
    assert ".git" in result.outcomes[0].error


def test_symlinked_path_is_canonicalised_to_its_target(tmp_path):
    _write(tmp_path, "real/mod.py", "X = 1\n")
    try:
        (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    result = plan_edits(tmp_path, [Edit("alias/mod.py", "X = 1\n", "X = 2\n")])
    assert result.ok, result.errors
    assert list(result.new_contents) == ["real/mod.py"]       # what the scope guard sees
    assert result.outcomes[0].safe_path == "real/mod.py"


def test_case_insensitive_filesystem_uses_on_disk_name(tmp_path):
    _write(tmp_path, "pkg/mod.py", "X = 1\n")
    if not (tmp_path / "PKG" / "MOD.PY").exists():
        pytest.skip("case-sensitive file system")
    result = plan_edits(tmp_path, [Edit("PKG/MOD.PY", "X = 1\n", "X = 2\n")])
    assert result.ok and list(result.new_contents) == ["pkg/mod.py"]


def test_ordinary_paths_still_work_and_are_safe(tmp_path):
    _write(tmp_path, "src/app.py", "A = 1\n")
    _write(tmp_path, "tests/test_app.py", "def test_a():\n    assert True\n")
    result = plan_edits(tmp_path, [
        Edit("src/app.py", "A = 1\n", "A = 2\n"),
        Edit("tests/test_app.py", "    assert True\n", "    assert 1\n"),
        Edit(".github_notes/new.md", "", "x\n", create=True),   # merely starts with .git
        Edit("docs/v1.2/guide.md", "", "y\n", create=True),
    ])
    assert result.ok, result.errors
    assert all(o.safe_path == o.edit.file and not o.path_rejected for o in result.outcomes)
    assert resolve_repo_path(tmp_path, "./src\\app.py") == "src/app.py"
    assert resolve_repo_path(tmp_path, "../etc/passwd") is None
    assert result.to_dict()["edits"][0]["path_rejected"] is False


# ---------------------------------------------------------------------------
# Phantom trailing blank line in SEARCH
# ---------------------------------------------------------------------------

def test_trailing_blank_line_in_search_at_eof_still_matches(tmp_path):
    _write(tmp_path, "m.py", "def f():\n    return 1\n")
    # The model copied an empty "last line" that the file does not have (window > file).
    result = plan_edits(tmp_path, [Edit("m.py", "def f():\n    return 1\n\n", "def f():\n    return 2\n\n")])
    assert result.ok, result.errors
    assert result.new_contents["m.py"] == "def f():\n    return 2\n"   # no blank line added


def test_trailing_blank_line_stripped_only_from_search_when_replace_has_none(tmp_path):
    _write(tmp_path, "m.py", "A = 1\nB = 2\n")
    result = plan_edits(tmp_path, [Edit("m.py", "A = 1\nB = 2\n\n\n", "A = 1\nB = 3\n")])
    assert result.ok, result.errors
    assert result.new_contents["m.py"] == "A = 1\nB = 3\n"


def test_stripped_retry_keeps_ambiguity_checks(tmp_path):
    _write(tmp_path, "m.py", "x = 1\ny = 2\nx = 1\n")
    result = plan_edits(tmp_path, [Edit("m.py", "x = 1\n\n", "x = 3\n\n")])
    assert result.outcomes[0].error_kind == "ambiguous_match"


# ---------------------------------------------------------------------------
# CANNOT_FIX only when it really is the marker
# ---------------------------------------------------------------------------

def test_prose_starting_with_cannot_fix_does_not_stop_a_valid_edit():
    text = _dedent("""
        INTENT:
        Cannot fix the caller directly, so patch the helper instead.

        FILE: util.py
        <<<<<<< SEARCH
        x = 1
        =======
        x = 2
        >>>>>>> REPLACE
    """)
    parsed = parse_edit_response(text)
    assert parsed.cannot_fix is None and len(parsed.edits) == 1
    assert parsed.format_errors == []
    assert "Cannot fix the caller directly" in parsed.intent


def test_cannot_fix_marker_before_file_blocks_still_counts():
    text = "CANNOT_FIX: needs a product decision\n\nFILE: util.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"
    parsed = parse_edit_response(text)
    assert parsed.cannot_fix == "needs a product decision"
    assert any("both CANNOT_FIX" in e for e in parsed.format_errors)


def test_cannot_fix_after_file_blocks_is_ignored_and_prose_counts_without_edits():
    after = "FILE: util.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\nCANNOT_FIX: unsure\n"
    assert parse_edit_response(after).cannot_fix is None
    alone = parse_edit_response("Cannot fix: the bug is in a vendored binary.")
    assert alone.cannot_fix == "the bug is in a vendored binary." and alone.edits == []
