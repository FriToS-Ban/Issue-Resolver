"""
editing.py — Robust SEARCH/REPLACE editing engine for the repository-aware pipeline.

Pipeline position
-----------------
    LLM answer ──parse_edit_response()──▶ ParsedResponse(intent, edits, cannot_fix, ...)
               ──plan_edits(root, edits)──▶ EditBatchResult   (pure: reads files, edits in memory)
               ──write_edits(root, result)──▶ files on disk  (only when result.ok)

``plan_edits`` never touches the disk, so a batch is atomic: either every edit applies
and every touched file still parses, or nothing is written and the caller gets precise,
model-readable errors (``EditBatchResult.errors`` / ``to_dict()``) to feed back to the LLM.

Why this replaces the legacy ``patch_generator.py`` for the new pipeline
----------------------------------------------------------------------
The legacy generator silently used the first match, had a loose whole-line fuzzy matcher
and could not tell the model *why* a block failed. This engine:

* detects ambiguity (SEARCH matching several places) and reports every start line;
* supports ``(occurrences: N)`` to replace exactly N identical occurrences;
* only falls back to whitespace-tolerant matching when the tolerant match is unique,
  and re-indents the replacement when the match was found at a different indentation;
* recognises edits that are already present (re-applied / partially applied attempts);
* on a miss, reports the closest region of the *current* file with line numbers and the
  first differing line, so the model can correct its SEARCH text;
* validates Python / JSON / TOML syntax of every changed file before anything is written.

Matching rules (per edit, applied in order to the evolving in-memory content)
-----------------------------------------------------------------------------
1. CRLF is normalised to LF (and a UTF-8 BOM is set aside) for matching only.
2. Exact match. A SEARCH that ends with a newline is line-based, so a match must start at
   the beginning of a line (``"x = 1\\n"`` never matches inside ``"max = 1\\n"``). An
   occurrence that already sits inside an identical copy of the REPLACE text (REPLACE
   contains SEARCH and was applied before) is not counted. ``n == expected`` → replace;
   ``n > expected`` (or overlapping matches) → ``ambiguous_match``; ``0 < n < expected``
   → ``occurrence_mismatch``.
3. ``n == 0`` → line windows ignoring trailing whitespace (``trailing_ws``), then ignoring a
   common leading indentation while keeping relative indentation (``indent``). Accepted only
   when the number of windows equals ``expected_occurrences`` (exactly one by default).
4. Still not found and SEARCH ends with blank lines → steps 2-3 are retried with those
   trailing blank lines removed from SEARCH (and as many from REPLACE): models copy a phantom
   empty last line from rendered file blocks, which can never match at the end of a file.
5. Still not found → ``already_applied`` if the REPLACE text is present (exactly or
   whitespace-tolerantly) *where the SEARCH text best matches* the file; REPLACE appearing
   only somewhere else is no evidence. Otherwise ``missing_search`` with ``best_match``.
6. Empty SEARCH creates a file that does not exist yet (``create`` / ``(new)``); on an
   existing file it is ``empty_search`` (``file_exists`` when the edit asked to create).
   SEARCH == REPLACE is ``no_change``.
7. After all edits every changed file goes through ``validate_syntax``; a syntax error that
   the edits introduced (the original file parsed fine) fails the batch (``syntax_error``).

Line endings, trailing newlines and encodings
---------------------------------------------
Files are read as UTF-8 with ``surrogateescape`` so undecodable bytes round-trip unchanged.
``EditBatchResult.new_contents`` is LF-normalised but already carries the original
trailing-newline state and BOM; ``write_edits`` converts back to the file's original line
ending style (CRLF when CRLF dominates; mixed files are normalised to that style).
``unified_diff`` is always LF and ``git apply``-compatible.
"""

from __future__ import annotations

import difflib
import json
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path, PurePosixPath
from typing import Any

from fix_issue.scope_guard import unsafe_component_reason

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

APPLIED = "applied"
ALREADY_APPLIED = "already_applied"
FAILED = "failed"
STATUSES = (APPLIED, ALREADY_APPLIED, FAILED)

MATCH_MODES = ("exact", "trailing_ws", "indent", "new_file")

ERROR_KINDS = (
    "missing_search",
    "ambiguous_match",
    "occurrence_mismatch",
    "file_not_found",
    "file_exists",
    "empty_search",
    "unsafe_path",
    "syntax_error",
    "no_change",
)

_OK_STATUSES = (APPLIED, ALREADY_APPLIED)
_MAX_LISTED_LINES = 50          # cap for "matches at lines ..." lists
_MAX_SHOWN_REGION_LINES = 40    # cap for the closest-region excerpt in error messages


EDIT_FORMAT_INSTRUCTIONS = """\
## Output format

Reply in exactly this format and nothing else:

INTENT:
<1-3 sentences: what you will change and why>

FILE: path/to/file.py
<<<<<<< SEARCH
<exact lines currently in the file>
=======
<the lines that replace them>
>>>>>>> REPLACE

Rules:
- FILE is the path relative to the repository root, written with forward slashes.
- SEARCH must reproduce the current file content exactly: the same characters, indentation, \
blank lines and comments. Never abbreviate with "..." or placeholders.
- SEARCH must match exactly one place in the file. Include just enough unchanged surrounding \
lines (usually 2-5) to make it unique, and no more.
- To change every one of N identical occurrences at once, write the header as \
`FILE: path/to/file.py (occurrences: N)`. The edit fails if the file does not contain exactly N.
- You may write several blocks, for one file or for several files. Blocks for the same file \
are applied in order, each to the result of the previous one, so they must not overlap.
- To delete lines, leave the REPLACE section empty. To insert lines, SEARCH for the adjacent \
existing lines and repeat them in REPLACE together with the new lines.
- To create a new file, write `FILE: path/to/new_file.py (new)`, leave the SEARCH section \
empty and put the complete file content in the REPLACE section.
- Make the smallest change that fixes the issue. Do not reformat or rewrite unrelated code.
- Do not write explanations outside INTENT.

If you cannot produce a correct fix, reply with this single line instead:
CANNOT_FIX: <one-sentence reason>
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Edit:
    file: str                    # repo-relative POSIX path
    search: str                  # exact existing text ("" only when creating a new file)
    replace: str
    expected_occurrences: int = 1
    create: bool = False         # new file


@dataclass
class EditOutcome:
    edit: Edit
    status: str                  # "applied" | "already_applied" | "failed"
    matched: str = ""            # "exact" | "trailing_ws" | "indent" | "new_file" | ""
    occurrences_found: int = 0
    line: int | None = None      # 1-based line where the (first) match starts
    error: str | None = None
    error_kind: str | None = None
    best_match: dict | None = None
    # True when the path itself was refused (error_kind "unsafe_path"): ``edit.file`` is then the
    # model's raw, untrusted string and must never be opened (it may point outside the repo).
    path_rejected: bool = False

    @property
    def ok(self) -> bool:
        return self.status in _OK_STATUSES

    @property
    def safe_path(self) -> str | None:
        """Canonical repo-relative path that is safe to read, or None when the path was rejected."""
        return None if self.path_rejected else self.edit.file

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.edit.file,
            "path_rejected": self.path_rejected,
            "create": self.edit.create,
            "expected_occurrences": self.edit.expected_occurrences,
            "search_lines": len(_split_lines(self.edit.search)),
            "replace_lines": len(_split_lines(self.edit.replace)),
            "status": self.status,
            "matched": self.matched,
            "occurrences_found": self.occurrences_found,
            "line": self.line,
            "error_kind": self.error_kind,
            "error": self.error,
            "best_match": dict(self.best_match) if self.best_match else None,
        }


@dataclass
class EditBatchResult:
    ok: bool
    outcomes: list[EditOutcome] = field(default_factory=list)
    original_contents: dict[str, str | None] = field(default_factory=dict)
    new_contents: dict[str, str] = field(default_factory=dict)
    validation: dict[str, str | None] = field(default_factory=dict)
    diff: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary; never contains full file contents."""
        added, removed = _diff_stats(self.diff)
        return {
            "ok": self.ok,
            "files_changed": list(self.new_contents),
            "new_files": [p for p in self.new_contents if self.original_contents.get(p) is None],
            "lines_added": added,
            "lines_removed": removed,
            "validation": dict(self.validation),
            "errors": list(self.errors),
            "edits": [o.to_dict() for o in self.outcomes],
        }


@dataclass
class ParsedResponse:
    intent: str = ""
    edits: list[Edit] = field(default_factory=list)
    cannot_fix: str | None = None
    format_errors: list[str] = field(default_factory=list)
    raw: str = ""


class EditWriteError(RuntimeError):
    """Raised by ``write_edits`` when it refuses to write (batch not ok, stale file, bad path)."""


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------

def _lf(text: str) -> str:
    return text.replace("\r\n", "\n")


def _split_lines(text: str) -> list[str]:
    """Split on "\\n" only (not str.splitlines' extra separators); a final "\\n" ends the last line."""
    if not text:
        return []
    if text.endswith("\n"):
        text = text[:-1]
    return text.split("\n")


def _join_lines(lines: list[str]) -> str:
    return "".join(line + "\n" for line in lines)


def _keepends(text: str) -> list[str]:
    """Lines with their "\\n" kept; the last line has none if the text lacks a final newline."""
    if not text:
        return []
    parts = text.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def _ensure_eol(text: str) -> str:
    return text if not text or text.endswith("\n") else text + "\n"


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _format_line_list(lines: list[int]) -> str:
    shown = ", ".join(str(n) for n in lines[:_MAX_LISTED_LINES])
    if len(lines) > _MAX_LISTED_LINES:
        shown += f", ... ({len(lines) - _MAX_LISTED_LINES} more)"
    return shown


def _uses_crlf(raw: str) -> bool:
    crlf = raw.count("\r\n")
    return crlf > 0 and crlf >= raw.count("\n") - crlf


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="surrogateescape")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
_SEARCH_RE = re.compile(r"^\s*<{5,9}\s*SEARCH\s*$", re.IGNORECASE)
_DIVIDER_RE = re.compile(r"^\s*={7}\s*$")
_REPLACE_RE = re.compile(r"^\s*>{5,9}\s*REPLACE\s*$", re.IGNORECASE)
_DECOR = r"(?:[#>*`\-]+\s*)*"   # markdown decoration allowed before a keyword: **, ##, -, >, `
_FILE_RE = re.compile(rf"^\s*{_DECOR}FILE\s*[*`]*\s*:\s*(?P<rest>.*?)\s*$", re.IGNORECASE)
_INTENT_RE = re.compile(rf"^\s*{_DECOR}INTENT\s*[*`]*\s*:\s*[*`]*\s*(?P<rest>.*?)\s*$", re.IGNORECASE)
_CANNOT_RE = re.compile(rf"^\s*{_DECOR}CANNOT[_ ]FIX\b\s*[*`]*\s*:?\s*(?P<rest>.*?)\s*$", re.IGNORECASE)
# The marker exactly as the format specifies it ("CANNOT_FIX:" starting its own line), as
# opposed to prose such as "Cannot fix the caller directly, so patch the helper".
_CANNOT_MARKER_RE = re.compile(rf"^\s*{_DECOR}CANNOT_FIX\s*[*`]*\s*:", re.IGNORECASE)
_SUFFIX_RE = re.compile(r"\(\s*([^()]*?)\s*\)\s*$")
_OCC_RE = re.compile(r"(?:occurrences?|occurences?|count|times)\s*[:=]?\s*(\d+)")


def _clean_path(path: str) -> str:
    """Normalise a model-written path: quotes/backticks, backslashes, ``./`` prefixes, ``//``."""
    p = path.strip().strip("*`'\"").strip()
    p = p.replace("\\", "/")
    p = re.sub(r"/{2,}", "/", p)
    while p.startswith("./"):
        p = p[2:]
    return p


def _parse_file_spec(rest: str) -> tuple[str, int | None, bool]:
    """Split ``path (occurrences: N) (new)`` into (path, occurrences, create)."""
    spec = rest
    occurrences: int | None = None
    create = False
    for _ in range(4):  # at most a couple of trailing "(...)" tags
        spec = spec.strip().strip("*`'\"").strip()
        m = _SUFFIX_RE.search(spec)
        if not m:
            break
        tag = m.group(1).lower()
        occ = _OCC_RE.fullmatch(tag)
        if tag in ("new", "new file", "create", "created"):
            create = True
        elif occ:
            occurrences = int(occ.group(1))
        else:
            break  # unknown parenthetical: leave it as part of the path
        spec = spec[: m.start()]
    return _clean_path(spec), occurrences, create


def _strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text)
    close = None
    for close in _THINK_CLOSE_RE.finditer(text):
        pass
    return text[close.end():] if close else text  # orphan "</think>": drop what precedes it


def parse_edit_response(text: str) -> ParsedResponse:
    """
    Parse an LLM answer written in the EDIT_FORMAT_INSTRUCTIONS format.

    Tolerates markdown fences around the whole answer or around individual blocks,
    ``**FILE:**`` / backticked / backslashed / ``./``-prefixed paths, CRLF, extra spaces in
    marker lines and ``<think>…</think>`` preambles. Blocks with problems (no FILE line,
    unterminated) are dropped and described in ``format_errors``; they are never guessed.
    Paths are only cleaned here — safety checks happen in ``plan_edits``.

    A CANNOT_FIX line ends the run, so when the answer also contains edits it only counts if
    it is the literal ``CANNOT_FIX:`` marker on its own line before any FILE line; a prose line
    that merely starts with "cannot fix" (e.g. inside INTENT) is kept as text.
    """
    raw = text or ""
    lines = _strip_thinking(_lf(raw)).split("\n")

    edits: list[Edit] = []
    errors: list[str] = []
    intent_parts: list[str] = []
    intent_seen = intent_open = False
    cannot_fix: str | None = None
    cannot_candidates: list[tuple[str, bool]] = []   # (reason, is the marker before any FILE line)
    saw_file = False

    cur_file: str | None = None
    cur_occ: int | None = None
    cur_create = False
    cur_file_line = 0
    cur_file_used = True

    state = "outside"               # outside | search | replace
    block_start = 0
    search_buf: list[str] = []
    replace_buf: list[str] = []

    def set_file(rest: str, lineno: int) -> None:
        nonlocal cur_file, cur_occ, cur_create, cur_file_line, cur_file_used
        path, occ, create = _parse_file_spec(rest)
        cur_file, cur_occ, cur_create, cur_file_line, cur_file_used = path or None, occ, create, lineno, False
        if not path:
            errors.append(f"line {lineno}: FILE line has no path.")
        if occ is not None and occ < 1:
            errors.append(f"line {lineno}: occurrences must be at least 1 (got {occ}).")

    def warn_unused_file() -> None:
        if cur_file and not cur_file_used:
            errors.append(
                f"line {cur_file_line}: 'FILE: {cur_file}' is not followed by a "
                "<<<<<<< SEARCH / ======= / >>>>>>> REPLACE block."
            )

    def recover_file(buf: list[str], lineno: int) -> None:
        # A block that never terminated may have swallowed the next block's FILE line.
        for line in reversed(buf):
            m = _FILE_RE.match(line)
            if m:
                set_file(m.group("rest"), lineno)
                return

    def emit() -> None:
        nonlocal cur_file_used
        if cur_file is None:
            errors.append(
                f"line {block_start}: SEARCH/REPLACE block has no preceding 'FILE: <path>' line; block ignored."
            )
            return
        search = "" if not any(s.strip() for s in search_buf) else _join_lines(search_buf)
        edits.append(Edit(
            file=cur_file,
            search=search,
            replace=_join_lines(replace_buf),
            expected_occurrences=1 if cur_occ is None else cur_occ,
            create=cur_create,
        ))
        cur_file_used = True

    for lineno, line in enumerate(lines, 1):
        if state == "outside":
            m = _FILE_RE.match(line)
            if m:
                warn_unused_file()
                set_file(m.group("rest"), lineno)
                intent_open = False
                saw_file = True
            elif _SEARCH_RE.match(line):
                state, block_start, search_buf = "search", lineno, []
                intent_open = False
            elif _DIVIDER_RE.match(line) or _REPLACE_RE.match(line):
                errors.append(f"line {lineno}: stray '{line.strip()}' marker outside a SEARCH/REPLACE block.")
            elif (m := _CANNOT_RE.match(line)) is not None:
                marker = bool(_CANNOT_MARKER_RE.match(line))
                cannot_candidates.append((m.group("rest").strip(" *`") or "(no reason given)",
                                          marker and not saw_file))
                if marker:
                    intent_open = False
                elif intent_open:
                    intent_parts.append(line.strip())   # prose; becomes CANNOT_FIX only without edits
            elif not intent_seen and (m := _INTENT_RE.match(line)) is not None:
                intent_seen = intent_open = True
                if m.group("rest"):
                    intent_parts.append(m.group("rest"))
            elif intent_open and not _FENCE_RE.match(line) and line.strip():
                intent_parts.append(line.strip())
            # anything else outside a block (prose, fences, blank lines) is ignored
        elif state == "search":
            if _DIVIDER_RE.match(line):
                state, replace_buf = "replace", []
            elif _REPLACE_RE.match(line):
                errors.append(
                    f"line {block_start}: block reached '>>>>>>> REPLACE' without a '=======' divider; block ignored."
                )
                state = "outside"
            elif _SEARCH_RE.match(line):
                errors.append(f"line {block_start}: SEARCH block is not terminated; block ignored.")
                recover_file(search_buf, lineno)
                block_start, search_buf = lineno, []
            else:
                search_buf.append(line)
        else:  # replace
            if _REPLACE_RE.match(line):
                emit()
                state = "outside"
            elif _SEARCH_RE.match(line):
                errors.append(
                    f"line {block_start}: SEARCH/REPLACE block is missing its '>>>>>>> REPLACE' line; block ignored."
                )
                recover_file(replace_buf, lineno)
                state, block_start, search_buf = "search", lineno, []
            else:
                replace_buf.append(line)

    if state != "outside":
        missing = "'======='" if state == "search" else "'>>>>>>> REPLACE'"
        errors.append(f"line {block_start}: SEARCH/REPLACE block is not terminated (missing {missing}); block ignored.")
    else:
        warn_unused_file()

    if edits:
        cannot_fix = next((reason for reason, marker in cannot_candidates if marker), None)
    elif cannot_candidates:
        cannot_fix = cannot_candidates[0][0]

    if cannot_fix is not None and edits:
        errors.append("Response contains both CANNOT_FIX and SEARCH/REPLACE blocks.")
    if not edits and cannot_fix is None:
        errors.append("No complete SEARCH/REPLACE blocks found in the response.")

    return ParsedResponse(
        intent=" ".join(intent_parts).strip(),
        edits=edits,
        cannot_fix=cannot_fix,
        format_errors=errors,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def _check_path(root: Path, path: str) -> tuple[str | None, str | None]:
    """
    Return ``(canonical_relative_path, None)`` or ``(None, reason)`` for unsafe paths.

    The returned path is the target's real location relative to the resolved root (symlinks,
    letter case and Windows short names resolved), so the file that is read, written and
    scope-checked is the file that actually changes. Components that could alias another
    name on Windows/macOS (``.git.``, ``.git ``, ``GIT~1``, ``x:stream``, invisible format
    characters — see ``scope_guard.unsafe_component_reason``) are rejected before resolving,
    and a target whose canonical path enters ``.git`` is rejected after resolving.
    """
    p = _clean_path(path or "")
    if not p:
        return None, "the path is empty"
    if any(ord(c) < 32 for c in p):
        return None, "the path contains control characters"
    if p.startswith("/") or _DRIVE_RE.match(p) or p == "~" or p.startswith("~/"):
        return None, "absolute paths are not allowed; use a path relative to the repository root"
    parts = [part for part in p.split("/") if part not in ("", ".")]
    if not parts:
        return None, "the path points at the repository root"
    if ".." in parts:
        return None, "'..' components are not allowed (the path must stay inside the repository)"
    for part in parts:
        reason = unsafe_component_reason(part)
        if reason:
            return None, reason
    rel = "/".join(parts)
    try:
        root_resolved = Path(root).resolve()
        target = (root_resolved / rel).resolve()
    except (OSError, RuntimeError) as exc:
        return None, f"the path cannot be resolved ({exc})"
    if target == root_resolved or not target.is_relative_to(root_resolved):
        return None, "the path resolves outside the repository (symlink?)"
    canonical = target.relative_to(root_resolved).parts
    if not canonical:
        return None, "the path points at the repository root"
    for part in canonical:
        reason = unsafe_component_reason(part)
        if reason:
            return None, f"the path resolves to '{'/'.join(canonical)}': {reason}"
    return "/".join(canonical), None


def resolve_repo_path(root: Path, path: str) -> str | None:
    """Canonical repo-relative path for ``path`` when it is safe to read or write, else None.

    Use it before reading any model- or output-supplied path from disk (e.g. in recovery)."""
    return _check_path(root, path)[0]


def _suggest_paths(root: Path, rel: str) -> list[str]:
    """Close file-name matches in the nearest existing parent directory (for file_not_found)."""
    try:
        root_resolved = Path(root).resolve()
        parent = (root_resolved / rel).parent
        while parent != root_resolved and not parent.is_dir():
            parent = parent.parent
        names = sorted(p.name for p in parent.iterdir() if p.name != ".git")[:5000]
        base = parent.relative_to(root_resolved).as_posix()
    except (OSError, ValueError):
        return []
    close = difflib.get_close_matches(PurePosixPath(rel).name, names, n=3, cutoff=0.6)
    return [name if base == "." else f"{base}/{name}" for name in close]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass
class _FileState:
    """Working copy of one file during plan_edits."""
    path: str
    original: str | None          # raw text as read (None = did not exist)
    content: str = ""             # LF, no BOM, ends with "\n" unless empty
    bom: str = ""
    trailing_newline: bool = True
    created: bool = False

    @property
    def exists(self) -> bool:
        return self.original is not None or self.created

    def final_text(self) -> str:
        text = self.content
        if not self.trailing_newline and text.endswith("\n"):
            text = text[:-1]
        return self.bom + text


def _load_state(root: Path, rel: str) -> _FileState:
    target = Path(root) / rel
    if target.is_dir():
        raise IsADirectoryError(f"'{rel}' is a directory")
    if not target.is_file():
        return _FileState(rel, None)
    raw = _read_text(target)
    text = _lf(raw)
    bom = ""
    if text.startswith("﻿"):
        bom, text = "﻿", text[1:]
    return _FileState(rel, raw, _ensure_eol(text), bom, trailing_newline=text == "" or text.endswith("\n"))


def _find_exact(content: str, search: str) -> list[int]:
    """Offsets of every (possibly overlapping) occurrence; line-based SEARCH must start a line."""
    anchored = search.endswith("\n")
    found: list[int] = []
    pos = content.find(search)
    while pos != -1:
        if not anchored or pos == 0 or content[pos - 1] == "\n":
            found.append(pos)
        pos = content.find(search, pos + 1)
    return found


def _is_embedded(content: str, pos: int, search: str, replace: str) -> bool:
    """True if the SEARCH occurrence at ``pos`` lies inside an already-applied copy of REPLACE."""
    k = replace.find(search)
    while k != -1:
        start = pos - k
        if start >= 0 and content.startswith(replace, start):
            return True
        k = replace.find(search, k + 1)
    return False


def _overlapping(starts: list[int], length: int) -> bool:
    return any(b < a + length for a, b in zip(starts, starts[1:]))


def _dedent(lines: list[str]) -> list[str]:
    """Remove the common leading indentation (keeping relative indentation) and trailing spaces."""
    nonblank = [line for line in lines if line.strip()]
    if not nonblank:
        return ["" for _ in lines]
    cut = min(len(_leading_ws(line)) for line in nonblank)
    return [line[cut:].rstrip() if line.strip() else "" for line in lines]


def _find_windows(c_lines: list[str], s_lines: list[str], mode: str) -> list[int]:
    """0-based start indexes of line windows equal to ``s_lines`` under a tolerant ``mode``."""
    n = len(s_lines)
    if n == 0 or n > len(c_lines):
        return []
    if mode == "trailing_ws":
        key = [line.rstrip() for line in s_lines]
        norm = [line.rstrip() for line in c_lines]
        return [i for i in range(len(c_lines) - n + 1) if norm[i] == key[0] and norm[i:i + n] == key]
    # mode == "indent": equal after stripping (cheap filter), then equal relative indentation
    key = _dedent(s_lines)
    s_stripped = [line.strip() for line in s_lines]
    stripped = [line.strip() for line in c_lines]
    return [
        i for i in range(len(c_lines) - n + 1)
        if stripped[i] == s_stripped[0]
        and stripped[i:i + n] == s_stripped
        and _dedent(c_lines[i:i + n]) == key
    ]


def _reindent(lines: list[str], s_ind: str, w_ind: str) -> list[str]:
    """Shift ``lines`` from the SEARCH anchor indentation ``s_ind`` to the matched ``w_ind``."""
    if s_ind == w_ind:
        return list(lines)
    delta = len(w_ind) - len(s_ind)
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append(line)
        elif line.startswith(s_ind):
            out.append(w_ind + line[len(s_ind):])
        elif delta > 0:  # shallower than the anchor: shift right by the width delta
            out.append(w_ind[:delta] + line)
        else:            # shift left, never removing more than the line's own indentation
            out.append(line[min(-delta, len(_leading_ws(line))):])
    return out


def _replace_locations(content: str, replace: str) -> list[tuple[str, int, int]]:
    """Every place the REPLACE text already exists: (mode, first line, last line), 1-based."""
    if not replace.strip():
        return []
    n = max(1, len(_split_lines(replace)))
    positions = _find_exact(content, replace)
    if positions:
        return [("exact", _line_of(content, p), _line_of(content, p) + n - 1) for p in positions]
    c_lines, r_lines = _split_lines(content), _split_lines(replace)
    for mode in ("trailing_ws", "indent"):
        hits = _find_windows(c_lines, r_lines, mode)
        if hits:
            return [(mode, h + 1, h + n) for h in hits]
    return []


def _find_replace_present(content: str, search: str, replace: str) -> tuple[str, int] | None:
    """
    (mode, 1-based line) if the edit looks already applied: the REPLACE text exists *where the
    SEARCH text best matches* the current file. REPLACE merely appearing somewhere (say, in
    another function) is not evidence that this edit was applied, so that is a missing SEARCH.
    """
    locations = _replace_locations(content, replace)
    if not locations:
        return None
    best = locate_best_match(content, search)
    if best is None:
        return None
    for mode, first, last in locations:
        if first <= best["end_line"] and last >= best["start_line"]:
            return mode, first
    return None


def _strip_trailing_blank_lines(edit: Edit) -> Edit | None:
    """
    The edit with trailing blank lines removed from SEARCH (and as many from REPLACE), or None.

    Models often copy a phantom empty last line from a rendered file block into SEARCH; at the
    end of a file that line does not exist, so the SEARCH can never match as written.
    """
    s_lines = _split_lines(edit.search)
    k = 0
    while k < len(s_lines) and not s_lines[len(s_lines) - 1 - k].strip():
        k += 1
    if k == 0 or k == len(s_lines):
        return None
    r_lines = _split_lines(edit.replace)
    j = 0
    while j < min(k, len(r_lines)) and not r_lines[len(r_lines) - 1 - j].strip():
        j += 1
    search = _join_lines(s_lines[:len(s_lines) - k])
    replace = _join_lines(r_lines[:len(r_lines) - j])
    if search == replace:
        return None
    return dc_replace(edit, search=search, replace=replace)


_TOKEN_RE = re.compile(r"\w{2,}")


def _candidate_starts(key: list[str], s_key: list[str]) -> list[int] | None:
    """Window starts worth scoring: anchored on identical / similar lines shared with SEARCH."""
    total = len(key)
    index: dict[str, list[int]] = defaultdict(list)
    for i, line in enumerate(key):
        if len(line) >= 3:
            index[line].append(i)
    votes: Counter[int] = Counter()
    for j, line in enumerate(s_key):
        hits = index.get(line, []) if len(line) >= 3 else []
        if 0 < len(hits) <= 50:  # very common lines (e.g. "return None") carry no signal
            for i in hits:
                votes[i - j] += 1
    if not votes:
        # No identical line: vote through rare identifier/number tokens shared with SEARCH
        # lines (linear in file size, unlike pairwise difflib scoring of every line).
        token_index: dict[str, set[int]] = defaultdict(set)
        for i, line in enumerate(key):
            for token in _TOKEN_RE.findall(line):
                token_index[token].add(i)
        for j, line in enumerate(s_key):
            for token in set(_TOKEN_RE.findall(line)):
                hits = token_index.get(token, ())
                if 0 < len(hits) <= 50:
                    for i in hits:
                        votes[i - j] += 1
    if not votes:
        return list(range(total)) if total <= 1000 else None
    starts = {st + d for st, _ in votes.most_common(25) for d in range(-2, 3)}
    return sorted(s for s in starts if 0 <= s < total)


def locate_best_match(content: str, search: str) -> dict | None:
    """
    Closest region of ``content`` to ``search``: windows of len(search lines) ± 2 lines,
    scored with difflib on whitespace-stripped lines. Returns
    ``{"start_line", "end_line", "similarity", "text"}`` (1-based, inclusive; similarity 0..1)
    or None when there is nothing to compare (empty file / blank SEARCH, or a file over 1000
    lines sharing no line or rare token with SEARCH). Candidate windows are anchored on lines
    or rare tokens shared with SEARCH, so the cost stays roughly linear in file size.
    """
    c_lines = _split_lines(_lf(content or "").lstrip("﻿"))
    s_lines = _split_lines(_lf(search or ""))
    if not c_lines or not any(line.strip() for line in s_lines):
        return None
    key = [line.strip() for line in c_lines]
    s_key = [line.strip() for line in s_lines]
    starts = _candidate_starts(key, s_key)
    if starts is None:
        return None
    n = len(s_lines)
    sizes = sorted({min(max(1, n + d), len(c_lines)) for d in range(-2, 3)})

    matcher = difflib.SequenceMatcher(None, autojunk=False)
    matcher.set_seq2("\n".join(s_key))
    best_ratio, best_start, best_size = -1.0, 0, sizes[0]
    for start in starts:
        for size in sizes:
            if start + size > len(c_lines):
                continue
            matcher.set_seq1("\n".join(key[start:start + size]))
            if matcher.real_quick_ratio() <= best_ratio or matcher.quick_ratio() <= best_ratio:
                continue
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best_ratio, best_start, best_size = ratio, start, size
    if best_ratio < 0:
        return None
    return {
        "start_line": best_start + 1,
        "end_line": best_start + best_size,
        "similarity": round(best_ratio, 3),
        "text": "\n".join(c_lines[best_start:best_start + best_size]),
    }


def _first_difference(search: str, best: dict) -> str:
    """One-line hint naming the first SEARCH line that differs from the closest region."""
    s_lines = _split_lines(search)
    r_lines = best["text"].split("\n")
    matcher = difflib.SequenceMatcher(None, s_lines, r_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        file_line = best["start_line"] + j1
        if tag == "replace":
            return f"First difference: SEARCH line {i1 + 1} is {s_lines[i1]!r} but file line {file_line} is {r_lines[j1]!r}."
        if tag == "delete":
            return f"First difference: SEARCH line {i1 + 1} {s_lines[i1]!r} does not exist in the file near line {file_line}."
        if tag == "insert":
            return f"First difference: file line {file_line} {r_lines[j1]!r} is missing from SEARCH."
    return ""


def _missing_message(label: str, search: str, best: dict | None) -> str:
    head = (
        f"{label}: SEARCH text was not found in the current file "
        "(exact, trailing-whitespace-insensitive and indentation-insensitive matching all failed)."
    )
    if not best:
        return head + " No similar region exists; check the file path and copy SEARCH lines from the current file."
    region = best["text"].split("\n")
    shown = "\n".join(
        f"{best['start_line'] + i:>5} | {line}" for i, line in enumerate(region[:_MAX_SHOWN_REGION_LINES])
    )
    if len(region) > _MAX_SHOWN_REGION_LINES:
        shown += "\n      | ..."
    parts = [
        head,
        f"Closest region is lines {best['start_line']}-{best['end_line']} "
        f"(similarity {best['similarity']:.2f}):",
        shown,
    ]
    hint = _first_difference(search, best)
    if hint:
        parts.append(hint)
    parts.append("Copy the SEARCH lines verbatim from the current file content.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _fail(edit: Edit, kind: str, message: str, **extra: Any) -> EditOutcome:
    return EditOutcome(edit=edit, status=FAILED, error=message, error_kind=kind, **extra)


def _match_and_apply(state: _FileState, edit: Edit, label: str) -> EditOutcome | None:
    """Exact, then whitespace-tolerant matching of ``edit`` against ``state`` (applied in place).

    Returns the outcome (applied, already applied inside an identical REPLACE copy, or an
    ambiguity / occurrence failure), or None when the SEARCH text was not found at all."""
    rel, search, replace, expected = state.path, edit.search, edit.replace, edit.expected_occurrences
    content = state.content

    # --- 1. exact match -------------------------------------------------------------
    positions = _find_exact(content, search)
    if positions:
        embedded = {p for p in positions if search in replace and _is_embedded(content, p, search, replace)}
        live = [p for p in positions if p not in embedded]
        if not live:
            return EditOutcome(edit=edit, status=ALREADY_APPLIED, matched="exact",
                               line=_line_of(content, min(embedded)))
        lines = [_line_of(content, p) for p in live]
        note = f" ({len(embedded)} further occurrence(s) already contain the REPLACE text)" if embedded else ""
        if len(live) > expected or _overlapping(live, len(search)):
            extra = f", or write 'FILE: {rel} (occurrences: {len(live)})' to replace all of them" \
                if not _overlapping(live, len(search)) else ""
            return _fail(edit, "ambiguous_match",
                         f"{label}: SEARCH text matches {len(live)} places (starting at lines "
                         f"{_format_line_list(lines)}) but {expected} expected{note}. Add more surrounding "
                         f"lines to SEARCH so it matches exactly one place{extra}.",
                         matched="exact", occurrences_found=len(live), line=lines[0])
        if len(live) < expected:
            return _fail(edit, "occurrence_mismatch",
                         f"{label}: expected {expected} occurrences of the SEARCH text but found {len(live)} "
                         f"(starting at lines {_format_line_list(lines)}){note}.",
                         matched="exact", occurrences_found=len(live), line=lines[0])
        pieces, last = [], 0
        for p in live:
            pieces += [content[last:p], replace]
            last = p + len(search)
        pieces.append(content[last:])
        state.content = _ensure_eol("".join(pieces))
        return EditOutcome(edit=edit, status=APPLIED, matched="exact",
                           occurrences_found=len(live), line=lines[0])

    # --- 2. whitespace-tolerant line windows --------------------------------------------
    s_lines = _split_lines(search)
    if any(line.strip() for line in s_lines):
        c_lines = _split_lines(content)
        for mode in ("trailing_ws", "indent"):
            hits = _find_windows(c_lines, s_lines, mode)
            if not hits:
                continue
            lines = [h + 1 for h in hits]
            what = "ignoring trailing whitespace" if mode == "trailing_ws" else "ignoring indentation"
            if len(hits) > expected or _overlapping(hits, len(s_lines)):
                return _fail(edit, "ambiguous_match",
                             f"{label}: SEARCH text matches {len(hits)} places when {what} (starting at lines "
                             f"{_format_line_list(lines)}) but {expected} expected. Copy SEARCH exactly from the "
                             "file and add surrounding lines so it matches exactly one place.",
                             matched=mode, occurrences_found=len(hits), line=lines[0])
            if len(hits) < expected:
                return _fail(edit, "occurrence_mismatch",
                             f"{label}: expected {expected} occurrences of the SEARCH text but found {len(hits)} "
                             f"when {what} (starting at lines {_format_line_list(lines)}).",
                             matched=mode, occurrences_found=len(hits), line=lines[0])
            r_lines = _split_lines(replace)
            anchor = next(i for i, line in enumerate(s_lines) if line.strip())
            s_ind = _leading_ws(s_lines[anchor])
            out = list(c_lines)
            for h in reversed(hits):
                new = _reindent(r_lines, s_ind, _leading_ws(c_lines[h + anchor])) if mode == "indent" else r_lines
                out[h:h + len(s_lines)] = new
            state.content = _join_lines(out)
            return EditOutcome(edit=edit, status=APPLIED, matched=mode,
                               occurrences_found=len(hits), line=lines[0])
    return None


def _plan_one(root: Path, states: dict[str, _FileState], edit: Edit, index: int) -> EditOutcome:
    rel, reason = _check_path(root, edit.file)
    if rel is None:
        return _fail(edit, "unsafe_path", f"Edit #{index} ({edit.file!r}): unsafe path: {reason}.",
                     path_rejected=True)

    edit = dc_replace(edit, file=rel, search=_lf(edit.search or ""), replace=_lf(edit.replace or ""))
    label = f"Edit #{index} ({rel})"
    search, replace, expected = edit.search, edit.replace, edit.expected_occurrences

    if expected < 1:
        return _fail(edit, "occurrence_mismatch", f"{label}: expected occurrences must be at least 1 (got {expected}).")

    state = states.get(rel)
    if state is None:
        try:
            state = _load_state(root, rel)
        except IsADirectoryError:
            return _fail(edit, "file_not_found", f"{label}: '{rel}' is a directory, not a file.")
        except OSError as exc:
            return _fail(edit, "file_not_found", f"{label}: cannot read '{rel}': {exc}.")
        states[rel] = state

    # --- empty SEARCH: create a new file (or fill an existing empty one) ---------------
    if search == "":
        if not state.exists:
            state.content, state.created, state.trailing_newline = _ensure_eol(replace), True, True
            return EditOutcome(edit=edit, status=APPLIED, matched="new_file", line=1)
        if edit.create:
            return _fail(edit, "file_exists",
                         f"{label}: cannot create '{rel}': the file already exists. "
                         "Edit it with a SEARCH/REPLACE block that copies the lines to change.")
        if state.content == "":
            state.content = _ensure_eol(replace)
            return EditOutcome(edit=edit, status=APPLIED, matched="exact", line=1)
        return _fail(edit, "empty_search",
                     f"{label}: SEARCH is empty but '{rel}' already exists. Put the exact lines to change "
                     "in SEARCH (to insert code, SEARCH for the neighbouring lines and repeat them in REPLACE).")

    if not state.exists:
        message = (f"{label}: file '{rel}' does not exist. To create it, write 'FILE: {rel} (new)' "
                   "with an empty SEARCH section.")
        suggestions = _suggest_paths(root, rel)
        if suggestions:
            message += " Did you mean: " + ", ".join(suggestions) + "?"
        return _fail(edit, "file_not_found", message)

    if search == replace:
        return _fail(edit, "no_change", f"{label}: SEARCH and REPLACE are identical, so the edit changes nothing.")

    outcome = _match_and_apply(state, edit, label)
    stripped = None
    if outcome is None:
        # A phantom trailing blank line copied from a rendered file block (see
        # _strip_trailing_blank_lines): retry without it, with the same matching rules.
        stripped = _strip_trailing_blank_lines(edit)
        if stripped is not None:
            outcome = _match_and_apply(state, stripped, label)
    if outcome is not None:
        return outcome
    content = state.content

    # --- 3. already applied (REPLACE present where SEARCH best matches)? -----------------
    for candidate in (edit, stripped):
        if candidate is None:
            continue
        present = _find_replace_present(content, candidate.search, candidate.replace)
        if present:
            return EditOutcome(edit=candidate, status=ALREADY_APPLIED, matched=present[0], line=present[1])

    # --- 4. genuinely missing -----------------------------------------------------------
    best = locate_best_match(content, search)
    return _fail(edit, "missing_search", _missing_message(label, search, best), best_match=best)


def plan_edits(root: Path, edits: list[Edit]) -> EditBatchResult:
    """
    Apply ``edits`` in order to in-memory copies of the files under ``root``.

    Pure: reads files but never writes. ``ok`` is True only if every edit is applied or
    already applied and every changed file passes ``validate_syntax`` (an empty edit list
    is not ok). Call ``write_edits`` with the result only when ``ok`` is True.
    """
    root = Path(root)
    states: dict[str, _FileState] = {}
    outcomes = [_plan_one(root, states, edit, i) for i, edit in enumerate(edits, 1)]

    original_contents = {path: s.original for path, s in states.items()}
    new_contents: dict[str, str] = {}
    for path, s in states.items():
        final = s.final_text()
        if (s.original is None and s.created) or (s.original is not None and final != _lf(s.original)):
            new_contents[path] = final

    validation: dict[str, str | None] = {}
    for path, text in new_contents.items():
        error = validate_syntax(path, text)
        original = states[path].original
        if error and original is not None and validate_syntax(path, original) is not None:
            error = None  # the file was already unparsable before our edits; do not blame them
        validation[path] = error
        if error:
            for number, outcome in enumerate(outcomes, 1):
                if outcome.status == APPLIED and outcome.edit.file == path:
                    outcome.status, outcome.error_kind = FAILED, "syntax_error"
                    outcome.error = f"Edit #{number} ({path}) leaves the file invalid: {error}"

    errors: list[str] = [
        o.error for o in outcomes if o.status == FAILED and o.error and o.error_kind != "syntax_error"
    ]
    errors += [f"Syntax check failed after applying the edits: {e}" for e in validation.values() if e]
    if not edits:
        errors.append("No edits to apply.")

    ok = bool(edits) and all(o.status in _OK_STATUSES for o in outcomes) and not any(validation.values())
    return EditBatchResult(
        ok=ok,
        outcomes=outcomes,
        original_contents=original_contents,
        new_contents=new_contents,
        validation=validation,
        diff=unified_diff(original_contents, new_contents),
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _to_disk_text(text: str, original: str | None) -> str:
    """Restore the original file's trailing-newline state and line-ending style."""
    text = _lf(text)
    if original:
        if original.endswith("\n") and text and not text.endswith("\n"):
            text += "\n"
        elif not original.endswith("\n") and text.endswith("\n"):
            text = text[:-1]
        if _uses_crlf(original):
            text = text.replace("\n", "\r\n")
    return text


def write_edits(root: Path, result: EditBatchResult, *, force: bool = False) -> list[str]:
    """
    Write ``result.new_contents`` under ``root`` and return the written relative paths.

    Preserves each file's original line endings (CRLF vs LF) and trailing-newline state and
    creates parent directories for new files. Refuses (``EditWriteError``, nothing written)
    when the batch is not ok (unless ``force``), when a path is unsafe, or when a file
    changed on disk since ``plan_edits`` read it.
    """
    if not result.ok and not force:
        raise EditWriteError("Refusing to write an edit batch that is not ok: " + "; ".join(result.errors[:3]))
    root = Path(root)
    planned: list[tuple[str, Path, str]] = []
    for path, text in result.new_contents.items():
        rel, reason = _check_path(root, path)
        if rel is None:
            raise EditWriteError(f"Unsafe path {path!r}: {reason}.")
        target = root / rel
        original = result.original_contents.get(path)
        if original is None:
            if target.exists():
                raise EditWriteError(f"'{rel}' appeared on disk after the edits were planned; re-plan.")
        elif not target.is_file() or _read_text(target) != original:
            raise EditWriteError(f"'{rel}' changed on disk after the edits were planned; re-plan.")
        planned.append((rel, target, _to_disk_text(text, original)))

    written: list[str] = []
    for rel, target, data in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.encode("utf-8", errors="surrogateescape"))
        written.append(rel)
    return written


# ---------------------------------------------------------------------------
# Diff + validation
# ---------------------------------------------------------------------------

def unified_diff(original: dict[str, str | None], new: dict[str, str]) -> str:
    """
    ``git apply``-compatible unified diff (LF) from ``original`` to ``new``.

    Each file gets ``diff --git`` / ``--- a/p`` / ``+++ b/p`` headers (``new file mode`` and
    ``--- /dev/null`` for files absent from ``original``), three lines of context and
    ``\\ No newline at end of file`` markers where needed. Unchanged files are skipped.
    """
    parts: list[str] = []
    for path, new_text in new.items():
        old_text = original.get(path)
        is_new = old_text is None
        a, b = _lf(old_text or ""), _lf(new_text)
        if not is_new and a == b:
            continue
        out = [f"diff --git a/{path} b/{path}\n"]
        if is_new:
            out.append("new file mode 100644\n")
        out.append("--- /dev/null\n" if is_new else f"--- a/{path}\n")
        out.append(f"+++ b/{path}\n")
        hunks = difflib.unified_diff(_keepends(a), _keepends(b), n=3, lineterm="\n")
        for i, line in enumerate(hunks):
            if i < 2:  # difflib's own ---/+++ header lines; ours are above
                continue
            if line.endswith("\n"):
                out.append(line)
            else:
                out.append(line + "\n\\ No newline at end of file\n")
        parts.append("".join(out))
    return "".join(parts)


def _diff_stats(diff: str) -> tuple[int, int]:
    added = removed = 0
    in_hunk = False
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            added += 1
        elif in_hunk and line.startswith("-"):
            removed += 1
    return added, removed


def validate_syntax(path: str, content: str) -> str | None:
    """
    Syntax-check ``content`` by file type: ``.py``/``.pyi`` via ``compile()``, ``.json`` via
    ``json.loads``, ``.toml`` via ``tomllib``. Returns an error message with the line number,
    or None when valid or the type is not checked.
    """
    suffix = PurePosixPath(path.replace("\\", "/")).suffix.lower()
    text = _lf(content)
    if text.startswith("﻿"):
        text = text[1:]
    if suffix in (".py", ".pyi"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                compile(text, path, "exec", dont_inherit=True)
        except SyntaxError as exc:
            where = f"line {exc.lineno}" + (f", column {exc.offset}" if exc.offset else "")
            source = (exc.text or "").strip("\n")
            message = f"{path}: {type(exc).__name__} at {where}: {exc.msg}"
            return message + (f"\n    {source.strip()}" if source.strip() else "")
        except (ValueError, UnicodeError) as exc:
            return f"{path}: cannot be compiled: {exc}"
    elif suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
    elif suffix == ".toml" and tomllib is not None:
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            return f"{path}: invalid TOML: {exc}"
    return None
