"""
retrieval/mentions.py — Explicit evidence in the issue text: file paths, stack-trace frames,
code identifiers and exception types. These are the strongest retrieval signals and are
resolved against the real repository file list so absolute or partial paths still match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fix_issue.retrieval.text import COMMON_CODE_NAMES, STOP

_CODE_EXT = (
    "py|pyi|js|jsx|ts|tsx|mjs|cjs|java|kt|go|rb|rs|c|cc|cpp|h|hpp|cs|php|swift|scala|"
    "toml|yaml|yml|json|cfg|ini|md|rst|sql|sh"
)
_PATH_RE = re.compile(rf"(?<![\w/.-])((?:[A-Za-z]:)?[\w./\\-]*[\w-]+\.(?:{_CODE_EXT}))\b")
_PY_FRAME = re.compile(r'File "([^"]+)", line (\d+)(?:, in ([\w<>.]+))?')
_JS_FRAME = re.compile(r"at (?:([\w$.<>\[\] ]+?) \()?((?:[A-Za-z]:)?[^\s():]+\.(?:js|jsx|ts|tsx|mjs|cjs)):(\d+)(?::\d+)?\)?")
_JAVA_FRAME = re.compile(r"at ([\w$.]+)\.([\w$<>]+)\((\w+\.(?:java|kt)):(\d+)\)")
_GO_FRAME = re.compile(r"^\s*((?:[A-Za-z]:)?[^\s:]+\.go):(\d+)", re.MULTILINE)
_BACKTICK = re.compile(r"`([^`\n]{1,120})`")
_DOTTED = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b")
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_SNAKE = re.compile(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b")
_CAMEL = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+|[a-z]+(?:[A-Z][a-z0-9]+)+)\b")
_ERROR = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Warning|Declined|Fault))\b")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class Frame:
    path: str          # resolved repo-relative path
    line: int | None
    function: str | None
    raw: str


@dataclass
class Mentions:
    paths: list[str] = field(default_factory=list)            # resolved repo paths, in order of appearance
    frames: list[Frame] = field(default_factory=list)          # resolved stack frames (project files only)
    identifiers: list[str] = field(default_factory=list)       # code-like identifiers (incl. dotted)
    error_types: list[str] = field(default_factory=list)
    unresolved_paths: list[str] = field(default_factory=list)  # mentioned but not in repo

    def to_dict(self) -> dict:
        return {
            "paths": self.paths,
            "frames": [f.__dict__ for f in self.frames],
            "identifiers": self.identifiers,
            "error_types": self.error_types,
            "unresolved_paths": self.unresolved_paths,
        }


def resolve_path(raw: str, repo_files: list[str], _cache: dict | None = None) -> str | None:
    """Map an absolute/partial/Windows path or dotted module onto a repository file."""
    cand = raw.strip().strip("'\"()[]<>,;").replace("\\", "/")
    while cand.startswith("./"):
        cand = cand[2:]
    if not cand:
        return None
    files_set = set(repo_files)
    if cand in files_set:
        return cand
    parts = [p for p in cand.split("/") if p and p != "."]
    # Longest suffix match on path components (handles /home/u/proj/pkg/mod.py, site-packages,
    # Windows paths). The file name itself must match; ties are ambiguous and rejected.
    best: list[str] = []
    best_len = 0
    for f in repo_files:
        fparts = f.split("/")
        n = 0
        while n < min(len(parts), len(fparts)) and parts[-1 - n] == fparts[-1 - n]:
            n += 1
        if n > best_len:
            best, best_len = [f], n
        elif n == best_len and n > 0:
            best.append(f)
    if best_len >= 1 and len(best) == 1:
        return best[0]
    return None


def resolve_module(dotted: str, repo_files: list[str]) -> str | None:
    """'pkg.sub.mod' -> 'pkg/sub/mod.py' or 'pkg/sub/mod/__init__.py' (also under src/)."""
    rel = dotted.replace(".", "/")
    for suffix in (f"{rel}.py", f"{rel}/__init__.py"):
        for f in repo_files:
            if f == suffix or f.endswith("/" + suffix):
                return f
    return None


def extract_mentions(text: str, repo_files: list[str]) -> Mentions:
    m = Mentions()
    text = text or ""
    seen_paths: set[str] = set()

    def add_path(p: str | None) -> None:
        if p and p not in seen_paths:
            seen_paths.add(p)
            m.paths.append(p)

    # --- stack frames -------------------------------------------------------
    for fm in _PY_FRAME.finditer(text):
        resolved = resolve_path(fm.group(1), repo_files)
        if resolved:
            m.frames.append(Frame(resolved, int(fm.group(2)), fm.group(3), fm.group(0)))
            add_path(resolved)
    for fm in _JS_FRAME.finditer(text):
        resolved = resolve_path(fm.group(2), repo_files)
        if resolved:
            func = (fm.group(1) or "").strip().split(".")[-1] or None
            m.frames.append(Frame(resolved, int(fm.group(3)), func, fm.group(0)))
            add_path(resolved)
    for fm in _JAVA_FRAME.finditer(text):
        resolved = resolve_path(fm.group(3), repo_files)
        if resolved:
            m.frames.append(Frame(resolved, int(fm.group(4)), fm.group(2), fm.group(0)))
            add_path(resolved)
    for fm in _GO_FRAME.finditer(text):
        resolved = resolve_path(fm.group(1), repo_files)
        if resolved:
            m.frames.append(Frame(resolved, int(fm.group(2)), None, fm.group(0)))
            add_path(resolved)

    # --- explicit paths -----------------------------------------------------
    for pm in _PATH_RE.finditer(text):
        raw = pm.group(1)
        resolved = resolve_path(raw, repo_files)
        if resolved:
            add_path(resolved)
        elif "/" in raw or "\\" in raw:
            if raw not in m.unresolved_paths:
                m.unresolved_paths.append(raw)

    # --- identifiers --------------------------------------------------------
    idents: list[str] = []

    def add_ident(tok: str) -> None:
        tok = tok.strip().strip("().,:;")
        if not tok or len(tok) < 3:
            return
        low = tok.lower()
        if low in STOP or tok in idents:
            return
        if "." not in tok and low in COMMON_CODE_NAMES and "_" not in tok:
            return
        idents.append(tok)

    for bm in _BACKTICK.finditer(text):
        inner = bm.group(1)
        for dm in _DOTTED.finditer(inner):
            dotted = dm.group(1)
            mod = resolve_module(dotted, repo_files)
            if mod:
                add_path(mod)
            add_ident(dotted)
        for tok in _IDENT.findall(inner):
            add_ident(tok)
    for dm in _DOTTED.finditer(text):
        dotted = dm.group(1)
        if re.fullmatch(rf".*\.(?:{_CODE_EXT})", dotted):
            continue  # a file name, handled above
        if re.fullmatch(r"(?:e\.g|i\.e|etc|vs)\..*", dotted, re.IGNORECASE):
            continue
        mod = resolve_module(dotted, repo_files)
        if mod:
            add_path(mod)
        add_ident(dotted)
    for rx in (_CALL, _SNAKE, _CAMEL):
        for im in rx.finditer(text):
            add_ident(im.group(1))
    for f in m.frames:
        if f.function and f.function not in ("<module>", "<lambda>"):
            add_ident(f.function)
    m.identifiers = idents[:60]

    errs: list[str] = []
    for em in _ERROR.finditer(text):
        if em.group(1) not in errs:
            errs.append(em.group(1))
    m.error_types = errs
    return m
