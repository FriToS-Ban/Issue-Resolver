"""
retrieval/index.py — Lightweight repository index: files, symbols, imports, references.

Inspired by RepoSense's per-file "facts" + global resolution split, but deliberately
smaller and dependency-free:

  * Python files are parsed with ``ast``: classes, functions, methods (with line spans,
    signatures, docstrings), module imports (absolute + relative), and the names each
    symbol references (calls, attribute accesses, bare names).
  * Other languages (JS/TS, Go, Java/Kotlin, Rust, Ruby, C#) use conservative regexes for
    definitions and imports — enough for definition lookup and a coarse import graph.

Global structures:
  defs[name]        -> [Symbol]  (by short name and by qualname)
  refs[name]        -> {(path, enclosing qualname)}  — who references a name (callers)
  imports[path]     -> {resolved repo paths}
  importers[path]   -> {paths that import it}
"""

from __future__ import annotations

import ast
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from fix_issue.retrieval.mentions import resolve_module
from fix_issue.retrieval.text import tokenize

SOURCE_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".java": "java", ".kt": "kotlin", ".rs": "rust", ".rb": "ruby", ".cs": "csharp",
    ".php": "php", ".swift": "swift", ".scala": "scala", ".c": "c", ".h": "c", ".cpp": "cpp",
    ".hpp": "cpp", ".cc": "cpp",
}
TEXT_EXT = {".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".sql", ".sh"}

SKIP_DIRS = re.compile(
    r"(^|/)(node_modules|\.git|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|dist|build|"
    r"[^/]*\.egg-info|venv|\.venv|env|\.tox|coverage|\.nyc_output|vendor|third_party|site-packages|"
    r"\.idea|\.vscode|target|out)(/|$)",
    re.IGNORECASE,
)
MAX_FILE_BYTES = 400_000
TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec)(/|$)|(^|/)test_[^/]*$|_test\.[a-z]+$|\.(spec|test)\.[jt]sx?$")
_DRIVE = re.compile(r"^[A-Za-z]:")


# ---------------------------------------------------------------------------
# Containment: never read outside the clone
# ---------------------------------------------------------------------------
# Everything read here ends up in an LLM prompt (and from there possibly in a public PR), so a
# symlink or junction in the clone — committed by the repo, or created by test code the model
# wrote — must never pull in a file from outside it (~/.ssh, ~/.aws, the tool's .env, ...).

def _is_link(path: Path) -> bool:
    """Symlink or (Windows) junction / mount point."""
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)   # Python 3.12+
        return bool(is_junction and is_junction(path))
    except OSError:
        return True


def _inside(path: Path, root_resolved: Path) -> bool:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return False
    return resolved != root_resolved and resolved.is_relative_to(root_resolved)


def _walk_repo_files(root: Path):
    """Regular files under ``root``; symlinks/junctions and anything resolving outside are skipped."""
    try:
        root_resolved = root.resolve()
    except (OSError, RuntimeError):
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        rel_dir = here.relative_to(root).as_posix()
        kept = []
        for d in dirnames:
            sub = here / d
            rel = d if rel_dir == "." else f"{rel_dir}/{d}"
            if SKIP_DIRS.search(rel + "/") or _is_link(sub) or not _inside(sub, root_resolved):
                continue
            kept.append(d)
        dirnames[:] = kept
        for name in filenames:
            p = here / name
            # Contents are only read through safe_repo_file (which also resolves the file).
            if _is_link(p) or not p.is_file():
                continue
            yield p


def safe_repo_file(root: Path, rel: str) -> Path | None:
    """
    ``root/rel`` when it is a regular file inside the clone, else None.

    Rejects absolute paths, ``..``, and any symlink/junction on the way (resolve() would
    follow them), and requires the resolved file to be inside the resolved root. Use before
    every read of a path that came from the index, the plan (LLM output) or the issue.
    """
    if not rel or "\x00" in rel:
        return None
    norm = rel.replace("\\", "/")
    if norm.startswith("/") or _DRIVE.match(norm):
        return None
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return None
    root = Path(root)
    try:
        root_resolved = root.resolve()
    except (OSError, RuntimeError):
        return None
    cur = root
    for part in parts:
        cur = cur / part
        if _is_link(cur):
            return None
    try:
        if not cur.is_file():
            return None
    except OSError:
        return None
    return cur if _inside(cur, root_resolved) else None


@dataclass
class Symbol:
    name: str
    qualname: str
    kind: str            # class | function | method | constant
    path: str
    start: int           # 1-based inclusive
    end: int
    signature: str = ""
    doc: str = ""
    refs: set[str] = field(default_factory=set)   # names referenced inside the body

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}"


@dataclass
class FileInfo:
    path: str
    language: str
    text: str
    symbols: list[Symbol] = field(default_factory=list)
    raw_imports: list[str] = field(default_factory=list)
    module_refs: set[str] = field(default_factory=set)  # names referenced at module level
    is_test: bool = False
    parse_error: str | None = None

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()


class RepoIndex:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.files: dict[str, FileInfo] = {}
        self.all_paths: list[str] = []
        self.defs: dict[str, list[Symbol]] = defaultdict(list)
        self.refs: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.imports: dict[str, set[str]] = defaultdict(set)
        self.importers: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, root: Path, max_files: int = 6000) -> "RepoIndex":
        idx = cls(root)
        paths: list[str] = []
        for p in sorted(_walk_repo_files(Path(root))):
            rel = p.relative_to(root).as_posix()
            if SKIP_DIRS.search(rel):
                continue
            paths.append(rel)
            if len(paths) >= max_files:
                break
        idx.all_paths = paths
        for rel in paths:
            ext = Path(rel).suffix.lower()
            if ext not in SOURCE_EXT and ext not in TEXT_EXT:
                continue
            p = safe_repo_file(Path(root), rel)   # re-checked right before the read
            if p is None:
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lang = SOURCE_EXT.get(ext, "text")
            fi = FileInfo(rel, lang, text, is_test=bool(TEST_PATH.search(rel)))
            if lang == "python":
                _extract_python(fi)
                # `test_runner.py` outside a tests/ dir is a test only if it defines tests.
                if fi.is_test and not re.search(r"(^|/)(tests?|__tests__|spec)/", rel):
                    fi.is_test = any(s.name.startswith("test") or s.name.startswith("Test") for s in fi.symbols)
            elif lang != "text":
                _extract_regex(fi)
            idx.files[rel] = fi
        idx._link()
        return idx

    # ------------------------------------------------------------------
    def _link(self) -> None:
        for fi in self.files.values():
            for sym in fi.symbols:
                self.defs[sym.name].append(sym)
                if sym.qualname != sym.name:
                    self.defs[sym.qualname].append(sym)
                for r in sym.refs:
                    self.refs[r].add((fi.path, sym.qualname))
            for r in fi.module_refs:
                self.refs[r].add((fi.path, ""))
            for raw in fi.raw_imports:
                target = self._resolve_import(fi, raw)
                if target and target != fi.path:
                    self.imports[fi.path].add(target)
                    self.importers[target].add(fi.path)

    def _resolve_import(self, fi: FileInfo, raw: str) -> str | None:
        if fi.language == "python":
            return resolve_module(raw, self.all_paths) if raw else None
        if raw.startswith("."):
            base = Path(fi.path).parent
            cand = (base / raw).as_posix()
            cand = _normpath(cand)
            for suffix in ("", ".ts", ".tsx", ".js", ".jsx", ".mjs", "/index.ts", "/index.js"):
                if cand + suffix in self.files:
                    return cand + suffix
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def lookup(self, ident: str) -> list[Symbol]:
        """Definitions for an identifier: exact qualname/name, then last dotted part, then case-insensitive."""
        if ident in self.defs:
            return list(self.defs[ident])
        if "." in ident:
            parts = ident.split(".")
            for k in range(len(parts) - 1):
                tail = ".".join(parts[k + 1:])
                if tail in self.defs:
                    return list(self.defs[tail])
        low = ident.lower()
        hits = [s for name, syms in self.defs.items() if name.lower() == low for s in syms]
        return hits

    def callers(self, name: str) -> set[tuple[str, str]]:
        short = name.split(".")[-1]
        return set(self.refs.get(short, set()))

    def symbol_at(self, path: str, line: int) -> Symbol | None:
        fi = self.files.get(path)
        if not fi:
            return None
        best = None
        for s in fi.symbols:
            if s.start <= line <= s.end and (best is None or (s.end - s.start) < (best.end - best.start)):
                best = s
        return best

    def symbol_text(self, sym: Symbol) -> str:
        fi = self.files.get(sym.path)
        if not fi:
            return ""
        return "\n".join(fi.lines[sym.start - 1: sym.end])

    def source_paths(self) -> list[str]:
        return [p for p, f in self.files.items() if f.language != "text"]

    def stats(self) -> dict:
        return {
            "files_indexed": len(self.files),
            "source_files": len(self.source_paths()),
            "symbols": sum(len(f.symbols) for f in self.files.values()),
            "import_edges": sum(len(v) for v in self.imports.values()),
            "parse_errors": [f.path for f in self.files.values() if f.parse_error],
        }


# ---------------------------------------------------------------------------
# Python extraction
# ---------------------------------------------------------------------------

def _module_name(path: str) -> str:
    mod = path[:-3] if path.endswith(".py") else path
    if mod.endswith("/__init__"):
        mod = mod[: -len("/__init__")]
    return mod.replace("/", ".")


def _extract_python(fi: FileInfo) -> None:
    try:
        tree = ast.parse(fi.text)
    except (SyntaxError, ValueError) as exc:
        fi.parse_error = str(exc)
        _extract_regex(fi)
        return
    module = _module_name(fi.path)
    package = module if fi.path.endswith("__init__.py") else module.rsplit(".", 1)[0] if "." in module else ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                fi.raw_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".") if package else []
                if node.level > 1:
                    base_parts = base_parts[: len(base_parts) - (node.level - 1)]
                base = ".".join(base_parts)
                mod = f"{base}.{node.module}" if node.module and base else (node.module or base)
            else:
                mod = node.module or ""
            if mod:
                fi.raw_imports.append(mod)
            for alias in node.names:
                if alias.name != "*" and mod:
                    fi.raw_imports.append(f"{mod}.{alias.name}")  # may be a submodule
                fi.module_refs.add(alias.name)

    def refs_of(node: ast.AST) -> set[str]:
        """Names a symbol depends on: everything it calls, plus free (non-local) names it
        loads. Local variables/parameters and plain attribute reads are excluded — they are
        the main source of spurious edges (a local `path` is not a use of `X.path`)."""
        local: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
                if arg is not None:
                    local.add(arg.arg)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                local.add(sub.id)
            elif isinstance(sub, (ast.For, ast.comprehension)) and isinstance(getattr(sub, "target", None), ast.Name):
                local.add(sub.target.id)
        out: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                if isinstance(fn, ast.Name):
                    out.add(fn.id)
                elif isinstance(fn, ast.Attribute):
                    out.add(fn.attr)
            elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id not in local:
                out.add(sub.id)
            elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id[:1].isupper():
                out.add(sub.attr)  # Class.attr / Module-level constant access
        return out - {"self", "cls"}

    def visit(body: list[ast.stmt], prefix: str, in_class: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}{node.name}"
                try:
                    sig = f"def {node.name}({ast.unparse(node.args)})"
                    if node.returns is not None:
                        sig += f" -> {ast.unparse(node.returns)}"
                except Exception:  # pragma: no cover
                    sig = f"def {node.name}(...)"
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                sym = Symbol(node.name, qual, "method" if in_class else "function", fi.path,
                             start, node.end_lineno or node.lineno, sig,
                             (ast.get_docstring(node) or "")[:300], refs_of(node) - {node.name})
                fi.symbols.append(sym)
                visit(node.body, f"{qual}.", False)
            elif isinstance(node, ast.ClassDef):
                qual = f"{prefix}{node.name}"
                bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                sym = Symbol(node.name, qual, "class", fi.path, start, node.end_lineno or node.lineno,
                             f"class {node.name}({bases})" if bases else f"class {node.name}",
                             (ast.get_docstring(node) or "")[:300],
                             set().union(*(refs_of(b) for b in node.bases)) if node.bases else set())
                fi.symbols.append(sym)
                visit(node.body, f"{qual}.", True)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and not prefix:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        fi.symbols.append(Symbol(t.id, t.id, "constant", fi.path, node.lineno,
                                                 node.end_lineno or node.lineno, t.id))
                fi.module_refs |= refs_of(node)
            elif not prefix:
                fi.module_refs |= refs_of(node)

    visit(tree.body, "", False)


# ---------------------------------------------------------------------------
# Regex extraction for other languages
# ---------------------------------------------------------------------------

_DEF_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {
    "javascript": [
        ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)", re.M)),
        ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)", re.M)),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>", re.M)),
        ("method", re.compile(r"^\s{2,}(?:async\s+)?(?:static\s+)?([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{", re.M)),
    ],
    "go": [
        ("function", re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", re.M)),
        ("class", re.compile(r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)", re.M)),
    ],
    "java": [
        ("class", re.compile(r"^\s*(?:public|private|protected)?\s*(?:abstract\s+|final\s+|static\s+)*(?:class|interface|enum|record)\s+([A-Za-z_]\w*)", re.M)),
        ("method", re.compile(r"^\s+(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?[\w<>\[\], ]+\s+([a-z]\w*)\s*\(", re.M)),
    ],
    "rust": [
        ("function", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)", re.M)),
        ("class", re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)", re.M)),
    ],
    "ruby": [
        ("function", re.compile(r"^\s*def\s+(?:self\.)?([A-Za-z_]\w*[?!]?)", re.M)),
        ("class", re.compile(r"^\s*(?:class|module)\s+([A-Z]\w*)", re.M)),
    ],
    "python": [
        ("function", re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", re.M)),
        ("class", re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M)),
    ],
}
_DEF_PATTERNS["typescript"] = _DEF_PATTERNS["javascript"]
_DEF_PATTERNS["kotlin"] = _DEF_PATTERNS["java"]
_DEF_PATTERNS["csharp"] = _DEF_PATTERNS["java"]
_IMPORT_PATTERNS = [
    re.compile(r"""^\s*import\s+(?:[^'"]+\s+from\s+)?['"]([^'"]+)['"]""", re.M),
    re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)"""),
    re.compile(r"""^\s*import\s+"([^"]+)\"""", re.M),
]
_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "function", "constructor", "else"}


def _extract_regex(fi: FileInfo) -> None:
    patterns = _DEF_PATTERNS.get(fi.language, [])
    lines = fi.text.splitlines()
    starts: list[tuple[int, str, str]] = []
    for kind, rx in patterns:
        for m in rx.finditer(fi.text):
            name = m.group(1)
            if name in _KEYWORDS:
                continue
            line = fi.text.count("\n", 0, m.start()) + 1
            starts.append((line, kind, name))
    starts.sort()
    seen: set[tuple[int, str]] = set()
    uniq = [s for s in starts if (s[0], s[2]) not in seen and not seen.add((s[0], s[2]))]
    for i, (line, kind, name) in enumerate(uniq):
        end = uniq[i + 1][0] - 1 if i + 1 < len(uniq) else len(lines)
        end = max(line, min(end, line + 200))
        body = "\n".join(lines[line - 1:end])
        fi.symbols.append(Symbol(name, name, kind, fi.path, line, end,
                                 lines[line - 1].strip()[:200] if line - 1 < len(lines) else name,
                                 "", set(tokenize_identifiers(body)) - {name}))
    for rx in _IMPORT_PATTERNS:
        fi.raw_imports.extend(m.group(1) for m in rx.finditer(fi.text))
    fi.module_refs |= set(tokenize_identifiers(fi.text[:2000]))


_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize_identifiers(text: str) -> list[str]:
    return _ID.findall(text)


def _normpath(p: str) -> str:
    parts: list[str] = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts)


def symbol_document(index: RepoIndex, sym: Symbol, max_chars: int = 1500) -> str:
    """Text used for lexical/embedding retrieval of one symbol."""
    head = f"{sym.path} {sym.qualname}\n{sym.signature}\n{sym.doc}\n"
    return (head + index.symbol_text(sym))[:max_chars]


def file_tokens(fi: FileInfo) -> list[str]:
    return tokenize(fi.path.replace("/", " ").replace(".", " ")) * 3 + tokenize(fi.text[:60_000])
