"""
validators.py — Discover and run the target repository's OWN quality tools, diff-aware.

A patch should satisfy the repository's conventions, not ours. So validation only runs
tools the repository itself configures (and that are installed), only on the files the
patch changed, and only fails on findings the patch *introduced*:

``discover_tools(root, cfg) -> (tools, skipped)``
    Inspects the repo's config files. A tool is runnable only if the repo CONFIGURES it and
    it is INSTALLED; configured-but-unavailable, disabled, and gated tools are returned in
    ``skipped`` with a reason. The built-in ``py-syntax`` check is always included.

``run_validators(root, tools, changed_files, base_ref) -> [ValidatorResult]``
    Runs each tool on the changed files (filtered by extension) in the working tree, and on
    the same files at ``base_ref`` in a temporary ``git worktree add --detach`` checkout that
    is always removed afterwards (``git worktree remove --force`` + ``prune``, even on
    errors). Findings are compared by ``Finding.key()`` — (tool, path, code, normalized
    message), deliberately WITHOUT line numbers, so a pre-existing finding that merely moved
    because lines were inserted above it is not "new". Keys are compared as a multiset: a
    second identical finding in the same file is new. Formatters work the same way: a file
    already unformatted at base doesn't count, a file that was formatted and no longer is,
    does. Files that did not exist at base: every finding is new.

Tools that cannot run (not installed, timed out, baseline worktree could not be created,
baseline run crashed) report ``ran=False`` with a ``skipped_reason`` and ``ok=True``: an
unavailable tool never fails validation, but the reason is recorded. Internal errors while
running one tool are reported the same way and never abort the others.

Discovery
---------
===========  =========  ===========================================================  ===============================
tool         kind       configured by                                                installed when
===========  =========  ===========================================================  ===============================
py-syntax    static     always (in-process ``compile()``, no subprocess)             —
ruff         lint       [tool.ruff] in pyproject.toml, ruff.toml, .ruff.toml          ``python -m ruff`` / ruff on PATH
ruff-format  format     [tool.ruff.format] in pyproject.toml, [format] in ruff.toml   (as ruff)
flake8       lint       .flake8, [flake8] in setup.cfg or tox.ini                     ``python -m flake8`` / PATH
mypy         typecheck  [tool.mypy], mypy.ini, .mypy.ini, [mypy] in setup.cfg        ``python -m mypy`` / PATH
pyright      typecheck  pyrightconfig.json, [tool.pyright]                            pyright on PATH
black        format     [tool.black]                                                  ``python -m black`` / PATH
eslint       lint       .eslintrc*, eslint.config.*, "eslintConfig" in package.json   node_modules/.bin/eslint
tsc          typecheck  tsconfig.json                                                 node_modules/.bin/tsc
go-vet       lint       go.mod                                                        go on PATH
npm-lint     lint       package.json "lint" script      (allow_project_scripts)       npm on PATH
make-lint    lint       Makefile "lint:" target         (allow_project_scripts)       make on PATH
tox-lint     lint       [testenv:lint] in tox.ini       (allow_project_scripts)       tox on PATH
===========  =========  ===========================================================  ===============================

``[validation] run_linters / run_type_checkers / run_formatters`` switch whole kinds off.
Project scripts run arbitrary repository code, so they are only used with
``[validation] allow_project_scripts = true``.

Hygiene: every subprocess goes through ``sandbox.run_command`` (sanitized env, timeout, no
shell); Python tools run as ``python -P -m <tool>`` so a module planted in the repo cannot
shadow the tool; the baseline worktree is created with hooks disabled; file arguments that
start with ``-`` are passed as ``./-name``. For eslint/tsc/project scripts the repo's
``node_modules`` is linked into the baseline worktree (and unlinked before removal) so
plugins resolve the same way in both runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from fix_issue.config import ValidationConfig
from fix_issue.sandbox import python_executable, run_command

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str            # "ruff", "flake8", "mypy", "pyright", "black", "ruff-format", "eslint", "tsc", "go-vet", "py-syntax", ...
    kind: str            # lint | format | typecheck | static
    command: list[str]   # argv prefix; files are appended when per_file
    per_file: bool
    reason: str          # why it was discovered, e.g. "[tool.ruff] in pyproject.toml"
    project_script: bool = False   # npm run / make / tox: arbitrary repo code


_NUM_POS = re.compile(r"\b\d+:\d+(?::\d+)?\b")
_LINE_REF = re.compile(r"\b(line|lines|col|column)\s+\d+", re.IGNORECASE)
_WS = re.compile(r"\s+")


def _normalize_message(message: str) -> str:
    text = _LINE_REF.sub(lambda m: f"{m.group(1)} N", message)
    text = _NUM_POS.sub("N:N", text)
    return _WS.sub(" ", text).strip()


@dataclass
class Finding:
    tool: str
    path: str
    line: int | None
    code: str
    message: str

    def key(self) -> tuple:
        """Identity across runs: line numbers excluded so shifted lines don't count as new."""
        return (self.tool, self.path, self.code, _normalize_message(self.message))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidatorResult:
    tool: str
    kind: str
    ran: bool
    ok: bool                     # True when ran=False: an unavailable tool never fails validation
    skipped_reason: str | None = None
    new_findings: list[Finding] = field(default_factory=list)
    baseline_count: int = 0
    after_count: int = 0
    output_tail: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_PY_EXTS = (".py", ".pyi")
_JS_EXTS = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts", ".vue")
_TS_EXTS = (".ts", ".tsx", ".mts", ".cts")
_TOOL_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "py-syntax": (".py",),
    "ruff": _PY_EXTS,
    "ruff-format": _PY_EXTS,
    "flake8": (".py",),
    "mypy": _PY_EXTS,
    "pyright": _PY_EXTS,
    "black": _PY_EXTS,
    "eslint": _JS_EXTS,
    "tsc": _TS_EXTS,
    "go-vet": (".go",),
}
_NODE_TOOLS = {"eslint", "tsc"}
_ESLINT_CONFIGS = (
    ".eslintrc", ".eslintrc.js", ".eslintrc.cjs", ".eslintrc.json", ".eslintrc.yml", ".eslintrc.yaml",
    "eslint.config.js", "eslint.config.mjs", "eslint.config.cjs",
    "eslint.config.ts", "eslint.config.mts", "eslint.config.cts",
)
_MAKEFILES = ("Makefile", "makefile", "GNUmakefile")
_PROJECT_SCRIPT_SKIP = "project script; set [validation] allow_project_scripts = true"

_which = shutil.which   # indirection so tests can monkeypatch availability


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class _TomlFile:
    """Parsed TOML with a header-regex fallback when parsing is impossible."""

    def __init__(self, path: Path) -> None:
        text = _read_text(path) if path.is_file() else ""
        self.data: dict | None = None
        if text and tomllib is not None:
            try:
                self.data = tomllib.loads(text)
            except ValueError:
                self.data = None
        self.headers = {m.group(1).strip() for m in
                        re.finditer(r"^\s*\[\[?\s*([^\]\n]+?)\s*\]\]?", text, re.MULTILINE)}

    def has(self, dotted: str) -> bool:
        if self.data is not None:
            node: object = self.data
            for part in dotted.split("."):
                if not isinstance(node, dict) or part not in node:
                    return False
                node = node[part]
            return True
        return any(h == dotted or h.startswith(dotted + ".") for h in self.headers)


def _ini_sections(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {m.group(1).strip().lower()
            for m in re.finditer(r"^\s*\[([^\]\n]+)\]", _read_text(path), re.MULTILINE)}


def _python_module_prefix(module: str) -> list[str]:
    # -P (3.11+): don't put the cwd on sys.path, so a planted `ruff.py` in the repo can't shadow the tool.
    flags = ["-P"] if sys.version_info >= (3, 11) else []
    return [python_executable(), *flags, "-m", module]


def _resolve_python_tool(module: str | None, exe: str) -> list[str] | None:
    """Command prefix for a Python tool: importable by our interpreter, else on PATH; None if neither."""
    if module:
        prefix = _python_module_prefix(module)
        res = run_command([*prefix, "--version"], cwd=Path(tempfile.gettempdir()), timeout=60)
        if res.ok:
            return prefix
    found = _which(exe)
    return [found] if found else None


def _node_tool(root: Path, bin_name: str, js_parts: tuple[str, ...]) -> list[str] | None:
    """Command for a node_modules tool; prefers `node <entry>.js` over the .bin shim."""
    bin_dir = root / "node_modules" / ".bin"
    shims = [bin_dir / f"{bin_name}.cmd", bin_dir / bin_name] if os.name == "nt" else \
            [bin_dir / bin_name, bin_dir / f"{bin_name}.cmd"]
    existing = [s for s in shims if s.is_file()]
    if not existing:
        return None
    entry = root.joinpath("node_modules", *js_parts)
    node = _which("node")
    if node and entry.is_file():
        return [node, str(entry.resolve())]
    return [str(existing[0].resolve())]


def discover_tools(root: Path, cfg: ValidationConfig | None = None) -> tuple[list[ToolSpec], list[dict]]:
    """Return (runnable tools, skipped [{"tool", "reason"}]) for the repository at `root`."""
    cfg = cfg if cfg is not None else ValidationConfig()
    root = Path(root)
    tools: list[ToolSpec] = [ToolSpec("py-syntax", "static", [], True,
                                      "built-in: changed Python files are compiled in-process")]
    skipped: list[dict] = []
    enabled = {
        "lint": (cfg.run_linters, "run_linters"),
        "typecheck": (cfg.run_type_checkers, "run_type_checkers"),
        "format": (cfg.run_formatters, "run_formatters"),
    }
    py_cache: dict[str, list[str] | None] = {}

    def python_tool(module: str | None, exe: str) -> Callable[[], list[str] | None]:
        def resolve() -> list[str] | None:
            if exe not in py_cache:
                py_cache[exe] = _resolve_python_tool(module, exe)
            return py_cache[exe]
        return resolve

    def on_path(exe: str) -> Callable[[], list[str] | None]:
        def resolve() -> list[str] | None:
            found = _which(exe)
            return [found] if found else None
        return resolve

    def consider(name: str, kind: str, reason: str | None, resolve: Callable[[], list[str] | None],
                 args: list[str], per_file: bool = True, missing: str = "not installed",
                 project_script: bool = False) -> None:
        if reason is None:
            return
        on, flag = enabled[kind]
        if not on:
            skipped.append({"tool": name, "reason": f"disabled by [validation] {flag} = false"})
            return
        prefix = resolve()
        if not prefix:
            skipped.append({"tool": name, "reason": f"configured ({reason}) but {missing}"})
            return
        tools.append(ToolSpec(name, kind, [*prefix, *args], per_file, reason, project_script))

    pyproject = _TomlFile(root / "pyproject.toml")
    setup_cfg = _ini_sections(root / "setup.cfg")
    tox_ini = _ini_sections(root / "tox.ini")
    ruff_name = next((n for n in ("ruff.toml", ".ruff.toml") if (root / n).is_file()), None)
    ruff_toml = _TomlFile(root / ruff_name) if ruff_name else None
    pkg: dict = {}
    if (root / "package.json").is_file():
        try:
            loaded = json.loads(_read_text(root / "package.json"))
            pkg = loaded if isinstance(loaded, dict) else {}
        except ValueError:
            pkg = {}

    # --- Python -----------------------------------------------------------
    ruff = python_tool("ruff", "ruff")
    reason = ("[tool.ruff] in pyproject.toml" if pyproject.has("tool.ruff")
              else f"{ruff_name} present" if ruff_name else None)
    consider("ruff", "lint", reason, ruff,
             ["check", "--output-format", "json", "--no-cache", "--force-exclude"])

    if pyproject.has("tool.ruff.format"):
        reason = "[tool.ruff.format] in pyproject.toml"
    elif ruff_toml is not None and ruff_toml.has("format"):
        reason = f"[format] in {ruff_name}"
    else:
        reason = None
    consider("ruff-format", "format", reason, ruff, ["format", "--check", "--no-cache", "--force-exclude"])

    if (root / ".flake8").is_file():
        reason = ".flake8 present"
    elif "flake8" in setup_cfg:
        reason = "[flake8] in setup.cfg"
    elif "flake8" in tox_ini:
        reason = "[flake8] in tox.ini"
    else:
        reason = None
    consider("flake8", "lint", reason, python_tool("flake8", "flake8"), [])

    if pyproject.has("tool.mypy"):
        reason = "[tool.mypy] in pyproject.toml"
    elif (root / "mypy.ini").is_file():
        reason = "mypy.ini present"
    elif (root / ".mypy.ini").is_file():
        reason = ".mypy.ini present"
    elif "mypy" in setup_cfg:
        reason = "[mypy] in setup.cfg"
    else:
        reason = None
    consider("mypy", "typecheck", reason, python_tool("mypy", "mypy"),
             ["--no-error-summary", "--show-error-codes", "--no-pretty", "--no-color-output",
              "--cache-dir", os.devnull])

    if (root / "pyrightconfig.json").is_file():
        reason = "pyrightconfig.json present"
    elif pyproject.has("tool.pyright"):
        reason = "[tool.pyright] in pyproject.toml"
    else:
        reason = None
    consider("pyright", "typecheck", reason, on_path("pyright"), ["--outputjson"])

    consider("black", "format", "[tool.black] in pyproject.toml" if pyproject.has("tool.black") else None,
             python_tool("black", "black"), ["--check"])

    # --- JavaScript / TypeScript -------------------------------------------
    eslint_cfg = next((n for n in _ESLINT_CONFIGS if (root / n).is_file()), None)
    reason = (f"{eslint_cfg} present" if eslint_cfg
              else '"eslintConfig" in package.json' if "eslintConfig" in pkg else None)
    consider("eslint", "lint", reason, lambda: _node_tool(root, "eslint", ("eslint", "bin", "eslint.js")),
             ["-f", "json"], missing="node_modules/.bin/eslint not found (run npm install)")

    consider("tsc", "typecheck", "tsconfig.json present" if (root / "tsconfig.json").is_file() else None,
             lambda: _node_tool(root, "tsc", ("typescript", "bin", "tsc")),
             ["--noEmit", "--pretty", "false"], per_file=False,
             missing="node_modules/.bin/tsc not found (run npm install)")

    # --- Go ------------------------------------------------------------------
    consider("go-vet", "lint", "go.mod present" if (root / "go.mod").is_file() else None,
             on_path("go"), ["vet", "./..."], per_file=False, missing="go not found on PATH")

    # --- Project scripts (arbitrary repo code; opt-in) ----------------------
    scripts: list[tuple[str, str, str, list[str]]] = []
    if isinstance(pkg.get("scripts"), dict) and isinstance(pkg["scripts"].get("lint"), str):
        scripts.append(("npm-lint", 'package.json "lint" script', "npm", ["run", "--silent", "lint"]))
    makefile = next((n for n in _MAKEFILES if (root / n).is_file()), None)
    if makefile and re.search(r"^lint\s*:(?!=)", _read_text(root / makefile), re.MULTILINE):
        scripts.append(("make-lint", f'"lint" target in {makefile}', "make", ["lint"]))
    if "testenv:lint" in tox_ini:
        scripts.append(("tox-lint", "[testenv:lint] in tox.ini", "tox", ["-e", "lint"]))
    for name, reason, exe, args in scripts:
        if not cfg.allow_project_scripts:
            skipped.append({"tool": name, "reason": _PROJECT_SCRIPT_SKIP})
            continue
        consider(name, "lint", reason, on_path(exe), args, per_file=False,
                 missing=f"{exe} not found on PATH", project_script=True)

    return tools, skipped


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_GENERIC_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?:(?P<col>\d+):)?\s+(?P<code>[A-Z][A-Za-z0-9_-]*\d+)\s+(?:\[\*\]\s+)?(?P<msg>.*)$")
_MYPY_RE = re.compile(
    r"^(?P<path>.+?):(?:(?P<line>\d+):)?(?:\d+:)?\s*(?P<sev>error|warning|note):\s*(?P<msg>.*?)"
    r"(?:\s+\[(?P<code>[a-z0-9_-]+)\])?\s*$")
_TSC_RE = re.compile(
    r"^(?P<path>.+?)\((?P<line>\d+),(?P<col>\d+)\):\s*(?P<sev>error|warning)\s+(?P<code>TS\d+):\s*(?P<msg>.*)$")
_TSC_GLOBAL_RE = re.compile(r"^(?:error|warning)\s+(?P<code>TS\d+):\s*(?P<msg>.*)$")
_ESLINT_UNIX_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<msg>.*?)\s*\[(?:Error|Warning)(?:/(?P<code>[^\]]+))?\]\s*$")
_PYRIGHT_TEXT_RE = re.compile(
    r"^\s*(?P<path>.+?):(?P<line>\d+):(?P<col>\d+)\s+-\s+(?P<sev>error|warning):\s*(?P<msg>.*?)(?:\s+\((?P<code>\w+)\))?\s*$")
_GO_VET_RE = re.compile(r"^(?:vet:\s*)?(?P<path>.+?\.go):(?P<line>\d+):(?:(?P<col>\d+):)?\s*(?P<msg>.*)$")
_REFORMAT_RE = re.compile(r"^\s*would reformat:?\s+(?P<path>.+?)\s*$", re.IGNORECASE)
_FORMAT_ERROR_RE = re.compile(
    r"^error:\s*(?:cannot format|failed to (?:parse|format))\s+(?P<path>.+?\.pyi?)(?::(?P<line>\d+)(?::\d+)?)?:\s*(?P<msg>.*)$",
    re.IGNORECASE)


class _Paths:
    """Normalizes tool-reported paths to repo-relative POSIX paths for a run directory."""

    def __init__(self, cwd: Path) -> None:
        bases = [cwd.absolute()]
        with contextlib.suppress(OSError):
            bases.append(cwd.resolve())
        self.bases = list(dict.fromkeys(bases))

    def rel(self, raw: object) -> str:
        s = str(raw or "").strip().strip('"').strip("'")
        if not s:
            return ""
        p = Path(s)
        if p.is_absolute():
            candidates = [p]
            with contextlib.suppress(OSError):
                candidates.append(p.resolve())
            for cand in candidates:
                for base in self.bases:
                    try:
                        return cand.relative_to(base).as_posix()
                    except ValueError:
                        continue
            return p.as_posix()
        s = s.replace("\\", "/")
        while s.startswith("./"):
            s = s[2:]
        return s


def _int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _extract_json(output: str, typ: type) -> object | None:
    opener = "[" if typ is list else "{"
    decoder = json.JSONDecoder()
    idx = output.find(opener)
    attempts = 0
    while idx != -1 and attempts < 50:
        attempts += 1
        try:
            obj, _ = decoder.raw_decode(output, idx)
        except ValueError:
            idx = output.find(opener, idx + 1)
            continue
        if isinstance(obj, typ):
            return obj
        idx = output.find(opener, idx + 1)
    return None


def _parse_lines(tool: str, output: str, paths: _Paths, regexes: list[re.Pattern[str]],
                 default_code: str = "error") -> list[Finding]:
    out: list[Finding] = []
    for line in output.splitlines():
        for rx in regexes:
            m = rx.match(line)
            if not m:
                continue
            groups = m.groupdict()
            if groups.get("sev") == "note":
                break
            code = groups.get("code") or groups.get("sev") or default_code
            out.append(Finding(tool, paths.rel(groups.get("path")), _int(groups.get("line")),
                               code, (groups.get("msg") or "").strip()))
            break
    return out


def _parse_ruff(tool: str, output: str, paths: _Paths) -> list[Finding]:
    data = _extract_json(output, list)
    if data is None:
        return _parse_lines(tool, output, paths, [_GENERIC_RE])
    out = []
    for item in data:  # type: ignore[union-attr]
        if not isinstance(item, dict):
            continue
        loc = item.get("location") or {}
        out.append(Finding(tool, paths.rel(item.get("filename")), _int(loc.get("row")),
                           str(item.get("code") or "syntax-error"), str(item.get("message") or "")))
    return out


def _parse_formatter(tool: str, output: str, paths: _Paths) -> list[Finding]:
    out = []
    for line in output.splitlines():
        m = _REFORMAT_RE.match(line)
        if m:
            out.append(Finding(tool, paths.rel(m.group("path")), None, "format", "would reformat"))
            continue
        m = _FORMAT_ERROR_RE.match(line)
        if m:
            out.append(Finding(tool, paths.rel(m.group("path")), _int(m.group("line")),
                               "parse-error", m.group("msg").strip()))
    return out


def _parse_mypy(tool: str, output: str, paths: _Paths) -> list[Finding]:
    return _parse_lines(tool, output, paths, [_MYPY_RE])


def _parse_pyright(tool: str, output: str, paths: _Paths) -> list[Finding]:
    data = _extract_json(output, dict)
    if not isinstance(data, dict) or "generalDiagnostics" not in data:
        return _parse_lines(tool, output, paths, [_PYRIGHT_TEXT_RE])
    out = []
    for diag in data.get("generalDiagnostics") or []:
        if not isinstance(diag, dict) or diag.get("severity") not in ("error", "warning"):
            continue
        start = ((diag.get("range") or {}).get("start") or {})
        line = _int(start.get("line"))
        out.append(Finding(tool, paths.rel(diag.get("file")), line + 1 if line is not None else None,
                           str(diag.get("rule") or diag.get("severity")), str(diag.get("message") or "")))
    return out


def _parse_eslint(tool: str, output: str, paths: _Paths) -> list[Finding]:
    data = _extract_json(output, list)
    if data is None:
        return _parse_lines(tool, output, paths, [_ESLINT_UNIX_RE], default_code="eslint")
    out = []
    for entry in data:  # type: ignore[union-attr]
        if not isinstance(entry, dict):
            continue
        rel = paths.rel(entry.get("filePath"))
        for msg in entry.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            code = msg.get("ruleId") or ("parse-error" if msg.get("fatal") else "eslint")
            out.append(Finding(tool, rel, _int(msg.get("line")), str(code), str(msg.get("message") or "")))
    return out


def _parse_tsc(tool: str, output: str, paths: _Paths) -> list[Finding]:
    found = _parse_lines(tool, output, paths, [_TSC_RE])
    for line in output.splitlines():
        m = _TSC_GLOBAL_RE.match(line.strip())
        if m:
            found.append(Finding(tool, "", None, m.group("code"), m.group("msg").strip()))
    return found


def _parse_go_vet(tool: str, output: str, paths: _Paths) -> list[Finding]:
    return _parse_lines(tool, output, paths, [_GO_VET_RE], default_code="vet")


def _parse_generic(tool: str, output: str, paths: _Paths) -> list[Finding]:
    return _parse_lines(tool, output, paths, [_TSC_RE, _MYPY_RE, _GENERIC_RE, _ESLINT_UNIX_RE])


_PARSERS: dict[str, Callable[[str, str, _Paths], list[Finding]]] = {
    "ruff": _parse_ruff,
    "ruff-format": _parse_formatter,
    "black": _parse_formatter,
    "flake8": _parse_generic,
    "mypy": _parse_mypy,
    "pyright": _parse_pyright,
    "eslint": _parse_eslint,
    "tsc": _parse_tsc,
    "go-vet": _parse_go_vet,
}


def parse_output(tool: str, output: str, cwd: Path) -> list[Finding]:
    """Parse a tool's output into findings with paths relative to `cwd` (the run directory)."""
    return _PARSERS.get(tool, _parse_generic)(tool, output, _Paths(Path(cwd)))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

_MAX_OUTPUT = 2_000_000
_TAIL = 2000


@dataclass
class _Run:
    findings: list[Finding]
    output: str
    returncode: int | None
    duration: float
    skip_reason: str | None = None
    crashed: bool = False


def _py_syntax(cwd: Path, files: list[str]) -> list[Finding]:
    out: list[Finding] = []
    for rel in files:
        if not rel.lower().endswith(".py"):
            continue
        try:
            source = (cwd / rel).read_bytes()
        except OSError:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                compile(source, rel, "exec", dont_inherit=True)
            except SyntaxError as exc:
                out.append(Finding("py-syntax", rel, exc.lineno, type(exc).__name__, exc.msg or str(exc)))
            except (ValueError, TypeError) as exc:
                out.append(Finding("py-syntax", rel, None, type(exc).__name__, str(exc)))
            except (RecursionError, MemoryError):
                out.append(Finding("py-syntax", rel, None, "RecursionError", "source is too deeply nested to compile"))
    return out


def _safe_arg(rel: str) -> str:
    return f"./{rel}" if rel.startswith("-") else rel


def _execute(tool: ToolSpec, cwd: Path, files: list[str], timeout: int, prefix: list[str] | None) -> _Run:
    cmd = list(tool.command)
    if tool.per_file:
        cmd += [_safe_arg(f) for f in files]
    res = run_command(cmd, cwd, timeout=timeout, prefix=prefix, max_output=_MAX_OUTPUT)
    if res.not_found:
        return _Run([], res.output, None, res.duration_seconds, skip_reason=f"command not found: {cmd[0]}")
    if res.timed_out:
        return _Run([], res.output, None, res.duration_seconds, skip_reason=f"timed out after {timeout}s")
    findings = parse_output(tool.name, res.output, cwd)
    crashed = False
    if res.returncode != 0 and not findings:
        crashed = True
        findings = [Finding(tool.name, "", None, f"exit-{res.returncode}",
                            f"{tool.name} exited with code {res.returncode} and no parseable findings")]
    return _Run(findings, res.output, res.returncode, res.duration_seconds, crashed=crashed)


def _new_findings(before: list[Finding], after: list[Finding]) -> list[Finding]:
    remaining = Counter(f.key() for f in before)
    new: list[Finding] = []
    for f in after:
        k = f.key()
        if remaining[k] > 0:
            remaining[k] -= 1
        else:
            new.append(f)
    return new


def _tail(text: str) -> str:
    return text[-_TAIL:] if text else ""


def _skip(tool: ToolSpec, reason: str, duration: float = 0.0, output: str = "") -> ValidatorResult:
    return ValidatorResult(tool.name, tool.kind, ran=False, ok=True, skipped_reason=reason,
                           output_tail=_tail(output), duration_seconds=round(duration, 3))


def _compare(tool: ToolSpec, before: list[Finding], after: list[Finding], output: str,
             duration: float) -> ValidatorResult:
    new = _new_findings(before, after)
    return ValidatorResult(tool.name, tool.kind, ran=True, ok=not new, new_findings=new,
                           baseline_count=len(before), after_count=len(after),
                           output_tail=_tail(output), duration_seconds=round(duration, 3))


def _run_one(tool: ToolSpec, root: Path, base_dir: Path | None, base_error: str | None,
             files: list[str], timeout: int, prefix: list[str] | None) -> ValidatorResult:
    if tool.name == "py-syntax":
        start = time.monotonic()
        after = _py_syntax(root, files)
        before: list[Finding] = []
        note = ""
        if base_dir is not None:
            before = _py_syntax(base_dir, [f for f in files if (base_dir / f).is_file()])
        else:
            note = f"baseline unavailable ({base_error}); every finding counts as new\n"
        text = note + "\n".join(f"{f.path}:{f.line}: {f.code} {f.message}" for f in after)
        return _compare(tool, before, after, text, time.monotonic() - start)

    if base_dir is None:
        return _skip(tool, base_error or "baseline worktree unavailable")

    after_run = _execute(tool, root, files, timeout, prefix)
    if after_run.skip_reason:
        return _skip(tool, after_run.skip_reason, after_run.duration, after_run.output)

    duration = after_run.duration
    if tool.per_file:
        base_files = [f for f in files if (base_dir / f).is_file()]
        if not base_files:  # every changed file is new: everything is new
            return _compare(tool, [], after_run.findings, after_run.output, duration)
    else:
        base_files = []
    base_run = _execute(tool, base_dir, base_files, timeout, prefix)
    duration += base_run.duration
    if base_run.skip_reason:
        return _skip(tool, f"baseline run: {base_run.skip_reason}", duration, after_run.output)
    if base_run.crashed and not after_run.crashed:
        return _skip(tool, f"baseline run failed (exit {base_run.returncode}), so new findings cannot be "
                           f"told apart from pre-existing ones", duration,
                     f"--- baseline output ---\n{_tail(base_run.output)}")
    return _compare(tool, base_run.findings, after_run.findings, after_run.output, duration)


def _make_temp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="fix-issue-base-"))


def _is_dir_link(path: Path) -> bool:
    if os.path.islink(path):
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _link_dir(src: Path, dst: Path) -> bool:
    try:
        os.symlink(src, dst, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        try:
            import _winapi  # type: ignore[import-not-found]
            _winapi.CreateJunction(str(src), str(dst))
            return True
        except (ImportError, AttributeError, OSError):
            pass
    return False


def _unlink_dir_link(path: Path) -> None:
    """Remove a symlink/junction WITHOUT touching its target; raises OSError if that's not possible."""
    if not _is_dir_link(path):
        raise OSError(f"{path} is not a link")
    try:
        os.rmdir(path) if os.name == "nt" else os.unlink(path)
    except OSError:
        os.unlink(path) if os.name == "nt" else os.rmdir(path)


@contextlib.contextmanager
def _baseline_worktree(root: Path, base_ref: str, link_env: bool) -> Iterator[tuple[Path | None, str | None]]:
    """Yield (worktree path, None) or (None, error). The worktree is always removed afterwards."""
    tmp = _make_temp_dir()
    wt = tmp / "base"
    links: list[Path] = []
    try:
        if not base_ref or base_ref.startswith("-"):
            yield None, f"invalid base ref {base_ref!r}"
            return
        hooks = tmp / "no-hooks"
        hooks.mkdir(exist_ok=True)
        res = run_command(["git", "-c", f"core.hooksPath={hooks}", "worktree", "add", "--detach",
                           str(wt), base_ref], cwd=root, timeout=300)
        if not res.ok:
            yield None, f"could not create baseline worktree at {base_ref}: {res.output.strip()[-300:]}"
            return
        if link_env:
            src, dst = root / "node_modules", wt / "node_modules"
            if src.is_dir() and not _is_dir_link(src) and not dst.exists() and _link_dir(src.resolve(), dst):
                links.append(dst)
        yield wt, None
    finally:
        unlinked = True
        for link in links:
            try:
                _unlink_dir_link(link)
            except OSError:
                unlinked = False
        if wt.exists() and unlinked:
            run_command(["git", "worktree", "remove", "--force", str(wt)], cwd=root, timeout=120)
        if wt.exists():
            # rmtree does not follow symlinks/junctions (Python >= 3.8), so a link left behind is safe.
            shutil.rmtree(wt, ignore_errors=True)
        run_command(["git", "worktree", "prune"], cwd=root, timeout=60)
        shutil.rmtree(tmp, ignore_errors=True)


def _clean_changed(files: list[str]) -> list[str]:
    out: list[str] = []
    for f in files:
        p = str(f).strip().strip('"').replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        if not p or p.startswith("/") or re.match(r"^[A-Za-z]:", p) or ".." in p.split("/"):
            continue
        if p not in out:
            out.append(p)
    return out


def run_validators(root: Path, tools: list[ToolSpec], changed_files: list[str], base_ref: str = "HEAD",
                   timeout: int = 120, prefix: list[str] | None = None) -> list[ValidatorResult]:
    """Run `tools` diff-aware on `changed_files`; see the module docstring. Results follow `tools` order."""
    root = Path(root)
    changed = _clean_changed(changed_files)
    results: list[ValidatorResult | None] = []
    plans: list[tuple[int, ToolSpec, list[str]]] = []
    for idx, tool in enumerate(tools):
        exts = _TOOL_EXTENSIONS.get(tool.name)
        relevant = [f for f in changed if exts is None or f.lower().endswith(exts)]
        existing = [f for f in relevant if (root / f).is_file()]
        if (tool.per_file and not existing) or (not tool.per_file and not relevant):
            results.append(_skip(tool, "no relevant changed files"))
        else:
            results.append(None)
            plans.append((idx, tool, existing))

    if plans:
        link_env = any(t.name in _NODE_TOOLS or t.project_script for _, t, _ in plans)
        with _baseline_worktree(root, base_ref, link_env) as (base_dir, base_error):
            for idx, tool, files in plans:
                try:
                    results[idx] = _run_one(tool, root, base_dir, base_error, files, timeout, prefix)
                except Exception as exc:  # one broken tool/parser must never abort validation
                    results[idx] = _skip(tool, f"internal error while running {tool.name}: {exc!r}")
    return [r for r in results if r is not None]
