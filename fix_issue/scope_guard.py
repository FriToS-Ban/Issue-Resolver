"""
scope_guard.py — Enforce scope limits BEFORE running any tests or opening a PR.

Two entry points
----------------
``check(diff_text, repo_root, max_diff_lines)``  (legacy, all-or-nothing)
    Raises ``ScopeViolation`` if the diff touches a path listed in the target repo's
    ``.fix-issue-ignore`` or has more than ``max_diff_lines`` raw +/- lines. The CLI turns
    that into a "needs review — too large / out of scope" outcome without burning a test run.

``evaluate(diff_text, repo_root, ...) -> ScopeReport``  (path- and severity-aware)
    Classifies every touched path and turns every concern into a ``ScopeFinding``:

      block   the patch must not be proposed
      review  the patch may be proposed, but only flagged for human review
      info    recorded in the run log only

    ``report.ok`` is False when any finding blocks; ``report.requires_review`` is True when
    any finding asks for review.

Default policy
--------------
==========  ==============================================================  ==================================
category    default patterns                                                outcome
==========  ==============================================================  ==================================
unsafe_path any component that is ``.git`` (any case), ends in ``.`` or a    block (hard-coded; repo rules
            space, looks like an 8.3 short name (``GIT~1``), contains ``:``,  cannot override it)
            ``..``, control or invisible format characters
policy      ``.fix-issue-scope.toml`` / ``.fix-issue-ignore`` (any level)    block (hard-coded)
ignored     ``.fix-issue-ignore`` (fnmatch on the full path or basename)     block
secrets     .env .env.* *.pem *.key **/secrets/** *credentials* .npmrc       block
            .pypirc .netrc .git-credentials .yarnrc(.yml) pip.conf
ci          .github/** .gitlab-ci.yml .gitlab/** .circleci/** Jenkinsfile   block
            .travis.yml azure-pipelines.yml .pre-commit-config.yaml
            CODEOWNERS bitbucket-pipelines.yml .drone.yml appveyor.yml ...
git_metadata .gitattributes .gitmodules                                     block
lockfile    poetry.lock package-lock.json yarn.lock pnpm-lock.yaml          block
            Pipfile.lock uv.lock Cargo.lock go.sum
migrations  **/migrations/** **/alembic/** **/*migration*                   new file: review;
                                                                            modify/delete/rename existing: block
tests       tests/** **/tests/** test_*.py *_test.py conftest.py            size weight 0.5
            **/__tests__/** *.spec.* *.test.*
docs        docs/** *.md *.rst                                              size weight 0.25
auth        **/auth/** **/*auth*.py **/permission* **/oauth/**              review, and > 20 changed lines
            **/security/** **/login*                                        in the category: block
payments    **/payment*/** **/payment* **/billing/** **/stripe/**           review, and > 20 changed lines
            **/invoice*                                                     in the category: block
build       setup.py setup.cfg pyproject.toml tox.ini noxfile.py Makefile   review
            package.json MANIFEST.in requirements*.txt Pipfile ...
source      everything else                                                 size weight 1.0
public_api  non-test Python, when base/new contents are passed: removed     review (names the symbol)
            public function/class/method, changed parameter list (names,
            order, removed params/defaults, new required params), changed
            ``__all__``, changed package ``__init__.py`` exports. Body-only
            changes (and changed default *values*) are not API changes.
deletion    deleted or renamed files                                        review
binary      ``GIT binary patch`` / ``Binary files ... differ``              block
special_file symlink (mode 120000) or submodule gitlink (160000) entry      block
size        effective lines = sum(changed lines x weight) > limit           block
size        effective changed characters > limit x 200 (one huge line must  block
            not slip under a line budget)
triage      path matches ``ScopeConstraints.forbidden_paths`` (tests exempt) block
triage      path outside a non-empty ``allowed_paths`` (tests/docs exempt)  review (triage only saw the
                                                                            issue text, so it cannot block)
==========  ==============================================================  ==================================

Rules are tried in order and the first match decides a path's category, so a test or doc
*about* auth (``tests/test_auth.py``, ``docs/auth.md``) is a test/doc, while a key file under
``tests/`` is still a secret. For renames both the old and the new path are classified and
the stricter rule wins. The size limit is ``min(max_diff_lines, constraints.max_diff_lines)``:
triage can tighten it, never loosen it.

Pattern syntax (rules and triage paths): gitignore-like, case-insensitive, matched against
the repo-relative POSIX path. A pattern without ``/`` matches any single path component
(``*.pem``, ``Jenkinsfile``); a pattern with ``/`` is anchored at the repo root; ``**/``
matches any number of directories; ``*`` and ``?`` do not cross ``/``; a pattern that
matches a directory matches everything below it. (``.fix-issue-ignore`` keeps its original
fnmatch semantics.)

Repository overrides
--------------------
A target repo may add rules in ``.fix-issue-scope.toml``::

    [[rule]]
    category  = "generated"
    patterns  = ["src/generated/**"]
    severity  = "block"     # block | review | info  (default: info)
    max_lines = 10          # optional changed-line budget for the category (over -> block)
    weight    = 1.0         # optional size weight

Repo rules are tried before the defaults, so a repo may also re-classify its own paths;
``.fix-issue-ignore`` and triage constraints always apply on top. Problems in the file are
reported as ``info`` findings (category ``config``) and the defaults still apply.

Where the policy is read from
-----------------------------
The agent must not be able to rewrite the policy that judges it, so neither policy file is
ever read from the *working tree* of a git checkout (an earlier attempt, or test code the
model wrote, may have changed it). ``load_policy`` / ``evaluate`` take each file, in order,
from: an explicit ``policy=`` snapshot; the caller's ``base_contents`` (the orchestrator
computes those from the base commit); ``git cat-file blob HEAD:<file>`` (the base commit,
since edits stay uncommitted until the PR is pushed). Only a directory that is not a git
checkout at all (no ``.git``) falls back to the working-tree files. ``.fix-issue-ignore``
patterns found in the working tree are *added* to the base ones, since they can only make
the policy stricter. Edits to either policy file are always blocked (category ``policy``).
"""

from __future__ import annotations

import ast
import fnmatch
import functools
import os
import re
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


class ScopeViolation(Exception):
    """Raised when a diff violates scope limits. Message is user-readable."""

    def __init__(self, message: str = "", report: ScopeReport | None = None) -> None:
        super().__init__(message)
        self.report = report


def check(
    diff_text: str,
    repo_root: Path,
    max_diff_lines: int = 150,
) -> None:
    """
    Run all scope checks against `diff_text`.

    Parameters
    ----------
    diff_text:      Unified diff string to validate.
    repo_root:      Root of the temp clone (used to find .fix-issue-ignore).
    max_diff_lines: Maximum number of ±-prefixed lines allowed in the diff.

    Raises
    ------
    ScopeViolation  — with a human-readable explanation.
    """
    _check_diff_size(diff_text, max_diff_lines)
    _check_protected_paths(diff_text)
    _check_ignored_paths(diff_text, repo_root)


# ---------------------------------------------------------------------------
# Size check
# ---------------------------------------------------------------------------

def _check_diff_size(diff_text: str, max_diff_lines: int) -> None:
    """Count lines that add or remove code (+ or - prefix, excluding --- and +++ headers)."""
    change_lines = [
        line for line in diff_text.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("---")
        and not line.startswith("+++")
    ]
    actual = len(change_lines)
    if actual > max_diff_lines:
        raise ScopeViolation(
            f"Diff is too large: {actual} changed lines (limit is {max_diff_lines}). "
            f"This issue likely requires a larger refactor — needs human review. "
            f"Increase FIX_ISSUE_MAX_DIFF_LINES in ~/.fix-issue.toml to override."
        )


# ---------------------------------------------------------------------------
# Ignored paths check
# ---------------------------------------------------------------------------

_DIFF_FILE_RE = re.compile(r"^(?:---|\+\+\+) (?:a/|b/)(.+)$", re.MULTILINE)


def _parse_changed_paths(diff_text: str) -> list[str]:
    """Extract file paths from `--- a/...` and `+++ b/...` diff headers."""
    paths: set[str] = set()
    for m in _DIFF_FILE_RE.finditer(diff_text):
        p = m.group(1).strip()
        if p != "/dev/null":
            paths.add(p)
    return sorted(paths)


IGNORE_FILE = ".fix-issue-ignore"


def _ignore_patterns_from_text(text: str | None) -> list[str]:
    """Lines starting with # are comments. Blank lines are skipped."""
    if not text:
        return []
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_ignore_patterns(repo_root: Path) -> list[str]:
    """
    Read .fix-issue-ignore from the repo root's WORKING TREE.
    Only used directly for non-git directories and to add (never remove) patterns on top of
    the base commit's file; see ``load_policy``.
    """
    ignore_file = Path(repo_root) / IGNORE_FILE
    try:
        if not ignore_file.is_file():
            return []
        return _ignore_patterns_from_text(ignore_file.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return []


def _check_protected_paths(diff_text: str) -> None:
    """Legacy check: unsafe (alias) paths and the scope-policy files are never editable."""
    for path in _parse_changed_paths(diff_text):
        reason = protected_path_reason(path)
        if reason:
            raise ScopeViolation(f"Diff touches '{path}': {reason}. This path is off-limits for the agent.")


def _check_ignored_paths(diff_text: str, repo_root: Path) -> None:
    patterns = load_policy(repo_root).ignore_patterns
    if not patterns:
        return

    changed_paths = _parse_changed_paths(diff_text)
    for path in changed_paths:
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path.split("/")[-1], pattern):
                raise ScopeViolation(
                    f"Diff touches '{path}' which matches the ignored pattern '{pattern}' "
                    f"in .fix-issue-ignore. This path is off-limits for the agent."
                )


# ===========================================================================
# Path- and severity-aware evaluation
# ===========================================================================

SEVERITIES = ("info", "review", "block")
_RANK = {s: i for i, s in enumerate(SEVERITIES)}
SCOPE_CONFIG_FILE = ".fix-issue-scope.toml"
POLICY_FILES = (SCOPE_CONFIG_FILE, IGNORE_FILE)
DEFAULT_CATEGORY = "source"
_EXEMPT_FROM_ALLOWED = ("tests", "docs")
# Triage only saw the issue text; the implementer is told to add a regression test, so a
# triage-forbidden glob such as "tests/**" must not block test files. Hard-coded blocks,
# .fix-issue-ignore and repo/default rules still apply to them.
_EXEMPT_FROM_FORBIDDEN = ("tests",)
# Character budget per allowed (effective) changed line: the line limit alone lets one
# 350 KB line through a 10-line cap. 200 is ~2.5x a long source line.
CHARS_PER_LINE_BUDGET = 200


@dataclass
class ScopeConstraints:
    """Issue-specific scope suggestions (from triage). They can only tighten the policy."""
    max_diff_lines: int | None = None
    forbidden_paths: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=list)
    source: str = "triage"


@dataclass
class ScopeFinding:
    severity: str        # "block" | "review" | "info"
    category: str        # secrets | ci | lockfile | auth | payments | migrations | public_api | tests
                         # | docs | source | ignored | size | triage | binary | deletion | config
    path: str | None
    message: str


@dataclass
class ScopeReport:
    ok: bool                     # no "block" findings
    requires_review: bool        # any "review" finding
    findings: list[ScopeFinding]
    raw_lines: int               # changed +/- lines
    effective_lines: float       # weighted by category
    limit: int                   # effective limit used
    categories: dict[str, str]   # path -> category
    raw_chars: int = 0           # characters on changed +/- lines (without the +/- marker)
    effective_chars: float = 0.0 # weighted like effective_lines
    char_limit: int = 0          # limit x CHARS_PER_LINE_BUDGET

    def to_dict(self) -> dict:
        data = asdict(self)
        data["effective_lines"] = round(self.effective_lines, 2)
        data["effective_chars"] = round(self.effective_chars, 2)
        return data

    def blocking_message(self) -> str:
        return "\n".join(f.message for f in self.findings if f.severity == "block")

    def review_message(self) -> str:
        return "\n".join(f.message for f in self.findings if f.severity == "review")


@dataclass
class PathRule:
    category: str
    patterns: list[str]
    severity: str = "info"           # block | review | info
    max_lines: int | None = None     # changed-line budget for the whole category
    weight: float = 1.0              # size weight
    source: str = "default"          # "default" | ".fix-issue-scope.toml"

    def match(self, path: str) -> str | None:
        """Return the first of this rule's patterns that matches `path`, else None."""
        for pattern in self.patterns:
            if path_matches(path, pattern):
                return pattern
        return None


# Severity model for the built-in rules (first match wins, so order matters):
#   block  — files that act with more authority than the code under review, where a fix to an
#            issue never needs the agent's hand: credentials and registry/auth config (.npmrc,
#            .pypirc, ... can carry tokens or redirect package installs), CI and repository
#            automation (everything under .github/ — workflows, composite actions, CODEOWNERS,
#            dependabot, issue forms — plus other CI systems and pre-commit hooks, which run
#            code on CI runners and contributors' machines, often with secrets), git metadata
#            (.gitattributes filters/diff drivers, .gitmodules URLs change what git checks out
#            and shows in review), and lockfiles.
#   review — build/packaging/dependency manifests (setup.py, pyproject.toml, package.json,
#            Makefile, tox.ini, requirements*.txt, ...): a legitimate fix sometimes has to touch
#            them (a missing dependency, a test-config tweak), but they run at install/build
#            time and change the supply chain, so a human must look. Listed after tests/docs so a
#            fixture such as tests/fixtures/package.json stays a test file.
DEFAULT_RULES: tuple[PathRule, ...] = (
    PathRule("secrets", [".env", ".env.*", "*.pem", "*.key", "**/secrets/**", "*credentials*",
                         ".npmrc", ".pypirc", ".netrc", "_netrc", ".git-credentials", ".yarnrc",
                         ".yarnrc.yml", "pip.conf", "pip.ini"],
             "block"),
    PathRule("ci", [".github/**", ".gitlab-ci.yml", ".gitlab/**", ".circleci/**", "Jenkinsfile",
                    ".travis.yml", "azure-pipelines.yml", "azure-pipelines.yaml", ".azure-pipelines/**",
                    "bitbucket-pipelines.yml", ".drone.yml", ".buildkite/**", "appveyor.yml",
                    ".appveyor.yml", ".pre-commit-config.yaml", ".pre-commit-hooks.yaml", "CODEOWNERS"],
             "block"),
    PathRule("git_metadata", [".gitattributes", ".gitmodules"], "block"),
    PathRule("lockfile", ["poetry.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
                          "Pipfile.lock", "uv.lock", "Cargo.lock", "go.sum"], "block"),
    # Severity is decided per file: new migration -> review, existing one touched -> block.
    PathRule("migrations", ["**/migrations/**", "**/alembic/**", "**/*migration*"], "review"),
    PathRule("tests", ["tests/**", "**/tests/**", "test_*.py", "*_test.py", "conftest.py",
                       "**/__tests__/**", "*.spec.*", "*.test.*"], "info", weight=0.5),
    PathRule("docs", ["docs/**", "*.md", "*.rst"], "info", weight=0.25),
    PathRule("auth", ["**/auth/**", "**/*auth*.py", "**/permission*", "**/oauth/**",
                      "**/security/**", "**/login*"], "review", max_lines=20),
    PathRule("payments", ["**/payment*/**", "**/payment*", "**/billing/**", "**/stripe/**",
                          "**/invoice*"], "review", max_lines=20),
    PathRule("build", ["setup.py", "setup.cfg", "pyproject.toml", "tox.ini", "noxfile.py", "Makefile",
                       "GNUmakefile", "package.json", "MANIFEST.in", "requirements*.txt", "Pipfile",
                       "Dockerfile", "CMakeLists.txt", "Cargo.toml", "go.mod", "Gemfile", "*.gemspec",
                       "pom.xml", "build.gradle", "build.gradle.kts"], "review"),
)

_CATEGORY_TEXT = {
    "secrets": ("a secrets/credentials file", "The agent must never modify secrets."),
    "ci": ("CI / repository automation configuration",
           "CI and automation changes are off-limits for the agent."),
    "git_metadata": ("git metadata (attributes / submodules)",
                     "It changes how git checks out and shows the repository; off-limits for the agent."),
    "lockfile": ("a dependency lockfile",
                 "Lockfiles must be regenerated by maintainers, not edited by the agent."),
    "auth": ("authentication/authorization code",
             "Security-sensitive changes always need human review."),
    "payments": ("payments/billing code", "Payment changes always need human review."),
    "build": ("a build/packaging/dependency file",
              "Build and dependency changes run at install time and need human review."),
}


# ---------------------------------------------------------------------------
# Path aliasing (hard-coded, not overridable by repository rules)
# ---------------------------------------------------------------------------

# Windows 8.3 short names ("GIT~1", "GITHUB~1", "WORKFL~1", "PRE-CO~1.YAM"): on NTFS they open
# the long-named file or directory, so they can smuggle ".git" / ".github" past name checks.
_SHORT_NAME_RE = re.compile(r"^[^~./\\]{1,6}~[0-9]+(?:\.[^.~]{0,3})?$")
# Code points HFS+ ignores in file names (".g\u200cit" opens ".git" on macOS), as in git's
# core.protectHFS.
_IGNORABLE_RE = re.compile("[\u200c-\u200f\u202a-\u202e\u2066-\u2069\u206a-\u206f\ufeff]")


def unsafe_component_reason(part: str) -> str | None:
    """
    Why one path component could alias something the agent must not write, or None.

    Same idea as git's core.protectNTFS/protectHFS: Windows strips trailing dots and spaces
    (".git." and ".git " are ".git"), 8.3 short names alias long names, ':' selects an NTFS
    alternate data stream (".git::$INDEX_ALLOCATION" is the .git directory), and macOS ignores
    some invisible code points. Rejecting these outright is simpler and safer than normalising.
    """
    if any(ord(c) < 32 or ord(c) == 127 for c in part):
        return "the path contains control characters"
    if part == "..":
        return "'..' components are not allowed (the path must stay inside the repository)"
    if _IGNORABLE_RE.search(part):
        return "the path contains invisible Unicode format characters (macOS ignores them, so the name can alias another file)"
    if ":" in part:
        return "':' is not allowed in paths"
    if part != part.rstrip(". "):
        return (f"path component {part!r} ends with '.' or a space (Windows strips those, so it would "
                "alias another file)")
    if part.lower() == ".git":
        return "paths inside .git/ are not allowed"
    if _SHORT_NAME_RE.match(part):
        return (f"path component {part!r} looks like a Windows 8.3 short name (e.g. GIT~1), which can "
                "alias another file or directory")
    return None


def unsafe_path_reason(path: str) -> str | None:
    """First ``unsafe_component_reason`` of any component of a repo-relative path, or None."""
    for part in path.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        reason = unsafe_component_reason(part)
        if reason:
            return reason
    return None


def is_policy_file(path: str) -> bool:
    """True for the scope-policy files (``.fix-issue-scope.toml`` / ``.fix-issue-ignore``) anywhere."""
    name = _norm_path(path).rsplit("/", 1)[-1].lower()
    return any(name == p.lower() for p in POLICY_FILES)


def protected_path_reason(path: str) -> str | None:
    """Reason a path is hard-blocked regardless of any rule (alias paths, policy files), or None."""
    reason = unsafe_path_reason(path)
    if reason:
        return reason
    if is_policy_file(path):
        return ("it is a scope-policy file (.fix-issue-scope.toml / .fix-issue-ignore); the agent may "
                "never change the rules that judge its own patch")
    return None


def _hard_category(path: str) -> str | None:
    if unsafe_path_reason(path):
        return "unsafe_path"
    if is_policy_file(path):
        return "policy"
    return None


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def _norm_path(path: str) -> str:
    p = path.strip().replace("\\", "/")
    if len(p) >= 2 and p[0] == p[-1] == '"':
        p = p[1:-1]
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


@functools.lru_cache(maxsize=2048)
def _compile_glob(pattern: str) -> re.Pattern[str]:
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
            continue
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        c = pattern[i]
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[" and pattern.find("]", i + 2) != -1:
            j = pattern.find("]", i + 2)
            body = pattern[i + 1:j]
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append("[" + body.replace("\\", "\\\\") + "]")
            i = j + 1
            continue
        else:
            out.append(re.escape(c))
        i += 1
    try:
        return re.compile("".join(out) + r"\Z", re.IGNORECASE | re.DOTALL)
    except re.error:
        return re.compile(re.escape(pattern) + r"\Z", re.IGNORECASE)


def path_matches(path: str, pattern: str) -> bool:
    """gitignore-like match of a repo-relative path (see module docstring)."""
    path = _norm_path(path)
    pat = pattern.strip().replace("\\", "/")
    while pat.startswith("./"):
        pat = pat[2:]
    pat = pat.strip("/")
    if not path or not pat:
        return False
    rx = _compile_glob(pat)
    parts = [p for p in path.split("/") if p]
    if "/" not in pat:
        return any(rx.match(part) for part in parts)
    # Anchored: the pattern may match the path itself or any parent directory of it.
    return any(rx.match("/".join(parts[:k])) for k in range(1, len(parts) + 1))


def _match_rule(path: str, rules: list[PathRule] | tuple[PathRule, ...]) -> tuple[PathRule, str] | None:
    for rule in rules:
        pattern = rule.match(path)
        if pattern is not None:
            return rule, pattern
    return None


def classify_path(path: str, rules: list[PathRule] | None = None) -> str:
    """Category of `path` under `rules` (default: the built-in rules).

    Alias paths (``unsafe_path``) and the policy files (``policy``) are classified before any
    rule, so no rule can make them look like ordinary source."""
    hard = _hard_category(path)
    if hard:
        return hard
    hit = _match_rule(path, DEFAULT_RULES if rules is None else rules)
    return hit[0].category if hit else DEFAULT_CATEGORY


def _ignored_match(path: str, patterns: list[str]) -> str | None:
    """Same semantics as the legacy .fix-issue-ignore check."""
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path.split("/")[-1], pattern):
            return pattern
    return None


def _first_match(path: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        if path_matches(path, pattern):
            return pattern
    return None


# ---------------------------------------------------------------------------
# Rules loading
# ---------------------------------------------------------------------------

@dataclass
class ScopePolicy:
    """The repository's scope policy as of the base commit: repo rules + defaults, ignore patterns."""
    rules: list[PathRule]
    ignore_patterns: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)    # reported as info findings
    read_error: str | None = None                        # policy could not be read: review finding
    source: str = "defaults"                             # explicit | base_contents | git:<ref> | working tree


class _PolicyReadError(Exception):
    pass


# Environment variables that would make git read a different repository / object store.
_GIT_REDIRECT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                     "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE",
                     "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT")


def _git_base_files(repo_root: Path, ref: str, names: list[str]) -> dict[str, str | None]:
    """Contents of `names` at commit `ref` (None = absent there), via one ``git cat-file --batch``.

    Reads blobs only (no textconv/filters), ignores ``refs/replace`` and never discovers a
    parent repository (GIT_CEILING_DIRECTORIES). Raises _PolicyReadError when git fails."""
    env = {k: v for k, v in os.environ.items() if k not in _GIT_REDIRECT_ENV}
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        env["GIT_CEILING_DIRECTORIES"] = str(Path(repo_root).resolve().parent)
        proc = subprocess.run(
            ["git", "cat-file", "--batch"], cwd=str(repo_root), env=env, capture_output=True,
            input="".join(f"{ref}:{name}\n" for name in names).encode("utf-8"), timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _PolicyReadError(str(exc)) from exc
    if proc.returncode != 0:
        raise _PolicyReadError((proc.stderr or b"").decode("utf-8", "replace").strip()[:300]
                               or f"git exited with {proc.returncode}")
    data, pos, out = proc.stdout, 0, {}
    try:
        for name in names:
            nl = data.index(b"\n", pos)
            header = data[pos:nl].decode("utf-8", "replace").split()
            pos = nl + 1
            if len(header) == 3 and header[2].isdigit():      # "<oid> <type> <size>"
                size = int(header[2])
                out[name] = data[pos:pos + size].decode("utf-8", "replace") if header[1] == "blob" else None
                pos += size + 1
            else:                                              # "<ref>:<name> missing" / ambiguous
                out[name] = None
    except ValueError as exc:
        raise _PolicyReadError(f"unexpected git cat-file output ({exc})") from exc
    return out


def _policy_texts(repo_root: Path, ref: str, base_contents: dict[str, str | None] | None
                  ) -> tuple[dict[str, str | None], str, str | None]:
    """(file name -> base text or None, source label, read error) for both policy files."""
    texts: dict[str, str | None] = {}
    sources: list[str] = []
    if base_contents:
        normalised = {_norm_path(k): v for k, v in base_contents.items()}
        for name in POLICY_FILES:
            if name in normalised:
                texts[name] = normalised[name]
        if texts:
            sources.append("base_contents")
    missing = [name for name in POLICY_FILES if name not in texts]
    error: str | None = None
    if missing:
        if not (repo_root / ".git").exists():
            # Not a git checkout (e.g. a plain directory in unit tests): no base commit to read.
            for name in missing:
                p = repo_root / name
                try:
                    texts[name] = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
                except OSError:
                    texts[name] = None
            sources.append("working tree")
        else:
            try:
                texts.update(_git_base_files(repo_root, ref, missing))
                sources.append(f"git:{ref}")
            except _PolicyReadError as exc:
                for name in missing:
                    texts[name] = None
                error = (f"Could not read {' / '.join(missing)} from the base commit ({exc}); repository "
                         f"scope rules were NOT applied (the working-tree copies are never trusted).")
    return texts, "+".join(sources) or "defaults", error


def load_policy(repo_root: Path, ref: str = "HEAD",
                base_contents: dict[str, str | None] | None = None) -> ScopePolicy:
    """
    The scope policy of `repo_root` as of commit `ref` (see "Where the policy is read from").

    Call it once right after the workspace is created (before any repository or model code
    runs) and pass the result to ``evaluate(..., policy=...)`` to pin the policy for a run.
    """
    repo_root = Path(repo_root)
    texts, source, error = _policy_texts(repo_root, ref, base_contents)
    rules, problems = _rules_from_text(texts.get(SCOPE_CONFIG_FILE))
    ignore = _ignore_patterns_from_text(texts.get(IGNORE_FILE))
    if source != "working tree":
        # Working-tree ignore patterns can only add restrictions, so honouring them is safe
        # (and keeps an uncommitted .fix-issue-ignore effective).
        ignore += [p for p in _load_ignore_patterns(repo_root) if p not in ignore]
    return ScopePolicy(rules, ignore, problems, error, source)


def load_rules(repo_root: Path) -> list[PathRule]:
    """Repo rules from `.fix-issue-scope.toml` (if any) followed by the defaults."""
    return load_policy(Path(repo_root)).rules


def _load_rules_with_warnings(repo_root: Path) -> tuple[list[PathRule], list[str]]:
    policy = load_policy(Path(repo_root))
    return policy.rules, policy.problems


def _rules_from_text(text: str | None) -> tuple[list[PathRule], list[str]]:
    defaults = [replace(r, patterns=list(r.patterns)) for r in DEFAULT_RULES]
    if text is None:
        return defaults, []
    if tomllib is None:  # pragma: no cover
        return defaults, [f"{SCOPE_CONFIG_FILE} ignored: no TOML parser available."]
    try:
        raw = tomllib.loads(text)
    except (ValueError, RecursionError) as exc:
        return defaults, [f"Could not parse {SCOPE_CONFIG_FILE} ({exc}); using default scope rules only."]

    entries = raw.get("rule", [])
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return defaults, [f"{SCOPE_CONFIG_FILE}: 'rule' must be an array of tables ([[rule]])."]

    custom: list[PathRule] = []
    problems: list[str] = []
    for idx, entry in enumerate(entries, 1):
        rule, entry_problems = _rule_from_toml(entry, idx)
        problems.extend(entry_problems)
        if rule is not None:
            custom.append(rule)
    return custom + defaults, problems


def _rule_from_toml(entry: object, idx: int) -> tuple[PathRule | None, list[str]]:
    where = f"{SCOPE_CONFIG_FILE} rule #{idx}"
    if not isinstance(entry, dict):
        return None, [f"{where} is not a table; skipped."]
    category = entry.get("category")
    if not isinstance(category, str) or not category.strip():
        return None, [f"{where} has no 'category'; skipped."]
    patterns = entry.get("patterns", entry.get("pattern"))
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns) or not patterns:
        return None, [f"{where} ('{category}') has no valid 'patterns'; skipped."]

    problems: list[str] = []
    severity = entry.get("severity", "info")
    if severity not in SEVERITIES:
        problems.append(f"{where} ('{category}') has invalid severity {severity!r}; using 'review'.")
        severity = "review"
    max_lines = entry.get("max_lines")
    if max_lines is not None and (isinstance(max_lines, bool) or not isinstance(max_lines, int) or max_lines < 0):
        problems.append(f"{where} ('{category}') has invalid max_lines {max_lines!r}; ignored.")
        max_lines = None
    weight = entry.get("weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        problems.append(f"{where} ('{category}') has invalid weight {weight!r}; using 1.0.")
        weight = 1.0
    rule = PathRule(category.strip(), list(patterns), severity, max_lines, float(weight),
                    source=SCOPE_CONFIG_FILE)
    return rule, problems


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------

@dataclass
class _FileChange:
    old_path: str | None = None
    new_path: str | None = None
    added: int = 0
    removed: int = 0
    chars: int = 0               # characters on +/- lines (without the marker)
    is_new: bool = False
    is_deleted: bool = False
    is_binary: bool = False
    is_special: bool = False     # symlink (120000) or submodule gitlink (160000) entry
    is_rename: bool = False
    git_header: bool = False
    saw_file_header: bool = False

    @property
    def path(self) -> str | None:
        return self.new_path or self.old_path

    @property
    def lines(self) -> int:
        return self.added + self.removed

    def paths(self) -> list[str]:
        out = [p for p in (self.path, self.old_path) if p]
        return list(dict.fromkeys(out))


_HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
_GIT_HEADER_RE = re.compile(r'^diff --git "?a/(.+?)"? "?b/(.+?)"?\s*$')
_BINARY_RE = re.compile(r"^Binary files (.+?) and (.+?) differ\s*$")
_SPECIAL_MODE_RE = re.compile(
    r"^(?:(?:new file|deleted file|new|old) mode|index [0-9a-f]+\.\.[0-9a-f]+) (?:120000|160000)\s*$")


def _header_path(raw: str) -> str | None:
    raw = raw.split("\t", 1)[0].strip()          # difflib may append a timestamp
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        raw = raw[1:-1]
    if raw == "/dev/null":
        return None
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    return _norm_path(raw) or None


def _parse_diff(diff_text: str) -> tuple[list[_FileChange], _FileChange]:
    """Split a unified diff into per-file changes; lines outside any file go to `orphan`."""
    files: list[_FileChange] = []
    orphan = _FileChange()
    cur: _FileChange | None = None
    lines = diff_text.splitlines()
    n = len(lines)
    in_hunk = False
    in_binary = False
    old_left: int | None = None
    new_left: int | None = None

    def count(tag: str, text: str) -> None:
        target = cur if cur is not None else orphan
        if tag == "+":
            target.added += 1
        else:
            target.removed += 1
        target.chars += max(0, len(text) - 1)

    i = 0
    while i < n:
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < n else None

        if line.startswith("diff --git "):
            cur = _FileChange(git_header=True)
            m = _GIT_HEADER_RE.match(line)
            if m:
                cur.old_path, cur.new_path = _norm_path(m.group(1)), _norm_path(m.group(2))
            files.append(cur)
            in_hunk = in_binary = False
            i += 1
            continue
        if in_binary:  # base85 payload of a GIT binary patch
            i += 1
            continue

        if in_hunk:
            looks_like_header = (
                line.startswith("--- ") and nxt is not None and nxt.startswith("+++ ")
                and (old_left is None or (i + 2 < n and lines[i + 2].startswith("@@")))
            )
            if not looks_like_header and not line.startswith("@@"):
                tag = line[:1]
                if tag in "+-" and tag:
                    count(tag, line)
                    if tag == "+" and new_left is not None:
                        new_left -= 1
                    elif tag == "-" and old_left is not None:
                        old_left -= 1
                elif tag != "\\" and old_left is not None and new_left is not None:
                    old_left -= 1
                    new_left -= 1
                if old_left is not None and new_left is not None and old_left <= 0 and new_left <= 0:
                    in_hunk = False
                i += 1
                continue
            in_hunk = False

        if line.startswith("@@"):
            m = _HUNK_RE.match(line)
            if m:
                old_left = int(m.group(1)) if m.group(1) is not None else 1
                new_left = int(m.group(2)) if m.group(2) is not None else 1
                in_hunk = old_left > 0 or new_left > 0
            else:  # e.g. a bare "@@" from a hand-written diff: count leniently
                old_left = new_left = None
                in_hunk = True
            i += 1
            continue

        if line.startswith("--- ") and nxt is not None and nxt.startswith("+++ "):
            old, new = _header_path(line[4:]), _header_path(nxt[4:])
            if cur is None or not cur.git_header or cur.saw_file_header:
                cur = _FileChange()  # difflib-style diff without "diff --git" lines
                files.append(cur)
            cur.saw_file_header = True
            cur.old_path = old
            cur.new_path = new
            if old is None:
                cur.is_new = True
            if new is None:
                cur.is_deleted = True
            i += 2
            continue

        m = _BINARY_RE.match(line)
        if m:
            old, new = _header_path(m.group(1)), _header_path(m.group(2))
            if cur is None:
                cur = _FileChange(old_path=old, new_path=new)
                files.append(cur)
            cur.is_binary = True
            cur.is_new = cur.is_new or old is None
            cur.is_deleted = cur.is_deleted or new is None
            i += 1
            continue

        if cur is not None:
            if _SPECIAL_MODE_RE.match(line):
                cur.is_special = True
            if line.startswith("GIT binary patch"):
                cur.is_binary = True
                in_binary = True
            elif line.startswith("new file mode"):
                cur.is_new = True
            elif line.startswith("deleted file mode"):
                cur.is_deleted = True
            elif line.startswith("rename from "):
                cur.old_path, cur.is_rename = _norm_path(line[len("rename from "):]), True
            elif line.startswith("rename to "):
                cur.new_path, cur.is_rename = _norm_path(line[len("rename to "):]), True
            elif line.startswith("copy to "):
                cur.new_path = _norm_path(line[len("copy to "):])

        # Lenient: +/- lines outside a recognised hunk still count (hand-written diffs).
        if (line.startswith("+") or line.startswith("-")) and not line.startswith(("---", "+++")):
            count(line[0], line)
        i += 1

    merged: dict[str, _FileChange] = {}
    for fc in files:
        if fc.is_new and not fc.is_rename:
            fc.old_path = None
        if fc.is_deleted:
            fc.new_path = None
        key = fc.path
        if key is None:
            orphan.added += fc.added
            orphan.removed += fc.removed
            orphan.chars += fc.chars
            continue
        if key in merged:
            m = merged[key]
            m.added += fc.added
            m.removed += fc.removed
            m.chars += fc.chars
            m.is_binary |= fc.is_binary
            m.is_special |= fc.is_special
            m.is_rename |= fc.is_rename
            m.old_path = m.old_path or fc.old_path
        else:
            merged[key] = fc
    return list(merged.values()), orphan


# ---------------------------------------------------------------------------
# Public API comparison (Python, via ast)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Sig:
    positional: tuple[str, ...]
    positional_defaults: tuple[bool, ...]
    kwonly: tuple[tuple[str, bool], ...]
    vararg: str | None
    kwarg: str | None
    is_async: bool
    kind: str                     # function | method | staticmethod | classmethod | property

    def render(self) -> str:
        parts = [f"{n}=..." if d else n for n, d in zip(self.positional, self.positional_defaults)]
        if self.vararg:
            parts.append("*" + self.vararg)
        elif self.kwonly:
            parts.append("*")
        parts += [f"{n}=..." if d else n for n, d in self.kwonly]
        if self.kwarg:
            parts.append("**" + self.kwarg)
        return "(" + ", ".join(parts) + ")"


def _is_public(name: str) -> bool:
    return not name.startswith("_") or (len(name) > 4 and name.startswith("__") and name.endswith("__"))


def _method_kind(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name == "staticmethod":
            return "staticmethod"
        if name == "classmethod":
            return "classmethod"
        if name in ("property", "cached_property", "setter", "getter", "deleter"):
            return "property"
    return "method"


def _sig_of(node: ast.FunctionDef | ast.AsyncFunctionDef, in_class: bool) -> _Sig:
    a = node.args
    positional = [x.arg for x in (*a.posonlyargs, *a.args)]
    n_defaults = len(a.defaults)
    pos_defaults = [False] * (len(positional) - n_defaults) + [True] * n_defaults
    kwonly = tuple((x.arg, d is not None) for x, d in zip(a.kwonlyargs, a.kw_defaults))
    return _Sig(
        tuple(positional), tuple(pos_defaults), kwonly,
        a.vararg.arg if a.vararg else None, a.kwarg.arg if a.kwarg else None,
        isinstance(node, ast.AsyncFunctionDef),
        _method_kind(node) if in_class else "function",
    )


def _sig_change(old: _Sig, new: _Sig) -> str | None:
    """Why `new` is not call-compatible with `old`, or None."""
    if old.kind != new.kind:
        return f"{old.kind} became {new.kind}"
    if old.kind == "property":
        return None
    if old.is_async != new.is_async:
        return "changed between sync and async"
    op, od = list(old.positional), list(old.positional_defaults)
    np_, nd = list(new.positional), list(new.positional_defaults)
    if old.kind in ("method", "classmethod") and op and np_:
        op, od, np_, nd = op[1:], od[1:], np_[1:], nd[1:]   # self/cls may be renamed freely
    if np_[:len(op)] != op:
        return "positional parameters renamed, reordered or removed"
    for name, had, has in zip(op, od, nd):
        if had and not has:
            return f"parameter '{name}' lost its default"
    for name, has in zip(np_[len(op):], nd[len(op):]):
        if not has:
            return f"new required parameter '{name}'"
    new_kw = dict(new.kwonly)
    new_pos = dict(zip(np_, nd))
    for name, had in old.kwonly:
        if name in new_kw:
            has = new_kw[name]
        elif name in new_pos:
            has = new_pos[name]
        else:
            return f"keyword-only parameter '{name}' removed"
        if had and not has:
            return f"parameter '{name}' lost its default"
    old_kw = {name for name, _ in old.kwonly}
    for name, has in new.kwonly:
        if name not in old_kw and not has:
            return f"new required keyword-only parameter '{name}'"
    if old.vararg and not new.vararg:
        return f"'*{old.vararg}' removed"
    if old.kwarg and not new.kwarg:
        return f"'**{old.kwarg}' removed"
    return None


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for elt in target.elts:
            out |= _target_names(elt)
        return out
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return set()


def _bound_names(node: ast.stmt) -> set[str]:
    """Names a statement binds in its enclosing scope (recursing into if/try/with blocks)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Import):
        return {a.asname or a.name.split(".")[0] for a in node.names}
    if isinstance(node, ast.ImportFrom):
        return {f"*:{node.module or ''}" if a.name == "*" else (a.asname or a.name) for a in node.names}
    if isinstance(node, ast.Assign):
        out: set[str] = set()
        for t in node.targets:
            out |= _target_names(t)
        return out
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _target_names(node.target)
    out = set()
    for attr in ("body", "orelse", "finalbody"):
        for child in getattr(node, attr, None) or []:
            if isinstance(child, ast.stmt):
                out |= _bound_names(child)
    for handler in getattr(node, "handlers", None) or []:
        for child in handler.body:
            out |= _bound_names(child)
    return out


def _literal_names(node: ast.AST | None) -> list[str] | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return None
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def _all_value(tree: ast.Module) -> object:
    """frozenset of names in a literal __all__, a 'dynamic:...' string, or None if absent."""
    names: list[str] = []
    dynamic: list[str] = []
    seen = False
    for node in tree.body:
        value = None
        augment = False
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            value = node.value
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            value, augment = node.value, True
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "__all__"):
            call = node.value
            if call.func.attr == "extend" and call.args:
                value, augment = call.args[0], True
            elif call.func.attr == "append" and call.args:
                value, augment = ast.List(elts=[call.args[0]], ctx=ast.Load()), True
            else:
                continue
        else:
            continue
        seen = True
        literal = _literal_names(value)
        if literal is None:
            dynamic.append(ast.dump(value) if value is not None else "")
            continue
        names = names + literal if augment else literal
    if not seen:
        return None
    if dynamic:
        return "dynamic:" + "|".join(dynamic) + "|" + ",".join(sorted(names))
    return frozenset(names)


@dataclass
class _ModuleApi:
    functions: dict[str, _Sig] = field(default_factory=dict)
    classes: dict[str, dict[str, _Sig]] = field(default_factory=dict)
    class_bound: dict[str, set[str]] = field(default_factory=dict)
    bound: set[str] = field(default_factory=set)
    all_: object = None

    @property
    def exports(self) -> set[str]:
        return {n for n in self.bound if not n.startswith("_")}


def _module_api(tree: ast.Module) -> _ModuleApi:
    api = _ModuleApi()
    for node in tree.body:
        api.bound |= _bound_names(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            api.functions[node.name] = _sig_of(node, in_class=False)
        elif isinstance(node, ast.ClassDef) and _is_public(node.name):
            methods: dict[str, _Sig] = {}
            bound: set[str] = set()
            for item in node.body:
                bound |= _bound_names(item)
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(item.name):
                    methods[item.name] = _sig_of(item, in_class=True)
            api.classes[node.name] = methods
            api.class_bound[node.name] = bound
    api.all_ = _all_value(tree)
    return api


def _parse_python(source: str) -> ast.Module | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return ast.parse(source)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return None


def _fmt_names(names: list[str] | set[str], limit: int = 10) -> str:
    items = sorted(names)
    text = ", ".join(items[:limit])
    return text + (f", ... (+{len(items) - limit})" if len(items) > limit else "")


def _public_api_findings(path: str, base_src: str | None, new_src: str | None) -> list[ScopeFinding]:
    out: list[ScopeFinding] = []

    def review(message: str) -> None:
        out.append(ScopeFinding("review", "public_api", path, message))

    if base_src is None:
        return out          # new module: purely additive
    base_tree = _parse_python(base_src)
    if base_tree is None:
        return out
    old = _module_api(base_tree)
    is_init = path.rsplit("/", 1)[-1] == "__init__.py"

    if new_src is None:
        symbols = set(old.functions) | set(old.classes) | (old.exports if is_init else set())
        if symbols:
            review(f"Deletes module '{path}', which defines public symbols: {_fmt_names(symbols)}. "
                   f"Code importing them will break — public API change needs human review.")
        return out

    new_tree = _parse_python(new_src)
    if new_tree is None:
        return [ScopeFinding("info", "public_api", path,
                             f"Could not parse the new version of '{path}'; its public API was not compared.")]
    new = _module_api(new_tree)

    removed: set[str] = set()
    for name, sig in old.functions.items():
        if name in new.functions:
            why = _sig_change(sig, new.functions[name])
            if why:
                review(f"Changes the signature of public function `{name}` in '{path}': "
                       f"{name}{sig.render()} -> {name}{new.functions[name].render()} ({why}).")
        elif name in new.classes:
            review(f"Public function `{name}` in '{path}' is now a class.")
        elif name not in new.bound:
            removed.add(name)
            review(f"Removes public function `{name}` from '{path}'.")

    for cname, methods in old.classes.items():
        if cname not in new.classes:
            if cname not in new.bound:
                removed.add(cname)
                review(f"Removes public class `{cname}` from '{path}'.")
            continue
        new_methods = new.classes[cname]
        for mname, sig in methods.items():
            if mname in new_methods:
                why = _sig_change(sig, new_methods[mname])
                if why:
                    review(f"Changes the signature of public method `{cname}.{mname}` in '{path}': "
                           f"{mname}{sig.render()} -> {mname}{new_methods[mname].render()} ({why}).")
            elif mname not in new.class_bound.get(cname, set()):
                review(f"Removes public method `{cname}.{mname}` from '{path}'.")

    if old.all_ != new.all_:
        if isinstance(old.all_, frozenset) and isinstance(new.all_, frozenset):
            added, gone = new.all_ - old.all_, old.all_ - new.all_
            detail = "; ".join(x for x in (f"added: {_fmt_names(added)}" if added else "",
                                           f"removed: {_fmt_names(gone)}" if gone else "") if x)
            review(f"Changes `__all__` in '{path}' ({detail}).")
        else:
            review(f"Changes `__all__` in '{path}'.")

    if is_init:
        added = new.exports - old.exports
        gone = old.exports - new.exports - removed
        if added or gone:
            detail = "; ".join(x for x in (f"added: {_fmt_names(added)}" if added else "",
                                           f"removed: {_fmt_names(gone)}" if gone else "") if x)
            review(f"Changes the exports of package '{path}' ({detail}).")
    return out


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

def _existed_at_base(fc: _FileChange, base: dict[str, str | None] | None) -> bool:
    key = fc.old_path or fc.path
    if base is not None and key in base:
        return base[key] is not None
    return not fc.is_new


def _strictest(fc: _FileChange, rules: list[PathRule]) -> tuple[PathRule, str] | None:
    best: tuple[PathRule, str] | None = None
    for p in fc.paths():
        hit = _match_rule(p, rules)
        if hit and (best is None or _RANK[hit[0].severity] > _RANK[best[0].severity]):
            best = hit
    return best


def evaluate(
    diff_text: str,
    repo_root: Path,
    max_diff_lines: int = 150,
    constraints: ScopeConstraints | None = None,
    base_contents: dict[str, str | None] | None = None,
    new_contents: dict[str, str] | None = None,
    *,
    policy: ScopePolicy | None = None,
) -> ScopeReport:
    """
    Classify every path the diff touches and report block / review / info findings.

    base_contents / new_contents (repo-relative path -> text, None = file absent) enable the
    Python public-API comparison and sharpen the "existing migration" check.

    ``policy`` pins the repository policy (``load_policy(root, base_ref)`` taken before any
    repository code ran). Without it the policy is read from ``base_contents`` / the HEAD
    commit — never from working-tree edits (see the module docstring).
    """
    repo_root = Path(repo_root)
    if policy is None:
        policy = load_policy(repo_root, "HEAD", base_contents)
    rules, ignore_patterns = policy.rules, policy.ignore_patterns
    files, orphan = _parse_diff(diff_text)
    base = {_norm_path(k): v for k, v in base_contents.items()} if base_contents is not None else None
    new = {_norm_path(k): v for k, v in new_contents.items()} if new_contents is not None else None

    findings: list[ScopeFinding] = [ScopeFinding("info", "config", SCOPE_CONFIG_FILE, msg)
                                    for msg in policy.problems]
    if policy.read_error:
        findings.append(ScopeFinding("review", "config", SCOPE_CONFIG_FILE, policy.read_error))
    categories: dict[str, str] = {}
    raw_lines = orphan.lines
    effective = float(orphan.lines)
    raw_chars = orphan.chars
    effective_chars = float(orphan.chars)
    budget_used: dict[str, int] = {}
    budget_limit: dict[str, int] = {}
    budget_paths: dict[str, list[str]] = {}
    forbidden = [p for p in (constraints.forbidden_paths if constraints else []) if p and p.strip()]
    allowed = [p for p in (constraints.allowed_paths if constraints else []) if p and p.strip()]
    source = constraints.source if constraints else "triage"

    for fc in files:
        path = fc.path
        assert path is not None
        hard = next(((p, r) for p in fc.paths() if (r := protected_path_reason(p))), None)
        if hard:
            # Alias paths and policy files: blocked before (and regardless of) any rule.
            hard_path, reason = hard
            rule, pattern = None, None
            category = _hard_category(hard_path) or "unsafe_path"
            findings.append(ScopeFinding(
                "block", category, path,
                f"Diff touches '{hard_path}': {reason}. This path is off-limits for the agent."))
        else:
            hit = _strictest(fc, rules)
            rule, pattern = hit if hit else (None, None)
            category = rule.category if rule else DEFAULT_CATEGORY
        categories[path] = category
        weight = rule.weight if rule else 1.0
        raw_lines += fc.lines
        effective += fc.lines * weight
        raw_chars += fc.chars
        effective_chars += fc.chars * weight

        for p in fc.paths():
            pat = _ignored_match(p, ignore_patterns)
            if pat:
                findings.append(ScopeFinding(
                    "block", "ignored", path,
                    f"Diff touches '{p}' which matches the ignored pattern '{pat}' in "
                    f".fix-issue-ignore. This path is off-limits for the agent."))
                break

        for p in (fc.paths() if category not in _EXEMPT_FROM_FORBIDDEN else ()):
            pat = _first_match(p, forbidden)
            if pat:
                findings.append(ScopeFinding(
                    "block", "triage", path,
                    f"Diff touches '{p}', which {source} marked as off-limits for this issue "
                    f"(forbidden path '{pat}')."))
                break

        if fc.is_special:
            findings.append(ScopeFinding(
                "block", "special_file", path,
                f"Diff adds or changes a symlink or submodule entry at '{path}'; the agent only proposes "
                f"regular text files (a symlink can expose files outside the repository)."))

        if fc.is_binary:
            findings.append(ScopeFinding(
                "block", "binary", path,
                f"Diff contains a binary patch for '{path}'; the agent only proposes text changes."))

        if rule is not None and pattern is not None:
            if rule.category == "migrations":
                if _existed_at_base(fc, base):
                    verb = "deletes" if fc.is_deleted else ("renames" if fc.is_rename else "modifies")
                    findings.append(ScopeFinding(
                        "block", "migrations", path,
                        f"Diff {verb} the existing migration '{fc.old_path or path}' (matches "
                        f"'{pattern}'). Applied migrations must never be rewritten; add a new one."))
                else:
                    findings.append(ScopeFinding(
                        "review", "migrations", path,
                        f"Diff adds a new migration '{path}' (matches '{pattern}'); schema changes "
                        f"need human review."))
            elif rule.severity in ("block", "review"):
                desc, consequence = _CATEGORY_TEXT.get(
                    rule.category,
                    (f"in the '{rule.category}' category",
                     f"({rule.source} rule, severity {rule.severity})."))
                findings.append(ScopeFinding(
                    rule.severity, rule.category, path,
                    f"'{path}' is {desc} (matches '{pattern}'). {consequence}"))
            if rule.max_lines is not None:
                budget_used[category] = budget_used.get(category, 0) + fc.lines
                budget_limit[category] = min(budget_limit.get(category, rule.max_lines), rule.max_lines)
                budget_paths.setdefault(category, []).append(path)

        if fc.is_deleted:
            findings.append(ScopeFinding("review", "deletion", path,
                                         f"Diff deletes '{path}'; file deletions need human review."))
        elif fc.is_rename and fc.old_path and fc.old_path != path:
            findings.append(ScopeFinding("review", "deletion", path,
                                         f"Diff renames '{fc.old_path}' to '{path}' (the old path "
                                         f"disappears); renames need human review."))

    for category, used in budget_used.items():
        limit = budget_limit[category]
        if used > limit:
            findings.append(ScopeFinding(
                "block", category, None,
                f"{used} changed lines in {category} paths ({', '.join(budget_paths[category])}) "
                f"exceed the strict budget of {limit} lines for that category."))

    if base is not None and new is not None:
        for fc in files:
            path = fc.path
            assert path is not None
            if categories.get(path) == "tests" or not path.endswith((".py", ".pyi")):
                continue
            if fc.is_new:
                base_src = None
            elif (fc.old_path or path) in base:
                base_src = base[fc.old_path or path]
            else:
                continue
            if fc.is_deleted:
                new_src = None
            elif path in new:
                new_src = new[path]
            else:
                continue
            if base_src is not None and new_src is None and not fc.is_deleted and not fc.is_rename:
                continue
            findings.extend(_public_api_findings(path, base_src, new_src))

    if allowed:
        for fc in files:
            path = fc.path
            assert path is not None
            if categories[path] in _EXEMPT_FROM_ALLOWED:
                continue
            if _first_match(path, allowed) is None:
                findings.append(ScopeFinding(
                    "review", "triage", path,
                    f"'{path}' is outside the paths {source} expected ({', '.join(allowed)}). "
                    f"Triage only saw the issue text, so this is flagged for human review "
                    f"rather than blocked."))

    limit = max_diff_lines
    tightened = False
    if constraints is not None and constraints.max_diff_lines is not None and constraints.max_diff_lines > 0:
        if constraints.max_diff_lines < max_diff_lines:
            limit = constraints.max_diff_lines
            tightened = True
            findings.append(ScopeFinding(
                "info", "size", None,
                f"{source} tightened the size limit from {max_diff_lines} to {limit} lines."))
        elif constraints.max_diff_lines > max_diff_lines:
            findings.append(ScopeFinding(
                "info", "size", None,
                f"Ignored {source} size limit of {constraints.max_diff_lines} lines: issue-specific "
                f"constraints can only tighten the configured limit ({max_diff_lines})."))
    if effective > limit:
        findings.append(ScopeFinding(
            "block", "size", None,
            f"Diff is too large: {effective:g} effective changed lines ({raw_lines} raw; tests count "
            f"x0.5, docs x0.25) exceed the limit of {limit}"
            f"{f' set by {source}' if tightened else ''}. This issue likely requires a larger "
            f"change — needs human review."))
    char_limit = max(0, limit) * CHARS_PER_LINE_BUDGET
    if effective_chars > char_limit:
        findings.append(ScopeFinding(
            "block", "size", None,
            f"Diff changes too much text: {effective_chars:g} effective changed characters "
            f"({raw_chars} raw, weighted like lines) exceed the budget of {char_limit} "
            f"({CHARS_PER_LINE_BUDGET} per allowed line x {limit}). Very long lines (generated or "
            f"minified content) are not a small fix — needs human review."))

    return ScopeReport(
        ok=not any(f.severity == "block" for f in findings),
        requires_review=any(f.severity == "review" for f in findings),
        findings=findings,
        raw_lines=raw_lines,
        effective_lines=effective,
        limit=limit,
        categories=categories,
        raw_chars=raw_chars,
        effective_chars=effective_chars,
        char_limit=char_limit,
    )
