"""
repo_manager.py — Git workspaces the pipeline edits, tests, commits and pushes.

    GitWorkspace   git operations on an existing working tree (diff, reset, file state)
    RepoManager    context manager: fresh temp clone of a GitHub repo (always cleaned up)
    LocalWorkspace context manager over an existing local repo (benchmark fixtures)

Usage (always via context manager so the temp dir is always cleaned up):

    with RepoManager("owner/repo", token, "fix-issue-42") as repo:
        repo.apply_patch(diff_text)   # raises ApplyError on hard failure
        ...
    # temp dir is gone here, even if an exception fired mid-pipeline

Git hygiene
-----------
The repository's own code (its test suite, its linters' plugins, a model-written
``conftest.py``) runs in the working tree BEFORE we diff, commit and push. It can plant hooks
in ``.git/hooks`` or settings in ``.git/config`` (``core.fsmonitor``, ``filter.*.clean``,
``diff.*.textconv``, ``core.sshCommand``, ``url.*.insteadOf``, ``include.path`` ...) that git
would execute — during the push, with the credentials in its environment. So every git
command goes through ``GitWorkspace._run`` / ``git_command``:

  * ``-c core.hooksPath=<empty dir owned by this process>`` (no hook ever runs) and
    ``--no-verify`` on commit and push; ``-c core.fsmonitor=false``; ``-c gc.auto=0`` and
    ``-c maintenance.auto=false`` (no detached background jobs),
  * the git control files are snapshotted when the workspace is set up — before any
    repository code runs — and restored byte-for-byte before every git command:
    ``config`` is rewritten if it changed, a planted ``commondir``/``config.worktree`` is
    removed, and a ``.git`` that was swapped for a link or a file aborts the command,
  * the environment is ``sandbox.safe_env()`` (an allowlist; no tokens or API keys) plus the
    proxy / CA / author variables git itself may need,
  * a wall-clock timeout, with the whole process tree killed (``sandbox.run_process``),
  * failures raise ``GitCommandError`` (a ``RuntimeError``) whose message, command, stdout and
    stderr have the token — plain and base64 — redacted.

The token never appears in a URL or in ``.git/config``: clone and push authenticate with a
per-command ``-c http.https://github.com/.extraheader=...`` argument (so it IS on git's command
line, base64-encoded, for the duration of those two commands), the push goes to an
explicit URL (so a planted remote or url rewrite in the repo config cannot redirect it) with
TLS verification forced on, and only the commit we created is pushed, on top of the base.

What this cannot stop: repository code runs as the same OS user without a container, so it
can still read files under HOME and modify the user's global git config; see sandbox.py.
"""

from __future__ import annotations

import atexit
import base64
import contextlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from fix_issue.sandbox import python_executable, run_process, safe_env

_GIT_TIMEOUT = 300
_CLONE_TIMEOUT = 900
_PUSH_TIMEOUT = 600

# Variables git itself may need on top of the sandbox allowlist (network + identity). They come
# from the user's own environment and are only given to git, never to repository code.
_GIT_ENV_PASSTHROUGH = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "GIT_SSL_CAINFO", "GIT_SSL_CAPATH",
    "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
    "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM",
    # The user's own SSH / signing setup (a global url.insteadOf may route github.com over SSH,
    # commit.gpgsign may be on). Given to git only, never to repository code.
    "SSH_AUTH_SOCK", "SSH_AGENT_PID", "GIT_SSH", "GIT_SSH_COMMAND", "GIT_SSH_VARIANT",
    "GNUPGHOME", "GPG_TTY", "DISPLAY",
)


class ApplyError(Exception):
    """Raised when `git apply` and `git apply --3way` both fail."""


class GitCommandError(RuntimeError):
    """A git command failed, timed out or was refused. Message, command and output are redacted."""

    def __init__(self, message: str, *, cmd: list[str] | None = None, returncode: int | None = None,
                 stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.cmd = list(cmd or [])
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

_AUTH_HEADER_RE = re.compile(r"(?i)(authorization:\s*(?:basic|bearer|token)\s+)\S+")
_URL_CREDENTIALS_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@")


def token_forms(token: str) -> list[str]:
    """Every spelling of ``token`` that may reach argv or git output (plain and base64)."""
    if not token:
        return []
    pair = f"x-access-token:{token}"
    return [token, pair, base64.b64encode(pair.encode()).decode(), base64.b64encode(token.encode()).decode()]


def redact(text: str, secrets: list[str] | tuple[str, ...] = ()) -> str:
    """Remove ``secrets`` (and any Authorization header value / URL credentials) from ``text``."""
    if not text:
        return text
    for secret in sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True):
        text = text.replace(secret, "***")
    text = _AUTH_HEADER_RE.sub(r"\1***", text)
    return _URL_CREDENTIALS_RE.sub(r"\1***@", text)


# ---------------------------------------------------------------------------
# Hardened git invocation
# ---------------------------------------------------------------------------

_EMPTY_HOOKS: Path | None = None


def _is_link(path: Path) -> bool:
    isjunction = getattr(os.path, "isjunction", None)
    return os.path.islink(path) or bool(isjunction and isjunction(path))


def _empty_hooks_dir() -> Path:
    """A directory that holds no hooks; recreated if anything was planted in (or swapped for) it."""
    global _EMPTY_HOOKS
    d = _EMPTY_HOOKS
    try:
        ok = d is not None and not _is_link(d) and d.is_dir() and not any(d.iterdir())
    except OSError:
        ok = False
    if not ok:
        if d is not None and not _is_link(d):
            shutil.rmtree(d, ignore_errors=True)
        d = Path(tempfile.mkdtemp(prefix="fix-issue-nohooks-"))
        _EMPTY_HOOKS = d
        atexit.register(shutil.rmtree, str(d), True)
    return d


def git_hardening_args() -> list[str]:
    """``-c`` options every git invocation gets (they override any config file)."""
    return [
        "-c", f"core.hooksPath={_empty_hooks_dir()}",
        "-c", "core.fsmonitor=false",
        "-c", "gc.auto=0",
        "-c", "maintenance.auto=false",
    ]


def git_env() -> dict[str, str]:
    """Environment for git: the sandbox allowlist plus proxy / CA / identity variables."""
    env = safe_env()
    present = {k.upper() if os.name == "nt" else k for k in env}
    for name in _GIT_ENV_PASSTHROUGH:
        key = name.upper() if os.name == "nt" else name
        value = os.environ.get(name)
        if value is not None and key not in present:
            env[name] = value
            present.add(key)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    # refs/replace (writable by repository code) would let `git diff <base>` show a different
    # base tree than the one that gets committed on, hiding changes from the scope checks.
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    return env


def _rmtree(path: Path | str) -> None:
    """Delete a working tree, including git's read-only object files (Windows refuses those)."""
    def retry(func, target, *_exc) -> None:
        with contextlib.suppress(OSError):
            os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
            func(target)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:  # pragma: no cover
        shutil.rmtree(path, onerror=retry)


def _remove_path(path: Path) -> None:
    if _is_link(path):
        try:
            path.unlink()
        except OSError:
            os.rmdir(path)  # a directory junction / symlink on Windows
    elif path.is_dir():
        _rmtree(path)
    elif path.exists():
        try:
            path.unlink()
        except PermissionError:   # read-only file on Windows
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            path.unlink()


def _identity(path: Path) -> tuple[int, int]:
    st = os.stat(path)
    return st.st_dev, st.st_ino


class _GitDirGuard:
    """Byte-for-byte snapshot of a working tree's git control files, restored before every git command."""

    def __init__(self, root: Path, git_dir: Path, common_dir: Path) -> None:
        self.root = root
        self.dot_git = root / ".git"
        self.dot_git_is_dir = self.dot_git.is_dir() and not _is_link(self.dot_git)
        self.dot_git_bytes = None if self.dot_git_is_dir else self.dot_git.read_bytes()
        self.dot_git_id = _identity(self.dot_git)
        self.files: dict[Path, bytes | None] = {}
        for path in (common_dir / "config", git_dir / "config.worktree", git_dir / "commondir",
                     common_dir / "config.worktree"):
            if path not in self.files:
                self.files[path] = path.read_bytes() if path.is_file() and not _is_link(path) else None

    def restore(self) -> None:
        if self.dot_git_is_dir:
            try:
                same = (not _is_link(self.dot_git) and self.dot_git.is_dir()
                        and _identity(self.dot_git) == self.dot_git_id)
            except OSError:
                same = False
            if not same:
                raise GitCommandError(f"{self.dot_git} is no longer the original directory; refusing to run git")
        elif (_is_link(self.dot_git) or not self.dot_git.is_file()
              or self.dot_git.read_bytes() != self.dot_git_bytes):
            raise GitCommandError(f"{self.dot_git} was modified; refusing to run git")
        for path, content in self.files.items():
            try:
                self._restore_file(path, content)
            except OSError as exc:
                raise GitCommandError(f"could not restore {path.name} in {path.parent}: {exc}") from None

    @staticmethod
    def _restore_file(path: Path, content: bytes | None) -> None:
        exists = _is_link(path) or path.exists()
        if content is None:
            if exists:
                _remove_path(path)
            return
        if exists and not _is_link(path) and path.is_file():
            with contextlib.suppress(OSError):
                if path.read_bytes() == content:
                    return
        if exists:
            _remove_path(path)
        tmp = path.with_name(f"{path.name}.fix-issue-restore")
        if _is_link(tmp) or tmp.exists():
            _remove_path(tmp)
        tmp.write_bytes(content)
        os.replace(tmp, path)   # replaces the directory entry: never writes through a planted (hard)link


_GUARDS: dict[str, _GitDirGuard] = {}


def _guard_key(root: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(root)))


def git_command(root: Path | str, *args: str) -> list[str]:
    """
    Hardened argv for ``git <args>`` in ``root`` (for callers that run git via ``sandbox.run_command``).

    If a workspace registered ``root``, its git control files are restored first (this raises
    ``GitCommandError`` when ``.git`` itself was swapped).
    """
    guard = _GUARDS.get(_guard_key(root))
    if guard is not None:
        guard.restore()
    return ["git", *git_hardening_args(), *args]


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------

class GitWorkspace:
    """Git operations on a working tree rooted at ``self.root``."""

    root: Path | None
    _tmpdir: str | None

    def __init__(self) -> None:
        self.root = None
        self._tmpdir = None
        self.base_ref = "HEAD"
        self._secrets: list[str] = []
        self._guard: _GitDirGuard | None = None
        self._commit_sha: str | None = None

    # ------------------------------------------------------------------
    # Patch application (legacy unified-diff path)
    # ------------------------------------------------------------------

    def apply_patch(self, diff_text: str) -> None:
        """
        Try `git apply`, fall back to `git apply --3way`.
        If both fail, raise ApplyError (counts as a failed attempt, not a crash).
        """
        # Normalize diff to LF line endings to avoid CRLF mismatch on Windows
        diff_text_lf = diff_text.replace("\r\n", "\n")
        patch_file = self.root / "_fix_issue.patch"
        patch_file.write_bytes(diff_text_lf.encode("utf-8"))

        try:
            first_err_msg = ""
            try:
                self._run(["git", "apply", "--check", str(patch_file)])
                self._run(["git", "apply", str(patch_file)])
                return
            except GitCommandError as first_err:
                first_err_msg = (first_err.stderr or first_err.stdout or str(first_err)).strip()

            try:
                self._run(["git", "apply", "--3way", str(patch_file)])
                return
            except GitCommandError as second_err:
                second_err_msg = (second_err.stderr or second_err.stdout or str(second_err)).strip()
                details = f"git apply: {first_err_msg}\ngit apply --3way: {second_err_msg}"
                raise ApplyError(
                    f"`git apply` and `git apply --3way` both failed for the generated patch.\n"
                    f"Git Stderr / Details:\n{details}\n\n"
                    f"Patch preview (first 500 chars):\n{diff_text[:500]}"
                ) from None
        finally:
            patch_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Working-tree state
    # ------------------------------------------------------------------

    def diff(self) -> str:
        """Cumulative unified diff of the working tree vs. the base commit (incl. new files)."""
        self._run(["git", "add", "--intent-to-add", "."], check=False)
        # Fixed a/ b/ prefixes: diff.mnemonicPrefix / diff.noprefix in the user's (or a planted)
        # config would otherwise change the headers every path rule and the PR text parse.
        out = self._run(["git", "diff", "--no-color", "--no-ext-diff", "--no-textconv",
                         "--src-prefix=a/", "--dst-prefix=b/", self.base_ref], check=False)
        return out.stdout

    def changed_files(self) -> list[str]:
        out = self._run(["git", "status", "--porcelain", "--untracked-files=all"], check=False).stdout
        files = []
        for line in out.splitlines():
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            path = path.strip('"')
            if path and not _is_tool_artifact(path):
                files.append(path)
        return sorted(set(files))

    def base_content(self, rel_path: str) -> str | None:
        """File content at the base commit, or None if it did not exist."""
        res = self._run(["git", "show", f"{self.base_ref}:{rel_path}"], check=False)
        return res.stdout if res.returncode == 0 else None

    def reset_hard(self) -> None:
        """Discard every change made since the base commit."""
        self._run(["git", "reset", "-q", "--hard", self.base_ref], check=False)
        self._run(["git", "clean", "-fdq"], check=False)

    def commit_all(self, message: str) -> str:
        """Stage all changes and commit them directly on the base commit. Returns the new commit SHA."""
        if self.root:
            (self.root / "_fix_issue.patch").unlink(missing_ok=True)
        branch = getattr(self, "branch", None)
        if branch and re.fullmatch(r"[0-9a-f]{40,64}", self.base_ref or ""):
            # Whatever repository code did to HEAD or the branch ref, the commit sits on the base
            # (the index — what the reviewed diff was computed from — is left untouched).
            self._run(["git", "symbolic-ref", "HEAD", f"refs/heads/{branch}"])
            self._run(["git", "update-ref", f"refs/heads/{branch}", self.base_ref])
        # Untracked tool artifacts are kept out by .git/info/exclude. (A former blanket
        # `git rm --cached *.patch *.pyc __pycache__` also deleted TRACKED files of that kind from
        # the commit — changes the reviewed diff never contained.)
        self._run(["git", "add", "-A"])
        self._run(["git", "commit", "--no-verify", "-q", "-m", message])
        self._commit_sha = self._run(["git", "rev-parse", "HEAD"]).stdout.strip()
        return self._commit_sha

    # ------------------------------------------------------------------
    # Test command detection
    # ------------------------------------------------------------------

    def detect_test_command(self) -> list[str] | None:
        """
        Return a command (as a list) for running the repo's test suite.
        Tries common conventions in priority order.
        Returns None if nothing recognisable is found.

        Python commands use the interpreter running fix-issue (which has pytest
        installed) instead of whatever ``python`` happens to be first on PATH.
        """
        root = self.root
        py = python_executable()
        pytest_cmd = [py, "-m", "pytest", "--tb=short", "-q", "-p", "no:cacheprovider"]

        # 1. pytest.ini / setup.cfg [tool:pytest] / pyproject.toml [tool.pytest]
        if (
            (root / "pytest.ini").exists()
            or (root / "setup.cfg").exists()
            or _toml_has_pytest(root / "pyproject.toml")
        ):
            return pytest_cmd

        # 2. Bare setup.py with tests/
        if (root / "setup.py").exists() and (root / "tests").exists():
            return pytest_cmd

        # 3. package.json — look for "test" script
        pkg = root / "package.json"
        if pkg.exists():
            try:
                scripts = json.loads(pkg.read_text()).get("scripts", {})
                if "test" in scripts:
                    npm = "npm.cmd" if os.name == "nt" else "npm"
                    return [npm, "run", "test", "--", "--passWithNoTests"]
            except Exception:
                pass

        # 4. Makefile with a "test" target
        makefile = root / "Makefile"
        if makefile.exists():
            content = makefile.read_text(errors="replace")
            if re.search(r"^test\s*:", content, re.MULTILINE):
                return ["make", "test"]

        # 5. tox
        if (root / "tox.ini").exists():
            return ["tox"]

        # 6. go test
        if (root / "go.mod").exists():
            return ["go", "test", "./..."]

        # 7. A plain tests/ directory of pytest-style files
        tests_dir = root / "tests"
        if tests_dir.is_dir() and any(tests_dir.rglob("test_*.py")):
            return pytest_cmd

        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _protect(self) -> None:
        """Snapshot the git control files. Call once set-up is done, BEFORE any repository code runs."""
        out = self._run(["git", "rev-parse", "--absolute-git-dir", "--git-common-dir"]).stdout.splitlines()
        git_dir = Path(out[0].strip())
        common = Path(out[1].strip()) if len(out) > 1 and out[1].strip() else git_dir
        if not common.is_absolute():
            common = Path(self.root) / common
        self._guard = _GitDirGuard(Path(self.root), git_dir, common)
        _GUARDS[_guard_key(self.root)] = self._guard

    def _unprotect(self) -> None:
        if self._guard is not None and self.root is not None:
            key = _guard_key(self.root)
            if _GUARDS.get(key) is self._guard:
                del _GUARDS[key]
        self._guard = None

    def _redact(self, text: str) -> str:
        return redact(text, self._secrets)

    def _run(self, cmd: list[str], cwd: str | None = None, check: bool = True,
             timeout: float | None = None) -> subprocess.CompletedProcess:
        """Run ``cmd`` (``["git", ...]``) hardened; see the module docstring. Timeouts always raise."""
        if not cmd or cmd[0] != "git":
            raise ValueError(f"GitWorkspace only runs git, not {cmd[:1]!r}")
        if self._guard is not None:
            self._guard.restore()
        argv = ["git", *git_hardening_args(), *cmd[1:]]
        shown = [self._redact(a) for a in argv]
        label = self._redact(" ".join(cmd))[:300]
        limit = timeout or _GIT_TIMEOUT
        try:
            res = run_process(argv, Path(cwd or self.root), timeout=limit, env=git_env())
        except OSError as exc:
            raise GitCommandError(f"could not run `{label}`: {exc}", cmd=shown) from None
        stdout, stderr = res.stdout, res.stderr
        if res.timed_out:
            raise GitCommandError(f"`{label}` timed out after {limit:g}s", cmd=shown,
                                  stdout=self._redact(stdout), stderr=self._redact(stderr))
        if check and res.returncode != 0:
            err = self._redact(stderr or stdout)
            raise GitCommandError(f"`{label}` failed (exit {res.returncode}): {err.strip()[-1500:]}",
                                  cmd=shown, returncode=res.returncode,
                                  stdout=self._redact(stdout), stderr=err)
        return subprocess.CompletedProcess(shown, res.returncode, stdout, stderr)


class RepoManager(GitWorkspace):
    """
    Context manager that owns a fresh temp clone for one pipeline run.

    Parameters
    ----------
    repo:   "owner/repo" string
    token:  GitHub personal access token (for private repos + push)
    branch: The branch to create and commit fixes onto
    """

    github_url = "https://github.com/"   # class attribute so tests can point it at a local remote

    def __init__(self, repo: str, token: str, branch: str) -> None:
        super().__init__()
        self.repo = repo
        self.token = token
        self.branch = branch
        self._secrets = token_forms(token)

    def _remote_url(self) -> str:
        return f"{self.github_url}{self.repo}.git"

    def _auth_args(self) -> list[str]:
        """
        Per-command auth (never stored): an extra header scoped to the GitHub URL, TLS verification
        on, and — with a token — no credential helpers, so a rejected token fails instead of quietly
        falling back to the user's own stored GitHub credentials.
        """
        url = self._remote_url()
        # `url.<X>.insteadOf=<X>` is the longest possible match for our URL, so it beats any
        # user/global or planted rewrite (e.g. to SSH or another host): the URL is used as is.
        args = ["-c", "http.sslVerify=true", "-c", f"url.{url}.insteadOf={url}"]
        if self.token:
            basic = base64.b64encode(f"x-access-token:{self.token}".encode()).decode()
            args = ["-c", f"http.{self.github_url}.extraheader=AUTHORIZATION: basic {basic}",
                    "-c", "credential.helper=", *args]
        return args

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RepoManager":
        self._tmpdir = tempfile.mkdtemp(prefix="fix-issue-")
        try:
            self.root = Path(self._tmpdir)
            self._run(["git", *self._auth_args(), "clone", "-c", "core.autocrlf=false", "--depth=1",
                       self._remote_url(), "."], cwd=self._tmpdir, timeout=_CLONE_TIMEOUT)
            self._run(["git", "config", "core.autocrlf", "false"])
            _write_tool_excludes(self.root)
            self.base_ref = self._run(["git", "rev-parse", "HEAD"]).stdout.strip()
            # Create the fix branch
            self._run(["git", "checkout", "-q", "-b", self.branch])
            self._protect()
        except BaseException:
            self._cleanup()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cleanup()
        return False  # do not suppress exceptions

    def _cleanup(self) -> None:
        self._unprotect()
        if self._tmpdir and Path(self._tmpdir).exists():
            with contextlib.suppress(OSError):
                _rmtree(self._tmpdir)

    def push_branch(self) -> None:
        """Force-push exactly the commit we made to the fix branch, at an explicit URL, authenticated per command."""
        source = self._commit_sha or f"refs/heads/{self.branch}"
        self._run(["git", *self._auth_args(), "push", "--no-verify", "--force", "--recurse-submodules=no",
                   "--no-follow-tags",
                   self._remote_url(), f"{source}:refs/heads/{self.branch}"], timeout=_PUSH_TIMEOUT)


class LocalWorkspace(GitWorkspace):
    """
    Work on an existing local git repository (used by the benchmark).

    ``cleanup=True`` deletes the directory on exit. Push is a no-op that records intent.
    """

    def __init__(self, root: Path, branch: str = "fix-issue", cleanup: bool = False) -> None:
        super().__init__()
        self.root = Path(root)
        self._tmpdir = str(self.root)
        self.branch = branch
        self.cleanup = cleanup
        self.pushed = False

    def __enter__(self) -> "LocalWorkspace":
        try:
            _write_tool_excludes(self.root)
            self.base_ref = self._run(["git", "rev-parse", "HEAD"]).stdout.strip()
            self._run(["git", "checkout", "-q", "-B", self.branch])
            self._protect()
        except BaseException:
            self._unprotect()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._unprotect()
        if self.cleanup and self.root.exists():
            with contextlib.suppress(OSError):
                _rmtree(self.root)
        return False

    def push_branch(self) -> None:
        self.pushed = True


def _write_tool_excludes(root: Path) -> None:
    exclude_file = root / ".git" / "info" / "exclude"
    if exclude_file.parent.exists():
        existing = exclude_file.read_text(encoding="utf-8") if exclude_file.exists() else ""
        if "fix-issue tool internal artifacts" not in existing:
            with open(exclude_file, "a", encoding="utf-8") as f:
                f.write(
                    "\n# fix-issue tool internal artifacts\n_fix_issue.patch\n*.patch\n"
                    "__pycache__/\n*.pyc\n*.pyo\n.pytest_cache/\n.mypy_cache/\n.ruff_cache/\n"
                )


def _is_tool_artifact(path: str) -> bool:
    return (
        path.endswith((".pyc", ".pyo"))
        or "__pycache__" in path
        or path == "_fix_issue.patch"
        or ".pytest_cache" in path
    )


def _toml_has_pytest(path: Path) -> bool:
    if not path.exists():
        return False
    content = path.read_text(errors="replace")
    return "[tool.pytest" in content
