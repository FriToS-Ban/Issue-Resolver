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

Security note: the clone URL carries the token only for the clone itself. Immediately
afterwards the remote URL is rewritten without credentials, and push authenticates with a
one-shot ``http.extraheader`` so code executed by the repo's own test suite cannot read the
token from ``.git/config``.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from fix_issue.sandbox import python_executable


class ApplyError(Exception):
    """Raised when `git apply` and `git apply --3way` both fail."""


class GitWorkspace:
    """Git operations on a working tree rooted at ``self.root``."""

    root: Path | None
    _tmpdir: str | None

    def __init__(self) -> None:
        self.root = None
        self._tmpdir = None
        self.base_ref = "HEAD"

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
            except subprocess.CalledProcessError as first_err:
                first_err_msg = (first_err.stderr or first_err.stdout or str(first_err)).strip()

            try:
                self._run(["git", "apply", "--3way", str(patch_file)])
                return
            except subprocess.CalledProcessError as second_err:
                second_err_msg = (second_err.stderr or second_err.stdout or str(second_err)).strip()
                details = f"git apply: {first_err_msg}\ngit apply --3way: {second_err_msg}"
                raise ApplyError(
                    f"`git apply` and `git apply --3way` both failed for the generated patch.\n"
                    f"Git Stderr / Details:\n{details}\n\n"
                    f"Patch preview (first 500 chars):\n{diff_text[:500]}"
                ) from second_err
        finally:
            patch_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Working-tree state
    # ------------------------------------------------------------------

    def diff(self) -> str:
        """Cumulative unified diff of the working tree vs. the base commit (incl. new files)."""
        self._run(["git", "add", "--intent-to-add", "."], check=False)
        out = self._run(["git", "diff", "--no-color", "--no-ext-diff", self.base_ref], check=False)
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
        """Stage all changes and commit. Returns the new commit SHA."""
        if self.root:
            (self.root / "_fix_issue.patch").unlink(missing_ok=True)
        self._run(["git", "add", "-A"])
        # Unstage any accidental internal/bytecode artifacts if present
        subprocess.run(
            ["git", "rm", "-r", "--cached", "--ignore-unmatch", "_fix_issue.patch", "*.patch", "__pycache__", "*.pyc"],
            cwd=str(self.root),
            capture_output=True,
        )
        self._run(["git", "commit", "-m", message])
        result = self._run(["git", "rev-parse", "HEAD"])
        return result.stdout.strip()

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

    def _run(self, cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=cwd or str(self.root),
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )


class RepoManager(GitWorkspace):
    """
    Context manager that owns a fresh temp clone for one pipeline run.

    Parameters
    ----------
    repo:   "owner/repo" string
    token:  GitHub personal access token (for private repos + push)
    branch: The branch to create and commit fixes onto
    """

    def __init__(self, repo: str, token: str, branch: str) -> None:
        super().__init__()
        self.repo = repo
        self.token = token
        self.branch = branch

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RepoManager":
        self._tmpdir = tempfile.mkdtemp(prefix="fix-issue-")
        clone_url = f"https://x-access-token:{self.token}@github.com/{self.repo}.git"
        self._run(["git", "clone", "-c", "core.autocrlf=false", "--depth=1", clone_url, "."], cwd=self._tmpdir)
        self.root = Path(self._tmpdir)
        # Drop the credential from .git/config — repo code (tests) must not be able to read it.
        self._run(["git", "remote", "set-url", "origin", f"https://github.com/{self.repo}.git"])
        self._run(["git", "config", "core.autocrlf", "false"])
        _write_tool_excludes(self.root)
        self.base_ref = self._run(["git", "rev-parse", "HEAD"]).stdout.strip()
        # Create the fix branch
        self._run(["git", "checkout", "-b", self.branch])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._tmpdir and Path(self._tmpdir).exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        return False  # do not suppress exceptions

    def push_branch(self) -> None:
        """Force-push the fix branch to origin, authenticating via a one-shot header."""
        basic = base64.b64encode(f"x-access-token:{self.token}".encode()).decode()
        self._run([
            "git", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}",
            "push", "origin", self.branch, "--force",
        ])


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
        _write_tool_excludes(self.root)
        self.base_ref = self._run(["git", "rev-parse", "HEAD"]).stdout.strip()
        self._run(["git", "checkout", "-q", "-B", self.branch])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.cleanup and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
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
