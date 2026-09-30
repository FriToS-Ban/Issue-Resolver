"""
repo_manager.py — Clone a repo to a temp dir, detect its test command, apply patches.

Usage (always via context manager so the temp dir is always cleaned up):

    with RepoManager("owner/repo") as repo:
        repo.apply_patch(diff_text)   # raises ApplyError on hard failure
        ...
    # temp dir is gone here, even if an exception fired mid-pipeline
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


class ApplyError(Exception):
    """Raised when `git apply` and `git apply --3way` both fail."""


class RepoManager:
    """
    Context manager that owns a fresh temp clone for one pipeline run.

    Parameters
    ----------
    repo:   "owner/repo" string
    token:  GitHub personal access token (for private repos + push)
    branch: The branch to create and commit fixes onto
    """

    def __init__(self, repo: str, token: str, branch: str) -> None:
        self.repo = repo
        self.token = token
        self.branch = branch
        self._tmpdir: str | None = None
        self.root: Path | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RepoManager":
        self._tmpdir = tempfile.mkdtemp(prefix="fix-issue-")
        clone_url = f"https://x-access-token:{self.token}@github.com/{self.repo}.git"
        self._run(["git", "clone", "-c", "core.autocrlf=false", "--depth=1", clone_url, "."], cwd=self._tmpdir)
        self.root = Path(self._tmpdir)
        self._run(["git", "config", "core.autocrlf", "false"])
        # Configure local git exclude rules for tool artifacts
        exclude_file = self.root / ".git" / "info" / "exclude"
        if exclude_file.parent.exists():
            with open(exclude_file, "a", encoding="utf-8") as f:
                f.write("\n# fix-issue tool internal artifacts\n_fix_issue.patch\n*.patch\n__pycache__/\n*.pyc\n*.pyo\n")
        # Create the fix branch
        self._run(["git", "checkout", "-b", self.branch])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._tmpdir and Path(self._tmpdir).exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Patch application
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


    def commit_all(self, message: str) -> str:
        """Stage all changes and commit. Returns the new commit SHA."""
        # Ensure patch file is unlinked before staging
        if self.root:
            (self.root / "_fix_issue.patch").unlink(missing_ok=True)
        self._run(["git", "add", "-A"])
        # Unstage any accidental internal/bytecode artifacts if present
        subprocess.run(
            ["git", "rm", "-r", "--cached", "--ignore-unmatch", "_fix_issue.patch", "*.patch", "__pycache__", "*.pyc"],
            cwd=self._tmpdir,
            capture_output=True,
        )
        self._run(["git", "commit", "-m", message])
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self._tmpdir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def push_branch(self) -> None:
        """Force-push the fix branch to origin."""
        self._run(["git", "push", "origin", self.branch, "--force"])



    # ------------------------------------------------------------------
    # Test command detection
    # ------------------------------------------------------------------

    def detect_test_command(self) -> list[str] | None:
        """
        Return a shell command (as a list) for running the repo's test suite.
        Tries common conventions in priority order.
        Returns None if nothing recognisable is found.
        """
        root = self.root

        # 1. pytest.ini / setup.cfg [tool:pytest] / pyproject.toml [tool.pytest]
        if (
            (root / "pytest.ini").exists()
            or (root / "setup.cfg").exists()
            or _toml_has_pytest(root / "pyproject.toml")
        ):
            return ["python", "-m", "pytest", "--tb=short", "-q"]

        # 2. Bare setup.py with tests/
        if (root / "setup.py").exists() and (root / "tests").exists():
            return ["python", "-m", "pytest", "--tb=short", "-q"]

        # 3. package.json — look for "test" script
        pkg = root / "package.json"
        if pkg.exists():
            import json
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

        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=cwd or self._tmpdir,
            check=True,
            capture_output=True,
            text=True,
        )


def _toml_has_pytest(path: Path) -> bool:
    if not path.exists():
        return False
    content = path.read_text(errors="replace")
    return "[tool.pytest" in content
