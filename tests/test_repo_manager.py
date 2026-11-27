"""
repo_manager.py git hygiene: repository code that ran in the working tree (tests, a model-written
conftest.py) must not get git to run anything — no hooks, no planted fsmonitor / filter / textconv
/ url rewrites — and the GitHub token must never reach argv errors, tracebacks or .git/config.

Every "attack" is planted AFTER the workspace was entered, exactly when repository code runs. Each
test also proves the vector is live by firing it with plain, unhardened git at the end.
"""

from __future__ import annotations

import base64
import os
import stat
import subprocess
import tempfile
import traceback
from pathlib import Path

import pytest

from fix_issue import repo_manager
from fix_issue.repo_manager import (
    GitCommandError,
    LocalWorkspace,
    RepoManager,
    git_command,
    git_env,
    redact,
    token_forms,
)
from fix_issue.sandbox import ProcessResult

TOKEN = "ghp_FAKEtoken1234567890abcdefXYZ"
B64 = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Plain, UNHARDENED git (stands in for the user / the attacker)."""
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "commit.gpgsign=false", *args],
                          cwd=cwd, check=check, capture_output=True, text=True)


def _init_repo(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    for key, value in (("user.email", "bench@example.invalid"), ("user.name", "bench"),
                       ("core.autocrlf", "false"), ("commit.gpgsign", "false")):
        _git(root, "config", key, value)
    (root / "a.txt").write_text("one\n", newline="\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def _script(path: Path, marker: Path, tag: str, tail: str = "") -> Path:
    path.write_text(f"#!/bin/sh\necho {tag} >> '{marker.as_posix()}'\n{tail}", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


HOOKS = ("pre-commit", "commit-msg", "prepare-commit-msg", "post-commit", "post-checkout",
         "reference-transaction", "post-index-change", "pre-push", "pre-auto-gc", "post-rewrite")


def _plant_hooks(git_dir: Path, marker: Path) -> None:
    (git_dir / "hooks").mkdir(exist_ok=True)
    for name in HOOKS:
        _script(git_dir / "hooks" / name, marker, name)


def _plant_config(root: Path, marker: Path, scripts: Path) -> None:
    scripts.mkdir(exist_ok=True)
    fsmon = _script(scripts / "fsmonitor.sh", marker, "fsmonitor")
    clean = _script(scripts / "clean.sh", marker, "clean", "cat\n")
    textconv = _script(scripts / "textconv.sh", marker, "textconv", 'cat "$1"\n')
    ssh = _script(scripts / "ssh.sh", marker, "ssh")
    with open(root / ".git" / "config", "a", encoding="utf-8", newline="\n") as f:
        f.write(f"[core]\n\tfsmonitor = {fsmon.as_posix()}\n\tsshCommand = {ssh.as_posix()}\n"
                f"[filter \"evil\"]\n\tclean = sh '{clean.as_posix()}'\n\tsmudge = sh '{clean.as_posix()}'\n"
                f"[diff \"evil\"]\n\ttextconv = sh '{textconv.as_posix()}'\n")
    (root / ".gitattributes").write_text("*.txt filter=evil diff=evil\n", newline="\n")


@pytest.fixture
def git_identity(monkeypatch):
    for k in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(k, "fix-issue test")
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(k, "test@example.invalid")


# ---------------------------------------------------------------------------
# LocalWorkspace (benchmark) — same hardening as the GitHub clone
# ---------------------------------------------------------------------------

def test_planted_hooks_never_run(tmp_path):
    root = tmp_path / "repo"
    base = _init_repo(root)
    marker = tmp_path / "hooks-ran.txt"
    with LocalWorkspace(root, branch="fix") as ws:
        _plant_hooks(root / ".git", marker)                       # "repository code" runs now
        (root / "a.txt").write_text("two\n", newline="\n")
        assert "two" in ws.diff() and ws.changed_files() == ["a.txt"]
        ws.reset_hard()                                           # used on failed / dry-run paths
        assert (root / "a.txt").read_text() == "one\n"
        (root / "a.txt").write_text("three\n", newline="\n")
        sha = ws.commit_all("fix: x")
        ws.push_branch()
    assert not marker.exists(), marker.read_text()
    assert _git(root, "rev-parse", f"{sha}^").stdout.strip() == base
    _git(root, "commit", "-q", "--allow-empty", "-m", "control")  # control: the hooks are live
    assert marker.exists()


def test_planted_config_is_restored_and_has_no_effect(tmp_path):
    root = tmp_path / "repo"
    _init_repo(root)
    marker = tmp_path / "config-ran.txt"
    with LocalWorkspace(root, branch="fix") as ws:
        snapshot = (root / ".git" / "config").read_bytes()
        _plant_config(root, marker, tmp_path / "scripts")
        (root / "a.txt").write_text("two\n", newline="\n")
        diff = ws.diff()
        assert "+two" in diff
        assert set(ws.changed_files()) == {"a.txt", ".gitattributes"}
        assert ws.base_content("a.txt") == "one\n"
        assert (root / ".git" / "config").read_bytes() == snapshot
        ws.commit_all("fix: x")
        ws.reset_hard()
    assert not marker.exists(), marker.read_text()

    _plant_config(root, marker, tmp_path / "scripts")             # control: the vectors are live
    (root / "a.txt").write_text("four\n", newline="\n")
    _git(root, "status", "--porcelain", check=False)
    _git(root, "diff", "HEAD", check=False)
    fired = set(marker.read_text().split()) if marker.exists() else set()
    assert {"clean", "textconv"} <= fired, fired


def test_planted_commondir_is_removed_and_swapped_git_dir_aborts(tmp_path):
    root = tmp_path / "repo"
    _init_repo(root)
    with LocalWorkspace(root, branch="fix") as ws:
        evil = tmp_path / "evil-common"
        evil.mkdir()
        (root / ".git" / "commondir").write_text(str(evil))
        (root / ".git" / "config.worktree").write_text("[core]\n\tfsmonitor = /bin/false\n")
        assert ws.changed_files() == []
        assert not (root / ".git" / "commondir").exists()
        assert not (root / ".git" / "config.worktree").exists()

        config = root / ".git" / "config"
        snapshot = config.read_bytes()
        config.write_text("[core]\n\tfsmonitor = /evil\n")
        config.chmod(stat.S_IREAD)                                  # read-only does not stop the restore
        assert ws.changed_files() == [] and config.read_bytes() == snapshot

        (root / ".git").rename(root / ".git-real")                  # a .git file pointing elsewhere
        (root / ".git").write_text("gitdir: .git-real\n")
        with pytest.raises(GitCommandError, match="refusing to run git"):
            ws.diff()
        with pytest.raises(GitCommandError):
            git_command(root, "worktree", "prune")
        (root / ".git").unlink()
        _git(tmp_path, "clone", "-q", "--no-checkout", str(root / ".git-real"), str(tmp_path / "other"))
        (tmp_path / "other" / ".git").rename(root / ".git")         # a different real directory
        with pytest.raises(GitCommandError, match="no longer the original directory"):
            ws.changed_files()
    assert git_command(root, "status")[0] == "git"    # unregistered once the workspace exited


def test_commit_sits_on_the_base_even_if_repo_code_moved_the_branch(tmp_path):
    root = tmp_path / "repo"
    base = _init_repo(root)
    with LocalWorkspace(root, branch="fix") as ws:
        tree = _git(root, "write-tree").stdout.strip()
        rogue = _git(root, "commit-tree", tree, "-p", base, "-m", "rogue").stdout.strip()
        _git(root, "update-ref", "refs/heads/fix", rogue)
        _git(root, "symbolic-ref", "HEAD", "refs/heads/elsewhere")
        (root / "a.txt").write_text("fixed\n", newline="\n")
        sha = ws.commit_all("fix: x")
    assert _git(root, "rev-parse", f"{sha}^").stdout.strip() == base
    assert _git(root, "symbolic-ref", "HEAD").stdout.strip() == "refs/heads/fix"
    assert _git(root, "show", f"{sha}:a.txt").stdout == "fixed\n"


def test_git_env_is_the_allowlist_plus_git_network_settings(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "FAKE")
    monkeypatch.setenv("NVIDIA_API_KEY", "FAKE")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'core.hookspath'='/evil'")
    monkeypatch.setenv("GIT_DIR", "/evil")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    env = git_env()
    assert not {"GITHUB_TOKEN", "NVIDIA_API_KEY", "GIT_CONFIG_PARAMETERS", "GIT_DIR"} & set(env)
    assert env["HTTPS_PROXY"] == "http://proxy.local:3128"
    assert env["GIT_TERMINAL_PROMPT"] == "0" and env["PATH"] == os.environ["PATH"]


def test_hardening_args_and_empty_hooks_dir_is_kept_empty():
    args = repo_manager.git_hardening_args()
    assert "core.fsmonitor=false" in args and "gc.auto=0" in args
    hooks = Path(next(a for a in args if a.startswith("core.hooksPath=")).split("=", 1)[1])
    assert hooks.is_dir() and not any(hooks.iterdir())
    (hooks / "pre-commit").write_text("#!/bin/sh\n")                # planted by repository code
    again = Path(next(a for a in repo_manager.git_hardening_args()
                      if a.startswith("core.hooksPath=")).split("=", 1)[1])
    assert again.is_dir() and not any(again.iterdir())


def test_git_timeouts_raise(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _init_repo(root)
    ws = LocalWorkspace(root)
    monkeypatch.setattr(repo_manager, "run_process",
                        lambda *a, **k: ProcessResult(None, "", "", 1.0, timed_out=True))
    with pytest.raises(GitCommandError, match="timed out"):
        ws._run(["git", "status"], check=False)      # a timeout is never a silent empty result


def test_auth_is_a_per_command_header_with_tls_verification_and_no_credential_helpers():
    args = RepoManager("o/r", TOKEN, "b")._auth_args()
    assert f"http.https://github.com/.extraheader=AUTHORIZATION: basic {B64}" in args
    assert "credential.helper=" in args and "http.sslVerify=true" in args
    assert RepoManager("o/r", "", "b")._auth_args() == ["-c", "http.sslVerify=true"]


def test_redact():
    text = (f"fatal: https://x-access-token:{TOKEN}@github.com/o/r.git AUTHORIZATION: basic {B64} "
            f"http.extraheader=Authorization: Bearer abc.def {TOKEN}")
    out = redact(text, token_forms(TOKEN))
    assert TOKEN not in out and B64 not in out and "abc.def" not in out
    assert "https://***@github.com/o/r.git" in out
    assert redact("plain text", []) == "plain text"


# ---------------------------------------------------------------------------
# RepoManager (GitHub clone) against a local "GitHub"
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_github(tmp_path, monkeypatch, git_identity):
    src = tmp_path / "src"
    _init_repo(src)
    hosted = tmp_path / "github"
    (hosted / "owner").mkdir(parents=True)
    _git(tmp_path, "clone", "-q", "--bare", str(src), str(hosted / "owner" / "repo.git"))
    monkeypatch.setattr(RepoManager, "github_url", hosted.as_uri() + "/")
    made: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def mkdtemp(*a, **k):
        d = real_mkdtemp(*a, **k)
        made.append(d)
        return d

    monkeypatch.setattr(repo_manager.tempfile, "mkdtemp", mkdtemp)
    return hosted, made


def test_clone_commit_push_with_planted_hooks_and_config(tmp_path, fake_github):
    hosted, made = fake_github
    remote = hosted / "owner" / "repo.git"
    attacker = tmp_path / "attacker.git"
    _git(tmp_path, "init", "-q", "--bare", str(attacker))
    marker = tmp_path / "ran.txt"
    with RepoManager("owner/repo", TOKEN, "fix-issue-1") as rm:
        root = rm.root
        assert TOKEN not in (root / ".git" / "config").read_text() and B64 not in (root / ".git" / "config").read_text()
        _plant_hooks(root / ".git", marker)
        _plant_config(root, marker, tmp_path / "scripts")
        with open(root / ".git" / "config", "a", encoding="utf-8", newline="\n") as f:
            f.write(f"[url \"{attacker.as_uri()}\"]\n\tinsteadOf = {RepoManager.github_url}\n"
                    f"\tpushInsteadOf = {RepoManager.github_url}\n"
                    f"[remote \"origin\"]\n\tpushurl = {attacker.as_uri()}\n")
        (root / ".gitattributes").unlink()
        (root / "a.txt").write_text("fixed\n", newline="\n")
        sha = rm.commit_all("fix: resolve issue #1")
        rm.push_branch()
    assert not marker.exists(), marker.read_text()
    assert _git(remote, "rev-parse", "refs/heads/fix-issue-1").stdout.strip() == sha
    assert _git(attacker, "show-ref", check=False).stdout.strip() == ""
    assert made and not any(Path(d).exists() for d in made if "nohooks" not in d)


def test_clone_failure_is_redacted_runtime_error_and_temp_dir_removed(fake_github):
    _, made = fake_github
    with pytest.raises(GitCommandError) as info:
        with RepoManager("owner/does-not-exist", TOKEN, "fix-issue-1"):
            pass
    exc = info.value
    assert isinstance(exc, RuntimeError)     # Orchestrator.run catches RuntimeError
    rendered = "".join(traceback.format_exception(exc)) + repr(exc.cmd) + exc.stdout + exc.stderr
    for form in token_forms(TOKEN):
        assert form not in rendered
    assert "clone" in str(exc)
    assert made and not any(Path(d).exists() for d in made if "nohooks" not in d)


def test_push_failure_is_redacted(tmp_path, fake_github, monkeypatch):
    _, made = fake_github
    with RepoManager("owner/repo", TOKEN, "fix-issue-1") as rm:
        (rm.root / "a.txt").write_text("fixed\n", newline="\n")
        rm.commit_all("fix")
        monkeypatch.setattr(RepoManager, "github_url", (tmp_path / "nowhere").as_uri() + "/")
        with pytest.raises(GitCommandError) as info:
            rm.push_branch()
    rendered = "".join(traceback.format_exception(info.value)) + repr(info.value.cmd) + info.value.stderr
    assert "push" in rendered
    for form in token_forms(TOKEN):
        assert form not in rendered
    assert made and not any(Path(d).exists() for d in made if "nohooks" not in d)   # read-only objects too
