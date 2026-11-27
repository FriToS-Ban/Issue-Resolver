"""Regressions found by the review of the git hardening (replace refs, diff prefixes, commit scope,
URL rewrites, tags). Local repositories only; no network."""

from __future__ import annotations

from pathlib import Path

from fix_issue.repo_manager import LocalWorkspace, RepoManager, git_env
from fix_issue.scope_guard import evaluate
from tests.test_repo_manager import _git, _init_repo


def test_git_env_ignores_replace_refs():
    assert git_env()["GIT_NO_REPLACE_OBJECTS"] == "1"


def test_replace_ref_cannot_hide_a_workflow_from_the_diff(tmp_path):
    root = tmp_path / "repo"
    base = _init_repo(root)
    with LocalWorkspace(root, branch="fix") as ws:
        # Repository code fakes a base commit whose tree already contains its workflow.
        wf = root / ".github" / "workflows" / "ci.yml"
        wf.parent.mkdir(parents=True)
        wf.write_bytes(b"on: push\n")
        _git(root, "add", "-A")
        fake_tree = _git(root, "write-tree").stdout.strip()
        fake = _git(root, "commit-tree", fake_tree, "-m", "fake base").stdout.strip()
        _git(root, "reset", "-q")
        _git(root, "replace", base, fake)
        (root / "a.txt").write_bytes(b"fixed\n")
        diff = ws.diff()
    assert ".github/workflows/ci.yml" in diff
    assert not evaluate(diff, root).ok


def test_diff_prefixes_are_fixed_whatever_the_config(tmp_path):
    root = tmp_path / "repo"
    _init_repo(root)
    _git(root, "config", "diff.mnemonicPrefix", "true")
    _git(root, "config", "diff.noprefix", "true")
    with LocalWorkspace(root, branch="fix") as ws:
        (root / "a.txt").write_bytes(b"fixed\n")
        diff = ws.diff()
    assert "--- a/a.txt" in diff and "+++ b/a.txt" in diff


def test_commit_all_keeps_tracked_patch_and_pyc_files(tmp_path):
    root = tmp_path / "repo"
    _init_repo(root)
    (root / "patches").mkdir()
    (root / "patches" / "upstream.patch").write_bytes(b"--- a\n+++ b\n")
    (root / "vendored.pyc").write_bytes(b"\x00\x01")
    _git(root, "add", "-f", "-A")
    _git(root, "commit", "-q", "-m", "tracked artifacts")
    with LocalWorkspace(root, branch="fix") as ws:
        (root / "a.txt").write_bytes(b"fixed\n")
        sha = ws.commit_all("fix: x")
    files = _git(root, "ls-tree", "-r", "--name-only", sha).stdout.split()
    assert "patches/upstream.patch" in files and "vendored.pyc" in files
    changed = _git(root, "diff", "--name-only", f"{sha}^", sha).stdout.split()
    assert changed == ["a.txt"]


def test_auth_args_pin_the_remote_url_and_push_skips_tags():
    rm = RepoManager("octo/repo", token="", branch="fix-issue-1")
    url = rm._remote_url()
    args = rm._auth_args()
    assert f"url.{url}.insteadOf={url}" in args
    import inspect
    assert "--no-follow-tags" in inspect.getsource(RepoManager.push_branch)
