"""
github_client.py — Thin wrapper around the GitHub REST API.

All calls use plain `requests` + a GITHUB_TOKEN.
No PyGitHub or other SDK required.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import requests


class GitHubError(Exception):
    """Raised when a GitHub API call fails."""


class GitHubClient:
    BASE = "https://api.github.com"

    def __init__(self, token: str) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "fix-issue/0.1.0",
            }
        )

    # ------------------------------------------------------------------
    # Issues
    # ------------------------------------------------------------------

    def get_issue(self, repo: str, number: int) -> dict[str, Any]:
        """Return the full issue object for `owner/repo#number`."""
        resp = self._get(f"/repos/{repo}/issues/{number}")
        return resp

    def post_comment(self, repo: str, issue_number: int, body: str) -> dict[str, Any]:
        """Post a comment on an issue."""
        return self._post(
            f"/repos/{repo}/issues/{issue_number}/comments",
            {"body": body},
        )

    # ------------------------------------------------------------------
    # Repo contents
    # ------------------------------------------------------------------

    def list_tree(self, repo: str, ref: str = "HEAD") -> list[str]:
        """
        Return a flat list of all file paths in the repo at `ref`.
        Uses the Git Trees API with recursive=1 — one API call regardless of size.
        """
        # First resolve ref → sha
        ref_data = self._get(f"/repos/{repo}/git/ref/heads/{ref}")
        sha = ref_data["object"]["sha"]

        tree_data = self._get(
            f"/repos/{repo}/git/trees/{sha}",
            params={"recursive": "1"},
        )
        return [
            item["path"]
            for item in tree_data.get("tree", [])
            if item["type"] == "blob"
        ]

    def get_file_content(self, repo: str, path: str, ref: str = "HEAD") -> str:
        """Return the decoded text content of a file. Returns '' on binary or missing."""
        try:
            data = self._get(
                f"/repos/{repo}/contents/{path}",
                params={"ref": ref},
            )
        except GitHubError:
            return ""
        if data.get("encoding") == "base64":
            try:
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            except Exception:
                return ""
        return data.get("content", "")

    def get_default_branch(self, repo: str) -> str:
        data = self._get(f"/repos/{repo}")
        return data.get("default_branch", "main")

    # ------------------------------------------------------------------
    # Branches & PRs
    # ------------------------------------------------------------------

    def get_branch_sha(self, repo: str, branch: str) -> str:
        """Return the HEAD commit SHA of `branch`."""
        data = self._get(f"/repos/{repo}/branches/{branch}")
        return data["commit"]["sha"]

    def create_branch(self, repo: str, new_branch: str, from_sha: str) -> None:
        """Create a new branch pointing at `from_sha`."""
        self._post(
            f"/repos/{repo}/git/refs",
            {"ref": f"refs/heads/{new_branch}", "sha": from_sha},
        )

    def create_pr(
        self,
        repo: str,
        *,
        head: str,
        base: str,
        title: str,
        body: str,
        draft: bool = False,
    ) -> dict[str, Any]:

        """Open a pull request and return the PR object."""
        return self._post(
            f"/repos/{repo}/pulls",
            {
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": draft,
            },
        )



    def list_open_prs(self, repo: str, head_branch: str) -> list[dict[str, Any]]:
        """Check whether a PR already exists for a given head branch."""
        owner = repo.split("/")[0]
        return self._get(
            f"/repos/{repo}/pulls",
            params={"state": "open", "head": f"{owner}:{head_branch}"},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> Any:
        resp = self._session.get(f"{self.BASE}{path}", params=params, timeout=30)
        self._check(resp)
        return resp.json()

    def _post(self, path: str, payload: dict) -> Any:
        resp = self._session.post(
            f"{self.BASE}{path}", json=payload, timeout=30
        )
        self._check(resp)
        return resp.json()

    @staticmethod
    def _check(resp: requests.Response) -> None:
        if not resp.ok:
            try:
                msg = resp.json().get("message", resp.text[:300])
            except Exception:
                msg = resp.text[:300]
            raise GitHubError(f"GitHub API {resp.status_code}: {msg}")
