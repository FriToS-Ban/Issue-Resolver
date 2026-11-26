"""
adapters.py — GitHub implementations of the orchestrator's IssueSource / Publisher, plus a
recording publisher used by dry runs and the benchmark.
"""

from __future__ import annotations

from typing import Any

from fix_issue.github_client import GitHubClient, GitHubError


class GitHubIssueSource:
    def __init__(self, gh: GitHubClient, repo: str, issue_number: int) -> None:
        self.gh = gh
        self.repo = repo
        self.issue_number = issue_number

    def get_issue(self) -> dict[str, Any]:
        return self.gh.get_issue(self.repo, self.issue_number)

    def readme(self) -> str:
        for path in ("README.md", "README.rst", "README.txt", "readme.md"):
            try:
                content = self.gh.get_file_content(self.repo, path)
                if content:
                    return content[:2000]
            except Exception:
                continue
        return ""

    def recent_resolved(self, limit: int = 50) -> list[dict[str, Any]]:
        from fix_issue.duplicates import recent_resolved_from_github

        try:
            return recent_resolved_from_github(self.gh, self.repo, limit)
        except Exception:
            return []


class GitHubPublisher:
    """Posts comments / opens PRs; in dry-run mode only records what it would have done."""

    def __init__(self, gh: GitHubClient, repo: str, issue_number: int, base_branch: str,
                 dry_run: bool = False, reporter: Any = None) -> None:
        self.gh = gh
        self.repo = repo
        self.issue_number = issue_number
        self.base_branch = base_branch
        self.dry_run = dry_run
        self.ui = reporter
        self.actions: list[dict[str, Any]] = []

    def comment(self, body: str, kind: str) -> None:
        self.actions.append({"type": "comment", "kind": kind, "body": body})
        if self.dry_run:
            if self.ui:
                self.ui.warn(f"[DRY RUN] would post a '{kind}' comment")
                self.ui.info(body[:600])
            return
        try:
            self.gh.post_comment(self.repo, self.issue_number, body)
        except GitHubError as exc:
            if self.ui:
                self.ui.warn(f"Could not post comment: {exc}")

    def open_pr(self, *, head: str, title: str, body: str, draft: bool) -> str | None:
        self.actions.append({"type": "pr", "head": head, "title": title, "draft": draft})
        if self.dry_run:
            return None
        existing = self.gh.list_open_prs(self.repo, head)
        if existing:
            return existing[0]["html_url"]
        pr = self.gh.create_pr(self.repo, head=head, base=self.base_branch, title=title, body=body, draft=draft)
        return pr["html_url"]


class RecordingPublisher:
    """Never talks to GitHub — records intended comments/PRs (benchmark, tests)."""

    def __init__(self) -> None:
        self.actions: list[dict[str, Any]] = []

    def comment(self, body: str, kind: str) -> None:
        self.actions.append({"type": "comment", "kind": kind, "body": body})

    def open_pr(self, *, head: str, title: str, body: str, draft: bool) -> str | None:
        self.actions.append({"type": "pr", "head": head, "title": title, "draft": draft, "body": body})
        return f"local://{head}"
