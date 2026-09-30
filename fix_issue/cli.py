"""
cli.py — Entry point for the fix-issue CLI.

Usage:
    fix-issue <owner/repo> <issue-number> [options]

This module owns the full pipeline orchestration and the patch retry loop.
Each sub-module (triage, patch_generator, test_runner, etc.) does exactly one thing.
"""

from __future__ import annotations

import re
import sys
from typing import Any

import click
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule


from fix_issue import config as cfg_module
from fix_issue import logger
from fix_issue.file_selector import format_for_prompt, select_relevant_files
from fix_issue.github_client import GitHubClient, GitHubError
from fix_issue.llm_provider import LLMError, build_provider
from fix_issue.patch_generator import generate_patch
from fix_issue.pr_builder import PRContent, build_pr, generate_llm_summary
from fix_issue.repo_manager import ApplyError, RepoManager
from fix_issue.scope_guard import ScopeViolation, check as scope_check
from fix_issue.test_runner import run_tests
from fix_issue.triage import triage_issue

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()



@click.command()
@click.argument("repo")
@click.argument("issue_number", type=int)
@click.option("--dry-run", is_flag=True, help="Run everything but don't open a PR or post comments.")
@click.option("--max-retries", type=int, default=None, help="Override max patch attempts.")
@click.option("--max-diff-lines", type=int, default=None, help="Override max diff lines limit.")
@click.option("--config", "config_path", default=None, help="Path to a custom .fix-issue.toml.")
@click.version_option()
def main(
    repo: str,
    issue_number: int,
    dry_run: bool,
    max_retries: int | None,
    max_diff_lines: int | None,
    config_path: str | None,
) -> None:
    """
    Autonomously fix a GitHub issue and open a PR.

    REPO is in the form owner/repo (e.g. octocat/hello-world).
    ISSUE_NUMBER is the issue number to fix.
    """
    # ---------------------------------------------------------------
    # 0. Load and validate config
    # ---------------------------------------------------------------
    from pathlib import Path
    toml_path = Path(config_path) if config_path else Path.home() / ".fix-issue.toml"
    cfg = cfg_module.load(toml_path)

    if max_retries is not None:
        cfg.limits.max_retries = max_retries
    if max_diff_lines is not None:
        cfg.limits.max_diff_lines = max_diff_lines

    errors = cfg_module.validate(cfg)
    if errors:
        for e in errors:
            console.print(f"[bold red]Config error:[/] {e}")
        console.print(
            "\n[dim]Set up [bold]~/.fix-issue.toml[/] or environment variables. "
            "See [bold].env.example[/] for details.[/dim]"
        )
        sys.exit(1)

    # Validate repo format
    if not re.match(r"^[\w.\-]+/[\w.\-]+$", repo):
        console.print(f"[bold red]Invalid repo format:[/] '{repo}'. Expected owner/repo.")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]fix-issue[/] v0.1.0\n"
        f"[white]Repo:[/] {repo}   [white]Issue:[/] #{issue_number}   "
        f"[white]Model:[/] {cfg.llm.model}   "
        f"[white]Provider:[/] {cfg.llm.provider}"
        + ("   [yellow][DRY RUN][/]" if dry_run else ""),
        title="🤖 Autonomous Issue Resolver",
        border_style="cyan",
    ))

    # ---------------------------------------------------------------
    # Build clients
    # ---------------------------------------------------------------
    gh = GitHubClient(cfg.github.token)
    llm = build_provider(
        provider=cfg.llm.provider,
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
        base_url=cfg.llm.base_url,
    )

    # ---------------------------------------------------------------
    # 1. Fetch issue
    # ---------------------------------------------------------------
    _step("Fetching issue", f"#{issue_number} from {repo}")
    try:
        issue = gh.get_issue(repo, issue_number)
    except GitHubError as e:
        _fail(f"Could not fetch issue: {e}", repo, issue_number, "error", str(e))

    issue_title = issue.get("title", "")
    console.print(f"  [green]✓[/] [bold]{issue_title}[/]")

    # Fetch README snippet for triage context
    readme_snippet = ""
    for readme_path in ("README.md", "README.rst", "README.txt", "readme.md"):
        try:
            content = gh.get_file_content(repo, readme_path)
            if content:
                readme_snippet = content[:2000]
                break
        except Exception:
            pass

    # ---------------------------------------------------------------
    # 2. Triage
    # ---------------------------------------------------------------
    _step("Triaging issue", "classifying with LLM")
    try:
        triage = triage_issue(llm, issue, readme_snippet)
    except (LLMError, ValueError, requests.exceptions.RequestException, TimeoutError) as e:
        _fail(f"Triage failed: {e}", repo, issue_number, "error", str(e))


    _triage_display(triage)

    if triage.classification == "needs_info":
        _handle_needs_info(triage, gh, repo, issue_number, dry_run)
        logger.log_run(
            repo=repo, issue_number=issue_number, result="needs_info",
            classification="needs_info", triage_confidence=triage.confidence,
        )
        return

    if triage.classification in ("out_of_scope", "duplicate"):
        console.print(
            f"\n[yellow]⏭  Skipping:[/] Issue classified as [bold]{triage.classification}[/].\n"
            f"   Reason: {triage.reason}"
        )
        logger.log_run(
            repo=repo, issue_number=issue_number, result=triage.classification,
            classification=triage.classification, triage_confidence=triage.confidence,
        )
        return

    # ---------------------------------------------------------------
    # 3. Clone repo + select relevant files
    # ---------------------------------------------------------------
    default_branch = gh.get_default_branch(repo)
    fix_branch = f"fix-issue-{issue_number}"

    _step("Cloning repo", f"branch: {fix_branch}")

    with RepoManager(repo=repo, token=cfg.github.token, branch=fix_branch) as repo_mgr:
        _step("Selecting relevant files", "keyword scoring")
        relevant_files = select_relevant_files(
            issue_title=issue_title,
            issue_body=issue.get("body") or "",
            repo_root=repo_mgr.root,
            top_n=cfg.limits.relevant_files,
        )
        file_context = format_for_prompt(relevant_files)
        console.print(f"  [green]✓[/] {len(relevant_files)} relevant file(s) selected")

        # Detect test command once
        test_command = repo_mgr.detect_test_command()
        if test_command:
            console.print(f"  [green]✓[/] Test command detected: [dim]{' '.join(test_command)}[/]")
        else:
            console.print("  [yellow]⚠[/] No test command detected — patch will be applied but not tested")

        # ---------------------------------------------------------------
        # 4. Patch retry loop  (owned here in cli.py)
        # ---------------------------------------------------------------
        prior_failures: list[str] = []
        final_diff: str | None = None
        test_passed = False
        attempts = 0

        for attempt in range(1, cfg.limits.max_retries + 1):
            attempts = attempt
            console.print(Rule(f"[bold]Attempt {attempt} / {cfg.limits.max_retries}[/]", style="dim"))

            # 4a. Generate patch
            _step("Generating patch", f"attempt {attempt}")
            console.print("  [dim]Waiting on LLM response...[/dim]")
            try:
                diff = generate_patch(llm, issue, file_context, prior_failures, repo_mgr.root)

            except (LLMError, ValueError, requests.exceptions.RequestException, TimeoutError) as e:
                err_msg = f"LLM patch generation failed or timed out: {e}"
                console.print(f"  [red]✗[/] {err_msg}")
                prior_failures.append(err_msg)
                continue



            if diff is None:
                console.print("  [yellow]⚠[/] Model responded CANNOT_FIX — stopping.")
                logger.log_run(
                    repo=repo, issue_number=issue_number, result="cannot_fix",
                    classification="fixable", triage_confidence=triage.confidence,
                    attempts=attempts,
                )
                return

            console.print(f"  [green]✓[/] Diff generated ({_count_diff_lines(diff)} lines)")

            # 4b. Scope guard — BEFORE any test run
            _step("Checking scope", "diff size + ignored paths")
            try:
                scope_check(diff, repo_mgr.root, cfg.limits.max_diff_lines)
                console.print("  [green]✓[/] Scope checks passed")
            except ScopeViolation as e:
                console.print(f"  [red]✗[/] Scope violation: {e}")
                logger.log_run(
                    repo=repo, issue_number=issue_number, result="scope_violation",
                    classification="fixable", triage_confidence=triage.confidence,
                    attempts=attempts, error_message=str(e),
                )
                _post_scope_comment(gh, repo, issue_number, str(e), dry_run)
                return

            # 4c. Apply patch
            _step("Applying patch", "git apply")
            try:
                repo_mgr.apply_patch(diff)
                console.print("  [green]✓[/] Patch applied")
            except ApplyError as e:
                console.print(f"  [red]✗[/] Patch apply failed: {e}")
                prior_failures.append(f"Patch apply failed:\n{e}")
                continue  # count as failed attempt, try again

            # 4d. Run tests
            if test_command:
                _step("Running test suite", " ".join(test_command))
                result = run_tests(test_command, repo_mgr.root, cfg.limits.test_timeout_seconds)
                elapsed = f"{result.duration_seconds:.1f}s"
                if result.passed:
                    console.print(f"  [green]✓[/] Tests passed ({elapsed})")
                    test_passed = True
                    final_diff = diff
                    break
                else:
                    console.print(f"  [red]✗[/] Tests failed ({elapsed})")
                    console.print(f"[dim]{result.output[-600:]}[/dim]")
                    prior_failures.append(result.output)
                    # Reset working tree for next attempt
                    try:
                        repo_mgr._run(["git", "checkout", "."])
                        repo_mgr._run(["git", "clean", "-fd"])
                    except Exception:
                        pass
            else:
                # No test command — accept the patch if it applies cleanly
                console.print("  [yellow]⚠[/] No tests — accepting patch on clean apply")
                test_passed = False
                final_diff = diff
                break

        # ---------------------------------------------------------------
        # 5. Commit + push
        # ---------------------------------------------------------------
        if final_diff is None:
            console.print(
                f"\n[bold red]✗ All {cfg.limits.max_retries} attempts failed.[/] "
                f"Could not produce a passing patch."
            )
            _post_failure_comment(gh, repo, issue_number, prior_failures, dry_run)
            logger.log_run(
                repo=repo, issue_number=issue_number, result="error",
                classification="fixable", triage_confidence=triage.confidence,
                attempts=attempts,
                error_message=f"All {cfg.limits.max_retries} attempts failed",
            )
            sys.exit(1)

        if not dry_run:
            _step("Committing changes", fix_branch)
            commit_msg = f"fix: resolve issue #{issue_number} — {issue_title[:60]}"
            repo_mgr.commit_all(commit_msg)
            console.print("  [green]✓[/] Changes committed")

            _step("Pushing branch", fix_branch)
            repo_mgr.push_branch()
            console.print(f"  [green]✓[/] Pushed branch [bold]{fix_branch}[/]")

    # ---------------------------------------------------------------
    # 6. Build PR + open it
    # ---------------------------------------------------------------
    _step("Building PR", "generating summary")
    llm_summary = generate_llm_summary(llm, issue, final_diff)
    pr_content = build_pr(
        issue=issue,
        diff_text=final_diff,
        triage_reason=triage.reason,
        triage_confidence=triage.confidence,
        llm_summary=llm_summary,
        tests_passed=test_passed,
        test_command=test_command,
        attempt_count=attempts,
        repo=repo,
    )

    if dry_run:
        console.print("\n[yellow][DRY RUN] PR would have been opened:[/]")
        console.print(f"  Title: {pr_content.title}")
        console.print(f"  Confidence: {pr_content.confidence}/10")
        console.print("\n[dim]PR body:[/dim]")
        console.print(pr_content.body)
        console.print("\n[dim]Patch (final_diff):[/dim]")
        console.print(final_diff)
        logger.log_run(
            repo=repo, issue_number=issue_number, result="dry_run",
            classification="fixable", triage_confidence=triage.confidence,
            confidence=pr_content.confidence, attempts=attempts,
        )
        return

    _step("Opening PR", repo)
    try:
        # Check for an existing PR on this branch first
        existing = gh.list_open_prs(repo, fix_branch)
        if existing:
            pr_url = existing[0]["html_url"]
            console.print(f"  [yellow]⚠[/] PR already exists: {pr_url}")
        else:
            pr = gh.create_pr(
                repo=repo,
                head=fix_branch,
                base=default_branch,
                title=pr_content.title,
                body=pr_content.body,
            )
            pr_url = pr["html_url"]
            console.print(f"  [green]✓[/] PR opened: [link={pr_url}]{pr_url}[/link]")
    except GitHubError as e:
        _fail(f"Could not open PR: {e}", repo, issue_number, "error", str(e))

    logger.log_run(
        repo=repo, issue_number=issue_number, result="pr_opened",
        pr_url=pr_url, confidence=pr_content.confidence,
        classification="fixable", triage_confidence=triage.confidence,
        attempts=attempts,
    )

    console.print(Panel.fit(
        f"[bold green]✓ Done![/]\n"
        f"PR: [link={pr_url}]{pr_url}[/link]\n"
        f"Confidence: [bold]{pr_content.confidence}/10[/]  •  "
        f"Attempts: {attempts}  •  Tests: {'✅' if test_passed else '⚠️'}",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(label: str, detail: str = "") -> None:
    detail_str = f" [dim]{detail}[/dim]" if detail else ""
    console.print(f"\n[bold cyan]→[/] {label}{detail_str}")


def _fail(
    message: str,
    repo: str,
    issue_number: int,
    result: str,
    error_message: str,
) -> None:
    console.print(f"\n[bold red]✗ {message}[/]")
    logger.log_run(
        repo=repo, issue_number=issue_number, result=result,
        error_message=error_message,
    )
    sys.exit(1)


def _triage_display(triage: Any) -> None:
    colour = {
        "fixable": "green",
        "needs_info": "yellow",
        "out_of_scope": "red",
        "duplicate": "magenta",
    }.get(triage.classification, "white")
    console.print(
        f"  [{colour}]✓ {triage.classification.upper()}[/]  "
        f"(confidence: {triage.confidence:.0%})  —  {triage.reason}"
    )


def _handle_needs_info(triage: Any, gh: GitHubClient, repo: str, issue_number: int, dry_run: bool) -> None:
    question = triage.suggested_question or "Could you provide more details or a minimal reproduction case?"
    comment_body = (
        f"👋 Hi! I looked at this issue and need a bit more information before I can attempt a fix.\n\n"
        f"**{question}**\n\n"
        f"*(This comment was posted automatically by fix-issue. "
        f"Once you provide the details, re-run `fix-issue {repo} {issue_number}` to try again.)*"
    )
    if dry_run:
        console.print("\n[yellow][DRY RUN] Would post comment:[/]")
        console.print(comment_body)
    else:
        try:
            gh.post_comment(repo, issue_number, comment_body)
            console.print(f"  [green]✓[/] Clarifying question posted on issue #{issue_number}")
        except GitHubError as e:
            console.print(f"  [yellow]⚠[/] Could not post comment: {e}")


def _post_scope_comment(gh: GitHubClient, repo: str, issue_number: int, reason: str, dry_run: bool) -> None:
    body = (
        f"⚠️ I analysed this issue but the required change exceeds my configured scope limits:\n\n"
        f"> {reason}\n\n"
        f"A human review is needed for this one. "
        f"If you want to allow larger diffs, adjust `max_diff_lines` in `~/.fix-issue.toml` "
        f"or update `.fix-issue-ignore` in the repo.\n\n"
        f"*(Posted automatically by fix-issue.)*"
    )
    if dry_run:
        console.print("\n[yellow][DRY RUN] Would post scope comment[/]")
        return
    try:
        gh.post_comment(repo, issue_number, body)
    except GitHubError:
        pass


def _post_failure_comment(
    gh: GitHubClient,
    repo: str,
    issue_number: int,
    prior_failures: list[str],
    dry_run: bool,
) -> None:
    last_failure = prior_failures[-1][:1500] if prior_failures else "No details available."
    body = (
        f"🤖 I attempted to fix this issue but all retry attempts failed.\n\n"
        f"**Last test failure:**\n```\n{last_failure}\n```\n\n"
        f"This may need a more involved fix than I can safely produce automatically. "
        f"*(Posted automatically by fix-issue.)*"
    )
    if dry_run:
        console.print("\n[yellow][DRY RUN] Would post failure comment[/]")
        return
    try:
        gh.post_comment(repo, issue_number, body)
    except GitHubError:
        pass


def _count_diff_lines(diff_text: str) -> int:
    return sum(
        1 for line in diff_text.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("---")
        and not line.startswith("+++")
    )


if __name__ == "__main__":
    main()

