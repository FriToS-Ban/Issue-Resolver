"""
cli.py — Entry point for the fix-issue CLI.

    fix-issue <owner/repo> <issue-number> [options]      # unchanged: resolve an issue
    fix-issue resolve <owner/repo> <issue-number> ...    # same, explicit
    fix-issue runs list | show <id> | stats              # inspect run history
    fix-issue rerun <run-id> [overrides]                 # reproduce a run with new parameters
    fix-issue bench validate | retrieval | run | report | compare | review

The pipeline itself lives in orchestrator.py; this module only parses options, builds
clients, and renders progress/history.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from fix_issue import __version__
from fix_issue import config as cfg_module
from fix_issue.llm_provider import LLMProvider, build_provider
from fix_issue.orchestrator import Orchestrator, PipelineResult, Reporter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()


# ---------------------------------------------------------------------------
# Rich progress reporter
# ---------------------------------------------------------------------------

class ConsoleReporter(Reporter):
    def __init__(self, con: Console | None = None, prefix: str = "") -> None:
        self.c = con or console
        self.prefix = prefix

    def step(self, label: str, detail: str = "") -> None:
        d = f" [dim]{detail}[/dim]" if detail else ""
        self.c.print(f"\n{self.prefix}[bold cyan]→[/] {label}{d}")

    def ok(self, msg: str) -> None:
        self.c.print(f"{self.prefix}  [green]✓[/] {msg}")

    def warn(self, msg: str) -> None:
        self.c.print(f"{self.prefix}  [yellow]⚠[/] {msg}")

    def fail(self, msg: str) -> None:
        self.c.print(f"{self.prefix}  [red]✗[/] {msg}")

    def info(self, msg: str) -> None:
        self.c.print(f"{self.prefix}    [dim]{msg}[/dim]")

    def rule(self, title: str) -> None:
        self.c.print(Rule(f"[bold]{title}[/]", style="dim"))


# ---------------------------------------------------------------------------
# Group with a default command so `fix-issue owner/repo 123` keeps working
# ---------------------------------------------------------------------------

class DefaultGroup(click.Group):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["resolve", *args]
        return super().parse_args(ctx, args)


@click.group(cls=DefaultGroup)
@click.version_option(__version__)
def main() -> None:
    """Autonomous GitHub issue resolver — `fix-issue owner/repo 123` to fix an issue."""


def _load_cfg(config_path: str | None) -> cfg_module.Config:
    toml_path = Path(config_path) if config_path else Path.home() / ".fix-issue.toml"
    return cfg_module.load(toml_path)


def build_llms(cfg: cfg_module.Config) -> tuple[LLMProvider, LLMProvider | None]:
    llm = build_provider(cfg.llm.provider, cfg.llm.api_key, cfg.llm.model, cfg.llm.base_url,
                         temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens)
    provider, key, base, model = cfg_module.critic_settings(cfg)
    critic = None
    if cfg.critic.model or cfg.critic.provider:
        critic = build_provider(provider, key, model, base, temperature=cfg.llm.temperature,
                                max_tokens=cfg.llm.max_tokens)
    return llm, critic


def _apply_overrides(cfg: cfg_module.Config, **ov: Any) -> None:
    if ov.get("max_retries") is not None:
        cfg.limits.max_retries = ov["max_retries"]
    if ov.get("max_diff_lines") is not None:
        cfg.limits.max_diff_lines = ov["max_diff_lines"]
    if ov.get("profile"):
        cfg.pipeline.profile = ov["profile"]
    if ov.get("model"):
        cfg.llm.model = ov["model"]
    if ov.get("critic_model"):
        cfg.critic.model = ov["critic_model"]
    if ov.get("no_plan"):
        cfg.pipeline.planning = False
    if ov.get("no_critique"):
        cfg.pipeline.critique = False
    if ov.get("embedding_backend"):
        cfg.retrieval.embedding_backend = ov["embedding_backend"]
    if ov.get("temperature") is not None:
        cfg.llm.temperature = ov["temperature"]


_COMMON_OPTIONS = [
    click.option("--dry-run", is_flag=True, help="Run everything but don't push, open a PR or post comments."),
    click.option("--max-retries", type=int, default=None, help="Override max patch attempts."),
    click.option("--max-diff-lines", type=int, default=None, help="Override max diff lines limit."),
    click.option("--config", "config_path", default=None, help="Path to a custom .fix-issue.toml."),
    click.option("--profile", type=click.Choice(["full", "baseline"]), default=None,
                 help="full = planning/critique/layered retrieval; baseline = original pipeline."),
    click.option("--model", default=None, help="Override the implementer model."),
    click.option("--critic-model", default=None, help="Use a different model for the independent review."),
    click.option("--no-plan", is_flag=True, help="Skip the planning stage."),
    click.option("--no-critique", is_flag=True, help="Skip the independent review."),
    click.option("--embedding-backend", type=click.Choice(["local", "openai_compatible", "none"]), default=None),
    click.option("--temperature", type=float, default=None),
]


def _common(f):
    for opt in reversed(_COMMON_OPTIONS):
        f = opt(f)
    return f


@main.command("resolve")
@click.argument("repo")
@click.argument("issue_number", type=int)
@_common
def resolve_cmd(repo: str, issue_number: int, dry_run: bool, config_path: str | None, **ov: Any) -> None:
    """Fix ISSUE_NUMBER in REPO (owner/repo) and open a PR."""
    cfg = _load_cfg(config_path)
    _apply_overrides(cfg, **ov)
    result = run_github(cfg, repo, issue_number, dry_run)
    sys.exit(0 if result.result in ("pr_opened", "dry_run", "needs_info", "out_of_scope", "duplicate") else 1)


def run_github(cfg: cfg_module.Config, repo: str, issue_number: int, dry_run: bool) -> PipelineResult:
    from fix_issue.adapters import GitHubIssueSource, GitHubPublisher
    from fix_issue.duplicates import resolved_from_history
    from fix_issue.github_client import GitHubClient, GitHubError
    from fix_issue.repo_manager import RepoManager
    from fix_issue.run_history import list_runs

    errors = cfg_module.validate(cfg)
    if errors:
        for e in errors:
            console.print(f"[bold red]Config error:[/] {e}")
        console.print("\n[dim]Set up [bold]~/.fix-issue.toml[/] or environment variables. "
                      "See [bold].env.example[/] for details.[/dim]")
        sys.exit(1)
    if not re.match(r"^[\w.\-]+/[\w.\-]+$", repo):
        console.print(f"[bold red]Invalid repo format:[/] '{repo}'. Expected owner/repo.")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]fix-issue[/] v{__version__}  ·  profile [bold]{cfg.pipeline.profile}[/]\n"
        f"[white]Repo:[/] {repo}   [white]Issue:[/] #{issue_number}   "
        f"[white]Model:[/] {cfg.llm.model}   [white]Critic:[/] {cfg.critic.model or cfg.llm.model}"
        + ("   [yellow][DRY RUN][/]" if dry_run else ""),
        title="🤖 Autonomous Issue Resolver", border_style="cyan",
    ))
    reporter = ConsoleReporter()
    gh = GitHubClient(cfg.github.token)
    try:
        base_branch = gh.get_default_branch(repo)
    except GitHubError as exc:
        console.print(f"[bold red]✗ Could not reach repo:[/] {exc}")
        sys.exit(1)
    llm, critic = build_llms(cfg)
    branch = f"fix-issue-{issue_number}"
    history = resolved_from_history(list_runs(), repo)
    orch = Orchestrator(
        cfg, llm, GitHubIssueSource(gh, repo, issue_number),
        lambda: RepoManager(repo=repo, token=cfg.github.token, branch=branch),
        GitHubPublisher(gh, repo, issue_number, base_branch, dry_run=dry_run, reporter=reporter),
        critic_llm=critic, reporter=reporter, dry_run=dry_run, branch=branch, history_candidates=history,
        source_meta={"type": "github", "repo": repo, "issue": issue_number},
    )
    result = orch.run()
    _final_panel(result, dry_run)
    return result


def _final_panel(result: PipelineResult, dry_run: bool) -> None:
    if dry_run and result.pr_body:
        console.print("\n[yellow][DRY RUN] PR would have been opened:[/]")
        console.print(f"  Title: {result.pr_title}")
        console.print(f"  Confidence: {result.confidence}/10   Review: {result.verdict}")
        console.print("\n[dim]PR body:[/dim]")
        console.print(result.pr_body)
        console.print("\n[dim]Patch:[/dim]")
        console.print(result.final_diff or "")
    colour = "green" if result.result in ("pr_opened", "dry_run") else "yellow"
    console.print(Panel.fit(
        f"[bold {colour}]{result.result}[/]"
        + (f"\nPR: {result.pr_url}" if result.pr_url else "")
        + f"\nReview: {result.verdict or '—'}  •  Attempts: {result.attempts}  •  "
          f"Tests: {'✅' if result.tests_passed else '⚠️'}  •  Confidence: {result.confidence or '—'}"
        + f"\nRun: [bold]{result.run_id}[/]  (fix-issue runs show {result.run_id})",
        border_style=colour,
    ))


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------

@main.group("runs")
def runs_group() -> None:
    """Inspect structured run history (~/.fix-issue/runs, or a benchmark's runs directory)."""


_RUNS_DIR_HELP = ("Read run records from this directory instead of FIX_ISSUE_HOME/runs "
                  "(e.g. benchmark/results/runs/<results-stem>).")


def _runs_dir(path: str | None) -> Path | None:
    return Path(path) if path else None


def _load_run_any(run_id: str, runs_dir_opt: str | None) -> dict[str, Any]:
    """Load a run from --runs-dir, else FIX_ISSUE_HOME/runs, else any benchmark runs directory."""
    from fix_issue.run_history import load_run

    if runs_dir_opt:
        return load_run(run_id, Path(runs_dir_opt))
    try:
        return load_run(run_id)
    except FileNotFoundError as first:
        from fix_issue.bench.cases import DEFAULT_BENCH_DIR

        bench_runs = DEFAULT_BENCH_DIR / "results" / "runs"
        for d in sorted((p for p in bench_runs.glob("*") if p.is_dir()), reverse=True):
            try:
                return load_run(run_id, d)
            except FileNotFoundError as exc:
                if not str(exc).startswith("No run matching"):
                    raise
        raise first


@runs_group.command("list")
@click.option("--limit", type=int, default=20)
@click.option("--repo", default=None)
@click.option("--result", "result_filter", default=None)
@click.option("--json", "as_json", is_flag=True)
@click.option("--runs-dir", "runs_dir_opt", default=None, help=_RUNS_DIR_HELP)
def runs_list(limit: int, repo: str | None, result_filter: str | None, as_json: bool,
              runs_dir_opt: str | None) -> None:
    from fix_issue.run_history import list_runs, summarize

    rows = [summarize(r) for r in list_runs(_runs_dir(runs_dir_opt))]
    if repo:
        rows = [r for r in rows if r["repo"] == repo]
    if result_filter:
        rows = [r for r in rows if r["result"] == result_filter]
    rows = rows[:limit]
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    t = Table(title=f"fix-issue runs ({len(rows)})")
    for col in ("run_id", "started_at", "repo", "issue", "profile", "classification", "files_retrieved",
                "attempts", "verdict", "result", "duration"):
        t.add_column(col, overflow="fold")
    for r in rows:
        t.add_row(*[str(r.get(c) if r.get(c) is not None else "—") for c in (
            "run_id", "started_at", "repo", "issue", "profile", "classification", "files_retrieved",
            "attempts", "verdict", "result", "duration")])
    console.print(t)


_SECTIONS = ("issue", "triage", "controls", "similar_issues", "tooling", "baseline", "retrieval", "plans",
             "attempts", "critiques", "pr", "llm", "outcome", "events", "params")


@runs_group.command("show")
@click.argument("run_id")
@click.option("--section", type=click.Choice(_SECTIONS), default=None, help="Print one section as JSON.")
@click.option("--json", "as_json", is_flag=True, help="Print the whole record as JSON.")
@click.option("--runs-dir", "runs_dir_opt", default=None, help=_RUNS_DIR_HELP)
def runs_show(run_id: str, section: str | None, as_json: bool, runs_dir_opt: str | None) -> None:
    """Show one run (searches benchmark run directories too when not found in FIX_ISSUE_HOME)."""
    try:
        rec = _load_run_any(run_id, runs_dir_opt)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        sys.exit(1)
    if as_json or section:
        data = rec.get(section) if section else rec
        if section == "llm" and isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != "calls"} | {
                "calls": [{k: v for k, v in c.items() if k not in ("prompt_tail", "response")} for c in data.get("calls", [])]}
        click.echo(json.dumps(data, indent=2, default=str))
        return
    _print_run(rec)


def _print_run(rec: dict[str, Any]) -> None:
    out = rec.get("outcome") or {}
    tri = rec.get("triage") or {}
    console.print(Panel.fit(
        f"[bold]{rec.get('run_id')}[/]\n{rec.get('repo')} #{rec.get('issue_number')} — "
        f"{(rec.get('issue') or {}).get('title', '')}\nprofile {(rec.get('params') or {}).get('profile')} · "
        f"model {(rec.get('params') or {}).get('model')} · result [bold]{out.get('result', 'incomplete')}[/] · "
        f"{out.get('duration_seconds', '?')}s", border_style="cyan"))
    if tri:
        console.print(f"[bold]Triage[/] {tri.get('classification')} ({tri.get('confidence')}) — {tri.get('reason')}")
        extras = {k: tri.get(k) for k in ("complexity", "difficulty", "risk_level", "root_cause_category",
                                          "reproduction", "expected_files") if tri.get(k) is not None}
        if extras:
            console.print(f"  [dim]{extras}[/dim]")
    for line in (rec.get("controls") or {}).get("rationale", []):
        console.print(f"  [dim]control: {line}[/dim]")
    if rec.get("similar_issues"):
        console.print("[bold]Similar resolved issues[/]")
        for s in rec["similar_issues"]:
            console.print(f"  #{s.get('number')} {s.get('similarity')} {s.get('title')}")
    if rec.get("baseline"):
        b = rec["baseline"]
        console.print(f"[bold]Baseline tests[/] passed={b.get('passed')} {b.get('summary')}")
    ret = rec.get("retrieval") or {}
    if ret.get("files"):
        console.print(f"[bold]Retrieval[/] ({ret.get('embedding_backend', ret.get('method', ''))})")
        for f in ret["files"]:
            console.print(f"  {f.get('score', 0):>6.2f}  {f['path']}  [dim]{'; '.join(f.get('reasons', [])[:3])}[/dim]")
    for p in rec.get("plans") or []:
        console.print(f"[bold]Plan v{p.get('version')}[/] {p.get('root_cause')}")
        for s in p.get("steps", []):
            console.print(f"  {s.get('id')}: {s.get('description')} [dim]{s.get('files')}[/dim]")
        if p.get("warnings"):
            console.print(f"  [yellow]warnings: {p['warnings']}[/]")
    for a in rec.get("attempts") or []:
        console.print(f"[bold]Attempt {a.get('attempt')}[/] → {a.get('result')}")
        for e in a.get("edits", []) or []:
            console.print(f"  edit {e.get('step') or ''}: {e.get('summary') or 'failed'}  [dim]{(e.get('intent') or '')[:140]}[/dim]")
        if a.get("tests"):
            console.print(f"  tests: passed={a['tests'].get('passed')} {a['tests'].get('summary')}")
        if a.get("failure"):
            console.print(f"  [red]failure {a['failure'].get('category')}[/]: {a['failure'].get('message', '')[:300]}")
    for c in rec.get("critiques") or []:
        console.print(f"[bold]Review[/] {c.get('verdict')} (reviewer: {c.get('llm_verdict')}) — {c.get('summary')}")
        for g in c.get("gates", []):
            console.print(f"  [dim]gate: {g}[/dim]")
    llm = (rec.get("llm") or {}).get("implementer") or {}
    if llm:
        console.print(f"[bold]LLM[/] {llm.get('calls')} calls, {llm.get('seconds')}s, by purpose {llm.get('by_purpose')}")
    if out.get("pr_url"):
        console.print(f"[bold]PR[/] {out['pr_url']}")


@runs_group.command("stats")
@click.option("--runs-dir", "runs_dir_opt", default=None, help=_RUNS_DIR_HELP)
def runs_stats(runs_dir_opt: str | None) -> None:
    from collections import Counter

    from fix_issue.run_history import list_runs, summarize

    rows = [summarize(r) for r in list_runs(_runs_dir(runs_dir_opt))]
    if not rows:
        console.print("No runs recorded yet.")
        return
    t = Table(title=f"{len(rows)} runs")
    t.add_column("result")
    t.add_column("count", justify="right")
    for k, v in Counter(r["result"] for r in rows).most_common():
        t.add_row(str(k), str(v))
    console.print(t)
    verdicts = Counter(r["verdict"] for r in rows if r["verdict"])
    if verdicts:
        console.print(f"review verdicts: {dict(verdicts)}")
    att = [r["attempts"] for r in rows if r["attempts"]]
    if att:
        console.print(f"mean attempts: {sum(att) / len(att):.2f}")


@main.command("rerun")
@click.argument("run_id")
@click.option("--runs-dir", "runs_dir_opt", default=None, help=_RUNS_DIR_HELP)
@_common
def rerun_cmd(run_id: str, dry_run: bool, config_path: str | None, runs_dir_opt: str | None, **ov: Any) -> None:
    """Re-run a recorded run (same issue, same recorded parameters) with optional overrides.

    Benchmark runs are found in benchmark/results/runs/* automatically and, without --config,
    re-run with benchmark/bench.toml (not ~/.fix-issue.toml).
    """
    try:
        rec = _load_run_any(run_id, runs_dir_opt)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        sys.exit(1)
    if config_path is None and (rec.get("source") or {}).get("type") == "bench":
        from fix_issue.bench.cases import DEFAULT_BENCH_DIR

        config_path = str(DEFAULT_BENCH_DIR / "bench.toml")
    cfg = _load_cfg(config_path)
    p = rec.get("params") or {}
    cfg.pipeline.profile = p.get("profile", cfg.pipeline.profile)
    cfg.llm.model = p.get("model", cfg.llm.model)
    if p.get("temperature") is not None:
        cfg.llm.temperature = p["temperature"]
    for k, v in (p.get("limits") or {}).items():
        if hasattr(cfg.limits, k):
            setattr(cfg.limits, k, v)
    for k, v in (p.get("pipeline") or {}).items():
        if hasattr(cfg.pipeline, k):
            setattr(cfg.pipeline, k, v)
    if p.get("critic_model") and p.get("critic_model") != cfg.llm.model:
        cfg.critic.model = p["critic_model"]
    _apply_overrides(cfg, **ov)
    src = rec.get("source") or {}
    console.print(f"[dim]Re-running {run_id} ({src.get('type', 'github')}) with profile {cfg.pipeline.profile}, "
                  f"model {cfg.llm.model}[/dim]")
    if src.get("type") == "bench":
        from fix_issue.bench.cases import DEFAULT_BENCH_DIR, load_case
        from fix_issue.bench.runner import run_case
        from fix_issue.run_history import runs_dir

        case = load_case(DEFAULT_BENCH_DIR / "cases" / src["case_id"])
        graded = run_case(case, cfg, lambda: build_llms(cfg), runs_dir(), reporter=ConsoleReporter())
        click.echo(json.dumps({k: v for k, v in graded.items() if not k.endswith("_tail")}, indent=2))
        return
    run_github(cfg, rec["repo"], int(rec["issue_number"]), dry_run or bool((p or {}).get("dry_run")))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@main.group("bench")
def bench_group() -> None:
    """Reproducible evaluation benchmark on isolated fixture repositories."""


def _bench_dir(path: str | None) -> Path:
    from fix_issue.bench.cases import DEFAULT_BENCH_DIR

    return Path(path) if path else DEFAULT_BENCH_DIR


@bench_group.command("validate")
@click.option("--bench-dir", default=None)
@click.option("--cases", "only", multiple=True, help="Case ids, difficulties or fixtures to include.")
def bench_validate(bench_dir: str | None, only: tuple[str, ...]) -> None:
    """Check every case: clean fixture passes, injected bug is caught by hidden tests."""
    from fix_issue.bench.cases import load_cases
    from fix_issue.bench.validate import validate_case

    cases = load_cases(_bench_dir(bench_dir), list(only) or None)
    bad = 0
    for c in cases:
        r = validate_case(c)
        console.print(f"{'[green]OK[/]  ' if r.ok else '[red]FAIL[/]'} {c.id}")
        for p in r.problems:
            console.print(f"      [red]{p[:800]}[/]")
        bad += not r.ok
    console.print(f"\n{len(cases) - bad}/{len(cases)} cases valid")
    sys.exit(1 if bad else 0)


@bench_group.command("retrieval")
@click.option("--bench-dir", default=None)
@click.option("--cases", "only", multiple=True)
@click.option("--top-n", type=int, default=5)
@click.option("--config", "config_path", default=None, help="Config whose [retrieval] API embedding to include.")
@click.option("--out", default=None, help="Write JSON results here (a .md summary is written next to it).")
def bench_retrieval(bench_dir: str | None, only: tuple[str, ...], top_n: int, config_path: str | None,
                    out: str | None) -> None:
    """Evaluate retrieval alone (no LLM): keyword baseline vs layered variants."""
    from fix_issue.bench.cases import load_cases
    from fix_issue.bench.retrieval_eval import default_configs, evaluate_retrieval, render_retrieval_markdown
    from fix_issue.retrieval import build_embedder

    cases = [c for c in load_cases(_bench_dir(bench_dir), list(only) or None) if c.is_fix]
    api = None
    if config_path:
        cfg = _load_cfg(config_path)
        backend, base, key, model = cfg_module.embedding_settings(cfg)
        if backend == "openai_compatible":
            api = build_embedder(backend, base_url=base, api_key=key, model=model)
    cache = cfg_module.fix_issue_home() / "cache" / "embeddings.jsonl"
    data = evaluate_retrieval(cases, default_configs(api), top_n=top_n, cache_path=cache)
    md = render_retrieval_markdown(data)
    console.print(md)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(data, indent=2), encoding="utf-8")
        Path(out).with_suffix(".md").write_text(md, encoding="utf-8")
        console.print(f"[dim]wrote {out}[/dim]")


@bench_group.command("run")
@click.option("--bench-dir", default=None)
@click.option("--cases", "only", multiple=True)
@click.option("--config", "config_path", default=None, help="Benchmark config (e.g. benchmark/bench.toml).")
@click.option("--profile", type=click.Choice(["full", "baseline"]), default=None)
@click.option("--parallel", type=int, default=2)
@click.option("--out", default=None, help="Results JSON path (default benchmark/results/<profile>-<stamp>.json).")
@click.option("--verbose", is_flag=True, help="Stream pipeline progress for every case.")
def bench_run(bench_dir: str | None, only: tuple[str, ...], config_path: str | None, profile: str | None,
              parallel: int, out: str | None, verbose: bool) -> None:
    """Run the benchmark through the real pipeline with a real LLM and grade it."""
    from datetime import datetime

    from fix_issue.bench.cases import load_cases
    from fix_issue.bench.report import render_markdown
    from fix_issue.bench.runner import run_benchmark

    bdir = _bench_dir(bench_dir)
    cfg = _load_cfg(config_path or str(bdir / "bench.toml"))
    if profile:
        cfg.pipeline.profile = profile
    errors = [e for e in cfg_module.validate(cfg) if "GITHUB_TOKEN" not in e]
    if errors:
        for e in errors:
            console.print(f"[bold red]Config error:[/] {e}")
        sys.exit(1)
    cases = load_cases(bdir, list(only) or None)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(out) if out else bdir / "results" / f"{cfg.pipeline.profile}-{stamp}.json"
    console.print(f"Running {len(cases)} cases · profile {cfg.pipeline.profile} · model {cfg.llm.model} "
                  f"· critic {cfg.critic.model or cfg.llm.model} · parallel {parallel}\n→ {out_path}")

    def progress(r: dict) -> None:
        mark = "[green]✓[/]" if r.get("success") else "[red]✗[/]"
        console.print(f"{mark} {r['case_id']:<48} {r.get('difficulty', ''):<6} "
                      f"class={r.get('classification')} fixed={r.get('issue_fixed')} "
                      f"att={r.get('attempts')} review={r.get('critique_verdict')} "
                      f"{'' if r.get('success') else r.get('failure_category')} ({r.get('duration_seconds')}s)")

    rf = (lambda c: ConsoleReporter(prefix=f"[{c.id[:18]}] ")) if verbose else None
    data = run_benchmark(cases, cfg, lambda: build_llms(cfg), out_path, llm_mode="real",
                         parallel=parallel, reporter_factory=rf, progress=progress)
    md = render_markdown(data)
    out_path.with_suffix(".md").write_text(md, encoding="utf-8")
    console.print(md)


@bench_group.command("report")
@click.argument("results", type=click.Path(exists=True))
def bench_report(results: str) -> None:
    """Re-render a results file (applies manual verdicts if present)."""
    from fix_issue.bench.report import load_results, render_markdown

    data = load_results(Path(results))
    md = render_markdown(data)
    Path(results).with_suffix(".md").write_text(md, encoding="utf-8")
    console.print(md)


@bench_group.command("compare")
@click.argument("baseline", type=click.Path(exists=True))
@click.argument("improved", type=click.Path(exists=True))
@click.option("--out", default=None)
def bench_compare(baseline: str, improved: str, out: str | None) -> None:
    """Side-by-side metrics for two results files."""
    from fix_issue.bench.report import load_results, render_comparison

    md = render_comparison(load_results(Path(baseline)), load_results(Path(improved)))
    console.print(md)
    if out:
        Path(out).write_text(md, encoding="utf-8")


@bench_group.command("review")
@click.argument("results", type=click.Path(exists=True))
@click.option("--all", "include_all", is_flag=True, help="Review every fix case, not only flagged ones.")
@click.option("--set", "set_verdict", multiple=True, help="Non-interactive: CASE_ID=correct|incorrect|unsure")
@click.option("--note", default="")
def bench_review(results: str, include_all: bool, set_verdict: tuple[str, ...], note: str) -> None:
    """Record human correctness verdicts where automated checks are insufficient."""
    from fix_issue.bench.manual import interactive_review, save_verdict
    from fix_issue.run_history import load_run

    path = Path(results)
    if set_verdict:
        for item in set_verdict:
            cid, _, verdict = item.partition("=")
            save_verdict(path, cid.strip(), verdict.strip(), note)
            console.print(f"recorded {cid} = {verdict}")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    runs_dir = Path(data["meta"].get("runs_dir", ""))

    def diff_loader(r: dict) -> str:
        try:
            rec = load_run(r["run_id"], runs_dir)
            atts = rec.get("attempts") or []
            return next((a.get("diff") for a in reversed(atts) if a.get("diff")), "")
        except Exception:
            return ""

    n = interactive_review(path, ask=input, show=console.print, diff_loader=diff_loader, include_all=include_all)
    console.print(f"{n} verdict(s) recorded → run `fix-issue bench report {results}`")


if __name__ == "__main__":
    main()
