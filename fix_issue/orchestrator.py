"""
orchestrator.py — The issue-resolution pipeline, independent of where issues and code live.

One orchestrator, specialised components, bounded loops, everything recorded:

    fetch issue ─► similar resolved issues ─► triage (control signals)
        │                                          │ needs_info / out_of_scope / duplicate → comment, stop
        ▼                                          ▼
    workspace ─► discover tools ─► BASELINE tests ─► layered retrieval ─► plan (+1 investigation round)
        │
        ▼   attempts ≤ controls.max_retries (+ critique rounds)
    ┌── for each pending plan step (or one repair edit):
    │       generate INTENT + SEARCH/REPLACE edits on CURRENT files
    │       plan_edits (atomic, validated) ─► scope guard on cumulative diff ─► write
    │       failure → recovery analysis (fresh file state) → feedback → next attempt
    ├── tests vs baseline + diff-aware linters/type checkers
    │       failure → recovery (+ plan revision when evidence contradicts the plan)
    └── independent critique (+ deterministic gates)
            NEEDS_REVISION → feedback → another round (bounded) │ REJECTED → stop
        ▼
    commit/push ─► PR (draft when HUMAN_REVIEW) with structured explanation

Sources of issues/code and destinations of output are injected (GitHub or local benchmark
fixtures), so the benchmark runs exactly this code. ``profile="baseline"`` runs the
original keyword-retrieval + unified-diff pipeline for honest before/after comparison.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, ContextManager, Protocol

from fix_issue import editing
from fix_issue.config import Config, embedding_settings, fix_issue_home
from fix_issue.critic import Critique, critique_patch
from fix_issue.llm_provider import LLMError, LLMProvider, RecordingLLM, tagged
from fix_issue.planner import Plan, PlanStep, make_plan, plan_from_retrieval, revise_plan
from fix_issue.pr_builder import PRExplanation, build_pr, generate_llm_summary
from fix_issue.recovery import (
    FailureAnalysis,
    analyze_critique,
    analyze_edit_failure,
    analyze_format_failure,
    analyze_llm_error,
    analyze_scope_failure,
    analyze_test_failure,
    analyze_validator_failure,
)
from fix_issue.redaction import neutralize, redact
from fix_issue.repo_manager import ApplyError, GitWorkspace
from fix_issue.retrieval import EmbeddingCache, RepositoryRetriever, build_embedder
from fix_issue.run_history import RunRecorder
from fix_issue.scope_guard import ScopeConstraints, ScopeViolation, evaluate as scope_evaluate, load_policy
from fix_issue.test_runner import TestResult, run_tests
from fix_issue.triage import TriageResult, apply_duplicate_evidence, derive_controls, triage_issue

# ---------------------------------------------------------------------------
# Injected collaborators
# ---------------------------------------------------------------------------


class IssueSource(Protocol):
    repo: str
    issue_number: int

    def get_issue(self) -> dict[str, Any]: ...
    def readme(self) -> str: ...
    def recent_resolved(self, limit: int = 50) -> list[dict[str, Any]]: ...


class Publisher(Protocol):
    def comment(self, body: str, kind: str) -> None: ...
    def open_pr(self, *, head: str, title: str, body: str, draft: bool) -> str | None: ...


class Reporter:
    """Progress output. The base class is silent; the CLI subclasses it with rich output."""

    def step(self, label: str, detail: str = "") -> None: ...
    def ok(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...
    def fail(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def rule(self, title: str) -> None: ...


@dataclass
class PipelineResult:
    result: str            # pr_opened | dry_run | needs_info | out_of_scope | duplicate | cannot_fix |
                           # scope_violation | rejected | failed | error
    run_id: str
    pr_url: str | None = None
    verdict: str | None = None
    final_diff: str | None = None
    tests_passed: bool | None = None
    attempts: int = 0
    confidence: int | None = None
    pr_title: str | None = None
    pr_body: str | None = None
    error: str | None = None
    record: dict[str, Any] = field(default_factory=dict)


SYSTEM_IMPLEMENTER = """\
You are an expert software engineer implementing a planned fix for a GitHub issue in an existing
repository. Make the smallest correct change that fixes the ROOT CAUSE. Follow the plan, but if the
code you see contradicts the plan, say so in INTENT and do the correct thing. Never touch CI config,
lock files, secrets, or unrelated code. Keep the existing code style. Add or update a focused
regression test when the repository has tests and it is cheap to do so.

{format}
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    def __init__(
        self,
        cfg: Config,
        llm: LLMProvider,
        source: IssueSource,
        workspace_factory: Callable[[], ContextManager[GitWorkspace]],
        publisher: Publisher,
        *,
        critic_llm: LLMProvider | None = None,
        reporter: Reporter | None = None,
        recorder: RunRecorder | None = None,
        dry_run: bool = False,
        branch: str | None = None,
        history_candidates: list[dict] | None = None,
        source_meta: dict | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm if isinstance(llm, RecordingLLM) else RecordingLLM(llm)
        self.critic_llm = critic_llm or self.llm
        if not isinstance(self.critic_llm, RecordingLLM):
            self.critic_llm = RecordingLLM(self.critic_llm)
        self.source = source
        self.workspace_factory = workspace_factory
        self.publisher = publisher
        self.ui = reporter or Reporter()
        self.dry_run = dry_run
        self.branch = branch or f"fix-issue-{source.issue_number}"
        self.history_candidates = history_candidates or []
        self.profile = cfg.pipeline.profile
        self.rec = recorder or RunRecorder(
            repo=source.repo, issue_number=source.issue_number, params=self.params(),
            source=source_meta,
        )

    # ------------------------------------------------------------------
    def params(self) -> dict[str, Any]:
        c = self.cfg
        return {
            "profile": c.pipeline.profile,
            "dry_run": self.dry_run,
            "model": c.llm.model,
            "provider": c.llm.provider,
            "base_url": c.llm.base_url,
            "temperature": c.llm.temperature,
            "max_tokens": c.llm.max_tokens,
            "critic_model": c.critic.model or c.llm.model,
            "limits": c.limits.__dict__,
            "pipeline": c.pipeline.__dict__,
            "retrieval": {k: v for k, v in c.retrieval.__dict__.items() if "key" not in k},
            "validation": c.validation.__dict__,
        }

    def _finish(self, result: str, **kw: Any) -> PipelineResult:
        self.rec.set("llm", {
            "implementer": self.llm.summary(),
            "critic": self.critic_llm.summary() if self.critic_llm is not self.llm else None,
            "calls": self.llm.calls + ([] if self.critic_llm is self.llm else self.critic_llm.calls),
        })
        outcome_fields = {k: v for k, v in kw.items() if k in (
            "pr_url", "verdict", "tests_passed", "confidence", "error", "attempts", "failure_category")}
        if kw.get("final_diff"):
            outcome_fields["diff_lines"] = _count_changed(kw["final_diff"])
        self.rec.finish(result, **outcome_fields)
        return PipelineResult(result=result, run_id=self.rec.run_id, record=self.rec.record,
                              **{k: v for k, v in kw.items() if k in PipelineResult.__dataclass_fields__})

    # ------------------------------------------------------------------
    def run(self) -> PipelineResult:
        try:
            return self._run()
        except Exception as exc:  # any failure (git, GitHub API, provider quirks) is a recorded outcome
            import traceback

            msg = redact(f"{type(exc).__name__}: {exc}")[:2000]
            self.ui.fail(f"Pipeline error: {msg}")
            self.rec.event("error", msg[:1000])
            self.rec.event("traceback", redact(traceback.format_exc())[-3000:])
            return self._finish("error", error=msg)

    def _run(self) -> PipelineResult:
        cfg, ui, rec = self.cfg, self.ui, self.rec
        ui.step("Fetching issue", f"#{self.source.issue_number} from {self.source.repo}")
        issue = self.source.get_issue()
        rec.set("issue", {k: issue.get(k) for k in ("number", "title", "body", "html_url", "state")}
                | {"labels": [lbl.get("name") if isinstance(lbl, dict) else lbl for lbl in issue.get("labels", [])]})
        ui.ok(issue.get("title", ""))
        readme = self.source.readme()

        # --- duplicates + triage ------------------------------------------------
        similar: list[dict] = []
        embedder = self._embedder()
        if self.profile == "full":
            from fix_issue.duplicates import find_similar_issues

            candidates = list(self.source.recent_resolved(50)) + list(self.history_candidates)
            try:
                found = find_similar_issues(issue, candidates, embedder=embedder if embedder and embedder.semantic else None)
                similar = [s.to_dict() for s in found]
            except Exception as exc:
                rec.event("duplicates", f"similarity search failed: {exc}")
            rec.set("similar_issues", similar)

        ui.step("Triaging issue", "legacy classifier" if self.profile == "baseline" else "control signals")
        with tagged(self.llm, "triage"):
            triage = triage_issue(self.llm, issue, readme,
                                  similar_issues=similar if self.profile == "full" else None,
                                  legacy=self.profile == "baseline")
        if self.profile == "full":
            triage = apply_duplicate_evidence(triage, similar, cfg.pipeline.duplicate_threshold)
            controls = derive_controls(triage, cfg.limits, cfg.pipeline)
        else:
            controls = None
        rec.set("triage", _triage_dict(triage))
        rec.set("controls", controls.to_dict() if controls else None)
        ui.ok(f"{triage.classification.upper()} ({triage.confidence:.0%}) — {triage.reason}")
        if controls:
            for line in controls.rationale:
                ui.info(f"control: {line}")

        if triage.classification == "needs_info":
            q = triage.suggested_question or "Could you provide more details or a minimal reproduction case?"
            self.publisher.comment(_needs_info_comment(neutralize(q, 600), self.source.repo,
                                                       self.source.issue_number), "needs_info")
            return self._finish("needs_info")
        if triage.classification in ("out_of_scope", "duplicate"):
            ui.warn(f"Skipping: classified as {triage.classification}. {triage.reason}")
            if triage.classification == "duplicate" and getattr(triage, "duplicate_of", None):
                self.publisher.comment(
                    f"This looks like a duplicate of #{int(triage.duplicate_of)}: {neutralize(triage.reason, 600)}\n\n"
                    "*(Posted automatically by fix-issue.)*", "duplicate")
            return self._finish(triage.classification)

        with self.workspace_factory() as ws:
            if self.profile == "baseline":
                return self._run_legacy(ws, issue, triage)
            return self._run_full(ws, issue, triage, controls, embedder)

    # ------------------------------------------------------------------
    def _embedder(self):
        backend, base_url, key, model = embedding_settings(self.cfg)
        try:
            return build_embedder(backend, base_url=base_url, api_key=key, model=model)
        except ValueError as exc:
            self.rec.event("retrieval", f"embedding backend unavailable, using local: {exc}")
            return build_embedder("local")

    # ======================================================================
    # Full pipeline
    # ======================================================================
    def _run_full(self, ws: GitWorkspace, issue: dict, triage: TriageResult, controls, embedder) -> PipelineResult:
        cfg, ui, rec = self.cfg, self.ui, self.rec
        root = Path(ws.root)
        title, body = issue.get("title", ""), issue.get("body") or ""

        # Scope policy as of the base commit, read BEFORE any repository or model code runs, so
        # nothing the run writes (tests, conftest, edits) can loosen it.
        self._policy = load_policy(root, ws.base_ref)
        rec.set("scope_policy", {"source": self._policy.source, "error": self._policy.read_error})

        # --- tooling + baseline ------------------------------------------------
        test_cmd = ws.detect_test_command()
        from fix_issue.validators import discover_tools, run_validators

        tools, skipped_tools = discover_tools(root, cfg.validation)
        rec.set("tooling", {"test_command": test_cmd, "validators": [t.__dict__ for t in tools],
                            "skipped": skipped_tools,
                            "sandbox": cfg.validation.sandbox_prefix or "none (secrets stripped from env, timeouts)"})
        baseline_failed: set[str] = set()
        baseline: TestResult | None = None
        if test_cmd:
            ui.step("Baseline test run", " ".join(test_cmd))
            baseline = run_tests(test_cmd, root, cfg.limits.test_timeout_seconds, prefix=cfg.validation.sandbox_prefix or None)
            baseline_failed = set(baseline.failed_tests)
            rec.set("baseline", {"passed": baseline.passed, "summary": baseline.summary,
                                 "failed_tests": baseline.failed_tests, "seconds": round(baseline.duration_seconds, 1)})
            (ui.ok if baseline.passed else ui.warn)(f"baseline: {baseline.summary or ('passed' if baseline.passed else 'failing')}")
        else:
            ui.warn("No test command detected — the patch cannot be verified by tests")

        # --- retrieval ------------------------------------------------------------
        ui.step("Retrieving relevant code", f"layered retrieval ({getattr(embedder, 'name', 'no embeddings')})")
        cache_dir = Path(cfg.retrieval.cache_dir) if cfg.retrieval.cache_dir else fix_issue_home() / "cache"
        retriever = RepositoryRetriever(root, embedder=embedder,
                                        cache=EmbeddingCache(cache_dir / "embeddings.jsonl"),
                                        use_embeddings=embedder is not None)
        retrieval = retriever.retrieve(title, body, top_n=max(controls.context_files, cfg.limits.relevant_files))
        rec.set("retrieval", retrieval.to_dict())
        rec.set("retrieval_initial", {"files": [{"path": f.path, "score": round(f.score, 3)} for f in retrieval.files]})
        for f in retrieval.files:
            ui.info(f"{f.path}  ({f.score:.2f}) — {'; '.join(f.reasons[:2])}")
        repo_files = retriever.index.all_paths
        budget = cfg.pipeline.max_context_chars

        # --- planning ---------------------------------------------------------------
        plan: Plan
        context = retriever.render_context(retrieval, budget)
        if cfg.pipeline.planning:
            ui.step("Planning", "root cause, steps, tests, risks")
            plan = self._plan(issue, triage, controls, context, repo_files)
            if plan.needs_more_context:
                hint_paths = [p for p in plan.needs_more_context if p in retriever.index.files]
                hint_syms = [s for s in plan.needs_more_context if s not in hint_paths]
                hint_paths += [p for p in plan.files_to_inspect if p not in retrieval.paths()]
                ui.info(f"planner requested more context: {', '.join(plan.needs_more_context)[:200]}")
                retrieval = retriever.retrieve(title, body, top_n=max(controls.context_files, cfg.limits.relevant_files) + 2,
                                               hint_paths=hint_paths, hint_symbols=hint_syms)
                rec.set("retrieval", retrieval.to_dict() | {"expanded_for_plan": plan.needs_more_context})
                context = retriever.render_context(retrieval, budget, extra_paths=hint_paths)
                plan = self._plan(issue, triage, controls, context, repo_files, previous=plan)
        else:
            plan = plan_from_retrieval(issue, retrieval.paths())
            rec.append("plans", plan.to_dict())

        # --- execution loop -----------------------------------------------------------
        constraints = ScopeConstraints(
            max_diff_lines=controls.max_diff_lines,
            forbidden_paths=list(controls.scope_forbidden),
            allowed_paths=list(controls.scope_allowed),
        )
        decompose = controls.decompose and plan.is_multi_step()
        pending: list[PlanStep | None] = list(plan.ordered_steps()) if decompose else [None]
        feedback: str = ""
        attempt_budget = max(1, controls.max_retries)
        critique_rounds = cfg.pipeline.max_critique_rounds if cfg.pipeline.critique else 0
        plan_revisions = cfg.pipeline.max_plan_revisions
        attempt = 0
        last_failure: FailureAnalysis | None = None
        final: dict[str, Any] | None = None
        # The last change that passed validation and was sent for review. If a later revision round
        # fails, this candidate is published for HUMAN_REVIEW instead of being thrown away.
        candidate: dict[str, Any] | None = None
        preexisting_retry_used = False

        while attempt < attempt_budget:
            attempt += 1
            ui.rule(f"Attempt {attempt} / {attempt_budget}")
            att: dict[str, Any] = {"attempt": attempt, "plan_version": plan.version, "edits": []}
            failure: FailureAnalysis | None = None

            work = pending if pending else [None]
            for step in list(work):
                label = f"step {step.id}" if step else ("repair" if not pending else "single edit")
                ui.step("Generating edits", label)
                outcome = self._edit_round(ws, retriever, retrieval, issue, plan, step, feedback,
                                           constraints, repo_files, budget)
                att["edits"].append(outcome["record"])
                if outcome.get("cannot_fix"):
                    ui.warn(f"Model answered CANNOT_FIX: {outcome['cannot_fix']}")
                    rec.append("attempts", att | {"result": "cannot_fix"})
                    ws.reset_hard()
                    return self._finish("cannot_fix", attempts=attempt, error=outcome["cannot_fix"])
                if outcome.get("failure"):
                    failure = outcome["failure"]
                    break
                ui.ok(f"{label}: {outcome['record'].get('summary')}")
                feedback = ""
                if step is not None and step in pending:
                    pending.remove(step)
            if failure is None:
                pending = []

            diff = ws.diff()
            att["diff"] = diff[:20000]
            if failure is None and not diff.strip():
                failure = FailureAnalysis("edit_error", "The edits produced no change to any file.",
                                          retry_reason="Make an actual code change that fixes the issue.")

            tests: TestResult | None = None
            validator_results: list = []
            if failure is None:
                # --- validation --------------------------------------------------
                if test_cmd:
                    ui.step("Running tests", " ".join(test_cmd))
                    tests = run_tests(test_cmd, root, cfg.limits.test_timeout_seconds,
                                      prefix=cfg.validation.sandbox_prefix or None)
                    att["tests"] = _test_dict(tests, baseline_failed)
                    only_pre = _only_preexisting_failures(tests, baseline_failed)
                    if tests.passed:
                        ui.ok(f"tests passed ({tests.summary or f'{tests.duration_seconds:.1f}s'})")
                    elif only_pre and preexisting_retry_used:
                        # Unchanged pre-existing failures after a repair attempt: let the reviewer
                        # and a human decide (the critic caps this at HUMAN_REVIEW, gate G2).
                        ui.warn("only tests that already failed before the patch still fail: "
                                + ", ".join(sorted(tests.failed_tests)[:5]))
                        att["tests"]["only_preexisting_failures"] = True
                    else:
                        ui.fail(f"tests failed ({tests.summary})")
                        failure = analyze_test_failure(ws, tests, baseline_failed, plan.files_to_change, repo_files)
                        if only_pre:
                            preexisting_retry_used = True
                            failure = replace(failure, retry_reason=(
                                "No NEW test failures, but these tests already failed before your change and "
                                f"still fail: {', '.join(sorted(tests.failed_tests)[:10])}. If they reproduce this "
                                "issue, make them pass. If they are unrelated, keep your fix and say so in INTENT "
                                "(do not re-emit edits that are already applied)."))
                if failure is None and tools:
                    ui.step("Running linters / type checkers", ", ".join(t.name for t in tools))
                    validator_results = run_validators(root, tools, ws.changed_files(), ws.base_ref,
                                                       prefix=cfg.validation.sandbox_prefix or None)
                    att["validators"] = [v.to_dict() for v in validator_results]
                    if any(v.ran and not v.ok for v in validator_results):
                        failure = analyze_validator_failure([v for v in validator_results if v.ran and not v.ok], ws)
                        ui.fail("new linter/type-checker findings")
                    else:
                        ui.ok("no new findings")

            scope_now = None
            if failure is None:
                # The per-round guard checked each prospective edit; re-check the tree as it really is
                # AFTER tests and linters ran (their code may have created or changed files). Enforced
                # here, not only through the critic's gate G4.
                diff = ws.diff()
                att["diff"] = diff[:20000]
                scope_now = scope_evaluate(diff, root, controls.max_diff_lines, constraints,
                                           *self._contents_for(ws), policy=self._policy)
                if not scope_now.ok:
                    failure = replace(analyze_scope_failure(scope_now), retry_reason=(
                        "The working tree's cumulative change violates the scope policy (see above). "
                        "Remove or rework the offending changes so the whole diff stays in scope."))
                    ui.fail("cumulative change violates the scope policy")

            if failure is not None:
                att["failure"] = failure.to_dict()
                att["result"] = "failed"
                rec.append("attempts", att)
                rec.event("retry", f"attempt {attempt} failed: {failure.category}")
                ui.warn(f"attempt {attempt} failed: {failure.category}")
                last_failure = failure
                feedback = failure.feedback()
                if failure.plan_contradiction and plan_revisions > 0 and cfg.pipeline.planning:
                    plan_revisions -= 1
                    ui.step("Revising plan", failure.plan_contradiction[:120])
                    try:
                        with tagged(self.llm, "plan_revision"):
                            new_plan, _ = revise_plan(self.llm, issue, plan,
                                                      f"{failure.plan_contradiction}\n\n{failure.feedback()}",
                                                      retriever.render_context(retrieval, budget,
                                                                               extra_paths=failure.files),
                                                      repo_files)
                        plan = new_plan
                        rec.append("plans", plan.to_dict())
                        ws.reset_hard()
                        decompose = controls.decompose and plan.is_multi_step()
                        pending = list(plan.ordered_steps()) if decompose else [None]
                        # The reset discarded every edit: drop the failure's now-stale view of the tree
                        # (modified files, pre-reset excerpts, "your previous edits ARE applied").
                        after_reset = replace(failure, partial_changes=[], current_excerpts={}, retry_reason=(
                            "The working tree was reset to the base commit, so none of the earlier edits are "
                            "applied. Implement the revised plan against the CURRENT files."))
                        feedback = (f"(Plan revised to version {plan.version}; the working tree was reset.)\n"
                                    + after_reset.feedback())
                    except Exception as exc:
                        rec.event("plan_revision", f"failed: {exc}")
                continue

            # --- independent critique ----------------------------------------------
            test_report = _test_dict(tests, baseline_failed) if tests else {"ran": False}
            test_report["command"] = " ".join(test_cmd) if test_cmd else None
            test_report["tests_touched"] = [p for p in ws.changed_files() if _is_test_path(p)]
            crit: Critique | None = None
            if cfg.pipeline.critique:
                ui.step("Independent review", self.cfg.critic.model or self.cfg.llm.model)
                with tagged(self.critic_llm, "critique"):
                    crit = critique_patch(
                        self.critic_llm, issue=issue, acceptance_criteria=plan.acceptance_criteria
                        or list(getattr(triage, "acceptance_criteria", []) or []),
                        plan_text=plan.render(), diff=diff,
                        surrounding_code=_surrounding(ws, diff),
                        test_report=test_report, validator_report=[v.to_dict() for v in validator_results],
                        scope_report={**scope_now.to_dict(), "summary": _scope_summary(scope_now)},
                        strict=controls.critique_strictness == "strict" or controls.require_human_review,
                    )
                if crit.verdict == "NEEDS_REVISION" and critique_rounds <= 0:
                    # Escalate BEFORE recording, so the run record shows the verdict acted on.
                    crit.verdict = "HUMAN_REVIEW"
                    crit.gates.append("revision budget exhausted → HUMAN_REVIEW")
                rec.append("critiques", crit.to_dict() | {"attempt": attempt})
                ui.info(f"verdict {crit.verdict} (reviewer said {crit.llm_verdict}); {crit.summary[:160]}")
                for g in crit.gates:
                    ui.info(f"gate: {g}")
                if crit.verdict == "NEEDS_REVISION":
                    candidate = {"diff": diff, "tests": tests, "critique": crit, "scope": scope_now,
                                 "validators": validator_results, "attempt": attempt}
                    critique_rounds -= 1
                    attempt_budget += 1   # a review round is its own bounded budget
                    failure = analyze_critique(crit, ws)
                    rec.append("attempts", att | {"result": "needs_revision", "failure": failure.to_dict()})
                    feedback = failure.feedback()
                    last_failure = failure
                    continue
                if crit.verdict == "REJECTED":
                    rec.append("attempts", att | {"result": "rejected"})
                    ws.reset_hard()
                    self.publisher.comment(_rejected_comment(crit), "rejected")
                    return self._finish("rejected", verdict="REJECTED", attempts=attempt,
                                        tests_passed=tests.passed if tests else None, final_diff=diff)
            rec.append("attempts", att | {"result": "accepted"})
            final = {"diff": diff, "tests": tests, "critique": crit, "scope": scope_now,
                     "validators": validator_results, "attempt": attempt}
            break

        if final is None and candidate is not None:
            ws.reset_hard()
            try:
                ws.apply_patch(candidate["diff"])
            except Exception as exc:   # cannot restore -> fall through to the failure path
                rec.event("fallback", f"could not restore the reviewed candidate: {exc}")
            else:
                crit = candidate["critique"]
                crit.verdict = "HUMAN_REVIEW"
                crit.gates.append("the requested revision failed validation -> publishing the reviewed "
                                  "candidate for HUMAN_REVIEW")
                rec.event("fallback", f"revision round failed; restored candidate from attempt {candidate['attempt']}")
                rec.set("final_from_attempt", candidate["attempt"])
                ui.warn("revision failed - falling back to the reviewed candidate (HUMAN_REVIEW)")
                final = candidate

        if final is None:
            ws.reset_hard()
            category = last_failure.category if last_failure else "unknown"
            ui.fail(f"All attempts failed (last failure: {category}).")
            if category == "scope_violation" and last_failure:
                self.publisher.comment(_scope_comment(last_failure.message, self.rec.run_id), "scope_violation")
                return self._finish("scope_violation", attempts=attempt, error=last_failure.message[:1000],
                                    failure_category=category)
            self.publisher.comment(_failure_comment(last_failure, self.rec.run_id), "failed")
            return self._finish("failed", attempts=attempt, failure_category=category,
                                error=(last_failure.message[:1000] if last_failure else None))

        return self._publish(ws, issue, triage, plan, test_cmd, final, baseline_failed)

    # ------------------------------------------------------------------
    def _plan(self, issue, triage, controls, context, repo_files, previous: Plan | None = None) -> Plan:
        tri = {
            "classification": triage.classification, "reason": triage.reason,
            "complexity": getattr(triage, "complexity", None), "risk_level": getattr(triage, "risk_level", None),
            "root_cause_category": getattr(triage, "root_cause_category", None),
            "acceptance_criteria": getattr(triage, "acceptance_criteria", None),
        }
        if previous is not None:
            tri["previous_plan"] = previous.render()
        try:
            with tagged(self.llm, "plan"):
                plan, _ = make_plan(self.llm, issue, tri, context, repo_files, decompose=controls.decompose)
            if previous is not None:
                plan.version = previous.version + 1
                plan.revision_reason = plan.revision_reason or "re-planned with additional requested context"
                plan.needs_more_context = []   # only one investigation round
        except Exception as exc:
            self.rec.event("plan", f"planner failed: {exc}")
            self.ui.warn(f"planner failed ({exc}); continuing with a retrieval-based plan")
            plan = previous or plan_from_retrieval(issue, [])
        self.rec.append("plans", plan.to_dict())
        self.ui.ok(f"plan v{plan.version}: {plan.root_cause[:150]}")
        for s in plan.steps:
            self.ui.info(f"{s.id}: {s.description[:120]} [{', '.join(s.files)}]")
        return plan

    # ------------------------------------------------------------------
    def _contents_for(self, ws: GitWorkspace, overrides: dict[str, str] | None = None) -> tuple[dict, dict]:
        """(base_contents, new_contents) for every file changed in the working tree (+ overrides)."""
        root = Path(ws.root)
        paths = set(ws.changed_files()) | set(overrides or {})
        base: dict[str, str | None] = {}
        new: dict[str, str] = {}
        for p in sorted(paths):
            base[p] = ws.base_content(p)
            if overrides and p in overrides:
                new[p] = overrides[p]
            else:
                fp = root / p
                if fp.is_file():
                    new[p] = fp.read_text(encoding="utf-8", errors="replace")
        return base, new

    def _edit_round(self, ws, retriever, retrieval, issue, plan: Plan, step: PlanStep | None, feedback: str,
                    constraints: ScopeConstraints, repo_files: list[str], budget: int) -> dict[str, Any]:
        root = Path(ws.root)
        extra = list(dict.fromkeys((step.files + step.creates_files) if step else plan.files_to_change))
        context = retriever.render_context(retrieval, budget, extra_paths=extra, live=True)
        so_far = ws.diff()
        step_txt = (
            f"### Current step: {step.id}\n{step.description}\nFiles: {', '.join(step.files + step.creates_files)}\n"
            "Implement ONLY this step now (later steps will follow).\n" if step else
            "### Task\nImplement the whole plan (or the corrections requested below) now.\n"
        )
        prompt = (
            f"### Issue #{issue.get('number')} — {issue.get('title', '')}\n{(issue.get('body') or '').strip()[:5000]}\n\n"
            f"### Implementation plan (v{plan.version})\n{plan.render(step.id if step else None)}\n\n{step_txt}\n"
            f"### Current source (read from the working tree NOW; SEARCH text must match it exactly)\n{context}\n\n"
            + (f"### Changes already applied in this run (cumulative diff vs. base)\n```diff\n{so_far[:8000]}\n```\n\n"
               if so_far.strip() else "")
            + (f"{feedback}\n\n" if feedback else "")
            + "First write INTENT: (1-3 sentences), then the SEARCH/REPLACE blocks."
        )
        system = SYSTEM_IMPLEMENTER.format(format=editing.EDIT_FORMAT_INSTRUCTIONS)
        record: dict[str, Any] = {"step": step.id if step else None}
        try:
            with tagged(self.llm, f"edit:{step.id if step else 'all'}"):
                raw = self.llm.generate(prompt, system=system)
        except LLMError as exc:
            fa = analyze_llm_error(exc)
            return {"failure": fa, "record": record | {"error": str(exc)[:500]}}
        parsed = editing.parse_edit_response(raw)
        record["intent"] = parsed.intent
        if parsed.cannot_fix:
            return {"cannot_fix": parsed.cannot_fix, "record": record | {"cannot_fix": parsed.cannot_fix}}
        if not parsed.edits:
            fa = analyze_format_failure(parsed, ws)
            return {"failure": fa, "record": record | {"format_errors": parsed.format_errors}}
        batch = editing.plan_edits(root, parsed.edits)
        record["batch"] = batch.to_dict()
        if not batch.ok:
            return {"failure": analyze_edit_failure(batch, ws), "record": record}
        # Scope guard on the prospective CUMULATIVE change, before anything is written.
        base, new = self._contents_for(ws, overrides=batch.new_contents)
        prospective = editing.unified_diff(base, new)
        report = scope_evaluate(prospective, root, constraints.max_diff_lines or self.cfg.limits.max_diff_lines,
                                constraints, base, new, policy=getattr(self, "_policy", None))
        record["scope"] = report.to_dict()
        if not report.ok:
            return {"failure": analyze_scope_failure(report), "record": record}
        written = editing.write_edits(root, batch)
        applied = [o for o in batch.outcomes if o.status == "applied"]
        already = [o for o in batch.outcomes if o.status == "already_applied"]
        record["summary"] = (f"{len(applied)} edit(s) applied to {', '.join(written) or 'no files'}"
                             + (f", {len(already)} already present" if already else ""))
        return {"record": record}

    # ------------------------------------------------------------------
    def _publish(self, ws, issue, triage, plan: Plan, test_cmd, final: dict, baseline_failed) -> PipelineResult:
        cfg, ui = self.cfg, self.ui
        diff: str = final["diff"]
        tests: TestResult | None = final["tests"]
        crit: Critique | None = final["critique"]
        scope = final["scope"]
        # Without a review, a change whose tests do not fully pass still goes out only as a draft.
        verdict = crit.verdict if crit else ("HUMAN_REVIEW" if tests and not tests.passed else None)

        with tagged(self.llm, "pr_summary"):
            try:
                summary = generate_llm_summary(self.llm, issue, diff)
            except LLMError as exc:
                summary = f"(summary unavailable: {exc})"
        files_mod = _files_with_counts(diff)
        why = []
        if plan.root_cause and not plan.root_cause.startswith("(planning skipped)"):
            why.append(f"Addresses the planned root cause: {plan.root_cause}")
        why += [f"Acceptance criterion: {a}" for a in plan.acceptance_criteria[:5]]
        if crit and crit.answers.get("addresses_issue", {}).get("details"):
            why.append(f"Reviewer: {crit.answers['addresses_issue']['details']}")
        tests_lines = []
        if tests:
            tests_lines.append(f"`{' '.join(test_cmd)}` → {'passed' if tests.passed else 'FAILED'} ({tests.summary or 'no summary'})")
            fixed = sorted(baseline_failed - set(tests.failed_tests))
            if fixed:
                tests_lines.append(f"previously failing tests now pass: {', '.join(fixed[:10])}")
        else:
            tests_lines.append("No test suite detected — not verified by tests.")
        val_lines = [
            f"{v.tool} ({v.kind}): " + ("skipped — " + str(v.skipped_reason) if not v.ran else
                                        ("no new findings" if v.ok else f"{len(v.new_findings)} new findings"))
            for v in final["validators"]
        ]
        limitations = list(crit.missing_edge_cases[:5]) if crit else []
        if not tests:
            limitations.append("No automated tests ran.")
        if crit and crit.answers.get("tests_meaningful", {}).get("answer") in ("no", "partial"):
            limitations.append("Reviewer: existing tests only partially exercise the fixed behaviour — "
                               + crit.answers["tests_meaningful"].get("details", ""))
        if not cfg.validation.sandbox_prefix:
            limitations.append("Repository code (tests, linters) ran on the host without container isolation; "
                               "its environment was reduced to an allowlist, but it could read the user's files.")
        scope_notes = [f.message for f in scope.findings if f.severity == "review"]
        explanation = PRExplanation(
            root_cause=plan.root_cause if not plan.root_cause.startswith("(planning skipped)") else "",
            files_modified=files_mod, why_it_works=why, tests_executed=tests_lines, validators=val_lines,
            review_verdict=verdict, review_summary=crit.summary if crit else "",
            review_gates=crit.gates if crit else [], scope_notes=scope_notes, limitations=limitations,
            run_id=self.rec.run_id,
        )
        pr = build_pr(issue=issue, diff_text=diff, triage_reason=triage.reason,
                      triage_confidence=triage.confidence, llm_summary=summary,
                      tests_passed=bool(tests and tests.passed), test_command=test_cmd,
                      attempt_count=final["attempt"], repo=self.source.repo, explanation=explanation)
        self.rec.set("pr", {"title": pr.title, "body": pr.body, "draft": pr.draft, "confidence": pr.confidence})
        common = dict(verdict=verdict, final_diff=diff, tests_passed=bool(tests and tests.passed),
                      attempts=final["attempt"], confidence=pr.confidence, pr_title=pr.title, pr_body=pr.body)
        if self.dry_run:
            ui.warn("[DRY RUN] PR not opened")
            return self._finish("dry_run", **common)
        # Last gate before anything leaves the machine: commit_all stages the whole tree, so the
        # tree must still be exactly the change that was validated and reviewed.
        now = ws.diff()
        if now != diff:
            report = scope_evaluate(now, Path(ws.root), self.cfg.limits.max_diff_lines, None,
                                    *self._contents_for(ws), policy=getattr(self, "_policy", None))
            self.rec.event("publish", "working tree changed after review; not committing",
                           ok=report.ok, findings=[f.message for f in report.findings][:10])
            ws.reset_hard()
            return self._finish("scope_violation", error="The working tree changed after validation and "
                                "review; nothing was committed.", failure_category="scope_violation", **common)
        ui.step("Committing and pushing", self.branch)
        ws.commit_all(f"fix: resolve issue #{issue.get('number')} — {issue.get('title', '')[:60]}")
        ws.push_branch()
        url = self.publisher.open_pr(head=self.branch, title=pr.title, body=pr.body, draft=pr.draft)
        ui.ok(f"PR opened: {url}")
        return self._finish("pr_opened", pr_url=url, **common)

    # ======================================================================
    # Legacy pipeline (benchmark baseline): keyword files → unified diff → git apply
    # ======================================================================
    def _run_legacy(self, ws: GitWorkspace, issue: dict, triage: TriageResult) -> PipelineResult:
        from fix_issue.file_selector import format_for_prompt, select_relevant_files
        from fix_issue.patch_generator import generate_patch
        from fix_issue.scope_guard import check as legacy_scope_check

        cfg, ui, rec = self.cfg, self.ui, self.rec
        root = Path(ws.root)
        files = select_relevant_files(issue.get("title", ""), issue.get("body") or "", root,
                                      top_n=cfg.limits.relevant_files)
        rec.set("retrieval", {"files": [{"path": f.path, "score": f.score, "reasons": ["keyword score"]}
                                        for f in files], "method": "keyword"})
        context = format_for_prompt(files)
        test_cmd = ws.detect_test_command()
        rec.set("tooling", {"test_command": test_cmd})
        prior: list[str] = []
        attempts = 0
        for attempt in range(1, cfg.limits.max_retries + 1):
            attempts = attempt
            ui.rule(f"Attempt {attempt} / {cfg.limits.max_retries} (legacy)")
            att: dict[str, Any] = {"attempt": attempt}
            try:
                with tagged(self.llm, "legacy_patch"):
                    diff = generate_patch(self.llm, issue, context, prior, root)
            except (LLMError, ValueError) as exc:
                prior.append(f"LLM patch generation failed: {exc}")
                rec.append("attempts", att | {"result": "failed", "failure": {"category": "edit_error", "message": str(exc)[:2000]}})
                continue
            if diff is None:
                rec.append("attempts", att | {"result": "cannot_fix"})
                return self._finish("cannot_fix", attempts=attempts)
            att["diff"] = diff[:20000]
            try:
                legacy_scope_check(diff, root, cfg.limits.max_diff_lines)
            except ScopeViolation as exc:
                rec.append("attempts", att | {"result": "scope_violation", "failure": {"category": "scope_violation", "message": str(exc)}})
                return self._finish("scope_violation", attempts=attempts, error=str(exc), failure_category="scope_violation")
            try:
                ws.apply_patch(diff)
            except ApplyError as exc:
                prior.append(f"Patch apply failed:\n{exc}")
                rec.append("attempts", att | {"result": "failed", "failure": {"category": "apply_error", "message": str(exc)[:2000]}})
                continue
            if test_cmd:
                res = run_tests(test_cmd, root, cfg.limits.test_timeout_seconds)
                att["tests"] = _test_dict(res, set())
                if res.passed:
                    rec.append("attempts", att | {"result": "accepted"})
                    return self._legacy_done(ws, issue, triage, diff, True, test_cmd, attempts)
                prior.append(res.output)
                rec.append("attempts", att | {"result": "failed", "failure": {"category": "tests_failed", "message": res.output[-2000:]}})
                ws.reset_hard()
            else:
                rec.append("attempts", att | {"result": "accepted"})
                return self._legacy_done(ws, issue, triage, diff, False, test_cmd, attempts)
        ws.reset_hard()
        return self._finish("failed", attempts=attempts, failure_category="exhausted",
                            error=(prior[-1][:1000] if prior else None))

    def _legacy_done(self, ws, issue, triage, diff, passed, test_cmd, attempts) -> PipelineResult:
        with tagged(self.llm, "pr_summary"):
            try:
                summary = generate_llm_summary(self.llm, issue, diff)
            except LLMError as exc:
                summary = f"(summary unavailable: {exc})"
        pr = build_pr(issue=issue, diff_text=diff, triage_reason=triage.reason,
                      triage_confidence=triage.confidence, llm_summary=summary, tests_passed=passed,
                      test_command=test_cmd, attempt_count=attempts, repo=self.source.repo)
        common = dict(final_diff=diff, tests_passed=passed, attempts=attempts, confidence=pr.confidence,
                      pr_title=pr.title, pr_body=pr.body)
        if self.dry_run:
            return self._finish("dry_run", **common)
        ws.commit_all(f"fix: resolve issue #{issue.get('number')} — {issue.get('title', '')[:60]}")
        ws.push_branch()
        url = self.publisher.open_pr(head=self.branch, title=pr.title, body=pr.body, draft=False)
        return self._finish("pr_opened", pr_url=url, **common)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _only_preexisting_failures(t: TestResult, baseline_failed: set[str]) -> bool:
    """Tests failed, but every failure was already failing at baseline (nothing new broke)."""
    return bool(not t.passed and not t.timed_out and baseline_failed and t.failed_tests
                and set(t.failed_tests) <= baseline_failed)


def _count_changed(diff: str) -> int:
    return sum(1 for ln in diff.splitlines()
               if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")))


def _files_with_counts(diff: str) -> list[tuple[str, int]]:
    out: dict[str, int] = {}
    current = None
    for ln in diff.splitlines():
        if ln.startswith("+++ "):
            p = ln[4:].strip()
            current = p[2:] if p.startswith("b/") else (None if p == "/dev/null" else p)
            if current:
                out.setdefault(current, 0)
        elif ln.startswith("--- "):
            p = ln[4:].strip()
            if p != "/dev/null":
                current = p[2:] if p.startswith("a/") else p
                out.setdefault(current, 0)
        elif current and ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")):
            out[current] = out.get(current, 0) + 1
    return list(out.items())


def _is_test_path(p: str) -> bool:
    name = p.rsplit("/", 1)[-1]
    return p.startswith(("tests/", "test/")) or "/tests/" in p or name.startswith("test_") or name.endswith("_test.py")


def _test_dict(t: TestResult | None, baseline_failed: set[str]) -> dict[str, Any]:
    if t is None:
        return {"ran": False}
    return {
        "ran": True, "passed": t.passed, "summary": t.summary, "failed_tests": t.failed_tests,
        "baseline_failures": sorted(baseline_failed),
        "new_failures": sorted(set(t.failed_tests) - baseline_failed),
        "seconds": round(t.duration_seconds, 1), "timed_out": t.timed_out,
        "output_tail": t.output[-3000:],
    }


def _triage_dict(t: TriageResult) -> dict[str, Any]:
    d = {k: v for k, v in t.__dict__.items() if k != "raw"}
    d["raw"] = t.raw
    return d


def _surrounding(ws: GitWorkspace, diff: str, radius: int = 25) -> str:
    """Post-change code around every hunk (±radius lines) for the reviewer."""
    root = Path(ws.root)
    parts: list[str] = []
    current = None
    for ln in diff.splitlines():
        if ln.startswith("+++ "):
            p = ln[4:].strip()
            current = p[2:] if p.startswith("b/") else None
        elif ln.startswith("@@") and current:
            try:
                new_start = int(ln.split("+")[1].split(",")[0].split(" ")[0])
            except (IndexError, ValueError):
                continue
            fp = root / current
            if not fp.is_file():
                continue
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            a, b = max(1, new_start - radius), min(len(lines), new_start + 2 * radius)
            parts.append(f"### {current} (lines {a}-{b}, after the patch)\n```\n"
                         + "\n".join(f"{i:4d}  {lines[i - 1]}" for i in range(a, b + 1)) + "\n```")
    return "\n\n".join(parts)[:16000]


def _scope_summary(report) -> str:
    if not report.findings:
        return f"ok — {report.raw_lines} changed lines (weighted {report.effective_lines:.0f} / limit {report.limit})"
    return "; ".join(f"[{f.severity}] {f.message}" for f in report.findings)


# Public comments carry no raw model, test or tool output: that text can contain local paths,
# file contents printed by repository code, or (despite redaction) credentials. Details stay in
# the local run record, which the comment points to.

_FAILURE_WORDS = {
    "edit_error": "the generated edits did not apply cleanly to the current code",
    "missing_search": "the generated edits did not apply cleanly to the current code",
    "ambiguous_match": "the generated edits matched more than one place in the code",
    "syntax_error": "the edited file no longer parsed",
    "format_error": "the model's answer could not be parsed into edits",
    "tests_failed": "the change did not pass the repository's tests",
    "regression": "the change broke tests that passed before",
    "lint_failed": "the change introduced new linter or type-checker findings",
    "llm_error": "the language-model API call failed",
    "scope_violation": "the change was outside the configured scope policy",
    "critique_revision": "the independent review asked for changes that could not be completed",
}


def _needs_info_comment(question: str, repo: str, number: int) -> str:
    return (
        f"👋 Hi! I looked at this issue and need a bit more information before I can attempt a fix.\n\n"
        f"**{question}**\n\n"
        f"*(This comment was posted automatically by fix-issue. "
        f"Once you provide the details, re-run `fix-issue {repo} {number}` to try again.)*"
    )


def _scope_comment(reason: str, run_id: str | None = None) -> str:
    quoted = neutralize(reason, 1200).replace("\n", "\n> ")
    return (
        f"⚠️ I analysed this issue, but the change it needs is outside the scope policy I run under "
        f"(blocked paths, size limit or paths that need a human):\n\n"
        f"> {quoted}\n\n"
        f"A human review is needed for this one. The limits are `max_diff_lines` in the fix-issue "
        f"config and the repository's `.fix-issue-ignore` / `.fix-issue-scope.toml`."
        + (f" Local run record: `{run_id}`." if run_id else "")
        + "\n\n*(Posted automatically by fix-issue.)*"
    )


def _failure_comment(f: FailureAnalysis | None, run_id: str | None = None) -> str:
    cat = f.category if f else "unknown"
    why = _FAILURE_WORDS.get(cat, "see the run record for details")
    return (
        f"🤖 I attempted to fix this issue but every attempt failed (last failure: `{cat}` — {why}).\n\n"
        f"This may need a more involved fix than I can safely produce automatically."
        + (f" The full log is in the local run record `{run_id}` (`fix-issue runs show {run_id}`)." if run_id else "")
        + "\n\n*(Posted automatically by fix-issue.)*"
    )


def _rejected_comment(c: Critique) -> str:
    return (
        f"🤖 I produced a candidate fix, but an independent review rejected it, so no PR was opened.\n\n"
        f"**Reviewer summary:** {neutralize(c.summary, 1500)}\n\n"
        + ("\n".join(f"- {neutralize(r, 300)}" for r in c.required_changes[:10]) + "\n\n"
           if c.required_changes else "")
        + "*(Posted automatically by fix-issue.)*"
    )


def path_matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)
