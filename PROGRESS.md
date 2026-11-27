# Progress Log - Issue Resolver (`fix-issue`)

## What This Project Is
A local Python CLI (`fix-issue <owner/repo> <issue-number>`) that fetches a GitHub issue, triages it, clones the repo to a temp dir, generates a patch via an LLM, applies it, runs the test suite (retrying on failure up to MAX_RETRIES), and opens a PR with a confidence score. Runs manually, on one machine, no hosted infra. Provider-agnostic LLM layer — works with Anthropic, OpenAI-compatible APIs (including NVIDIA NIM), or Gemini via config, not code changes.

## Architecture (built and verified working)
- fix_issue/cli.py — owns the full pipeline + retry loop
- fix_issue/config.py — loads ~/.fix-issue.toml + env overrides
- fix_issue/github_client.py — GitHub REST API via plain `requests`, no SDK
- fix_issue/llm_provider.py — abstract LLMProvider + AnthropicProvider, OpenAICompatibleProvider (covers OpenAI/NIM/Ollama), GeminiProvider
- fix_issue/repo_manager.py — context-manager clone, git apply → --3way fallback
- fix_issue/file_selector.py — keyword-based relevance scoring
- fix_issue/triage.py — LLM classification: fixable/needs_info/out_of_scope/duplicate
- fix_issue/patch_generator.py — LLM unified diff generation
- fix_issue/scope_guard.py — pre-test check: .fix-issue-ignore + MAX_DIFF_LINES, runs AFTER patch generation (needs a diff to check), before apply/test
- fix_issue/test_runner.py — runs detected test command, 5min timeout
- fix_issue/pr_builder.py — PR body with confidence formula shown as a table
- fix_issue/logger.py — appends JSON line per run to ~/.fix-issue-runs.jsonl
- Config split: ~/.fix-issue.toml (secrets/personal limits, never in a repo) vs .fix-issue-ignore (repo scope rules, checked into VCS)
- Confidence formula: score = max(1, 10 − min(5, diff_lines//30) − (3 if tests failed) − (2 if triage_conf < 0.7)) — shown with full breakdown in every PR

## Verification Completed
- 47/47 unit tests passing across 4 test files
- pyproject.toml confirmed clean (no tomllib as a dependency — it's stdlib on Python 3.11+, requires-python correctly set to >=3.11)
- Retry loop indentation confirmed correct — both break statements are properly inside the for loop, tests failing correctly triggers retry not silent exit
- git apply failure path confirmed to append the ApplyError to prior_failures before retrying, so the LLM sees why the last patch didn't apply — same treatment as test failures
- --dry-run confirmed to run the full pipeline (triage → files → patch → scope_guard → apply/test loop) but skip commit/push/create_pr/post_comment, and prints the final diff + confidence score + would-be PR body for inspection
- LLM provider confirmed provider-agnostic: no hardcoded OpenAI URL, correct Bearer auth, standard chat-completions body shape — verified working live against NVIDIA NIM (model: meta/llama-3.1-8b-instruct) via a smoke test script that calls llm_provider.generate() directly, got a real response back
- ~/.fix-issue.toml confirmed populated with working NIM config (provider, base_url, api_key, model, github token)

## Current Blocker / Where We Stopped
The target repo has zero open issues, so there's nothing to run the pipeline against yet. Need to plant a deliberate small bug in the repo and open a corresponding GitHub issue describing it, so there's a genuinely small, well-scoped test case (typo, off-by-one, missing null check — not a feature) to validate the full pipeline against.

## Next Steps (in order)
1. Plant a small, deliberate bug in the repo (e.g. a typo in a comment/docstring, an off-by-one loop bound, a missing null check) — something easy enough that a small LLM should be able to fix it correctly, to test the pipeline mechanics rather than the model's coding ceiling
2. Open a GitHub issue describing that bug as if it were discovered normally (`gh issue create` or via the GitHub web UI)
3. Run `python -m fix_issue.cli owner/repo <issue-number> --dry-run` against it
4. Inspect the printed diff, confidence score breakdown, and PR body — confirm the diff actually fixes the planted bug and applies/passes tests cleanly
5. If dry-run output looks right, run it for real (drop --dry-run) and confirm a PR actually opens on GitHub with the expected content
6. Repeat steps 1-5 with 4-5 more planted or real small issues before trusting it on anything unplanned — this is the dogfood phase
7. After dogfooding is stable, review ~/.fix-issue-runs.jsonl for patterns in what fails or succeeds, and tighten .fix-issue-ignore based on what the agent tried to touch that it shouldn't have

## Known Limitations / Things to Watch
- meta/llama-3.1-8b-instruct is a small, general (non-coder-tuned) model — expect a higher rate of first-attempt patches that fail to apply cleanly; the retry loop and --3way fallback exist specifically for this, it's not a bug
- Auto-merge is intentionally out of scope — every PR is human-reviewed
- No hosted/webhook version exists yet — that's a deliberately later phase, requires actual hosting (Railway/Render/Fly.io) and is out of scope until the CLI is proven

## Log
- 2026-09-25: Created initial `PROGRESS.md` tracking project history, architecture, verifications (47/47 tests passing), current blocker (no open issues on target repo), and next steps for dogfooding.
- 2026-09-25: Handled 2 edge cases in `patch_generator.py`: (1) `_fuzzy_replace` now explicitly raises `ValueError` if a SEARCH block fails to match target file content, feeding error details back into `prior_failures` for retry; (2) Removed silent fallback for non-SEARCH/REPLACE outputs, raising `LLMError` to strictly enforce SEARCH/REPLACE block formatting.








- 2026-11-27: Repository-aware agent rebuild (branch `feat/repository-aware-agent`).
  - New pipeline in `fix_issue/orchestrator.py`: control-plane triage → baseline tests → layered retrieval (`fix_issue/retrieval/`) → validated plan (`planner.py`) → step-wise structured SEARCH/REPLACE edits (`editing.py`) with cumulative path/severity-aware scope checks (`scope_guard.evaluate`) → tests + diff-aware linters (`validators.py`) → recovery with fresh file state (`recovery.py`) → independent critique with deterministic gates (`critic.py`) → PR with structured explanation (draft on HUMAN_REVIEW). `--profile baseline` keeps the original pipeline.
  - Benchmark: `benchmark/` has 3 fixture repos and 23 cases (9 easy / 7 medium / 7 hard incl. 3 triage cases); `fix-issue bench validate` → 23/23 valid. Hidden tests separate "tests pass" from "issue fixed".
  - Retrieval benchmark (no LLM, 20 fix cases, top-5): keyword recall@5 0.82 / MRR 0.50 → layered 0.93 / 0.71; symbol recall 0.00 → 0.69. Misses: shopcart-m2 (utils/dates.py, two hops), shopcart-h2 (pricing.py).
  - Live smoke: 1 case (taskq-e1) on NIM `nvidia/nemotron-3-ultra-550b-a55b` + critic `mistralai/mistral-large-4-1t-a52b`: fixed, hidden tests pass, APPROVED, 1 attempt. The full 23-case baseline-vs-full benchmark has NOT been run yet.
  - Environment: `deepseek-ai/deepseek-v4.1-flash` (in ~/.fix-issue.toml) is end-of-life on NIM and the key there is rejected (403); a working key lives in the gitignored `.env` (NVIDIA_API_KEY). Use Python 3.12 (`C:\Python312\python.exe`) — `python` on PATH is 3.15 without pytest.
  - Next: `fix-issue bench run --profile baseline` and `--profile full` (≈15–25 min each), `fix-issue bench compare`, README rewrite, open the PR.
- 2026-11-28: Review + hardening pass on `feat/repository-aware-agent` (see `git log 1bc964f..HEAD`; each commit explains its fix). Correctness, security and regression reviews were run by sub-agents; all verified findings fixed with tests (514 passed, 1 skipped). Retrieval with real NIM embeddings now works (422 content-safety refusals handled) but adds no gain over lexical/structural layers (`benchmark/results/retrieval-api.md`). Duplicate thresholds recalibrated per scorer on `tests/data/duplicate_pairs.json`. Final baseline: 17/23 (`benchmark/results/baseline-20261128-final.md`, tool commit f87c42d). Final full run was in progress at 093804e (`benchmark/results/full-20261128.json`, 10/23 done, 9 ok at 04:36). Pre-fix diagnostic runs in `benchmark/results/diagnostic/`. PR not opened yet.
