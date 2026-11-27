# fix-issue

A local command-line tool that takes a GitHub issue, decides whether it can be fixed
automatically, finds the relevant code, writes a plan, makes small validated edits, runs the
repository's own tests and linters, has the result reviewed by a separate model call, and only
then opens a pull request. When it cannot fix an issue it says why in a comment on the issue.

```bash
fix-issue owner/repo 123 --dry-run     # run the whole pipeline, print the PR it would open
fix-issue owner/repo 123               # same, then push a branch and open the PR
```

Everything runs on your machine. There is no server. LLM calls go through plain HTTP to
Anthropic, Gemini, or any OpenAI-compatible endpoint (OpenAI, NVIDIA NIM, Ollama, LM Studio).
No PR is ever merged automatically.

There are two pipelines in the code:

- **full** (default): the repository-aware pipeline described below.
- **baseline** (`--profile baseline`): the original keyword-retrieval, single-diff pipeline,
  kept unchanged so the benchmark can compare the two on the same models and cases.

---

## Contents

- [How it works](#how-it-works)
- [Requirements and installation](#requirements-and-installation)
- [Configuration](#configuration)
- [Commands](#commands)
- [Profiles: full vs baseline](#profiles-full-vs-baseline)
- [Benchmark](#benchmark)
- [Security and limitations](#security-and-limitations)
- [Module reference](#module-reference)
- [Running the unit tests](#running-the-unit-tests)

---

## How it works

```
fetch issue ─► similar resolved issues ─► triage (+ control signals)
                                              │ needs_info / out_of_scope / duplicate → comment, stop
                                              ▼
clone ─► discover tests + linters ─► BASELINE test run ─► layered retrieval ─► plan (+1 investigation round)
                                              │
          attempts ≤ max_retries (triage may lower it; review rounds add attempts)
   ┌── for each plan step (or one edit round):
   │      model writes INTENT + SEARCH/REPLACE blocks against the CURRENT files
   │      edits applied atomically in memory ─► scope guard on the cumulative diff ─► write
   ├── tests (compared with the baseline run) + the repo's linters/type checkers (new findings only)
   │      any failure → recovery analysis from the current file state → next attempt
   │                    (plan revised when the evidence contradicts it)
   └── independent review + deterministic gates
          NEEDS_REVISION → another round (bounded) │ REJECTED → comment, stop
                                              ▼
commit ─► push fix-issue-<N> ─► PR (draft when the verdict is HUMAN_REVIEW) ─► run record
```

The pipeline lives in `fix_issue/orchestrator.py`. Where issues come from and where output goes
are injected, so the benchmark runs exactly the same code against local fixture repositories. The
steps below describe the full profile; [Profiles](#profiles-full-vs-baseline) lists what the
baseline does instead.

### 1. Similar resolved issues (duplicate evidence)

Before triage, the issue is compared with up to 50 recently closed issues of the repository (pull
requests and issues closed as "not planned" are skipped) and with earlier fix-issue runs on the
same repository that opened a PR. Similarity is `0.75 × embedding cosine + 0.25 × title-token
Jaccard`; the embedding uses the configured API model when `[retrieval] embedding_backend =
"openai_compatible"`, otherwise local n-gram vectors. The top matches are passed to triage as
evidence. A "duplicate" verdict is kept only when a listed issue reaches
`[pipeline] duplicate_threshold` for the scorer that produced its similarity (local n-gram and API
embedding scores are on different scales, so the defaults in `fix_issue/config.py` are per
scorer); otherwise it is downgraded to `fixable` (or `needs_info` when the report has no
reproduction) with confidence capped at 0.5.

### 2. Triage and control signals

Triage returns one JSON object, validated against a schema (`structured.py`): classification
(`fixable` / `needs_info` / `out_of_scope` / `duplicate`), confidence, complexity, difficulty, risk
level, reproduction completeness, root-cause category, expected number of files, recommended retry
and diff budgets, suggested allowed/forbidden paths, acceptance criteria and `duplicate_of`.
Invalid values fall back to safe defaults and are recorded as warnings.

`triage.derive_controls` turns that into control signals with fixed rules. Triage can only
tighten the configured limits, never loosen them:

| Signal | Rule |
|---|---|
| retry budget | `min(max_retries, recommended)`, at least 1; complex/hard issues keep the full configured budget |
| diff budget | `min(max_diff_lines, recommended)`, never below 10 |
| decomposition | step-by-step execution for moderate/complex issues or 3+ expected files |
| review strictness | strict when risk is high, triage confidence < 0.6, or there is no reproduction |
| scope | triage's forbidden paths block; its allowed paths only produce review findings |
| context size | `clamp(expected_files + 3, 4, 10)` files, at least `relevant_files` |

The model's confidence is only ever used to make review stricter. `needs_info` posts the
clarifying question as a comment, `duplicate` posts a pointer to the original issue (when one is
identified), `out_of_scope` stops without a comment.

### 3. Workspace, tooling, baseline

The repository is cloned into a temporary directory that is always removed. The test command is
detected from the repository (pytest configuration, `setup.py` + `tests/`, a `package.json`
`test` script, a Makefile `test` target, `tox.ini`, `go.mod`, or a `tests/` directory with
`test_*.py`). The repository's configured linters and type checkers are discovered (see step 8).
The tests are run once before any change, so failures that already existed are not blamed on the
patch.

### 4. Layered retrieval

`fix_issue/retrieval/` builds a small index of the repository (Python through `ast`; JS/TS, Go,
Java/Kotlin, Rust, Ruby and C# through conservative regexes) and scores files on several signals:

| Signal | Weight | Source |
|---|---|---|
| explicit | 3.0 | file paths and dotted module names in the issue text |
| traceback | 3.0 | files in stack-trace frames (innermost strongest) |
| symbol_def | 2.0 | files defining identifiers the issue mentions |
| lexical | 1.0 | BM25 over code-aware tokens |
| semantic | 1.0 | embedding similarity over symbol chunks |
| callee / caller | 0.9 / 0.7 | definitions called by, and files referencing, the focus symbols |
| imports / imported_by | 0.35 / 0.3 | import-graph neighbours of strong seeds |
| hint | 1.5 | files or symbols the planner asks for |

Graph expansions are scaled by the strength of the seed they came from. Files below 20% of the top
score are dropped (explicitly named and traceback files are always kept). Every selected file
carries the reasons it was chosen, which are shown during the run and stored in the run record.
Large files are rendered as an outline plus the full text of the relevant symbols, within
`[pipeline] max_context_chars`. Embeddings are pluggable: local n-gram vectors (default, offline,
not a semantic model), any OpenAI-compatible `/embeddings` endpoint, or none. API vectors are
cached on disk.

### 5. Plan and one investigation round

Before writing code the model commits to a plan: root cause, files to inspect, relevant symbols,
expected change per file, ordered steps with dependencies, tests that should validate the fix,
regression risks, acceptance criteria, and any extra context it still needs. The plan is checked
against the real repository: unknown files are dropped unless a step creates them, step ids must
be unique, dependencies must exist, cycles are broken and steps are ordered topologically. If the
planner asks for more context, retrieval runs again with those files and symbols as hints and the
plan is redone once. If planning fails, a plan built from the retrieval results is used. With
`--no-plan` that retrieval-based plan is used from the start.

### 6. Structured edits and recovery

The model answers with a short `INTENT:` and one or more blocks, possibly for several files:

```
FILE: path/to/file.py
<<<<<<< SEARCH
<exact lines currently in the file>
=======
<replacement lines>
>>>>>>> REPLACE
```

`FILE: path (new)` creates a file, `FILE: path (occurrences: N)` replaces exactly N identical
occurrences, and `CANNOT_FIX: <reason>` declines. `editing.py` parses this tolerantly (markdown
fences, `<think>` preambles, decorated paths) and applies a whole batch in memory before anything
is written: exact match first, then trailing-whitespace- or indentation-tolerant matching only
when that match is unique; ambiguous matches and wrong occurrence counts are errors; edits that
are already present are recognised; every changed Python/JSON/TOML file must still parse; paths
must stay inside the repository; original line endings are preserved. When the plan is decomposed,
each step is a separate edit round against the files as they are after the previous step.

Every failure (provider error, bad format, SEARCH text not found, ambiguous match, syntax error,
scope block, test failure or regression, new lint finding, reviewer revision request) is turned
into specific feedback by `recovery.py`: the exact error, the affected files re-read from disk, any
partial modifications already in the working tree, and the current code around the failure point.
The next attempt edits forward from the current state. When the evidence contradicts the plan
(for example a failing traceback in a file the plan never considered), the plan is revised (up to
`[pipeline] max_plan_revisions`) and the working tree is reset.

### 7. Scope guard

`scope_guard.evaluate` checks the prospective cumulative diff before it is written and classifies
every touched path. Each concern is a finding with a severity: **block** (the patch is not
proposed), **review** (allowed, but the PR goes to human review) or **info** (recorded only).

| Category | Default patterns | Outcome |
|---|---|---|
| ignored | `.fix-issue-ignore` in the target repo | block |
| secrets | `.env`, `.env.*`, `*.pem`, `*.key`, `**/secrets/**`, `*credentials*` | block |
| ci | `.github/workflows/**`, `.gitlab-ci.yml`, `.circleci/**`, `Jenkinsfile` | block |
| lockfile | `poetry.lock`, `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `Pipfile.lock`, `uv.lock`, `Cargo.lock`, `go.sum` | block |
| migrations | `**/migrations/**`, `**/alembic/**`, `**/*migration*` | new file: review; changing an existing one: block |
| auth | `**/auth/**`, `**/*auth*.py`, `**/permission*`, `**/oauth/**`, `**/security/**`, `**/login*` | review; more than 20 changed lines: block |
| payments | `**/payment*/**`, `**/payment*`, `**/billing/**`, `**/stripe/**`, `**/invoice*` | review; more than 20 changed lines: block |
| public API | removed public function/class/method, changed parameter list, changed `__all__` or package exports (Python) | review |
| deletion / rename | deleted or renamed files | review |
| binary | binary patches | block |
| size | weighted changed lines (tests ×0.5, docs ×0.25, source ×1.0) over the limit | block |
| triage forbidden | path matches a triage-suggested forbidden glob | block |
| triage allowed | path outside the triage-suggested allowed paths (tests/docs exempt) | review |

The size limit is `min(max_diff_lines, triage diff budget)`. A target repository can add its own
rules in `.fix-issue-scope.toml`; they are tried before the defaults:

```toml
[[rule]]
category  = "generated"
patterns  = ["src/generated/**"]
severity  = "block"     # block | review | info (default info)
max_lines = 10          # optional changed-line budget for the category (over it: block)
weight    = 1.0         # optional size weight
```

### 8. Tests and diff-aware linters

After the edits apply, the test suite runs and its failures are compared with the baseline run.
Then the repository's own quality tools run, but only tools the repository configures and that
are installed, only on the changed files, and only findings the patch introduced count. The same
files are checked at the base commit in a temporary `git worktree` (always removed), and findings
are compared without line numbers so a pre-existing warning that moved does not count as new.

| Tool | Configured by |
|---|---|
| py-syntax | always (in-process `compile()`) |
| ruff, ruff-format | `[tool.ruff]` / `[tool.ruff.format]`, `ruff.toml`, `.ruff.toml` |
| flake8 | `.flake8`, `[flake8]` in `setup.cfg` or `tox.ini` |
| mypy | `[tool.mypy]`, `mypy.ini`, `.mypy.ini`, `[mypy]` in `setup.cfg` |
| pyright | `pyrightconfig.json`, `[tool.pyright]` |
| black | `[tool.black]` |
| eslint, tsc | eslint config files / `tsconfig.json` (from `node_modules/.bin`) |
| go vet | `go.mod` |
| npm/make/tox lint scripts | only with `[validation] allow_project_scripts = true` |

A tool that cannot run is recorded as skipped with a reason; it never fails validation.

### 9. Independent review with deterministic gates

`critic.py` reviews the result with a separate system prompt and, if configured, a different model
(`[critic]` or `--critic-model`). The reviewer sees artifacts only: the issue, acceptance criteria,
the plan, the full diff, the post-change code around each hunk, test and linter results and the
scope report. It never sees the implementer's own explanation of why the patch works. It answers
six questions (addresses the issue, all relevant files updated, regressions, missing edge cases,
tests meaningful, change minimal) and gives a verdict: `APPROVED`, `NEEDS_REVISION`,
`HUMAN_REVIEW` or `REJECTED`.

The verdict then passes through gates that can only make it more conservative:

| Gate | Condition | Verdict at most |
|---|---|---|
| G1 | tests did not run | HUMAN_REVIEW |
| G2 | tests failed | NEEDS_REVISION |
| G3 | scope requires human review | HUMAN_REVIEW |
| G4 | scope blocked | REJECTED |
| G5 | new linter/type-checker findings | NEEDS_REVISION |
| G6 | approved, but the reviewer says the issue is not fully addressed | NEEDS_REVISION |
| G7 | approved, but the reviewer says the tests do not verify the fix | HUMAN_REVIEW |
| G8 | approved, but regressions are likely | NEEDS_REVISION |
| G9 | strict mode and any answer not positive | HUMAN_REVIEW |

`NEEDS_REVISION` sends the reviewer's required changes back as feedback for another round (up to
`[pipeline] max_critique_rounds`, each adding one attempt); when that budget is used up the verdict
becomes `HUMAN_REVIEW`. `REJECTED` posts the reviewer summary on the issue and opens no PR. If the
reviewer call fails, the verdict is `HUMAN_REVIEW`.

### 10. Pull request

The fix is committed and force-pushed to the branch `fix-issue-<N>`, and a PR is opened against the
default branch (if a PR from that branch is already open, its URL is returned). When the verdict is
`HUMAN_REVIEW` the PR is opened as a draft with a warning banner. The body has these sections:
Root Cause, Fix Summary, Changes, Confidence Score, Files Modified, Why This Should Work, Validation
Executed, Independent Review (verdict and gates), Scope Notes, Remaining Limitations.

The confidence score is a fixed formula, and the PR shows each term:

```
score = 10
score -= min(5, diff_lines // 30)        # -1 per 30 changed lines, at most -5
score -= 3 if tests did not pass
score -= 2 if triage confidence < 0.7
score -= 2 if the review verdict is HUMAN_REVIEW    # full profile only
score -= 1 if the scope guard asked for review      # full profile only
score = max(1, score)
```

### 11. Outcomes and run history

| Result | Meaning | Posted on the issue (full profile, not in dry runs) |
|---|---|---|
| `pr_opened` | PR opened (possibly as a draft) | the PR |
| `dry_run` | everything ran, nothing was pushed | nothing; the PR title, body and diff are printed |
| `needs_info` | triage needs more information | clarifying question |
| `duplicate` | duplicate backed by similarity evidence | pointer to the original |
| `out_of_scope` | not something to fix automatically | nothing |
| `cannot_fix` | the model declined | nothing |
| `scope_violation` | the last attempt was blocked by the scope guard | explanation |
| `rejected` | the independent review rejected the patch | reviewer summary |
| `failed` | attempts exhausted | last failure category and message |
| `error` | the pipeline stopped on an LLM, configuration or file-system error (recorded in the run) | nothing |

`fix-issue owner/repo N` exits 0 for `pr_opened`, `dry_run`, `needs_info`, `out_of_scope` and
`duplicate`, and 1 otherwise.

Every run writes a JSON record to `~/.fix-issue/runs/<run-id>.json` (root configurable with
`FIX_ISSUE_HOME`), updated after each stage: parameters, issue, triage and control signals,
similar issues, tooling, baseline tests, retrieval with reasons, every plan version, every attempt
(intent, edits, diff, scope report, tests, linters, failure analysis), every review round with its
gates, a per-call LLM log (purpose, sizes, latency, truncated prompt tail and response) and the
outcome. `fix-issue runs` reads these, and `fix-issue rerun` repeats a run with the recorded
parameters. A one-line summary per GitHub run is also still appended to `~/.fix-issue-runs.jsonl` (`FIX_ISSUE_HOME/runs.jsonl` when `FIX_ISSUE_HOME` is set).

---

## Requirements and installation

- Python 3.11 or newer
- `git` on your PATH
- An LLM API key (Anthropic, Gemini, or an OpenAI-compatible provider such as NVIDIA NIM)
- A GitHub token with read/write access to the repository's contents, issues and pull requests
  (only for `fix-issue owner/repo N`; the benchmark does not use GitHub)

```bash
git clone https://github.com/FriToS-Ban/Issue-Resolver.git
cd Issue-Resolver
pip install -e ".[dev]"
fix-issue --help
```

Install from a checkout in editable mode as shown. The benchmark reads `benchmark/` relative to
the source tree.

Tests and Python linters of the target repository are run with the interpreter that runs
fix-issue (`python -m pytest`, `python -P -m ruff`, ...), so the target repository's test
dependencies must be importable from that environment.

---

## Configuration

Settings are read in this order (later overrides earlier):

1. defaults in `fix_issue/config.py`
2. `~/.fix-issue.toml`, or the file given with `--config` (`fix-issue bench run` uses
   `benchmark/bench.toml` unless you pass `--config`)
3. environment variables, including a `.env` file in the current working directory (a `.env`
   value never replaces a variable that is already set in the real environment)
4. command-line options

Start from the templates:

```bash
cp .fix-issue.toml.example ~/.fix-issue.toml   # every key, with a working NVIDIA NIM setup
cp .env.example .env                           # keys and overrides; uncomment what you need
```

A variable that is set, even to an empty value, overrides the TOML file. Keep lines you don't use
commented out in `.env`.

### API keys

- `LLM_API_KEY` sets the `[llm]` key for any provider.
- `NVIDIA_API_KEY` is used as the `[llm]` key automatically when the `[llm]` base URL contains
  `nvidia.com` and `LLM_API_KEY` is not set at all.
- The critic and API embeddings fall back to the `[llm]` key, so one NIM key covers all three. If
  the critic or the embeddings use a different provider than `[llm]`, set `[critic] api_key` or
  `[retrieval] embedding_api_key`.
- `GITHUB_TOKEN` (or `[github] token`) is needed for real runs, not for the benchmark.

### Working NVIDIA NIM setup

These models work on NIM at the time of writing and are what `benchmark/bench.toml` uses.
`deepseek-ai/deepseek-v4.1-flash` is end-of-life on NIM and no longer works.

```toml
[llm]
provider = "openai_compatible"
base_url = "https://integrate.api.nvidia.com/v1"
model    = "nvidia/nemotron-3-ultra-550b-a55b"
api_key  = ""                    # empty: NVIDIA_API_KEY from the environment / .env is used

[critic]
model = "mistralai/mistral-large-4-1t-a52b"   # a different model family reviews the patch

[retrieval]
embedding_backend = "openai_compatible"
embedding_model   = "nvidia/nemotron-3.5-embed-1b"
```

Other providers:

```toml
# Anthropic
[llm]
provider = "anthropic"
api_key  = "sk-ant-..."
model    = "claude-opus-4-5"

# OpenAI (base_url is required for openai_compatible)
[llm]
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key  = "sk-..."
model    = "gpt-4o"

# Ollama
[llm]
provider = "openai_compatible"
base_url = "http://localhost:11434/v1"
api_key  = "ollama"             # any non-empty string
model    = "llama3.1"

# Gemini
[llm]
provider = "gemini"
api_key  = "AIza..."
model    = "gemini-2.5-pro"
```

### All settings

`[llm]`

| Key | Default | Meaning |
|---|---|---|
| `provider` | `anthropic` | `anthropic`, `openai_compatible` or `gemini` |
| `api_key` | empty | see [API keys](#api-keys) |
| `base_url` | empty | required for `openai_compatible`; ignored by the other providers |
| `model` | `claude-opus-4-5` | implementer model |
| `temperature` | provider default | also used by the critic; the benchmark uses 0.0 |
| `max_tokens` | 4096 | output tokens per call, implementer and critic |

`[github]`: `token` (default empty).

`[limits]`

| Key | Default | Meaning |
|---|---|---|
| `max_diff_lines` | 150 | full profile: weighted changed lines; baseline: raw `+`/`-` lines |
| `max_retries` | 3 | attempts per issue (triage may lower it, review rounds add to it) |
| `relevant_files` | 5 | files given to the model (a minimum in the full profile, exact in baseline) |
| `test_timeout_seconds` | 300 | per test-suite run |

`[critic]`: `provider`, `api_key`, `base_url`, `model`. Each empty value falls back to `[llm]`. A
separate reviewer is built only when `model` or `provider` is set (or `--critic-model` is given);
otherwise the implementer model reviews with the separate reviewer prompt.

`[pipeline]`

| Key | Default | Meaning |
|---|---|---|
| `profile` | `full` | `full` or `baseline` |
| `planning` | `true` | plan before editing (`--no-plan` turns it off) |
| `critique` | `true` | independent review (`--no-critique` turns it off) |
| `max_critique_rounds` | 1 | NEEDS_REVISION rounds before the verdict becomes HUMAN_REVIEW |
| `max_plan_revisions` | 1 | plan revisions triggered by contradicting evidence |
| `max_context_chars` | 60000 | source-context budget per prompt |
| `duplicate_threshold` | per-scorer table `DUPLICATE_THRESHOLDS` in `fix_issue/config.py` | similarity a resolved issue needs before a "duplicate" verdict is accepted: one number for every scorer, or a `[pipeline.duplicate_threshold]` table keyed by scorer (`local`, an embedding model name, `default`) that is merged over the defaults |

`[retrieval]`

| Key | Default | Meaning |
|---|---|---|
| `embedding_backend` | `local` | `local` (offline n-gram vectors), `openai_compatible`, or `none` |
| `embedding_model` | empty | required for `openai_compatible` |
| `embedding_base_url` | `[llm] base_url` | `/embeddings` endpoint base |
| `embedding_api_key` | `[llm] api_key` | |
| `cache_dir` | `~/.fix-issue/cache` | embedding cache for the pipeline |

If the API embedder is misconfigured, the pipeline records why and falls back to local vectors.

`[validation]`

| Key | Default | Meaning |
|---|---|---|
| `run_linters` | `true` | run configured linters |
| `run_type_checkers` | `true` | run configured type checkers |
| `run_formatters` | `true` | run configured formatter checks |
| `allow_project_scripts` | `false` | allow `npm run lint`, `make lint`, `tox -e lint` (arbitrary repository code) |
| `sandbox_prefix` | `[]` | argv prefix wrapped around test and linter commands, e.g. a `docker run ...` invocation; untested, see [limitations](#security-and-limitations) |

### Environment variables

| Variable | Overrides |
|---|---|
| `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | `[llm]` |
| `NVIDIA_API_KEY` | `[llm] api_key`, only for an `nvidia.com` base URL and when `LLM_API_KEY` is unset |
| `GITHUB_TOKEN` | `[github] token` |
| `FIX_ISSUE_MAX_DIFF_LINES`, `FIX_ISSUE_MAX_RETRIES` | `[limits]` |
| `FIX_ISSUE_PROFILE` | `[pipeline] profile` |
| `FIX_ISSUE_EMBEDDING_BACKEND` | `[retrieval] embedding_backend` |
| `FIX_ISSUE_HOME` | root for `runs/` and `cache/` (default `~/.fix-issue`) |

Environment variables also override `benchmark/bench.toml`, so unset `LLM_*` and `FIX_ISSUE_*`
variables before a benchmark run you want to compare. Set `FIX_ISSUE_HOME` in the real
environment rather than in `.env`: `fix-issue runs ...` and `fix-issue rerun` look up run records
before (or without) loading `.env`.

### Per-repository files

These live in the target repository, so the rules are visible to anyone reviewing the agent's PRs.

- `.fix-issue-ignore`: glob patterns (fnmatch, on the full path or the file name) the agent must
  never modify. A match blocks the patch. See `.fix-issue-ignore.example`.
- `.fix-issue-scope.toml`: extra scope rules, see [Scope guard](#7-scope-guard).

---

## Commands

Every command and option below exists in `fix_issue/cli.py`; `fix-issue <command> --help` prints
the same list.

### Resolve an issue

```bash
fix-issue owner/repo 123 [options]
fix-issue resolve owner/repo 123 [options]      # same, explicit
```

| Option | Effect |
|---|---|
| `--dry-run` | run everything, but don't push, open a PR or post comments |
| `--max-retries N` | override `[limits] max_retries` |
| `--max-diff-lines N` | override `[limits] max_diff_lines` |
| `--config PATH` | use this TOML file instead of `~/.fix-issue.toml` |
| `--profile full\|baseline` | pipeline to run |
| `--model NAME` | implementer model |
| `--critic-model NAME` | reviewer model (other `[critic]` fields still apply) |
| `--no-plan` | skip planning |
| `--no-critique` | skip the independent review (no gates, so no draft PRs either) |
| `--embedding-backend local\|openai_compatible\|none` | retrieval embeddings |
| `--temperature X` | sampling temperature |

`fix-issue --version` prints the version.

### Run history

```bash
fix-issue runs list [--limit 20] [--repo owner/repo] [--result pr_opened] [--json]
fix-issue runs show <run-id> [--section retrieval] [--json]
fix-issue runs stats
fix-issue rerun <run-id> [options]
```

- `runs list` shows id, start time, repo, issue, profile, classification, files retrieved,
  attempts, verdict, result and duration.
- `runs show` prints a readable summary of everything a run did and why. `--section` prints one
  part as JSON: `issue`, `triage`, `controls`, `similar_issues`, `tooling`, `baseline`,
  `retrieval`, `plans`, `attempts`, `critiques`, `pr`, `llm`, `outcome`, `events`, `params`.
  `--json` prints the whole record. A unique part of the run id is enough.
- `runs stats` counts results and review verdicts and shows the mean number of attempts.
- `rerun` takes the same options as `resolve`. It restores the recorded profile, model,
  temperature, critic model, `[limits]` and `[pipeline]` values, then applies your options.
  Provider, keys, retrieval and validation settings come from your current configuration. A run
  that was recorded as a dry run is repeated as a dry run.

### Benchmark commands

```bash
fix-issue bench validate  [--bench-dir DIR] [--cases ID|DIFFICULTY|FIXTURE ...]
fix-issue bench retrieval [--bench-dir DIR] [--cases ...] [--top-n 5] [--config PATH] [--out FILE.json]
fix-issue bench run       [--bench-dir DIR] [--cases ...] [--config PATH] [--profile full|baseline]
                          [--parallel 2] [--out FILE.json] [--verbose]
fix-issue bench report    RESULTS.json
fix-issue bench compare   BASELINE.json IMPROVED.json [--out FILE.md]
fix-issue bench review    RESULTS.json [--all] [--set CASE_ID=correct|incorrect|unsure ...] [--note TEXT]
```

See [Benchmark](#benchmark) for what each one does.

---

## Profiles: full vs baseline

| Stage | `full` (default) | `baseline` |
|---|---|---|
| Similar-issue search | yes | no |
| Triage | structured triage, duplicate evidence check, control signals | original v1 prompt; configured limits used as they are |
| Baseline test run | yes | no |
| Retrieval | layered (`fix_issue/retrieval/`) | keyword scoring (`file_selector.py`), top `relevant_files` |
| Planning | plan + one investigation round | none |
| Editing | step-wise SEARCH/REPLACE batches, validated before writing | `patch_generator.py` produces one unified diff, applied with `git apply` (then `--3way`) |
| Scope | `scope_guard.evaluate` on the cumulative diff, before writing | `scope_guard.check`: `.fix-issue-ignore` + raw line count; a violation ends the run |
| Linters / type checkers | diff-aware | none |
| Retries | categorised recovery from the current file state, plan revision | error text appended to the next prompt; tree reset after a test failure |
| Review | independent critique + gates | none |
| PR | structured explanation; draft on HUMAN_REVIEW | original body; never a draft |
| Comments on failure | `failed`, `scope_violation`, `rejected` | none |
| `sandbox_prefix` | applied to tests and linters | not applied |

The baseline is kept so that the benchmark measures the effect of the rebuild against the same
models and cases, not against a strawman.

---

## Benchmark

A reproducible evaluation on small fixture repositories, run through the real orchestrator with a
real LLM, and graded by the harness rather than by what the pipeline claims about itself.

### How it is built

```
benchmark/
  bench.toml                  config for `bench run` (NIM models, temperature 0.0; no secrets)
  fixtures/<name>/            clean fixture repositories with their own pytest suites
  cases/<id>/case.toml        one issue + the bug to inject + ground truth
  cases/<id>/hidden_tests/    tests that check the issue itself is fixed
  results/                    results JSON + Markdown (per-case run records under results/runs/, gitignored)
```

**Fixtures.** Three small Python packages: `shopcart` (cart, checkout, discounts, inventory, an
`auth` and a `payments` package, migrations; ships a `.fix-issue-ignore` for `shopcart/payments/**`),
`textkit` (text utilities and a CLI) and `taskq` (task queue, scheduling, retry, storage).

**Cases.** 23 cases: 9 easy, 7 medium, 7 hard; 10 on shopcart, 7 on taskq, 6 on textkit. 20 are fix
cases. 3 are triage cases where the right answer is not a patch: a feature request
(`out_of_scope`), a report that duplicates an already resolved issue (`duplicate`, with the
resolved candidates in `[[recent_resolved]]`), and a vague report (`needs_info`). A `case.toml`
holds:

- `id`, `fixture`, `difficulty`, `kind` (`fix` or `triage`), `expected_classification`,
  `root_cause_category`, `notes` (the reference fix, for humans)
- `[issue]`: number, title, labels, body, written like a real report
- `[[inject]]`: `file`, `search`, `replace`; optional `[[add_files]]`
- `[expected]`: ground-truth `files` and `symbols` (for retrieval), `forbidden_paths` and
  `max_diff_lines` (for scope)
- optional `requires_manual_review`, `visible_tests_fail_before`, `[[recent_resolved]]`

**Bug injection.** For each case the fixture is copied to a fresh temporary directory, the
injections are applied by plain exact string replacement (each search text must match exactly
once), and the result is committed to a new git repository. Injection deliberately does not use
the agent's editing engine, so the benchmark does not depend on the code under test.
(`.gitattributes` keeps benchmark files LF so the exact match works on every OS.)

**Hidden tests.** The files in `hidden_tests/` are copied into `tests/_hidden/` only after the
agent has finished. This separates "the visible suite passes" from "the issue is actually fixed".

**Grading.** The pipeline runs with a local workspace and a publisher that records instead of
posting, always as a dry run. The harness then re-runs the visible suite on the final working
tree, runs the hidden tests, checks the real diff against the case's forbidden paths and size cap
(the cap applies to non-test lines; regression tests the agent adds are counted separately),
checks retrieval against the expected files, and derives a failure category from the run record.
A fix case counts as a success only when the classification is correct, a patch is present, the
visible suite passes, the hidden tests pass, scope is respected and the pipeline finished with a
deliverable patch. A triage case succeeds when the classification is correct and no patch was
produced. Reported metrics include success rate (overall and by difficulty), classification
accuracy, issue-fixed rate, test-pass rate, "tests pass but not fixed", patch application rate,
edit-round apply rate, first-attempt success, scope respected, mean attempts, review verdicts,
reviewer false approvals and failure categories. Runs whose LLM is not a real model are labelled
and excluded from headline metrics.

### Running it

```bash
# 1. Prove every case is well-formed: clean fixture passes visible + hidden tests,
#    buggy baseline fails the hidden tests. No LLM.
fix-issue bench validate
fix-issue bench validate --cases easy --cases textkit   # every easy case plus every textkit case

# 2. Retrieval only, no LLM: keyword baseline vs layered (no embeddings) vs layered (local vectors).
fix-issue bench retrieval --out benchmark/results/retrieval.json
#    --config adds a layered-api variant using that file's [retrieval] openai_compatible embeddings.
fix-issue bench retrieval --config benchmark/bench.toml --out benchmark/results/retrieval-api.json

# 3. End to end with the real LLM (key from .env / environment, models from benchmark/bench.toml).
fix-issue bench run --profile baseline
fix-issue bench run --profile full
fix-issue bench run --profile full --cases taskq-e1-interval-seconds-unit --verbose

# 4. Compare two result files.
fix-issue bench compare benchmark/results/baseline-<stamp>.json benchmark/results/full-<stamp>.json \
    --out benchmark/results/compare.md

# 5. Record human verdicts where hidden tests are not enough, then re-render the report.
fix-issue bench review benchmark/results/full-<stamp>.json
fix-issue bench review benchmark/results/full-<stamp>.json --set textkit-m2-csv-quoted-fields=correct --note "checked by hand"
fix-issue bench report benchmark/results/full-<stamp>.json
```

- `--cases` takes case ids, difficulty names or fixture names and can be repeated.
- `bench run` writes `benchmark/results/<profile>-<stamp>.json` (or `--out`) after every case, a
  Markdown summary next to it, and the full run record of every case to
  `benchmark/results/runs/<results-name>/`. `--parallel` sets how many cases run at once. Those
  run records are not in `~/.fix-issue/runs`, so `fix-issue runs show` and `fix-issue rerun` do
  not find them; open the JSON files directly.
- `bench review` walks through fix cases that need a human look (cases marked
  `requires_manual_review`, and cases where the visible and hidden tests disagree; `--all` for every
  fix case), shows the automated grades and the diff, and stores verdicts in
  `<results>.manual.json`. `bench report` applies them (they override "issue fixed" for those cases)
  and counts reviews still pending.

### Results

**Retrieval (no LLM).** From `benchmark/results/retrieval-local.md`: 20 fix cases, top 5 files.

| Metric | keyword (baseline) | layered, no embeddings | layered, local vectors |
|---|---|---|---|
| recall@1 | 0.25 | 0.60 | 0.60 |
| recall@3 | 0.70 | 0.80 | 0.80 |
| recall@5 | 0.82 | 0.93 | 0.93 |
| MRR | 0.50 | 0.71 | 0.71 |
| all expected files found | 0.80 | 0.90 | 0.90 |
| symbol recall | 0.00 | 0.69 | 0.64 |
| recall@5, easy / medium / hard | 1.00 / 0.57 / 0.88 | 1.00 / 0.86 / 0.88 | 1.00 / 0.86 / 0.88 |
| mean files returned | 5.00 | 3.25 | 3.20 |
| mean context characters | 13924 | 11453 | 11224 |

The gain comes from mentions, symbols and the dependency graph; the local n-gram vectors add
nothing on this set (same recall and MRR, slightly lower symbol recall). Remaining misses:
`shopcart-m2` (`shopcart/utils/dates.py`, two hops from the issue text, missed by every variant)
and `shopcart-h2` (`shopcart/pricing.py` with layered retrieval; keyword retrieval missed
`shopcart/utils/money.py` instead).

**End to end: baseline vs full.**

<!-- BENCH-RESULTS -->
*Pending: results of `fix-issue bench run --profile baseline` vs `--profile full` will go here.*
<!-- /BENCH-RESULTS -->

---

## Security and limitations

What is done to limit risk:

- The GitHub token is used in the clone URL only for the clone itself; the remote URL is then
  rewritten without it, and the push authenticates with a one-shot `http.extraheader`, so code run
  by the repository's test suite cannot read the token from `.git/config`.
- Commands that run repository code (tests, linters) get an environment with secret-looking
  variables removed (`*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*API_KEY*`, `*_KEY`, `AWS_*`, `AZURE_*`,
  `GCP_*`, `GOOGLE_*`, ...), run as argument lists without a shell, have hard timeouts, and run
  inside the throwaway clone.
- Python linters and type checkers run as `python -P -m <tool>` so a module planted in the
  repository cannot shadow them; the base-commit worktree is created with hooks disabled; project
  lint scripts are opt-in.
- In the full profile, CI configuration, lock files and secrets files are never edited, and
  changes to auth and payment code are limited to 20 lines and, with the review enabled, end as
  `HUMAN_REVIEW` (draft PR). The baseline profile only enforces `.fix-issue-ignore` and the size
  limit.
- No PR is merged automatically. A PR is a draft whenever the review verdict is `HUMAN_REVIEW`.

What is not solved:

- **No container isolation.** The repository's tests and linters execute arbitrary code from the
  repository on your machine, with your user's file-system access. Secrets are stripped from the
  environment and there are timeouts, but nothing more. `[validation] sandbox_prefix` exists to
  wrap those commands (for example in `docker run ...`), but it is untested: commands use the host
  interpreter path and the host working directory, so a wrapper has to make both valid inside the
  container. It is not applied in the baseline profile or by the benchmark harness's own grading
  runs. Only run fix-issue on repositories whose code you are willing to execute.
- **The benchmark is small and Python-only.** Three fixture repositories with roughly 650 to 1,150
  lines of source each, 23 cases. Retrieval and validators support other languages, but nothing
  outside Python is measured, and results on large real-world repositories may differ.
- **Triage `allowed_paths` do not block.** Triage only sees the issue text, so a change outside the
  paths it suggested produces a review finding (and so a `HUMAN_REVIEW` verdict and a draft PR),
  not a block. Triage `forbidden_paths` do block.
- With `--no-critique` there is no verdict: the deterministic gates don't run and PRs are never
  drafts.
- The branch `fix-issue-<N>` is force-pushed, so running the same issue again replaces the earlier
  branch contents.
- Hidden tests are strong evidence that an issue is fixed, not proof. That is what `bench review`
  is for.

---

## Module reference

### `fix_issue/`

| Module | Responsibility |
|---|---|
| `__init__.py` | package version |
| `cli.py` | click entry point: `resolve` (default command), `runs`, `rerun`, `bench`; builds clients, renders progress and history |
| `orchestrator.py` | the pipeline: both profiles, bounded loops, everything recorded; issue source, workspace and publisher are injected |
| `adapters.py` | GitHub issue source and publisher for the orchestrator, plus a recording publisher for dry runs and the benchmark |
| `config.py` | dataclasses for every TOML section, `.env` loading, environment overrides, `NVIDIA_API_KEY` fallback, validation, critic/embedding fallbacks, per-scorer duplicate thresholds |
| `github_client.py` | GitHub REST API over plain `requests` (issues, comments, contents, branches, PRs) |
| `llm_provider.py` | provider-neutral LLM interface (Anthropic, OpenAI-compatible, Gemini) with bounded retries on 429/5xx/transport errors; `RecordingLLM` logs every call |
| `structured.py` | JSON extraction from LLM output and a small declarative schema validator, with one bounded repair round |
| `triage.py` | structured triage, `derive_controls` (control signals that only tighten limits), duplicate-evidence check; legacy prompt for the baseline |
| `duplicates.py` | similar-issue search against recently closed GitHub issues and past runs |
| `repo_manager.py` | git workspaces: `GitWorkspace` (diff, reset, base content), `RepoManager` (temp clone of a GitHub repo, token-safe push), `LocalWorkspace` (benchmark); test-command detection; `git apply` with `--3way` fallback |
| `planner.py` | schema-validated plans checked against the repository; `revise_plan`; retrieval-based fallback plan |
| `editing.py` | SEARCH/REPLACE parser and atomic in-memory edit engine with tolerant-but-unique matching, syntax validation and `git apply`-compatible diffs |
| `recovery.py` | failure categories and current-state feedback (exact error, fresh excerpts, partial changes, plan contradictions) |
| `scope_guard.py` | `evaluate`: path- and severity-aware scope report; `check`: legacy all-or-nothing check used by the baseline |
| `validators.py` | discovery of the repository's own linters/type checkers/formatters and diff-aware runs against a base-commit worktree |
| `test_runner.py` | runs the test command, parses pytest failures and the summary line |
| `sandbox.py` | secret-stripped environment, timeouts, no shell, optional `sandbox_prefix` |
| `critic.py` | independent review prompt, structured answers, deterministic gates G1-G9 |
| `pr_builder.py` | PR title and body, structured explanation sections, confidence formula, draft flag |
| `run_history.py` | per-run JSON records under `~/.fix-issue/runs`, listing, loading and summaries |
| `logger.py` | one-line JSON summary per run appended to `~/.fix-issue-runs.jsonl` (kept from v1) |
| `file_selector.py` | keyword file scoring (baseline profile; also the `keyword` variant of the retrieval benchmark) |
| `patch_generator.py` | baseline profile patch generation: SEARCH/REPLACE blocks turned into one unified diff |

### `fix_issue/retrieval/`

| Module | Responsibility |
|---|---|
| `__init__.py` | public exports |
| `index.py` | repository index: files, symbols with spans and signatures, imports, references (Python `ast`, regex for other languages) |
| `mentions.py` | explicit evidence in the issue text: paths, stack frames, identifiers, exception types, resolved against the real file list |
| `lexical.py` | BM25 over code-aware tokens |
| `embeddings.py` | local n-gram embedder, OpenAI-compatible embedder, on-disk vector cache, `build_embedder` |
| `retriever.py` | layered scoring, graph expansion, per-file reasons, context rendering (outline + focus symbols for large files), retrieval metrics |
| `text.py` | identifier splitting, tokenisation and stemming shared by all channels |

### `fix_issue/bench/`

| Module | Responsibility |
|---|---|
| `cases.py` | load and validate `case.toml`, copy fixtures, inject bugs, initialise the git repo, install hidden tests |
| `validate.py` | `bench validate`: clean fixture passes, buggy baseline fails the hidden tests |
| `retrieval_eval.py` | `bench retrieval`: retrieval variants against ground-truth files and symbols |
| `runner.py` | `bench run`: real orchestrator per case in a temp repo, independent grading, results JSON |
| `report.py` | metrics, Markdown report, side-by-side comparison |
| `manual.py` | human verdicts stored next to a results file |

---

## Running the unit tests

```bash
pip install -e ".[dev]"
pytest
```

| File | Covers |
|---|---|
| `tests/test_triage.py` | triage parsing and validation, control signals, duplicate evidence |
| `tests/test_duplicates.py` | similar-issue ranking and candidate sources |
| `tests/test_retrieval.py` | mentions, index, layered retrieval, embeddings cache |
| `tests/test_editing.py` | edit parsing, matching rules, atomic batches, line endings, diffs |
| `tests/test_scope_guard.py` | scope categories, severities, repo rules, public API detection, legacy check |
| `tests/test_validators.py` | tool discovery, output parsers, new-finding comparison, worktree cleanup |
| `tests/test_pipeline_components.py` | structured output, planner, critic gates, run recorder, PR explanation, benchmark metrics, `.env` / `NVIDIA_API_KEY` |
| `tests/test_embeddings_api.py` | API embedder handling of refused/blank inputs, cache robustness |
| `tests/test_bench_grading.py` | benchmark grading on a real fixture with hand-written trees (no LLM) |
| `tests/test_orchestrator.py` | the whole pipeline (both profiles) on a benchmark fixture, driven by a scripted test double instead of an LLM |
| `tests/test_file_selector.py` | keyword scoring (baseline) |
| `tests/test_confidence_score.py` | confidence formula and PR body |

The orchestrator tests check the plumbing (recovery, review rounds, scope blocks, triage
short-circuits), not how well a real model fixes issues; that is what `fix-issue bench run`
measures.

---

## Not done yet

- A GitHub Actions workflow (`on: issues.opened`) to trigger runs automatically
- Real isolation for repository code (container wrapper, tested)
- Benchmark cases in other languages and on larger repositories
- A hosted GitHub App / webhook receiver (needs an always-on host)

## License

MIT
