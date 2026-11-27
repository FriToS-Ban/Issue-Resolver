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
clone ─► pin the scope policy (base commit) ─► discover tests + linters ─► BASELINE test run
      ─► layered retrieval ─► plan (+1 investigation round)
                                              │
          attempts ≤ max_retries (triage may lower it; review rounds add attempts)
   ┌── for each plan step (or one edit round):
   │      model writes INTENT + SEARCH/REPLACE blocks against the CURRENT files
   │      edits applied atomically in memory ─► scope guard on the cumulative diff ─► write
   ├── tests (compared with the baseline run) + the repo's linters/type checkers (new findings only)
   │      any failure → recovery analysis from the current file state → next attempt
   │                    (plan revised when the evidence contradicts it)
   ├── scope guard again on the real working tree (test or linter code may have changed files)
   └── independent review + deterministic gates
          NEEDS_REVISION → another round (bounded) │ REJECTED → comment, stop
                                              ▼
tree still equal to the reviewed diff ─► commit on the base commit ─► push fix-issue-<N>
      ─► PR (draft when the verdict is HUMAN_REVIEW) ─► run record
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
reproduction) with confidence capped at 0.5. When the embedding API refuses the issue or a
candidate, the affected comparisons are scored with local vectors instead and recorded as such. The
defaults were chosen for zero false positives on a labelled set, which keeps recall low (see
[Security and limitations](#security-and-limitations)).

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

The repository is cloned (`--depth=1`) into a temporary directory, and the branch `fix-issue-<N>`
is created at the default branch's HEAD, which is the base commit for the rest of the run. The
directory is removed at the end, also when the clone fails, including git's read-only object
files (which a plain `shutil.rmtree` leaves behind on Windows).

Two things happen before any repository code runs:

- The scope policy (`.fix-issue-scope.toml` and `.fix-issue-ignore`) is read from the base commit
  with `git cat-file` and pinned for the whole run (see [step 7](#7-scope-guard)).
- The git control files are snapshotted: `.git/config`, and `config.worktree` / `commondir` if
  they exist.

The repository's tests and linters, and any `conftest.py` the model writes, run in the working
tree before fix-issue diffs, commits and pushes, so they could plant hooks or git config that git
would then execute. Every git command the workspace runs (diff, reset, commit, push) and the
validators' base-commit worktree commands are therefore hardened (the scope guard reads the policy
with a plain `git cat-file --batch`, which runs no hooks; in the full profile that happens before
any repository code runs):

- `-c core.hooksPath=<an empty directory fix-issue owns>`, so no hook runs, and `--no-verify` on
  commit and push; `-c core.fsmonitor=false`; `-c gc.auto=0` and `-c maintenance.auto=false`.
- Before each git command `.git/config` is restored byte for byte from the snapshot, so planted
  `filter.*`, `diff.*.textconv`, `core.sshCommand`, `url.*.insteadOf` or `include.path` entries
  never take effect. A planted `commondir` or `config.worktree` is removed. If `.git` itself was
  replaced (by a link, a file or another directory), the command is refused.
- git gets the allowlisted environment that repository code gets (step 8), plus the proxy,
  CA-bundle, author/committer and `GIT_CONFIG_GLOBAL` / `GIT_CONFIG_SYSTEM` /
  `GIT_CONFIG_NOSYSTEM` variables from your environment, and a timeout (300 s per command, 900 s
  for the clone, 600 s for the push) after which its process tree is killed.
- Diffs use `--no-ext-diff --no-textconv`.

The test command is detected from the repository (pytest configuration, `setup.py` + `tests/`, a
`package.json` `test` script, a Makefile `test` target, `tox.ini`, `go.mod`, or a `tests/`
directory with `test_*.py`). The repository's configured linters and type checkers are discovered
(see step 8). The tests are run once before any change, so failures that already existed are not
blamed on the patch.

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
cached on disk; inputs the endpoint refuses (for example NIM's HTTP 422 content screening) get no
vector instead of switching embeddings off for the whole repository. The index skips symlinks,
and every file read checks that the resolved path stays inside the clone.

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
are already present are recognised (only when the REPLACE text sits where the SEARCH text best
matches); every changed Python/JSON/TOML file must still parse; original line endings are
preserved. When the answer also contains edits, it is a decline only if a `CANNOT_FIX:` marker
line comes before the first `FILE:` line; prose such as "Cannot fix X, so..." in the INTENT does
not count. When the plan is decomposed, each step is a separate edit round against the files as
they are after the previous step.

Paths are checked before anything is read or written. Absolute paths, `..`, components ending in
`.` or a space, Windows 8.3 short names (`GIT~1`), `:` (NTFS alternate data streams), and control
or invisible Unicode format characters are refused, because on Windows or macOS they can alias
another name (`.git./hooks/pre-push` is `.git/hooks/pre-push` on NTFS). The path is then resolved
(symlinks, letter case, short names), and a target outside the repository or under `.git` is
refused. Reads, writes and scope checks use that canonical repo-relative path.

Every failure (provider error, bad format, SEARCH text not found, ambiguous match, syntax error,
scope block, test failure or regression, new lint finding, reviewer revision request) is turned
into specific feedback by `recovery.py`: the exact error, the affected files re-read from disk, any
partial modifications already in the working tree, and the current code around the failure point.
The next attempt edits forward from the current state. When the evidence contradicts the plan
(for example a failing traceback in a file the plan never considered), the plan is revised (up to
`[pipeline] max_plan_revisions`) and the working tree is reset.

### 7. Scope guard

`scope_guard.evaluate` classifies every path a diff touches. Each concern is a finding with a
severity: **block** (the patch is not proposed), **review** (allowed, but the PR goes to human
review) or **info** (recorded only). In the full profile it is applied at three points:

1. on the prospective cumulative diff of each edit round, before anything is written;
2. after the tests and linters ran, on the working tree as it really is (`git diff` against the
   base commit), because test or linter code may have created or changed files (a workflow,
   `.gitattributes`, a symlink). A violation fails the attempt like any other failure. This check
   is enforced even with `--no-critique`;
3. right before committing: the working tree must still equal the diff that was validated and
   reviewed, otherwise nothing is committed and the run ends as `scope_violation`.

| Category | Default patterns | Outcome |
|---|---|---|
| unsafe_path | a path component that is `.git` (any case), ends in `.` or a space, looks like an 8.3 short name (`GIT~1`), contains `:` or is `..`, or contains control or invisible Unicode format characters | block (hard-coded, repo rules cannot change it) |
| policy | `.fix-issue-scope.toml`, `.fix-issue-ignore` (at any level) | block (hard-coded) |
| ignored | patterns in `.fix-issue-ignore` (see below) | block |
| secrets | `.env`, `.env.*`, `*.pem`, `*.key`, `**/secrets/**`, `*credentials*`, `.npmrc`, `.pypirc`, `.netrc`, `_netrc`, `.git-credentials`, `.yarnrc`, `.yarnrc.yml`, `pip.conf`, `pip.ini` | block |
| ci | everything under `.github/` (workflows, actions, `CODEOWNERS`, dependabot, issue forms), `.gitlab-ci.yml`, `.gitlab/**`, `.circleci/**`, `Jenkinsfile`, `.travis.yml`, `azure-pipelines.yml`/`.yaml`, `.azure-pipelines/**`, `bitbucket-pipelines.yml`, `.drone.yml`, `.buildkite/**`, `appveyor.yml`, `.appveyor.yml`, `.pre-commit-config.yaml`, `.pre-commit-hooks.yaml`, `CODEOWNERS` | block |
| git_metadata | `.gitattributes`, `.gitmodules` | block |
| lockfile | `poetry.lock`, `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `Pipfile.lock`, `uv.lock`, `Cargo.lock`, `go.sum` | block |
| migrations | `**/migrations/**`, `**/alembic/**`, `**/*migration*` | new file: review; modifying, deleting or renaming an existing one: block |
| tests | `tests/**`, `**/tests/**`, `test_*.py`, `*_test.py`, `conftest.py`, `**/__tests__/**`, `*.spec.*`, `*.test.*` | info; size weight 0.5 |
| docs | `docs/**`, `*.md`, `*.rst` | info; size weight 0.25 |
| auth | `**/auth/**`, `**/*auth*.py`, `**/permission*`, `**/oauth/**`, `**/security/**`, `**/login*` | review; more than 20 changed lines in the category: block |
| payments | `**/payment*/**`, `**/payment*`, `**/billing/**`, `**/stripe/**`, `**/invoice*` | review; more than 20 changed lines in the category: block |
| build | `setup.py`, `setup.cfg`, `pyproject.toml`, `tox.ini`, `noxfile.py`, `Makefile`, `GNUmakefile`, `package.json`, `MANIFEST.in`, `requirements*.txt`, `Pipfile`, `Dockerfile`, `CMakeLists.txt`, `Cargo.toml`, `go.mod`, `Gemfile`, `*.gemspec`, `pom.xml`, `build.gradle`, `build.gradle.kts` | review |
| public API | removed public function/class/method, changed parameter list, changed `__all__` or package exports (non-test Python) | review |
| deletion / rename | deleted or renamed files | review |
| binary | binary patches | block |
| special_file | a symlink (mode 120000) or submodule (mode 160000) entry in the diff | block |
| size | weighted changed lines (tests ×0.5, docs ×0.25, everything else ×1.0) over the limit | block |
| size (characters) | weighted changed characters over limit × 200, so one very long line cannot pass a line budget | block |
| triage forbidden | path matches a triage-suggested forbidden glob (test files exempt) | block |
| triage allowed | path outside the triage-suggested allowed paths (tests/docs exempt) | review |
| config | a problem in `.fix-issue-scope.toml`: info; the policy could not be read from the base commit: review | info / review |

`unsafe_path` and `policy` are decided first. The path rules from `secrets` to `build` are then
tried in the order of the table and the first match decides a path's category, so
`tests/test_auth.py` is a test, a key file under `tests/` is still a secret, and
`tests/fixtures/package.json` is a test file, not a build file. `.fix-issue-ignore` is checked on
every path regardless of its category. Patterns are gitignore-like and case-insensitive: a pattern without `/`
matches any single path component, a pattern with `/` is anchored at the repository root, and a
pattern that matches a directory matches everything below it.

Triage only saw the issue text, while the implementer is asked to add a regression test, so
triage-suggested forbidden globs (for example `tests/**`) do not apply to test files. Every other
rule, `.fix-issue-ignore` and the hard-coded blocks still apply to them.

**Where the policy comes from.** The agent must not be able to change the rules that judge it.
Both policy files are read from the base commit when the workspace is created, before any
repository or model code runs, and that snapshot is used for every check in the run (the baseline
profile's legacy check reads them from the base commit on each check); the copies in the working
tree are never trusted. Only `.fix-issue-ignore` patterns found in the working tree
are added on top, since they can only make the policy stricter. If the files cannot be read from
the base commit, the repository rules are not applied and the run gets a review finding.

The size limit is `min(max_diff_lines, triage diff budget)`. A target repository can add its own
rules in `.fix-issue-scope.toml`; they are tried before the defaults, so a repository can also
re-classify its own paths (but not the hard-coded `unsafe_path` and `policy` categories), and
`.fix-issue-ignore` and triage constraints always apply on top:

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
| eslint, tsc | eslint config files / `tsconfig.json`; run with `node` from `node_modules/eslint/bin/eslint.js` / `node_modules/typescript/bin/tsc` |
| go vet | `go.mod` |
| npm/make/tox lint scripts | only with `[validation] allow_project_scripts = true` |

A tool that cannot run is recorded as skipped with a reason; it never fails validation.

**Tests that already failed before the patch.** When the suite fails but every failing test was
already failing in the baseline run (and the run did not time out), the attempt counts as failed
once: the next attempt gets feedback saying there are no new failures, listing the pre-existing
ones, and asking the model to make them pass if they reproduce the issue or to keep the fix
otherwise. If a later attempt again leaves only pre-existing failures, the change goes on to
the linters and the review instead of failing; gate G2 caps the verdict at `HUMAN_REVIEW`
([step 9](#9-independent-review-with-deterministic-gates)). Without the review such a change is
published only as a draft. Any new failure is still a normal test failure.

**How repository code is run.** Tests, linters and type checkers execute code from the target
repository, so every such command goes through `sandbox.run_command`:

- The environment is an allowlist: executable lookup and OS directories, temp, home and per-user
  profile directories, locale and terminal settings, TLS trust-store paths, and a fixed set of
  Python, Node, Go, Rust and Java variables. Everything else (GitHub and LLM tokens, database
  URLs, webhooks, agent sockets, CI tokens, registry credentials) is dropped. Variables named in
  `FIX_ISSUE_ENV_PASSTHROUGH` (names separated by commas, spaces or semicolons, for example
  `DATABASE_URL,JAVA_OPTS`) are passed through from your environment. `PYTHONDONTWRITEBYTECODE=1`,
  `CI=1` and `GIT_TERMINAL_PROMPT=0` are always set. `HOME` is not redirected.
- Commands are argument lists, never a shell, and run inside the clone.
- Timeouts are wall-clock (`[limits] test_timeout_seconds` for the test suite, 120 s per linter
  run). The child runs in a Windows Job object (created suspended, kill-on-close) or in its own
  POSIX session; on timeout, and again after a normal exit, the whole process tree is killed, so a
  grandchild cannot outlive the command. If no Job object can be created on Windows,
  `taskkill /T /F` is used. Output goes to temporary files, not pipes, so a grandchild holding the
  stream cannot keep fix-issue waiting.
- cmd.exe re-parses the arguments of `.cmd` and `.bat` programs. A `.bat`/`.cmd` command whose
  arguments contain anything cmd.exe could interpret is refused, and the validators never pass the
  changed (model-chosen) file names to a `.cmd`/`.bat` tool at all: eslint and tsc run through
  `node` (on POSIX the `node_modules/.bin` script is a fallback; the Windows `.cmd` shims are never
  used), and a per-file tool that resolves to a batch file is skipped with that reason.
- Python linters and type checkers run as `python -P -m <tool>`, so a module planted in the
  repository cannot shadow the tool.

This limits what repository code can reach, but it is not isolation; see
[Security and limitations](#security-and-limitations).

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
| G2 | tests failed | NEEDS_REVISION; HUMAN_REVIEW when the only failures are tests that already failed before the patch |
| G3 | scope requires human review | HUMAN_REVIEW |
| G4 | scope blocked | REJECTED |
| G5 | new linter/type-checker findings | NEEDS_REVISION |
| G6 | approved, but the reviewer says the issue is not fully addressed | NEEDS_REVISION |
| G7 | approved, but the reviewer says the tests do not verify the fix | HUMAN_REVIEW |
| G8 | approved, but regressions are likely | NEEDS_REVISION |
| G9 | strict mode and any answer not positive | HUMAN_REVIEW |

`NEEDS_REVISION` sends the reviewer's required changes back as feedback for another round (up to
`[pipeline] max_critique_rounds`, each adding one attempt); when that budget is used up the verdict
becomes `HUMAN_REVIEW`. The candidate that was sent for review is kept: if the revision attempts
then fail (edits that do not apply, failing tests, a scope block), the working tree is reset, that
reviewed candidate is re-applied and published with the verdict `HUMAN_REVIEW` (a draft PR), and
the run record notes the fallback. If it cannot be re-applied, the run ends as it would have
without it (`failed` or `scope_violation`). A `CANNOT_FIX` answer during the revision still ends
the run as `cannot_fix`, and a `REJECTED` verdict on the revision still rejects.
`REJECTED` posts the reviewer summary on the issue and opens no PR. If the reviewer call fails, the
verdict is `HUMAN_REVIEW`.

### 10. Pull request

Before committing, the working tree is diffed once more and must be exactly the diff that was
validated and reviewed (see step 7); otherwise nothing is committed. The commit is made with
`--no-verify` directly on the base commit: the fix branch is reset to the base SHA before staging,
whatever repository code did to `HEAD` or the branch in the meantime. Exactly that commit SHA is
force-pushed to `refs/heads/fix-issue-<N>` at the explicit URL `https://github.com/<owner>/<repo>.git`
(with `--no-verify` and `--recurse-submodules=no`), so a remote or URL rewrite planted in the
repository's config cannot redirect it.

The token is never put in a URL or in `.git/config`. Clone and push authenticate with a
per-command `-c http.https://github.com/.extraheader=AUTHORIZATION: basic <base64>`, together with
`-c credential.helper=` (a rejected token fails instead of falling back to the GitHub credentials
stored on your machine) and `-c http.sslVerify=true`. The header is on the git process's command
line for the duration of that command. Errors from git have the token removed, in plain and base64
form.

A PR is then opened against the default branch (if a PR from that branch is already open, its URL
is returned). When the verdict is `HUMAN_REVIEW` the PR is opened as a draft with a warning banner.
The body has these sections: Root Cause, Fix Summary, Changes, Confidence Score, Files Modified,
Why This Should Work, Validation Executed, Independent Review (verdict and gates), Scope Notes,
Remaining Limitations.

Model-written text in the body (root cause, summary, reasons, reviewer summary, gates, scope notes,
limitations) passes through `redaction.neutralize()`: configured keys and tokens and common
credential shapes are redacted, `@mentions` are broken with a zero-width space so nobody is
pinged, closing keywords followed by an issue reference (`fixes #12`) are broken the same way so
they cannot close or link other issues, and each field is length-capped. The `Resolves #<N>` line
for the issue being fixed is added outside that text.

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
| `duplicate` | duplicate backed by similarity evidence | pointer to the original (when one is identified) |
| `out_of_scope` | not something to fix automatically | nothing |
| `cannot_fix` | the model declined | nothing |
| `scope_violation` | the last attempt was blocked by the scope guard, or the working tree changed after review | the scope guard's blocking reasons and the run id (nothing when the tree changed after review) |
| `rejected` | the independent review rejected the patch | reviewer summary and required changes |
| `failed` | attempts exhausted | the last failure category, a fixed one-line description of it and the run id |
| `error` | the pipeline stopped on an exception of any kind (LLM, configuration, git, GitHub API, file system) | nothing |

Public comments never contain raw test or tool output or error messages: that text can include
local paths, file contents printed by repository code, provider errors or credentials. The
failure comment names the category and points to the local run record
(`fix-issue runs show <run-id>`).
Model-written text that does appear in comments (the clarifying question, the duplicate reason,
the reviewer summary and required changes, the scope reasons) goes through the same
`neutralize()` as the PR body. An `error` outcome keeps the redacted message and traceback in the
run record instead of ending in an unrecorded traceback.

`fix-issue owner/repo N` exits 0 for `pr_opened`, `dry_run`, `needs_info`, `out_of_scope` and
`duplicate`, and 1 otherwise.

Every run writes a JSON record to `~/.fix-issue/runs/<run-id>.json` (root configurable with
`FIX_ISSUE_HOME`), updated after each stage: parameters, issue, triage and control signals,
similar issues, where the scope policy was read from, tooling, baseline tests, retrieval with
reasons (and the initial retrieval before any planner-requested expansion), every plan version,
every attempt (intent, edits, diff, scope report, tests, linters, failure analysis), every review
round with its gates, a per-call LLM log (purpose, sizes, latency, the last 6000 characters of the
prompt and the first 6000 of the response) and the outcome. `fix-issue runs` reads these, and
`fix-issue rerun` repeats a run with the recorded parameters. A one-line summary per GitHub run is
also still appended to `~/.fix-issue-runs.jsonl` (`FIX_ISSUE_HOME/runs.jsonl` when
`FIX_ISSUE_HOME` is set).

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
3. environment variables, including a `.env` file in the current working directory. Only
   `FIX_ISSUE_*`, `LLM_*`, `NVIDIA_API_KEY` and `GITHUB_TOKEN` lines are loaded from `.env`; any
   other variable in it (`PATH`, `GIT_*`, proxies, ...) is ignored, and a `.env` value never
   replaces a variable that is already set in the real environment
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
- `NVIDIA_API_KEY` is used as the `[llm]` key when `LLM_API_KEY` is not set at all and the `[llm]`
  base URL is an `https` URL whose host is `nvidia.com` or a subdomain of it.
  `https://integrate.api.nvidia.com/v1` qualifies; `http://...` URLs and look-alike hosts such as
  `integrate.api.nvidia.com.example.org` do not.
- When the critic or the API embeddings have no key of their own, they get the `[llm]` key only
  if their base URL has the same host as `[llm] base_url`; an
  endpoint on `nvidia.com` (same rule as above) can use `NVIDIA_API_KEY`; otherwise no key is
  sent. With the NIM setup below one key covers all three.
- `[critic] base_url` falls back to `[llm] base_url` when empty. So a critic with a different
  `provider` (`anthropic` or `gemini`, which ignore `base_url`) and no `api_key` of its own counts
  as the same host and gets the `[llm]` key, which is then sent to that provider. Set
  `[critic] api_key` whenever the critic uses a different provider.
- The Gemini key is sent in the `x-goog-api-key` header, not in the URL.
- Every configured key and token is registered when the configuration is loaded. Provider errors,
  recorded LLM errors and `error` outcomes go through `redaction.redact()`, which removes those
  values (also in their base64 basic-auth form) and common credential shapes (GitHub tokens,
  `nvapi-`, `sk-` and `AIza` keys, bearer/basic headers, `key=`/`token=` query parameters,
  credentials in URLs). This is a safety net, not a guarantee.
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

`[critic]`: `provider`, `api_key`, `base_url`, `model`. An empty `provider`, `base_url` or
`model` falls back to `[llm]`; for `api_key` see [API keys](#api-keys). A separate reviewer is
built only when `model` or `provider` is set (or `--critic-model` is given); otherwise the
implementer model reviews with the separate reviewer prompt.

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
| `embedding_api_key` | empty | when empty, the `[llm]` key only for the same host, see [API keys](#api-keys) |
| `cache_dir` | `~/.fix-issue/cache` | embedding cache for the pipeline and for `bench retrieval --config` |

If the API embedder is misconfigured, the pipeline records why and falls back to local vectors.

`[validation]`

| Key | Default | Meaning |
|---|---|---|
| `run_linters` | `true` | run configured linters |
| `run_type_checkers` | `true` | run configured type checkers |
| `run_formatters` | `true` | run configured formatter checks |
| `allow_project_scripts` | `false` | allow `npm run lint`, `make lint`, `tox -e lint` (arbitrary repository code) |
| `sandbox_prefix` | `[]` | argv prefix wrapped around the full profile's test and linter commands, e.g. a `docker run ...` invocation; untested and not applied everywhere, see [limitations](#security-and-limitations) |

### Environment variables

| Variable | Overrides |
|---|---|
| `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | `[llm]` |
| `NVIDIA_API_KEY` | `[llm] api_key`, only for an `https` base URL on `nvidia.com` or a subdomain and when `LLM_API_KEY` is unset; also the key for a critic or embedding endpoint on such a host that has no key of its own |
| `GITHUB_TOKEN` | `[github] token` |
| `FIX_ISSUE_MAX_DIFF_LINES`, `FIX_ISSUE_MAX_RETRIES` | `[limits]` |
| `FIX_ISSUE_PROFILE` | `[pipeline] profile` |
| `FIX_ISSUE_EMBEDDING_BACKEND` | `[retrieval] embedding_backend` |
| `FIX_ISSUE_HOME` | root for `runs/` and `cache/` (default `~/.fix-issue`); when set, the one-line summary log is `FIX_ISSUE_HOME/runs.jsonl` instead of `~/.fix-issue-runs.jsonl` |
| `FIX_ISSUE_ENV_PASSTHROUGH` | not a setting: names of extra variables (separated by commas, spaces or semicolons) passed from your environment to repository code and git, which otherwise get only the allowlist described in [step 8](#8-tests-and-diff-aware-linters) |

Environment variables also override `benchmark/bench.toml`, so unset `LLM_*` and `FIX_ISSUE_*`
variables before a benchmark run you want to compare. Set `FIX_ISSUE_HOME` in the real
environment rather than in `.env`: `fix-issue runs ...` and `fix-issue rerun` look up run records
before (or without) loading `.env`.

### Per-repository files

These live in the target repository, so the rules are visible to anyone reviewing the agent's PRs.

- `.fix-issue-ignore`: glob patterns (fnmatch, on the full path or the file name) the agent must
  never modify. A match blocks the patch. See `.fix-issue-ignore.example`.
- `.fix-issue-scope.toml`: extra scope rules, see [Scope guard](#7-scope-guard).

Both are read from the base commit, and the agent can never edit either of them.
`.fix-issue-scope.toml` is never read from the working tree. Patterns from a `.fix-issue-ignore`
in the working tree are added to the base commit's, because they can only add restrictions.

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
| `--no-critique` | skip the independent review (no gates; the PR is a draft only when tests that failed before the patch still fail) |
| `--embedding-backend local\|openai_compatible\|none` | retrieval embeddings |
| `--temperature X` | sampling temperature |

`fix-issue --version` prints the version.

### Run history

```bash
fix-issue runs list [--limit 20] [--repo owner/repo] [--result pr_opened] [--json] [--runs-dir DIR]
fix-issue runs show <run-id> [--section retrieval] [--json] [--runs-dir DIR]
fix-issue runs stats [--runs-dir DIR]
fix-issue rerun <run-id> [--runs-dir DIR] [options]
```

- Run records are read from `FIX_ISSUE_HOME/runs` (default `~/.fix-issue/runs`). `--runs-dir`
  reads another directory instead, for example a benchmark run's
  `benchmark/results/runs/<results-name>/`.
- `runs list` shows id, start time, repo, issue, profile, classification, files retrieved,
  attempts, verdict, result and duration.
- `runs show` prints a readable summary of everything a run did and why. `--section` prints one
  part as JSON: `issue`, `triage`, `controls`, `similar_issues`, `tooling`, `baseline`,
  `retrieval`, `plans`, `attempts`, `critiques`, `pr`, `llm`, `outcome`, `events`, `params`.
  `--json` prints the whole record. A unique part of the run id is enough. Without `--runs-dir`,
  a run that is not in `FIX_ISSUE_HOME/runs` is also searched for in every
  `benchmark/results/runs/*/` directory.
- `runs stats` counts results and review verdicts and shows the mean number of attempts.
- `rerun` takes the same options as `resolve` and finds runs the same way as `runs show`. It
  restores the recorded profile, model, temperature, critic model, `[limits]` and `[pipeline]`
  values, then applies your options. Provider, keys, retrieval and validation settings come from
  the current configuration. A run that was recorded as a dry run is repeated as a dry run.
- A benchmark run is re-run through the benchmark harness on a fresh copy of its case (always a
  dry run, graded, the result printed as JSON, the new run record written to
  `FIX_ISSUE_HOME/runs`). Without `--config` it uses `benchmark/bench.toml`, not
  `~/.fix-issue.toml`.

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
| Scope | `scope_guard.evaluate` with the pinned policy: before writing, again on the real tree after tests and linters, and a tree-equals-reviewed-diff check before committing | `scope_guard.check` before applying the diff: unsafe (alias) paths and the policy files, `.fix-issue-ignore` from the base commit, raw line count; a violation ends the run |
| Linters / type checkers | diff-aware | none |
| Retries | categorised recovery from the current file state, plan revision | error text appended to the next prompt; tree reset after a test failure |
| Review | independent critique + gates | none |
| PR | structured explanation; draft on HUMAN_REVIEW | original body; never a draft |
| Comments on failure | `failed`, `scope_violation`, `rejected` | none |
| `sandbox_prefix` | applied to tests and linters | not applied |
| git hardening, environment allowlist, timeouts | yes | yes (same `GitWorkspace` and `sandbox.run_command`) |

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
  results/diagnostic/         two earlier runs made before the fixes on this branch, kept as evidence
                              for those fixes; not the reported result (see its README.md)
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
Retrieval is scored as recall@5 on the initial retrieval (before any planner-requested
expansion), so both profiles are measured on their first five files; cases where triage stopped
the run before retrieval have no recall value instead of 0. Grading runs git with hooks and
fsmonitor off, an allowlisted environment and a timeout (the `.git/config` restore is not applied
there, because the workspace has already been closed), runs pytest through the same runner as the
pipeline (allowlisted environment, process-tree kill, output decoded as UTF-8 with replacement),
and deletes the temporary case repository afterwards.

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
  `benchmark/results/runs/<results-name>/`. `--parallel` sets how many cases run at once.
  `fix-issue runs show <run-id>` and `fix-issue rerun <run-id>` find those records automatically;
  `runs list` and `runs stats` need `--runs-dir benchmark/results/runs/<results-name>`.
- `bench retrieval --config` uses that file's `[retrieval] cache_dir` for the embedding cache when
  it is set.
- `bench review` walks through fix cases that need a human look (cases marked
  `requires_manual_review`, and cases where the visible and hidden tests disagree; `--all` for every
  fix case), shows the automated grades and the diff, and stores verdicts in
  `<results>.manual.json`. `bench report` applies them: a `correct` or `incorrect` verdict on a fix
  case overrides "issue fixed", and success, first-attempt success, "tests pass but not fixed" and
  the failure category are re-derived from it, with the same conditions as automated success (so
  a run that ended in `error` stays a failure whatever the verdict). The automated values are kept as
  `issue_fixed_automated` and `failure_category_automated`. Reviews still pending are counted.

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

fix-issue runs the target repository's tests and linters on your machine, and those execute code
from the repository, including test code the model may have written. The measures below make that
harder to abuse. They are not isolation.

What is defended:

- **git cannot be turned against the run.** Every git command on the workspace runs with hooks
  disabled (`core.hooksPath` set to an empty directory fix-issue owns), `core.fsmonitor=false`, no
  automatic gc or maintenance, an allowlisted environment and a timeout. `.git/config` is
  restored from a snapshot taken before any repository code ran, before every git command, so
  filters, textconv drivers, `core.sshCommand` or URL rewrites planted there have no effect; a planted
  `commondir` / `config.worktree` is removed and a replaced `.git` aborts. Commit and push use
  `--no-verify`, the commit is made directly on the base commit, and exactly that SHA is pushed to
  an explicit URL ([step 3](#3-workspace-tooling-baseline), [step 10](#10-pull-request)).
- **The GitHub token is kept away from repository code.** It is never in a URL, in `.git/config`
  or in the environment of repository code. Clone and push authenticate with a per-command extra
  header, with `credential.helper` reset so a rejected token cannot fall back to your stored
  credentials. Git errors and tracebacks have it redacted. (Repository code can still read it from
  any file you keep it in, see below.)
- **Repository code gets a reduced environment and bounded run time**: an environment allowlist
  (extendable with `FIX_ISSUE_ENV_PASSTHROUGH`), no shell, wall-clock timeouts with the whole
  process tree killed (Windows Job object / POSIX session), output in temp files, `.cmd`/`.bat`
  programs never run with model-influenced arguments, Python tools run as `python -P -m`, and
  project lint scripts only with `allow_project_scripts` ([step 8](#8-tests-and-diff-aware-linters)).
- **Edits stay in the repository.** Paths that could alias `.git` or another file on Windows or
  macOS, `..`, absolute paths and targets that resolve outside the clone or under `.git` are
  refused ([step 6](#6-structured-edits-and-recovery)). Retrieval never reads files outside the
  clone, and recovery never reads refused or out-of-repo paths into the next prompt.
- **The agent cannot loosen its own scope policy.** The policy is read from the base commit
  before any repository or model code runs. The policy files themselves, everything under
  `.github/`, other CI systems and pre-commit, credential files, `.gitattributes`/`.gitmodules`,
  lock files, binary patches and symlink/submodule entries are blocked; build and packaging
  manifests, auth and payment code, new migrations, deletions and public API changes need review.
  The whole working tree is checked again after tests and linters ran, and must equal the reviewed
  diff before anything is committed ([step 7](#7-scope-guard)).
- **API keys are routed by host**: `NVIDIA_API_KEY` only to `https` hosts on `nvidia.com`, the
  `[llm]` key to a critic or embedding endpoint only on the same host (with the critic caveat in
  [API keys](#api-keys)); the Gemini key is sent in a header; `.env` can only set fix-issue's own
  variables; configured secrets and common credential shapes are redacted from provider errors,
  recorded LLM errors and `error` outcomes.
- **Public text carries no raw output.** Comments and PR bodies never include raw test or tool
  output or error messages; the model-written text they do include is neutralised (redaction, no
  `@mentions`, no closing keywords for other issues, length caps).
- **Every failure is recorded.** An exception of any type ends as a recorded, redacted `error`
  outcome with the traceback in the run record.
- No PR is merged automatically. A PR is a draft whenever the verdict is `HUMAN_REVIEW`.
- The baseline profile uses the same git hardening and command runner, but its scope check only
  covers alias paths, the policy files, `.fix-issue-ignore` and the size limit.

What is not defended:

- **No container isolation.** Repository code runs as your user, with your file-system and network
  access. It can read files under your home directory: `~/.fix-issue.toml` (your API keys, if you
  keep them there), `~/.git-credentials`, `~/.ssh/`, cloud credentials (`~/.aws/`,
  `~/.config/gcloud/`, `~/.azure/`), registry tokens in `~/.npmrc` and so on, and it can send them
  anywhere. It can edit your global or system git config, which fix-issue does not restore and
  which affects your other repositories (fix-issue's own git commands only override the keys it
  sets with `-c`). It can also escape the process-tree kill: on Windows by starting processes
  outside the Job object (WMI, scheduled tasks, services), on POSIX by leaving the process group
  (`setsid`, as in the usual daemon double fork).
- **`sandbox_prefix` is not a working sandbox.** `[validation] sandbox_prefix` exists to wrap
  commands in a container (for example `docker run ...`), but it is untested: commands use the
  host interpreter path and the host working directory, so a wrapper has to make both valid inside
  the container. It is applied only to the full profile's test and linter runs: not to the
  baseline profile's tests, not to the validators' `--version` probes, not to git, and not to the
  benchmark harness's grading runs. Only run fix-issue on repositories whose code you are willing
  to execute.
- **All git hooks are disabled**, including your own global hooks and git-lfs's `pre-push` hook.
  A repository that depends on them (for example git-lfs uploads on push) does not get them.
- **Pushing triggers CI before review.** The fix branch is pushed to the target repository itself,
  not to a fork. Its `on: push` workflows then run the model-written change with whatever secrets
  those workflows have, before anyone has looked at the PR. The scope guard keeps the agent out of
  the workflow files, not out of the code they execute.
- **`.env` is read from the current directory.** Do not run fix-issue inside an untrusted
  checkout: a `.env` there could set `LLM_BASE_URL` (your `LLM_API_KEY` would then be sent to that
  host), `FIX_ISSUE_ENV_PASSTHROUGH` (passing your variables to repository code) or
  `FIX_ISSUE_HOME`.
- **Run records contain private code.** Records under `FIX_ISSUE_HOME` (default
  `~/.fix-issue/runs/`) keep diffs, test output tails and, for every LLM call, the last 6000
  characters of the prompt and the first 6000 of the response, which include the repository's
  source. They are written with default file permissions and never deleted by fix-issue.
- **Redaction is a safety net.** It removes the configured secrets and common credential shapes,
  not every secret a repository or a provider error might contain.
- **Embeddings add no measurable retrieval gain on the benchmark repositories.** With the API
  embedding model from `benchmark/bench.toml` (`nvidia/nemotron-3.5-embed-1b`), layered retrieval
  has the same recall@1/3/5, MRR and symbol recall as without embeddings, and returns slightly
  more context (`benchmark/results/retrieval-api.md`); the local n-gram vectors add nothing either
  (see [Results](#results)). The repositories are small, so this says little about large ones.
- **Duplicate detection has low recall by design.** The per-scorer thresholds are set for zero
  false positives on a labelled calibration set (`tests/data/duplicate_pairs.json`, 30 hand-written
  pairs, 15 of them duplicates). At those thresholds the local scorer finds 2 of the 15 duplicates
  and the NIM embedding model 8 of 15; other API models use the uncalibrated 0.82. The benchmark's
  duplicate case (`shopcart-h4`) scores below the NIM threshold and is expected to fail.
- **The benchmark is small and Python-only.** Three fixture repositories with roughly 650 to 1,150
  lines of source each, 23 cases. Retrieval and validators support other languages, but nothing
  outside Python is measured, and results on large real-world repositories may differ.
- **Triage `allowed_paths` do not block.** Triage only sees the issue text, so a change outside the
  paths it suggested produces a review finding (and so a `HUMAN_REVIEW` verdict and a draft PR),
  not a block. Triage `forbidden_paths` do block.
- With `--no-critique` there is no verdict and the deterministic gates don't run. The scope checks
  still apply, but findings that only ask for review (auth, payments, build files, public API, ...)
  no longer make the PR a draft; a PR is a draft only when tests that failed before the patch
  still fail.
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
| `config.py` | dataclasses for every TOML section, filtered `.env` loading, environment overrides, exact-host `NVIDIA_API_KEY` matching, per-host key routing for the critic and embeddings, registering secrets for redaction, validation, per-scorer duplicate thresholds |
| `github_client.py` | GitHub REST API over plain `requests` (issues, comments, contents, branches, PRs) |
| `llm_provider.py` | provider-neutral LLM interface (Anthropic, OpenAI-compatible, Gemini with the key in a header) with bounded retries on 429/5xx/transport errors and redacted errors; `RecordingLLM` logs every call |
| `redaction.py` | `register_secret` / `redact`: removes configured keys and tokens (plain and base64) and common credential shapes from text; `neutralize`: `redact` plus broken `@mentions` and closing keywords and a length cap, for model text in comments and PR bodies |
| `structured.py` | JSON extraction from LLM output and a small declarative schema validator, with one bounded repair round |
| `triage.py` | structured triage, `derive_controls` (control signals that only tighten limits), duplicate-evidence check; legacy prompt for the baseline |
| `duplicates.py` | similar-issue search against recently closed GitHub issues and past runs; records which scorer produced each score; scores inputs the embedding API refuses locally |
| `repo_manager.py` | git workspaces: `GitWorkspace` (hardened git: no hooks, fsmonitor off, `.git/config` snapshot restored before every command, allowlisted env, timeouts, redacted `GitCommandError`; diff, reset, base content, commit on the base commit), `RepoManager` (temp clone of a GitHub repo with per-command auth, exact-SHA push to an explicit URL, cleanup also on failure), `LocalWorkspace` (benchmark); `git_command` / `git_env` for other modules; test-command detection; `git apply` with `--3way` fallback |
| `planner.py` | schema-validated plans checked against the repository; `revise_plan`; retrieval-based fallback plan |
| `editing.py` | SEARCH/REPLACE parser and atomic in-memory edit engine with tolerant-but-unique matching, path safety (canonical repo-relative paths; alias, `.git` and out-of-repo paths refused), syntax validation and `git apply`-compatible diffs |
| `recovery.py` | failure categories and current-state feedback (exact error, fresh excerpts, partial changes, plan contradictions); never reads refused or out-of-repo paths |
| `scope_guard.py` | `evaluate`: path- and severity-aware scope report (hard-coded `unsafe_path` and `policy` blocks, default and repo rules, line and character budgets, symlink/submodule entries, public API); `load_policy`: the policy as of the base commit; `check`: legacy all-or-nothing check used by the baseline |
| `validators.py` | discovery of the repository's own linters/type checkers/formatters and diff-aware runs against a base-commit worktree; `.cmd`/`.bat` tools never receive file names |
| `test_runner.py` | runs the test command through `sandbox.run_command`, parses pytest failures and the summary line |
| `sandbox.py` | environment allowlist (plus `FIX_ISSUE_ENV_PASSTHROUGH`); `run_process` / `run_command`: no shell, output in temp files, wall-clock timeouts, process-tree kill (Windows Job object or `taskkill`, POSIX session); `.bat`/`.cmd` argument guard; optional `sandbox_prefix` |
| `critic.py` | independent review prompt, structured answers, deterministic gates G1-G9 (G2 caps pre-existing failures at HUMAN_REVIEW) |
| `pr_builder.py` | PR title and body, structured explanation sections with model text neutralised, confidence formula, draft flag |
| `run_history.py` | per-run JSON records under `~/.fix-issue/runs` (or `FIX_ISSUE_HOME/runs`), listing, loading and summaries |
| `logger.py` | one-line JSON summary per GitHub run appended to `~/.fix-issue-runs.jsonl`, or to `FIX_ISSUE_HOME/runs.jsonl` when `FIX_ISSUE_HOME` is set (kept from v1) |
| `file_selector.py` | keyword file scoring (baseline profile; also the `keyword` variant of the retrieval benchmark) |
| `patch_generator.py` | baseline profile patch generation: SEARCH/REPLACE blocks turned into one unified diff |

### `fix_issue/retrieval/`

| Module | Responsibility |
|---|---|
| `__init__.py` | public exports |
| `index.py` | repository index: files, symbols with spans and signatures, imports, references (Python `ast`, regex for other languages); symlinks and paths resolving outside the root are skipped |
| `mentions.py` | explicit evidence in the issue text: paths, stack frames, identifiers, exception types, resolved against the real file list |
| `lexical.py` | BM25 over code-aware tokens |
| `embeddings.py` | local n-gram embedder, OpenAI-compatible embedder, on-disk vector cache, `build_embedder` |
| `retriever.py` | layered scoring, graph expansion, per-file reasons, context rendering (outline + focus symbols for large files), retrieval metrics |
| `text.py` | identifier splitting, tokenisation and stemming shared by all channels |

### `fix_issue/bench/`

| Module | Responsibility |
|---|---|
| `__init__.py` | package marker, so `fix_issue.bench` is included in a non-editable install (the cases and fixtures in `benchmark/` are still read from the source tree) |
| `cases.py` | load and validate `case.toml`, copy fixtures, inject bugs, initialise the git repo, install hidden tests |
| `validate.py` | `bench validate`: clean fixture passes, buggy baseline fails the hidden tests; `run_pytest` through the sandboxed runner |
| `retrieval_eval.py` | `bench retrieval`: retrieval variants against ground-truth files and symbols |
| `runner.py` | `bench run`: real orchestrator per case in a temp repo (deleted afterwards), independent grading with hardened git and the sandboxed pytest runner (size cap on non-test lines, recall@5 on the initial retrieval), results JSON |
| `report.py` | metrics, Markdown report, side-by-side comparison; applies manual verdicts and re-derives the fields that depend on them |
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
| `tests/test_duplicate_calibration.py` | duplicate thresholds against the labelled pairs in `tests/data/duplicate_pairs.json` (no false positives, the threshold rule), per-scorer threshold lookup, local scoring of inputs the API refuses; the NIM check is opt-in |
| `tests/test_retrieval.py` | mentions, index, layered retrieval, embeddings cache, symlinks and out-of-clone paths, no phantom last line in rendered context |
| `tests/test_embeddings_api.py` | API embedder handling of refused/blank inputs, cache robustness |
| `tests/test_editing.py` | edit parsing, matching rules, atomic batches, line endings, diffs, alias/`.git`/symlink path safety, `CANNOT_FIX` and "already applied" rules |
| `tests/test_scope_guard.py` | scope categories and severities, repo rules, policy read from the base commit, alias paths and policy files, character budget, symlink/submodule entries, triage globs vs test files, public API detection, legacy check |
| `tests/test_validators.py` | tool discovery, output parsers, new-finding comparison, worktree cleanup, `.cmd`/`.bat` handling, hooks and config planted by repository code ignored in the baseline worktree |
| `tests/test_sandbox.py` | environment allowlist, wall-clock timeouts that kill grandchildren, no stragglers after a normal exit, output capture, the `.bat`/`.cmd` guard |
| `tests/test_repo_manager.py` | git hardening against planted hooks, config, `commondir` and a swapped `.git`; commit on the base; per-command auth; token redaction on clone and push failures (each attack is also fired with plain git to show it is live) |
| `tests/test_config_security.py` | NVIDIA host matching, key routing for critic and embeddings, `.env` filtering, TOML boolean strings, redaction, Gemini key in a header |
| `tests/test_pipeline_components.py` | structured output, planner, critic gates (including G2 for pre-existing failures), recovery, run recorder, PR explanation, benchmark metrics, `.env` / `NVIDIA_API_KEY` |
| `tests/test_orchestrator.py` | the whole pipeline (both profiles) on a benchmark fixture, driven by a scripted test double instead of an LLM |
| `tests/test_bench_grading.py` | benchmark grading on a real fixture with hand-written trees (no LLM): size cap on non-test lines, manual verdicts and re-derived fields, non-ASCII pytest output |
| `tests/test_cli_runs.py` | `runs list` / `runs show` with `--runs-dir`, a missing run exits non-zero |
| `tests/test_file_selector.py` | keyword scoring (baseline) |
| `tests/test_confidence_score.py` | confidence formula and PR body |

At the time of writing the suite gives 504 passed, 1 skipped. The skipped test scores the
calibration set with NVIDIA NIM; it needs the network and a key and runs only with
`FIX_ISSUE_NIM_TESTS=1`.

The orchestrator tests check the plumbing (recovery, review rounds, scope blocks, triage
short-circuits), not how well a real model fixes issues; that is what `fix-issue bench run`
measures.

---

## Not done yet

- Real isolation for repository code: a tested container wrapper applied everywhere repository
  code runs (both profiles, benchmark grading), with its own home directory and git config
- Pushing the fix branch somewhere that does not run the repository's `on: push` workflows with
  secrets before review (a fork, for example)
- Restrictive permissions for run records, and an option not to keep prompt and response text
- Support for repositories that need git hooks (git-lfs)
- Duplicate detection with useful recall at zero false positives, calibrated on more data
- Benchmark cases in other languages and on larger repositories
- A GitHub Actions workflow (`on: issues.opened`) to trigger runs automatically
- A hosted GitHub App / webhook receiver (needs an always-on host)

## License

MIT
