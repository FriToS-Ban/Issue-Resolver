# fix-issue

**Autonomous GitHub Issue Resolution Agent**

A local CLI tool that fetches a GitHub issue, triages it with an LLM, generates a patch, runs the repo's own test suite, and opens a PR — all in one command, on your machine, with no server required.

```
fix-issue owner/repo 123
```

---

## How It Works

```
fix-issue owner/repo 123
      │
      ▼
┌──────────────┐    ┌───────────┐    ┌────────────────────────────────────┐
│ Fetch Issue  │───▶│  Triage   │───▶│  Retry Loop (up to MAX_RETRIES)    │
│ (GitHub API) │    │  (LLM)    │    │  ┌──────────────────────────────┐  │
└──────────────┘    └───────────┘    │  │ Generate patch (LLM)         │  │
                                     │  │ Scope check (BEFORE tests)   │  │
                                     │  │ git apply → git apply --3way │  │
                                     │  │ Run test suite               │  │
                                     │  │ On failure: feed output back │  │
                                     │  └──────────────────────────────┘  │
                                     └──────────────┬─────────────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │  Open PR          │
                                          │  (GitHub API)     │
                                          └──────────────────┘
```

**Pipeline steps:**
1. Fetch the issue via the GitHub API
2. Triage: classify as `fixable` / `needs_info` / `out_of_scope` / `duplicate`
3. Clone the repo to a temp dir (cleaned up automatically, even on errors)
4. Score repo files for relevance; pass top-N as LLM context
5. Retry loop: generate patch → scope check → `git apply` → run tests → retry on failure
6. Commit, push fix branch, open PR with structured description + confidence score
7. If can't fix: post a targeted clarifying comment instead

---

## Requirements

- Python 3.11+
- `git` on your PATH
- A GitHub personal access token (repo read/write + pull request scopes)
- An LLM API key (Anthropic, OpenAI-compatible, or Gemini)

---

## Installation

```bash
# Clone this repo
git clone https://github.com/your-org/fix-issue
cd fix-issue

# Install (editable, with dev tools)
pip install -e ".[dev]"

# Verify
fix-issue --help
```

---

## Configuration

fix-issue uses **two separate config files** — intentionally:

| File | Where | Purpose |
|---|---|---|
| `~/.fix-issue.toml` | Your home directory | Personal defaults: tokens, API keys, limits |
| `.fix-issue-ignore` | Target repo root | Glob patterns for paths the agent must never touch |

Personal secrets never touch a repo-committed file. Repo scope rules never live in your home dir.

### 1. Set up `~/.fix-issue.toml`

Copy the template and fill in your values:

```bash
cp .fix-issue.toml.example ~/.fix-issue.toml
```

```toml
[llm]
provider = "anthropic"           # anthropic | openai_compatible | gemini
api_key  = "sk-ant-..."
model    = "claude-opus-4-5"

[github]
token = "ghp_..."

[limits]
max_diff_lines = 150             # reject diffs larger than this (before running tests)
max_retries    = 3
```

**For other providers:**

```toml
# OpenAI
[llm]
provider = "openai_compatible"
api_key  = "sk-..."
model    = "gpt-4o"
# base_url defaults to https://api.openai.com/v1

# Ollama (local)
[llm]
provider = "openai_compatible"
api_key  = "ollama"              # any non-empty string
base_url = "http://localhost:11434/v1"
model    = "llama3.1"

# Gemini
[llm]
provider = "gemini"
api_key  = "AIza..."
model    = "gemini-2.5-pro"
```

You can also use environment variables (higher priority than the TOML):

```bash
export LLM_PROVIDER=anthropic
export LLM_API_KEY=sk-ant-...
export LLM_MODEL=claude-opus-4-5
export GITHUB_TOKEN=ghp_...
```

### 2. Add `.fix-issue-ignore` to target repos (optional)

```bash
# In the target repo:
cp /path/to/fix-issue/.fix-issue-ignore.example .fix-issue-ignore
# Edit to add your off-limits paths, then commit it
git add .fix-issue-ignore && git commit -m "chore: add fix-issue scope config"
```

Example `.fix-issue-ignore`:
```
# Paths the agent must never modify
auth/**
payments/**
**/migrations/**
.github/workflows/**
```

---

## Usage

```bash
# Basic usage
fix-issue owner/repo 123

# Dry run — run everything, print the PR body, but don't open a PR or post comments
fix-issue owner/repo 123 --dry-run

# Override limits for this run only
fix-issue owner/repo 123 --max-diff-lines 300 --max-retries 5

# Use a custom config file
fix-issue owner/repo 123 --config /path/to/custom.toml
```

### Example terminal output

```
╭────────────────────────────────────────────╮
│ 🤖 Autonomous Issue Resolver               │
│ Repo: octocat/hello-world   Issue: #42     │
│ Model: claude-opus-4-5   Provider: anthropic│
╰────────────────────────────────────────────╯

→ Fetching issue #42 from octocat/hello-world
  ✓ Fix the NullPointerException in login handler

→ Triaging issue classifying with LLM
  ✓ FIXABLE  (confidence: 91%)  — Small bug in auth/login.py, clear repro steps

→ Cloning repo branch: fix-issue-42
→ Selecting relevant files keyword scoring
  ✓ 3 relevant file(s) selected
  ✓ Test command detected: python -m pytest --tb=short -q

────────────── Attempt 1 / 3 ──────────────

→ Generating patch attempt 1
  ✓ Diff generated (8 lines)
→ Checking scope diff size + ignored paths
  ✓ Scope checks passed
→ Applying patch git apply
  ✓ Patch applied
→ Running test suite python -m pytest --tb=short -q
  ✓ Tests passed (12.3s)

→ Committing changes fix-issue-42
  ✓ Changes committed
→ Pushing branch fix-issue-42
  ✓ Pushed branch fix-issue-42
→ Building PR generating summary
→ Opening PR octocat/hello-world
  ✓ PR opened: https://github.com/octocat/hello-world/pull/99

╭─────────────────────────────────────────────╮
│ ✓ Done!                                     │
│ PR: https://github.com/octocat/hello-world/pull/99 │
│ Confidence: 9/10  •  Attempts: 1  •  Tests: ✅ │
╰─────────────────────────────────────────────╯
```

---

## Triage Outcomes

| Classification | What happens |
|---|---|
| `fixable` | Full fix loop runs, PR opened on success |
| `needs_info` | Posts a targeted clarifying question as a comment |
| `out_of_scope` | Logged, no action (large refactor / security-sensitive) |
| `duplicate` | Logged, no action |

---

## Confidence Score

Every PR includes an honest, formula-based confidence score:

```
score = 10
score -= min(5, diff_lines // 30)      # −1 per 30 lines, max −5
score -= 3  if not tests_passed        # −3 if tests didn't go green
score -= 2  if triage_confidence < 0.7 # −2 if triage was uncertain
score = max(1, score)
```

The PR body shows a table with each component and its penalty. No fake precision.

---

## Run Log

Every run appends a JSON line to `~/.fix-issue-runs.jsonl`:

```jsonc
{
  "timestamp": "2026-08-20T15:30:00Z",
  "repo": "octocat/hello-world",
  "issue": 42,
  "result": "pr_opened",
  "pr_url": "https://github.com/octocat/hello-world/pull/99",
  "confidence": 9,
  "classification": "fixable",
  "triage_confidence": 0.91,
  "attempts": 1
}
```

Query it with `jq`:

```bash
# All PRs opened this month
jq 'select(.result == "pr_opened")' ~/.fix-issue-runs.jsonl

# Average confidence across successful runs
jq -s '[.[] | select(.confidence)] | (map(.confidence) | add) / length' ~/.fix-issue-runs.jsonl
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

Test coverage:
- `test_scope_guard.py` — diff size limits, ignored path patterns
- `test_triage.py` — classification parsing, confidence clamping, mock LLM
- `test_file_selector.py` — keyword scoring, node_modules exclusion, top-N
- `test_confidence_score.py` — formula correctness, floor at 1, PR body structure

---

## Architecture

| Module | Responsibility |
|---|---|
| `cli.py` | Pipeline orchestration + **retry loop** |
| `config.py` | Load `~/.fix-issue.toml` + env vars |
| `github_client.py` | GitHub REST API (plain `requests`) |
| `llm_provider.py` | Provider-neutral LLM interface (Anthropic / OpenAI-compat / Gemini) |
| `repo_manager.py` | Context-manager clone + `git apply` with `--3way` fallback |
| `file_selector.py` | Keyword-based file relevance scoring |
| `triage.py` | Issue classification (takes `LLMProvider`, not a hardcoded SDK) |
| `patch_generator.py` | Unified diff generation (takes `LLMProvider`) |
| `scope_guard.py` | Pre-test scope checks (size + ignored paths) |
| `test_runner.py` | Run test suite, capture result |
| `pr_builder.py` | PR title/body + confidence score formula |
| `logger.py` | Append JSONL run record to `~/.fix-issue-runs.jsonl` |

---

## Scope Limits

The agent has **hard scope limits** that cannot be overridden by the LLM:

1. **Diff size** — diffs over `max_diff_lines` are rejected before any test run
2. **Ignored paths** — paths matching `.fix-issue-ignore` globs are always off-limits
3. **No auto-merge** — every PR requires human review
4. **No credentials in code** — tokens come from `~/.fix-issue.toml` or env vars only

---

## Roadmap

- **v1**: Wrap in a GitHub Actions workflow (`on: issues.opened`) for automatic triggering
- **v1**: SQLite run log + confidence trend dashboard
- **Later**: GitHub App with webhook receiver (needs a small always-on host)
- **Later**: Auto-merge under narrow, pre-configured conditions (diff size + confidence threshold)
- **Later**: Cross-issue pattern detection ("4th report of the same root cause")

---

## License

MIT
