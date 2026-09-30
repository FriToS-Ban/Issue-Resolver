"""
config.py — Load settings from ~/.fix-issue.toml and environment variables.

Priority (highest → lowest):
    1. Environment variables (FIX_ISSUE_* or LLM_* or GITHUB_TOKEN)
    2. ~/.fix-issue.toml
    3. Hard-coded defaults
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            tomllib = None  # type: ignore[assignment]

_DEFAULT_TOML = Path.home() / ".fix-issue.toml"


@dataclass
class LLMConfig:
    provider: str = "anthropic"         # anthropic | openai_compatible | gemini
    api_key: str = ""
    base_url: str = ""                  # only used when provider=openai_compatible
    model: str = "claude-opus-4-5"


@dataclass
class GitHubConfig:
    token: str = ""


@dataclass
class LimitsConfig:
    max_diff_lines: int = 150
    max_retries: int = 3
    relevant_files: int = 5             # how many files to pass to the LLM
    test_timeout_seconds: int = 300     # 5 minutes per test run


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)


def load(toml_path: Path = _DEFAULT_TOML) -> Config:
    """Load config from TOML file then overlay environment variables."""
    cfg = Config()

    # --- 1. Load TOML ---
    if toml_path.exists():
        if tomllib is None:
            raise RuntimeError(
                "Python <3.11 detected and neither 'tomllib' nor 'tomli' is installed. "
                "Run: pip install tomli"
            )
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))

        llm_raw = raw.get("llm", {})
        cfg.llm.provider = llm_raw.get("provider", cfg.llm.provider)
        cfg.llm.api_key  = llm_raw.get("api_key", cfg.llm.api_key)
        cfg.llm.base_url = llm_raw.get("base_url", cfg.llm.base_url)
        cfg.llm.model    = llm_raw.get("model", cfg.llm.model)

        gh_raw = raw.get("github", {})
        cfg.github.token = gh_raw.get("token", cfg.github.token)

        lim_raw = raw.get("limits", {})
        cfg.limits.max_diff_lines        = int(lim_raw.get("max_diff_lines", cfg.limits.max_diff_lines))
        cfg.limits.max_retries           = int(lim_raw.get("max_retries", cfg.limits.max_retries))
        cfg.limits.relevant_files        = int(lim_raw.get("relevant_files", cfg.limits.relevant_files))
        cfg.limits.test_timeout_seconds  = int(lim_raw.get("test_timeout_seconds", cfg.limits.test_timeout_seconds))

    # --- 2. Overlay environment variables ---
    cfg.llm.provider = os.environ.get("LLM_PROVIDER", cfg.llm.provider)
    cfg.llm.api_key  = os.environ.get("LLM_API_KEY", cfg.llm.api_key)
    cfg.llm.base_url = os.environ.get("LLM_BASE_URL", cfg.llm.base_url)
    cfg.llm.model    = os.environ.get("LLM_MODEL", cfg.llm.model)

    cfg.github.token = os.environ.get("GITHUB_TOKEN", cfg.github.token)

    cfg.limits.max_diff_lines = int(
        os.environ.get("FIX_ISSUE_MAX_DIFF_LINES", cfg.limits.max_diff_lines)
    )
    cfg.limits.max_retries = int(
        os.environ.get("FIX_ISSUE_MAX_RETRIES", cfg.limits.max_retries)
    )

    return cfg


def validate(cfg: Config) -> list[str]:
    """Return a list of human-readable error strings, empty if config is valid."""
    errors: list[str] = []
    if not cfg.github.token:
        errors.append("GITHUB_TOKEN is not set (env var or ~/.fix-issue.toml [github] token)")
    if not cfg.llm.api_key:
        errors.append("LLM_API_KEY is not set (env var or ~/.fix-issue.toml [llm] api_key)")
    if cfg.llm.provider == "openai_compatible" and not cfg.llm.base_url:
        errors.append(
            "LLM_PROVIDER=openai_compatible requires LLM_BASE_URL "
            "(e.g. http://localhost:11434/v1 for Ollama)"
        )
    return errors
