"""
config.py — Load settings from ~/.fix-issue.toml and environment variables.

Priority (highest → lowest):
    1. Environment variables (FIX_ISSUE_* or LLM_* or GITHUB_TOKEN)
       — a ``.env`` file in the working directory is loaded first WITHOUT overriding real
         environment variables. ``NVIDIA_API_KEY`` is used as the LLM key when the base URL
         points at NVIDIA NIM and ``LLM_API_KEY`` is not set.
    2. ~/.fix-issue.toml (or --config)
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
    temperature: float | None = None    # None = provider default; benchmark uses 0.0
    max_tokens: int = 4096


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
class CriticConfig:
    """Optional separate model for the independent reviewer (defaults to [llm])."""
    provider: str = ""
    api_key: str = ""
    base_url: str = ""
    model: str = ""


@dataclass
class PipelineConfig:
    profile: str = "full"               # full | baseline (legacy keyword/diff pipeline)
    planning: bool = True
    critique: bool = True
    max_critique_rounds: int = 1        # NEEDS_REVISION -> revise loops allowed
    max_plan_revisions: int = 1
    max_context_chars: int = 60_000     # source context budget per patch prompt
    duplicate_threshold: float = 0.82   # semantic similarity needed to accept "duplicate"


@dataclass
class RetrievalConfig:
    embedding_backend: str = "local"    # local | openai_compatible | none
    embedding_model: str = ""           # e.g. nvidia/nv-embedqa-e5-v5
    embedding_base_url: str = ""        # defaults to [llm] base_url
    embedding_api_key: str = ""         # defaults to [llm] api_key
    cache_dir: str = ""                 # defaults to ~/.fix-issue/cache


@dataclass
class ValidationConfig:
    run_linters: bool = True
    run_type_checkers: bool = True
    run_formatters: bool = True
    allow_project_scripts: bool = False # npm run / make / tox for lint (arbitrary code)
    sandbox_prefix: list[str] = field(default_factory=list)  # e.g. ["docker","run",...]


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def load_dotenv(path: Path | None = None) -> list[str]:
    """Load KEY=VALUE lines from a .env file into os.environ (existing vars win). Returns keys set."""
    path = path or (Path.cwd() / ".env")
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def load(toml_path: Path = _DEFAULT_TOML, dotenv: bool = True) -> Config:
    """Load config from TOML file then overlay environment variables."""
    if dotenv:
        load_dotenv()
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
        if "temperature" in llm_raw:
            cfg.llm.temperature = float(llm_raw["temperature"])
        cfg.llm.max_tokens = int(llm_raw.get("max_tokens", cfg.llm.max_tokens))

        gh_raw = raw.get("github", {})
        cfg.github.token = gh_raw.get("token", cfg.github.token)

        lim_raw = raw.get("limits", {})
        cfg.limits.max_diff_lines        = int(lim_raw.get("max_diff_lines", cfg.limits.max_diff_lines))
        cfg.limits.max_retries           = int(lim_raw.get("max_retries", cfg.limits.max_retries))
        cfg.limits.relevant_files        = int(lim_raw.get("relevant_files", cfg.limits.relevant_files))
        cfg.limits.test_timeout_seconds  = int(lim_raw.get("test_timeout_seconds", cfg.limits.test_timeout_seconds))

        _overlay(cfg.critic, raw.get("critic", {}))
        _overlay(cfg.pipeline, raw.get("pipeline", {}))
        _overlay(cfg.retrieval, raw.get("retrieval", {}))
        _overlay(cfg.validation, raw.get("validation", {}))

    # --- 2. Overlay environment variables ---
    cfg.llm.provider = os.environ.get("LLM_PROVIDER", cfg.llm.provider)
    cfg.llm.api_key  = os.environ.get("LLM_API_KEY", cfg.llm.api_key)
    cfg.llm.base_url = os.environ.get("LLM_BASE_URL", cfg.llm.base_url)
    cfg.llm.model    = os.environ.get("LLM_MODEL", cfg.llm.model)

    if (
        "LLM_API_KEY" not in os.environ
        and os.environ.get("NVIDIA_API_KEY")
        and "nvidia.com" in (cfg.llm.base_url or "")
    ):
        cfg.llm.api_key = os.environ["NVIDIA_API_KEY"]

    cfg.github.token = os.environ.get("GITHUB_TOKEN", cfg.github.token)

    cfg.limits.max_diff_lines = int(
        os.environ.get("FIX_ISSUE_MAX_DIFF_LINES", cfg.limits.max_diff_lines)
    )
    cfg.limits.max_retries = int(
        os.environ.get("FIX_ISSUE_MAX_RETRIES", cfg.limits.max_retries)
    )
    if os.environ.get("FIX_ISSUE_PROFILE"):
        cfg.pipeline.profile = os.environ["FIX_ISSUE_PROFILE"]
    if os.environ.get("FIX_ISSUE_EMBEDDING_BACKEND"):
        cfg.retrieval.embedding_backend = os.environ["FIX_ISSUE_EMBEDDING_BACKEND"]

    return cfg


def _overlay(target: object, raw: dict) -> None:
    """Copy known keys from a TOML table onto a dataclass, coercing to the field's type."""
    for key, value in raw.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, list):
            value = list(value)
        setattr(target, key, value)


def fix_issue_home() -> Path:
    """Directory for run history and caches (override with FIX_ISSUE_HOME)."""
    return Path(os.environ.get("FIX_ISSUE_HOME") or (Path.home() / ".fix-issue"))


def validate(cfg: Config) -> list[str]:
    """Return a list of human-readable error strings, empty if config is valid."""
    errors: list[str] = []
    if not cfg.github.token:
        errors.append("GITHUB_TOKEN is not set (env var or ~/.fix-issue.toml [github] token)")
    if not cfg.llm.api_key:
        errors.append("LLM_API_KEY is not set (env var or ~/.fix-issue.toml [llm] api_key)")
    if cfg.pipeline.profile not in ("full", "baseline"):
        errors.append(f"[pipeline] profile must be 'full' or 'baseline', got '{cfg.pipeline.profile}'")
    if cfg.retrieval.embedding_backend not in ("local", "openai_compatible", "none"):
        errors.append(
            "[retrieval] embedding_backend must be local | openai_compatible | none, "
            f"got '{cfg.retrieval.embedding_backend}'"
        )
    if cfg.llm.provider == "openai_compatible" and not cfg.llm.base_url:
        errors.append(
            "LLM_PROVIDER=openai_compatible requires LLM_BASE_URL "
            "(e.g. http://localhost:11434/v1 for Ollama)"
        )
    return errors


def critic_settings(cfg: Config) -> tuple[str, str, str, str]:
    """(provider, api_key, base_url, model) for the reviewer; falls back to [llm] per field."""
    c = cfg.critic
    return (
        c.provider or cfg.llm.provider,
        c.api_key or cfg.llm.api_key,
        c.base_url or cfg.llm.base_url,
        c.model or cfg.llm.model,
    )


def embedding_settings(cfg: Config) -> tuple[str, str, str, str]:
    """(backend, base_url, api_key, model) for retrieval embeddings."""
    r = cfg.retrieval
    return (
        r.embedding_backend,
        r.embedding_base_url or cfg.llm.base_url,
        r.embedding_api_key or cfg.llm.api_key,
        r.embedding_model,
    )
