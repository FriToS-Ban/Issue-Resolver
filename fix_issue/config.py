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
from urllib.parse import urlparse

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


# Similarity a listed resolved issue needs before a "duplicate" verdict is accepted, per scorer
# (the embedding backend that produced the similarity — the scales differ a lot). Calibrated on
# tests/data/duplicate_pairs.json (hand-written pairs, not the benchmark) with a precision-first
# rule: the largest recall with zero false positives, placed at the midpoint of the gap between
# the highest-scoring non-duplicate and the next duplicate (``duplicates.precision_first_threshold``).
#   local    offline n-gram scorer; gap taken across candidate-pool sizes (its IDF is pool-fitted)
#   <model>  a calibrated API embedding model (name with or without the "api:" prefix)
#   default  any other scorer — uncalibrated models keep the old conservative 0.82
DUPLICATE_THRESHOLDS: dict[str, float] = {
    "local": 0.41,
    "nvidia/nemotron-3.5-embed-1b": 0.53,
    "default": 0.82,
}


@dataclass
class PipelineConfig:
    profile: str = "full"               # full | baseline (legacy keyword/diff pipeline)
    planning: bool = True
    critique: bool = True
    max_critique_rounds: int = 1        # NEEDS_REVISION -> revise loops allowed
    max_plan_revisions: int = 1
    max_context_chars: int = 60_000     # source context budget per patch prompt
    # Similarity needed to accept "duplicate": a number applies to every scorer (the old
    # behaviour); a table maps scorer -> threshold and is merged over DUPLICATE_THRESHOLDS.
    duplicate_threshold: float | dict[str, float] = field(default_factory=lambda: dict(DUPLICATE_THRESHOLDS))


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


# Only the tool's own settings are taken from a .env file. Anything else (GIT_*, PATH, proxy
# variables...) would change how git and the repository's code behave for the whole run.
_DOTENV_PREFIXES = ("FIX_ISSUE_", "LLM_")
_DOTENV_KEYS = {"NVIDIA_API_KEY", "GITHUB_TOKEN"}


def load_dotenv(path: Path | None = None) -> list[str]:
    """Load fix-issue's own KEY=VALUE lines from a .env file into os.environ (existing vars win).

    Keys outside ``_DOTENV_PREFIXES``/``_DOTENV_KEYS`` are ignored. Returns the keys set.
    """
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
        if not (key in _DOTENV_KEYS or key.startswith(_DOTENV_PREFIXES)):
            continue
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
        and is_nvidia_endpoint(cfg.llm.base_url)
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

    from fix_issue.redaction import register_secret

    for secret in (cfg.llm.api_key, cfg.github.token, cfg.critic.api_key, cfg.retrieval.embedding_api_key,
                   os.environ.get("NVIDIA_API_KEY")):
        register_secret(secret)
    return cfg


def is_nvidia_endpoint(url: str | None) -> bool:
    """True only for https URLs whose host is nvidia.com or a subdomain of it (not a substring match)."""
    try:
        parsed = urlparse(url or "")
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme == "https" and (host == "nvidia.com" or host.endswith(".nvidia.com"))


def _host(url: str | None) -> str:
    try:
        return (urlparse(url or "").hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _key_for_endpoint(own_key: str, base_url: str, cfg: Config) -> str:
    """API key for a secondary endpoint (critic, embeddings) that has none of its own.

    The [llm] key is reused only when the endpoint is the same host as [llm]; an NVIDIA endpoint
    can use NVIDIA_API_KEY. A key is never sent to a host it was not configured for.
    """
    if own_key:
        return own_key
    if _host(base_url) == _host(cfg.llm.base_url):
        return cfg.llm.api_key
    if is_nvidia_endpoint(base_url) and os.environ.get("NVIDIA_API_KEY"):
        return os.environ["NVIDIA_API_KEY"]
    return ""


def _overlay(target: object, raw: dict) -> None:
    """Copy known keys from a TOML table onto a dataclass, coercing to the field's type."""
    for key, value in raw.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(current, bool):
            # bool("false") is True: accept TOML booleans and the usual spellings in strings.
            value = value.strip().lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, list):
            value = list(value)
        elif isinstance(current, dict) and isinstance(value, dict):
            value = {**current, **value}    # a partial table overrides only the keys it names
        setattr(target, key, value)


def duplicate_threshold_for(setting: object, scorer: str | None) -> float:
    """
    Threshold for a similarity produced by ``scorer`` (``SimilarIssue.scorer``, e.g.
    "local-ngram-768" or "api:nvidia/nemotron-3.5-embed-1b").

    ``setting`` is ``[pipeline] duplicate_threshold``: a number applies to every scorer; a
    mapping is looked up by exact scorer name, then the model name without "api:", then
    "local" for local scorers, then "default" — each falling back to DUPLICATE_THRESHOLDS.
    An unknown or missing scorer gets "default" (conservative).
    """
    if isinstance(setting, (int, float)) and not isinstance(setting, bool):
        return float(setting)
    table = {**DUPLICATE_THRESHOLDS, **(setting if isinstance(setting, dict) else {})}
    name = (scorer or "").strip()
    keys = [name, name.removeprefix("api:")] if name else []
    if name.startswith("local"):
        keys.append("local")
    keys.append("default")
    for k in keys:
        if k in table:
            try:
                return float(table[k])
            except (TypeError, ValueError):
                continue
    return DUPLICATE_THRESHOLDS["default"]


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
    dt = cfg.pipeline.duplicate_threshold
    dt_items = dt.items() if isinstance(dt, dict) else [("", dt)]
    for k, v in dt_items:
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0
        if not ok:
            where = f"[pipeline.duplicate_threshold] {k}" if k else "[pipeline] duplicate_threshold"
            errors.append(f"{where} must be a number between 0 and 1, got {v!r}")
    if cfg.llm.provider == "openai_compatible" and not cfg.llm.base_url:
        errors.append(
            "LLM_PROVIDER=openai_compatible requires LLM_BASE_URL "
            "(e.g. http://localhost:11434/v1 for Ollama)"
        )
    return errors


def critic_settings(cfg: Config) -> tuple[str, str, str, str]:
    """(provider, api_key, base_url, model) for the reviewer; falls back to [llm] per field."""
    c = cfg.critic
    provider = c.provider or cfg.llm.provider
    base_url = c.base_url or cfg.llm.base_url
    if c.api_key or provider == cfg.llm.provider and base_url == cfg.llm.base_url:
        key = c.api_key or cfg.llm.api_key       # same provider + endpoint as [llm]
    else:
        key = _key_for_endpoint("", base_url, cfg) if base_url else ""
    return (provider, key, base_url, c.model or cfg.llm.model)


def embedding_settings(cfg: Config) -> tuple[str, str, str, str]:
    """(backend, base_url, api_key, model) for retrieval embeddings."""
    r = cfg.retrieval
    base_url = r.embedding_base_url or cfg.llm.base_url
    return (r.embedding_backend, base_url, _key_for_endpoint(r.embedding_api_key, base_url, cfg), r.embedding_model)
