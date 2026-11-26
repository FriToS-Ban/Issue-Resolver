"""Configuration loading.

A config is a two-level mapping of sections to settings.  Anything not given by
the user falls back to :data:`DEFAULTS`.
"""

from __future__ import annotations

import copy
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping, Union

ConfigSource = Union[str, os.PathLike, Mapping[str, Any], None]

DEFAULTS: dict[str, dict[str, Any]] = {
    "queue": {"max_size": 10_000},
    "retry": {"max_attempts": 5, "base_delay": 1.0, "factor": 2.0, "max_delay": 300.0},
    "worker": {"name": "worker", "batch_size": 10, "poll_interval": "5s"},
    "storage": {"path": None},
}


class ConfigError(Exception):
    """The configuration file is missing, malformed or has unknown sections."""


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix == ".toml":
            return tomllib.loads(text)
        if path.suffix == ".json":
            return json.loads(text)
    except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    raise ConfigError(f"unsupported config format: {path.suffix or path.name}")


def load_config(source: ConfigSource = None) -> dict[str, Any]:
    """Load configuration from a file path, a mapping, or nothing (all defaults)."""
    if source is None:
        overrides: Mapping[str, Any] = {}
    elif isinstance(source, Mapping):
        overrides = source
    else:
        overrides = _read_file(Path(source))
    unknown = sorted(set(overrides) - set(DEFAULTS))
    if unknown:
        raise ConfigError(f"unknown config section(s): {', '.join(unknown)}")
    for name, section in overrides.items():
        if not isinstance(section, Mapping):
            raise ConfigError(f"section [{name}] must be a table")
    return merge(DEFAULTS, overrides)
