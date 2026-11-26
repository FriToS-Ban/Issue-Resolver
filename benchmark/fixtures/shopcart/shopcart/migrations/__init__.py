"""Schema migrations for the SQL storage backend.

Migrations are plain modules named ``NNNN_description.py`` exposing ``upgrade``
and ``downgrade`` functions that transform a schema dictionary.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def discover() -> list[ModuleType]:
    """Import all migration modules in order."""
    names = sorted(
        info.name for info in pkgutil.iter_modules(__path__) if info.name[:4].isdigit()
    )
    return [importlib.import_module(f"{__name__}.{name}") for name in names]


def apply_all(schema: dict | None = None) -> dict:
    result: dict = dict(schema or {})
    for module in discover():
        result = module.upgrade(result)
    return result
