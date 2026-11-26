"""Tiny ``{name}``-style string templates.

Placeholders are written ``{name}``, or ``{name.attr}`` to reach into nested
mappings and objects.  Literal braces are written doubled: ``{{`` and ``}}``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from textkit.errors import TemplateError

_TOKEN_RE = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\}")


def _lookup(context: Mapping[str, Any], dotted: str) -> Any:
    value: Any = context
    for part in dotted.split("."):
        if isinstance(value, Mapping):
            value = value[part]
        else:
            try:
                value = getattr(value, part)
            except AttributeError:
                raise KeyError(dotted) from None
    return value


def render(template: str, context: Mapping[str, Any] | None = None, **values: Any) -> str:
    """Substitute placeholders in ``template`` with values from ``context``/``values``.

    >>> render("Hello {name}!", name="Ada")
    'Hello Ada!'
    """
    data: dict[str, Any] = dict(context or {})
    data.update(values)

    def substitute(match: re.Match[str]) -> str:
        token = match.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        key = match.group(1)
        try:
            return str(_lookup(data, key))
        except KeyError:
            raise TemplateError(f"no value for placeholder {{{key}}}", key=key) from None

    return _TOKEN_RE.sub(substitute, template)


def placeholders(template: str) -> list[str]:
    """Placeholder names used in ``template``, in order of first appearance."""
    seen: list[str] = []
    for match in _TOKEN_RE.finditer(template):
        key = match.group(1)
        if key and key not in seen:
            seen.append(key)
    return seen


def missing_keys(template: str, context: Mapping[str, Any]) -> list[str]:
    """Placeholders in ``template`` that ``context`` cannot fill."""
    missing = []
    for key in placeholders(template):
        try:
            _lookup(context, key)
        except KeyError:
            missing.append(key)
    return missing
