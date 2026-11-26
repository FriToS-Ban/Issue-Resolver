"""URL slugs."""

from __future__ import annotations

import re

from textkit.normalize import strip_accents

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(text: str, separator: str = "-", max_length: int | None = None) -> str:
    """Turn arbitrary text into a lowercase, URL-friendly slug.

    >>> slugify("Crème Brûlée Recipe")
    'creme-brulee-recipe'
    """
    if not separator:
        raise ValueError("separator must not be empty")
    value = strip_accents(text).lower()
    value = _NON_ALNUM.sub(separator, value)
    value = value.strip(separator)
    if max_length is not None and len(value) > max_length:
        value = value[:max_length].rstrip(separator)
    return value


def unique_slug(text: str, taken: set[str], separator: str = "-") -> str:
    """Like :func:`slugify` but appends ``-2``, ``-3``... until the slug is unused."""
    base = slugify(text, separator) or "item"
    candidate = base
    counter = 2
    while candidate in taken:
        candidate = f"{base}{separator}{counter}"
        counter += 1
    return candidate
