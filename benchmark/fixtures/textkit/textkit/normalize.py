"""Normalisation helpers used by the other textkit modules."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")

_QUOTES = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "„": '"',
    }
)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and trim the ends."""
    return _WHITESPACE.sub(" ", text).strip()


def normalize_newlines(text: str) -> str:
    """Convert Windows and old Mac line endings to ``\\n``."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_quotes(text: str) -> str:
    """Replace typographic quotes with their ASCII equivalents."""
    return text.translate(_QUOTES)


def strip_accents(text: str) -> str:
    """Remove combining accents: ``"Crème brûlée"`` -> ``"Creme brulee"``."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))
