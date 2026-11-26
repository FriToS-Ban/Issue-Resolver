"""Word and sentence tokenisation."""

from __future__ import annotations

import re

from textkit.normalize import normalize_quotes, normalize_whitespace

_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def word_tokens(text: str, *, lowercase: bool = True) -> list[str]:
    """Split ``text`` into words.

    >>> word_tokens("Don't panic!")
    ["don't", 'panic']
    """
    tokens = _WORD_RE.findall(normalize_quotes(text))
    if lowercase:
        tokens = [tok.lower() for tok in tokens]
    return tokens


def word_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` character offsets of every word in ``text``."""
    return [m.span() for m in _WORD_RE.finditer(normalize_quotes(text))]


def sentence_split(text: str) -> list[str]:
    """Split ``text`` into sentences on ``.``, ``!`` and ``?``."""
    text = normalize_whitespace(text)
    if not text:
        return []
    return [part for part in _SENTENCE_BREAK.split(text) if part]


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Consecutive ``n``-token windows over ``tokens``."""
    if n < 1:
        raise ValueError("n must be at least 1")
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
