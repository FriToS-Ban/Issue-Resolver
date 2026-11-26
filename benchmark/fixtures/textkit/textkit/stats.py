"""Simple text statistics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from textkit.tokenize import sentence_split, word_tokens

DEFAULT_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i in is it its of on or that the "
    "this to was were will with you".split()
)


def word_count(text: str) -> int:
    return len(word_tokens(text))


def sentence_count(text: str) -> int:
    return len(sentence_split(text))


def word_frequencies(text: str, *, stopwords: Iterable[str] | None = None) -> Counter[str]:
    """Count how often each word occurs, ignoring ``stopwords``."""
    skip = {w.lower() for w in stopwords} if stopwords else set()
    return Counter(tok for tok in word_tokens(text) if tok not in skip)


def top_words(text: str, n: int = 10, *, stopwords: Iterable[str] | None = None) -> list[tuple[str, int]]:
    """The ``n`` most frequent words, ties broken alphabetically."""
    counts = word_frequencies(text, stopwords=stopwords)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def unique_words(text: str) -> list[str]:
    """Distinct words in ``text``, sorted."""
    return sorted(set(word_tokens(text)))


def average_word_length(text: str) -> float:
    tokens = word_tokens(text)
    if not tokens:
        return 0.0
    return sum(len(tok) for tok in tokens) / len(tokens)


def reading_time_minutes(text: str, words_per_minute: int = 200) -> int:
    """Estimated reading time, rounded up to whole minutes."""
    if words_per_minute < 1:
        raise ValueError("words_per_minute must be positive")
    words = word_count(text)
    return math.ceil(words / words_per_minute) if words else 0
