"""Identifier and headline case conversions."""

from __future__ import annotations

import re
from typing import Iterable

_WORD_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of", "on", "or", "the", "to", "via", "vs"}
)


def split_words(text: str) -> list[str]:
    """Split identifiers like ``"parseHTTPResponse_v2"`` into ``["parse", "HTTP", "Response", "v", "2"]``."""
    return _WORD_PARTS.findall(text)


def snake_case(text: str) -> str:
    return "_".join(word.lower() for word in split_words(text))


def kebab_case(text: str) -> str:
    return "-".join(word.lower() for word in split_words(text))


def pascal_case(text: str) -> str:
    return "".join(word[:1].upper() + word[1:].lower() for word in split_words(text))


def camel_case(text: str) -> str:
    pascal = pascal_case(text)
    return pascal[:1].lower() + pascal[1:]


def _capitalize(word: str) -> str:
    if len(word) > 1 and word.isupper():
        return word
    return word[:1].upper() + word[1:].lower()


def title_case(text: str, small_words: Iterable[str] = SMALL_WORDS) -> str:
    """Headline-style capitalisation.

    Short function words (articles, conjunctions, short prepositions) stay
    lowercase; acronyms such as ``NASA`` are kept as they are.
    """
    small = {w.lower() for w in small_words}
    words = text.split()
    last = len(words) - 1
    out = []
    for i, word in enumerate(words):
        lower = word.lower()
        if 0 < i < last and lower in small:
            out.append(lower)
        else:
            out.append(_capitalize(word))
    return " ".join(out)
