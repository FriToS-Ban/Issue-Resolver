"""Shortening text for previews, table cells and notifications."""

from __future__ import annotations


def truncate(text: str, limit: int, ellipsis: str = "...", *, word_boundary: bool = False) -> str:
    """Shorten ``text`` to ``limit`` characters, marking the cut with ``ellipsis``.

    With ``word_boundary=True`` the cut is moved back to the previous space so
    that words are not split in half.
    """
    if limit < 0:
        raise ValueError("limit must not be negative")
    if len(text) <= limit:
        return text
    if limit <= len(ellipsis):
        return ellipsis[:limit]
    cut = limit - len(ellipsis)
    head = text[:cut]
    if word_boundary:
        space = head.rfind(" ")
        if space > 0:
            head = head[:space]
    return head.rstrip() + ellipsis


def truncate_words(text: str, max_words: int, ellipsis: str = "...") -> str:
    """Keep the first ``max_words`` words of ``text``."""
    if max_words < 0:
        raise ValueError("max_words must not be negative")
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + ellipsis


def truncate_middle(text: str, limit: int, ellipsis: str = "...") -> str:
    """Shorten ``text`` by cutting out its middle, e.g. for long file paths."""
    if len(text) <= limit:
        return text
    if limit <= len(ellipsis):
        return ellipsis[:limit]
    keep = limit - len(ellipsis)
    front = (keep + 1) // 2
    back = keep - front
    return text[:front] + ellipsis + (text[-back:] if back else "")
