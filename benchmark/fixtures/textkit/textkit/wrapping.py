"""Greedy line wrapping."""

from __future__ import annotations

from textkit.normalize import normalize_whitespace


def wrap(text: str, width: int = 70, *, break_long_words: bool = True) -> list[str]:
    """Wrap ``text`` into lines of at most ``width`` characters.

    Whitespace is normalised first, so paragraphs should be wrapped one at a time.
    """
    if width < 1:
        raise ValueError("width must be at least 1")
    text = normalize_whitespace(text)
    if not text:
        return []
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        while break_long_words and len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:width])
            word = word[width:]
        if not word:
            continue
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def fill(text: str, width: int = 70) -> str:
    """Wrap ``text`` and join the lines with newlines."""
    return "\n".join(wrap(text, width))


def indent(text: str, prefix: str, *, first_line: str | None = None) -> str:
    """Prefix every line of ``text``; ``first_line`` overrides the first prefix."""
    out = []
    for i, line in enumerate(text.splitlines()):
        lead = first_line if (i == 0 and first_line is not None) else prefix
        out.append(f"{lead}{line}" if line else line)
    return "\n".join(out)
