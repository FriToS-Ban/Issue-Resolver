import pytest

from textkit.wrapping import fill, indent, wrap


def test_wrap_greedy():
    assert wrap("the quick brown fox jumps over the lazy dog", 10) == [
        "the quick",
        "brown fox",
        "jumps over",
        "the lazy",
        "dog",
    ]


def test_wrap_breaks_long_words():
    assert wrap("abcdefghij", 4) == ["abcd", "efgh", "ij"]


def test_wrap_keeps_long_words_when_asked():
    assert wrap("tiny enormousword", 5, break_long_words=False) == ["tiny", "enormousword"]


def test_wrap_empty():
    assert wrap("   ", 10) == []


def test_wrap_rejects_zero_width():
    with pytest.raises(ValueError):
        wrap("x", 0)


def test_fill():
    assert fill("a b c d", 3) == "a b\nc d"


def test_indent():
    assert indent("one\ntwo\n\nthree", "  ", first_line="- ") == "- one\n  two\n\n  three"
