import pytest

from textkit.truncation import truncate, truncate_middle, truncate_words


def test_short_text_is_unchanged():
    assert truncate("short", 10) == "short"


def test_text_exactly_at_limit_is_unchanged():
    assert truncate("exactly10!", 10) == "exactly10!"


def test_tiny_limit_returns_part_of_ellipsis():
    assert truncate("something long", 2) == ".."
    assert truncate("something long", 0) == ""


def test_negative_limit_rejected():
    with pytest.raises(ValueError):
        truncate("x", -1)


def test_truncate_words():
    assert truncate_words("one two three four", 2) == "one two..."
    assert truncate_words("one two", 5) == "one two"


def test_truncate_middle():
    assert truncate_middle("/very/long/path/to/file.txt", 15) == "/very/...le.txt"
    assert len(truncate_middle("abcdefghijklmnop", 9)) == 9
