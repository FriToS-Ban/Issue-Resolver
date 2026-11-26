import pytest

from textkit import slugify
from textkit.slug import unique_slug


@pytest.mark.parametrize(
    "text, slug",
    [
        ("  Hello World!  ", "hello-world"),
        ("(Draft) Release notes", "draft-release-notes"),
        ("--Already--Slugged--", "already-slugged"),
        ("...and then?", "and-then"),
        ("Hello World", "hello-world"),
    ],
)
def test_no_leading_or_trailing_separator(text, slug):
    assert slugify(text) == slug


def test_only_punctuation_gives_empty_slug():
    assert slugify("!!! ???") == ""


def test_custom_separator_is_stripped_too():
    assert slugify("  Hi there  ", separator="_") == "hi_there"


def test_max_length_with_leading_punctuation():
    assert slugify("¡Hola, señor!", max_length=4) == "hola"


def test_unique_slug_with_messy_input():
    assert unique_slug("  Report!  ", {"report"}) == "report-2"
