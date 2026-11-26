import pytest

from textkit.slug import slugify, unique_slug


@pytest.mark.parametrize(
    "text, slug",
    [
        ("Hello World", "hello-world"),
        ("Crème Brûlée Recipe", "creme-brulee-recipe"),
        ("Rock & Roll Hall of Fame", "rock-roll-hall-of-fame"),
        ("version 2.0 release notes", "version-2-0-release-notes"),
    ],
)
def test_slugify(text, slug):
    assert slugify(text) == slug


def test_custom_separator():
    assert slugify("Hello big World", separator="_") == "hello_big_world"


def test_max_length_does_not_end_with_separator():
    assert slugify("The quick brown fox", max_length=10) == "the-quick"


def test_empty_separator_rejected():
    with pytest.raises(ValueError):
        slugify("x", separator="")


def test_unique_slug():
    taken = {"report", "report-2"}
    assert unique_slug("Report", taken) == "report-3"
    assert unique_slug("Summary", taken) == "summary"
