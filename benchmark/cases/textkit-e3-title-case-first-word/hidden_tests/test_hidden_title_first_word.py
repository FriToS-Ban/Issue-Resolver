import pytest

from textkit import title_case
from textkit.cli import main


@pytest.mark.parametrize(
    "text, expected",
    [
        ("the lord of the rings", "The Lord of the Rings"),
        ("a tale of two cities", "A Tale of Two Cities"),
        ("of mice and men", "Of Mice and Men"),
        ("an introduction to NASA", "An Introduction to NASA"),
        ("war and peace", "War and Peace"),
        ("what are you looking at", "What Are You Looking At"),
    ],
)
def test_title_case(text, expected):
    assert title_case(text) == expected


@pytest.mark.parametrize("word", ["the", "a", "of"])
def test_single_small_word(word):
    assert title_case(word) == word.capitalize()


def test_two_small_words():
    assert title_case("to the") == "To The"


def test_cli_title_case(capsys):
    assert main(["case", "title", "the importance of being earnest"]) == 0
    assert capsys.readouterr().out.strip() == "The Importance of Being Earnest"
