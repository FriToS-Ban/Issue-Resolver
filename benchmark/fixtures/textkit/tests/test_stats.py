import pytest

from textkit.stats import (
    DEFAULT_STOPWORDS,
    average_word_length,
    reading_time_minutes,
    sentence_count,
    top_words,
    unique_words,
    word_count,
    word_frequencies,
)


def test_word_count():
    assert word_count("Hello there, general Kenobi.") == 4
    assert word_count("") == 0


def test_sentence_count():
    assert sentence_count("One. Two! Three?") == 3


def test_top_words_orders_by_count_then_alphabetically():
    text = "the cat and the hat and the bat"
    assert top_words(text, 3) == [("the", 3), ("and", 2), ("bat", 1)]


def test_top_words_with_stopwords():
    text = "the cat and the hat"
    assert top_words(text, 2, stopwords=DEFAULT_STOPWORDS) == [("cat", 1), ("hat", 1)]


def test_word_frequencies():
    freq = word_frequencies("a b a c a")
    assert freq["a"] == 3
    assert freq["b"] == 1


def test_unique_words_sorted():
    assert unique_words("pear apple pear fig") == ["apple", "fig", "pear"]


def test_average_word_length():
    assert average_word_length("aa bbbb") == pytest.approx(3.0)
    assert average_word_length("") == 0.0


def test_reading_time():
    assert reading_time_minutes("word " * 450) == 3
    assert reading_time_minutes("") == 0
