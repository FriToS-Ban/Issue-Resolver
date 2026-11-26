import pytest

from textkit.tokenize import ngrams, sentence_split, word_spans, word_tokens


def test_word_tokens_lowercases():
    assert word_tokens("The Cat sat") == ["the", "cat", "sat"]


def test_word_tokens_keep_case():
    assert word_tokens("The Cat", lowercase=False) == ["The", "Cat"]


def test_word_tokens_apostrophes():
    assert word_tokens("Don't panic!") == ["don't", "panic"]


def test_word_tokens_curly_apostrophe():
    assert word_tokens("It’s fine.") == ["it's", "fine"]


def test_word_spans():
    assert word_spans("hi there") == [(0, 2), (3, 8)]


def test_sentence_split():
    text = "It was late. Nobody answered!   Was anyone home? Yes."
    assert sentence_split(text) == ["It was late.", "Nobody answered!", "Was anyone home?", "Yes."]


def test_sentence_split_empty():
    assert sentence_split("   ") == []


def test_ngrams():
    assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    with pytest.raises(ValueError):
        ngrams(["a"], 0)
