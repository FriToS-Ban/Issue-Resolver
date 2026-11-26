from textkit import top_words, unique_words, word_count, word_frequencies, word_tokens
from textkit.cli import main
from textkit.tokenize import ngrams


def test_top_words_merges_punctuated_variants():
    assert top_words("Hello, world! Hello again. hello?", n=3) == [("hello", 3), ("again", 1), ("world", 1)]


def test_unique_words_ignores_surrounding_punctuation():
    text = 'He said: "stop," then (finally) stopped; stop!'
    assert unique_words(text) == ["finally", "he", "said", "stop", "stopped", "then"]


def test_frequencies_with_quotes_and_brackets():
    freq = word_frequencies("“Quoted” quoted 'quoted' [quoted] {quoted}")
    assert freq == {"quoted": 5}


def test_apostrophes_are_kept_inside_words():
    assert top_words("Don't stop. don't, DON'T!", n=1) == [("don't", 3)]


def test_word_count_unchanged():
    assert word_count("Hello, world! Hello again.") == 4


def test_word_tokens_drop_punctuation():
    assert word_tokens("Well, (maybe) not: yes; no.") == ["well", "maybe", "not", "yes", "no"]


def test_ngrams_have_no_punctuation():
    assert ngrams(word_tokens("red, green, blue"), 2) == [("red", "green"), ("green", "blue")]


def test_stats_command(tmp_path, capsys):
    path = tmp_path / "feedback.txt"
    path.write_text("Great app, great support. Great, great!", encoding="utf-8")
    assert main(["stats", str(path), "--top", "1"]) == 0
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.split() == ["4", "great"]
