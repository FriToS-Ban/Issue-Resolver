import pytest

from textkit.cli import main
from textkit.truncation import truncate

TEXT = "The quick brown fox jumps over the lazy dog"


@pytest.mark.parametrize("limit", [4, 5, 10, 20, 30, 42])
def test_result_never_exceeds_limit(limit):
    out = truncate(TEXT, limit)
    assert len(out) == limit
    assert out.endswith("...")


def test_expected_text():
    assert truncate(TEXT, 20) == "The quick brown f..."


def test_single_character_ellipsis():
    out = truncate(TEXT, 12, ellipsis="…")
    assert out == "The quick b…"
    assert len(out) == 12


def test_word_boundary_respects_limit():
    out = truncate(TEXT, 20, word_boundary=True)
    assert out == "The quick brown..."
    assert len(out) <= 20


def test_untouched_when_it_fits():
    assert truncate(TEXT, len(TEXT)) == TEXT


def test_cli_truncate(capsys):
    assert main(["truncate", TEXT, "--limit", "12"]) == 0
    assert len(capsys.readouterr().out.rstrip("\n")) == 12
