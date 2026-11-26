import pytest

from textkit.csvio import format_csv_line, parse_csv_line, read_rows
from textkit.errors import CSVError


def test_plain_fields():
    assert parse_csv_line("a,b,c") == ["a", "b", "c"]


def test_empty_fields():
    assert parse_csv_line("a,,c,") == ["a", "", "c", ""]


def test_simple_quoted_field():
    assert parse_csv_line('1,"Widget",9.99') == ["1", "Widget", "9.99"]


def test_trailing_newline_ignored():
    assert parse_csv_line("x,y\r\n") == ["x", "y"]


def test_custom_delimiter():
    assert parse_csv_line("a;b;c", delimiter=";") == ["a", "b", "c"]


def test_format_quotes_when_needed():
    assert format_csv_line(["plain", "has,comma", 'say "hi"', None, 3]) == 'plain,"has,comma","say ""hi""",,3'


def test_format_quotes_padded_values():
    assert format_csv_line([" padded"]) == '" padded"'


def test_read_rows():
    lines = ["id,name,price", "1,Widget,9.99", "", "2,Gadget,19.50"]
    assert read_rows(lines) == [
        {"id": "1", "name": "Widget", "price": "9.99"},
        {"id": "2", "name": "Gadget", "price": "19.50"},
    ]


def test_read_rows_field_count_mismatch():
    with pytest.raises(CSVError, match="line 2"):
        read_rows(["a,b", "1,2,3"])
