import pytest

from textkit import CSVError, format_csv_line, parse_csv_line, read_rows


def test_delimiter_inside_quotes():
    assert parse_csv_line('1,"Smith, John",Acme') == ["1", "Smith, John", "Acme"]


def test_doubled_quotes_are_escapes():
    assert parse_csv_line('"Prefers ""Jack""",x') == ['Prefers "Jack"', "x"]


def test_empty_quoted_field():
    assert parse_csv_line('a,"",c') == ["a", "", "c"]


def test_quoted_field_with_only_a_quote():
    assert parse_csv_line('""""') == ['"']


def test_custom_delimiter_inside_quotes():
    assert parse_csv_line('a;"b;c";d', delimiter=";") == ["a", "b;c", "d"]


def test_unterminated_quote_is_an_error():
    with pytest.raises(CSVError):
        parse_csv_line('1,"Smith, John,Acme')


@pytest.mark.parametrize(
    "fields",
    [
        ["plain", "has,comma", 'say "hi"', "", " padded "],
        ['"', ",", '""', "a,b,c"],
    ],
)
def test_round_trip_with_format(fields):
    assert parse_csv_line(format_csv_line(fields)) == fields


def test_read_rows_from_issue():
    lines = [
        "id,name,company,notes\n",
        '1,"Smith, John",Acme,"Prefers ""Jack"""\n',
        "2,Jane Doe,Initech,\n",
    ]
    assert read_rows(lines) == [
        {"id": "1", "name": "Smith, John", "company": "Acme", "notes": 'Prefers "Jack"'},
        {"id": "2", "name": "Jane Doe", "company": "Initech", "notes": ""},
    ]


def test_plain_lines_still_work():
    assert parse_csv_line("a,b,,d") == ["a", "b", "", "d"]
