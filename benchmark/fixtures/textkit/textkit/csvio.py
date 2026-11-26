"""Reading and writing individual CSV lines.

This is not a replacement for the :mod:`csv` module; it exists for places where
records arrive one line at a time (log shippers, sockets, chat bots).
"""

from __future__ import annotations

from typing import Iterable

from textkit.errors import CSVError


def parse_csv_line(line: str, delimiter: str = ",", quotechar: str = '"') -> list[str]:
    """Split one CSV record into fields.

    Fields may be wrapped in ``quotechar``; a quoted field can contain the
    delimiter, and a doubled quote character inside it stands for one literal
    quote.
    """
    line = line.rstrip("\r\n")
    fields: list[str] = []
    buf: list[str] = []
    in_quotes = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quotes:
            if ch == quotechar:
                if i + 1 < len(line) and line[i + 1] == quotechar:
                    buf.append(quotechar)
                    i += 1
                else:
                    in_quotes = False
            else:
                buf.append(ch)
        elif ch == quotechar:
            in_quotes = True
        elif ch == delimiter:
            fields.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    if in_quotes:
        raise CSVError(f"unterminated quoted field in {line!r}")
    fields.append("".join(buf))
    return fields


def format_csv_line(fields: Iterable[object], delimiter: str = ",", quotechar: str = '"') -> str:
    """Join ``fields`` into one CSV record, quoting where necessary."""
    out = []
    for field in fields:
        text = "" if field is None else str(field)
        if delimiter in text or quotechar in text or "\n" in text or text != text.strip():
            text = quotechar + text.replace(quotechar, quotechar * 2) + quotechar
        out.append(text)
    return delimiter.join(out)


def read_rows(lines: Iterable[str], delimiter: str = ",") -> list[dict[str, str]]:
    """Parse a header line followed by records into a list of dicts."""
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = parse_csv_line(line, delimiter)
        if header is None:
            header = [name.strip() for name in fields]
            continue
        if len(fields) != len(header):
            raise CSVError(
                f"expected {len(header)} fields, got {len(fields)}", line_number=number
            )
        rows.append(dict(zip(header, fields)))
    return rows
