"""The ``textkit`` command line tool."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from textkit.case import camel_case, kebab_case, pascal_case, snake_case, title_case
from textkit.errors import TextkitError
from textkit.slug import slugify
from textkit.stats import reading_time_minutes, top_words, word_count
from textkit.template import render
from textkit.truncation import truncate
from textkit.wrapping import fill

_CASES = {
    "snake": snake_case,
    "kebab": kebab_case,
    "camel": camel_case,
    "pascal": pascal_case,
    "title": title_case,
}


def _read_input(value: str | None) -> str:
    if value is None or value == "-":
        return sys.stdin.read()
    return value


def _parse_assignments(pairs: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise TextkitError(f"expected key=value, got {pair!r}")
        values[key] = value
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="textkit", description="Small text utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("slug", help="make a URL slug")
    p.add_argument("text")
    p.add_argument("--separator", default="-")

    p = sub.add_parser("wrap", help="wrap text to a width")
    p.add_argument("text", nargs="?")
    p.add_argument("--width", type=int, default=70)

    p = sub.add_parser("truncate", help="shorten text")
    p.add_argument("text")
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--words", action="store_true", help="cut at a word boundary")

    p = sub.add_parser("case", help="convert identifier or headline case")
    p.add_argument("style", choices=sorted(_CASES))
    p.add_argument("text")

    p = sub.add_parser("stats", help="word statistics for a file")
    p.add_argument("path", nargs="?")
    p.add_argument("--top", type=int, default=5)

    p = sub.add_parser("render", help="fill in a {placeholder} template")
    p.add_argument("template")
    p.add_argument("values", nargs="*", metavar="key=value")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "slug":
            print(slugify(args.text, args.separator))
        elif args.command == "wrap":
            print(fill(_read_input(args.text), args.width))
        elif args.command == "truncate":
            print(truncate(args.text, args.limit, word_boundary=args.words))
        elif args.command == "case":
            print(_CASES[args.style](args.text))
        elif args.command == "stats":
            if args.path in (None, "-"):
                text = sys.stdin.read()
            else:
                with open(args.path, encoding="utf-8") as fh:
                    text = fh.read()
            print(f"words: {word_count(text)}")
            print(f"reading time: {reading_time_minutes(text)} min")
            for word, count in top_words(text, args.top):
                print(f"{count:6d}  {word}")
        elif args.command == "render":
            print(render(args.template, _parse_assignments(args.values)))
    except TextkitError as exc:
        print(f"textkit: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
