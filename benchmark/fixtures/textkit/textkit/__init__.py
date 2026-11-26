"""textkit: small, dependency-free text utilities."""

from textkit.case import camel_case, kebab_case, pascal_case, snake_case, title_case
from textkit.csvio import format_csv_line, parse_csv_line, read_rows
from textkit.errors import CSVError, TemplateError, TextkitError
from textkit.slug import slugify
from textkit.stats import top_words, unique_words, word_count, word_frequencies
from textkit.template import placeholders, render
from textkit.tokenize import sentence_split, word_tokens
from textkit.truncation import truncate, truncate_middle, truncate_words
from textkit.wrapping import fill, wrap

__all__ = [
    "CSVError",
    "TemplateError",
    "TextkitError",
    "camel_case",
    "fill",
    "format_csv_line",
    "kebab_case",
    "parse_csv_line",
    "pascal_case",
    "placeholders",
    "read_rows",
    "render",
    "sentence_split",
    "slugify",
    "snake_case",
    "title_case",
    "top_words",
    "truncate",
    "truncate_middle",
    "truncate_words",
    "unique_words",
    "word_count",
    "word_frequencies",
    "word_tokens",
    "wrap",
]

__version__ = "0.9.1"
