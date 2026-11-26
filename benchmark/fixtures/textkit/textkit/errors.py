"""Exception types raised by textkit."""

from __future__ import annotations


class TextkitError(Exception):
    """Base class for all errors raised by textkit."""


class CSVError(TextkitError, ValueError):
    """A CSV line or file could not be parsed."""

    def __init__(self, message: str, *, line_number: int | None = None) -> None:
        if line_number is not None:
            message = f"line {line_number}: {message}"
        super().__init__(message)
        self.line_number = line_number


class TemplateError(TextkitError):
    """A template could not be rendered."""

    def __init__(self, message: str, *, key: str | None = None) -> None:
        super().__init__(message)
        self.key = key
