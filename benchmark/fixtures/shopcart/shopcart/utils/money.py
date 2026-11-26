"""Money helpers.

All internal arithmetic is done in integer cents.  Dollar amounts only appear at
the edges of the system: catalogue prices, receipts and reports.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Union

Number = Union[int, float, str, Decimal]

_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_CENT = Decimal("0.01")


def to_decimal(value: Number) -> Decimal:
    """Convert ``value`` to a :class:`Decimal` without picking up binary float noise."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("booleans are not amounts")
    if isinstance(value, float):
        return Decimal(repr(value))
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"not a valid amount: {value!r}") from exc


def round_half_up(value: Decimal) -> int:
    """Round ``value`` to the nearest integer, rounding halves away from zero."""
    return int(value.quantize(_ONE, rounding=ROUND_HALF_UP))


def to_cents(amount: Number) -> int:
    """Convert a dollar amount to integer cents."""
    return round_half_up(to_decimal(amount) * _HUNDRED)


def from_cents(cents: int) -> Decimal:
    """Convert integer cents back to a two-place dollar :class:`Decimal`."""
    return (Decimal(cents) / _HUNDRED).quantize(_CENT)


def format_money(amount: Number, symbol: str = "$") -> str:
    """Format a dollar amount for display, e.g. ``1234.5`` -> ``"$1,234.50"``."""
    cents = to_cents(amount)
    sign = "-" if cents < 0 else ""
    whole, frac = divmod(abs(cents), 100)
    return f"{sign}{symbol}{whole:,}.{frac:02d}"


def format_cents(cents: int, symbol: str = "$") -> str:
    """Format an integer number of cents for display."""
    return format_money(from_cents(cents), symbol)


def split_evenly(cents: int, parts: int) -> list[int]:
    """Split ``cents`` into ``parts`` integer shares that add up exactly.

    Any remainder is handed out one cent at a time to the first shares.
    """
    if parts < 1:
        raise ValueError("parts must be at least 1")
    share, remainder = divmod(cents, parts)
    return [share + (1 if i < remainder else 0) for i in range(parts)]
