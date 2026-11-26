"""Time sources.

Everything in taskq that needs "now" takes a zero-argument callable so tests and
simulations can substitute a :class:`FixedClock`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def utcnow() -> datetime:
    """The current time as an aware UTC datetime."""
    return datetime.now(timezone.utc)


class FixedClock:
    """A manually advanced clock."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def __call__(self) -> datetime:
        return self._now

    def advance(self, **delta: float) -> datetime:
        """Move the clock forward, e.g. ``clock.advance(minutes=5)``."""
        self._now += timedelta(**delta)
        return self._now

    def set(self, moment: datetime) -> None:
        self._now = moment
