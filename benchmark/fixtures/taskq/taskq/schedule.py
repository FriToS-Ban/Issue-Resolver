"""Interval parsing and recurring schedules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Union

IntervalLike = Union[str, int, float, timedelta]

_UNIT_SECONDS: dict[str, int] = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
}

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_PART = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z]+)")


def parse_interval(spec: IntervalLike) -> timedelta:
    """Parse ``"90s"``, ``"5m"``, ``"2h"``, ``"1d"``, ``"1h30m"`` or a number of seconds."""
    if isinstance(spec, timedelta):
        result = spec
    elif isinstance(spec, bool):
        raise TypeError("interval cannot be a boolean")
    elif isinstance(spec, (int, float)):
        result = timedelta(seconds=spec)
    elif isinstance(spec, str):
        result = _parse_text(spec)
    else:
        raise TypeError(f"cannot parse interval from {type(spec).__name__}")
    if result <= timedelta(0):
        raise ValueError(f"interval must be positive: {spec!r}")
    return result


def _parse_text(spec: str) -> timedelta:
    text = spec.strip().lower()
    if not text:
        raise ValueError("empty interval")
    if _NUMBER.fullmatch(text):
        return timedelta(seconds=float(text))
    total = 0.0
    pos = 0
    for match in _PART.finditer(text):
        if text[pos : match.start()].strip():
            raise ValueError(f"invalid interval: {spec!r}")
        number, unit = match.groups()
        if unit not in _UNIT_SECONDS:
            raise ValueError(f"unknown unit {unit!r} in interval {spec!r}")
        total += float(number) * _UNIT_SECONDS[unit]
        pos = match.end()
    if pos == 0 or text[pos:].strip():
        raise ValueError(f"invalid interval: {spec!r}")
    return timedelta(seconds=total)


def format_interval(delta: timedelta) -> str:
    """Inverse of :func:`parse_interval` for whole seconds: ``5400s`` -> ``"1h30m"``."""
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        raise ValueError("interval must be positive")
    parts = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        count, seconds = divmod(seconds, size)
        if count:
            parts.append(f"{count}{unit}")
    return "".join(parts)


def next_run(last_run: datetime | None, interval: IntervalLike, now: datetime) -> datetime:
    """When a job on ``interval`` should next run.

    Missed runs are not replayed: if several intervals have passed since
    ``last_run`` the next run is the first slot after ``now``.
    """
    step = parse_interval(interval)
    if last_run is None:
        return now
    candidate = last_run + step
    if candidate > now:
        return candidate
    missed = (now - last_run) // step
    return last_run + step * (missed + 1)


@dataclass
class Schedule:
    """A named recurring job."""

    name: str
    interval: IntervalLike
    last_run: datetime | None = None

    def due_at(self, now: datetime) -> datetime:
        return next_run(self.last_run, self.interval, now)

    def is_due(self, now: datetime) -> bool:
        return self.last_run is None or self.last_run + parse_interval(self.interval) <= now

    def mark_run(self, at: datetime) -> None:
        self.last_run = at
