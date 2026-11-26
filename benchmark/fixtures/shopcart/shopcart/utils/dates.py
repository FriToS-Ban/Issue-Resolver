"""Date helpers shared by coupons, reports and the storage layer."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Iterator, Union

DateLike = Union[date, datetime, str]

_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y")


def parse_date(value: DateLike) -> date:
    """Parse ``value`` into a :class:`date`.

    Accepts ``date`` and ``datetime`` objects as well as strings in ISO
    (``2024-03-01``), slashed (``2024/03/01``) or dotted European
    (``01.03.2024``) form.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in _FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {value!r}")


def end_of_day(day: date) -> datetime:
    """The last representable moment of ``day`` as a naive datetime."""
    return datetime.combine(day, time.max)


def is_expired(expires_on: DateLike, now: datetime | date | None = None) -> bool:
    """Return True if ``expires_on`` lies in the past.

    The expiry date itself is inclusive: something that expires on 1 March is
    usable until the end of that day.
    """
    deadline = end_of_day(parse_date(expires_on))
    if now is None:
        now = datetime.now()
    elif not isinstance(now, datetime):
        now = datetime.combine(now, time.min)
    remaining = deadline - now
    return remaining.total_seconds() < 0


def days_until(target: DateLike, today: date | None = None) -> int:
    """Whole days from ``today`` until ``target`` (negative when in the past)."""
    current = today if today is not None else date.today()
    return (parse_date(target) - current).days


def date_range(start: DateLike, end: DateLike) -> Iterator[date]:
    """Yield every date from ``start`` to ``end`` inclusive."""
    first, last = parse_date(start), parse_date(end)
    if last < first:
        raise ValueError("end date is before start date")
    for offset in range((last - first).days + 1):
        yield first + timedelta(days=offset)
