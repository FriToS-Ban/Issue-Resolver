from datetime import datetime, timedelta

import pytest

from taskq.schedule import Schedule, format_interval, next_run, parse_interval


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("5m", timedelta(minutes=5)),
        ("2h", timedelta(hours=2)),
        ("1d", timedelta(days=1)),
        ("1h30m", timedelta(hours=1, minutes=30)),
        ("1h 15min", timedelta(hours=1, minutes=15)),
        ("45 seconds", timedelta(seconds=45)),
        ("120", timedelta(seconds=120)),
        (30, timedelta(seconds=30)),
        (timedelta(minutes=1), timedelta(minutes=1)),
    ],
)
def test_parse_interval(spec, expected):
    assert parse_interval(spec) == expected


@pytest.mark.parametrize("spec", ["", "soon", "5 parsecs", "m5", "0m", "-5m"])
def test_parse_interval_rejects(spec):
    with pytest.raises(ValueError):
        parse_interval(spec)


def test_format_interval():
    assert format_interval(timedelta(hours=1, minutes=30)) == "1h30m"
    assert format_interval(timedelta(days=2, seconds=5)) == "2d5s"


def test_next_run_first_time_is_now():
    now = datetime(2024, 1, 1, 12, 0)
    assert next_run(None, "1h", now) == now


def test_next_run_after_last():
    last = datetime(2024, 1, 1, 12, 0)
    assert next_run(last, "1h", datetime(2024, 1, 1, 12, 10)) == datetime(2024, 1, 1, 13, 0)


def test_missed_runs_are_skipped():
    last = datetime(2024, 1, 1, 12, 0)
    assert next_run(last, "1h", datetime(2024, 1, 1, 15, 30)) == datetime(2024, 1, 1, 16, 0)


def test_schedule_due():
    job = Schedule("report", "15m")
    start = datetime(2024, 1, 1, 9, 0)
    assert job.is_due(start)
    job.mark_run(start)
    assert not job.is_due(start + timedelta(minutes=10))
    assert job.is_due(start + timedelta(minutes=15))
