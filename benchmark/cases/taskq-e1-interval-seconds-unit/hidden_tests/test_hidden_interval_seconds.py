from datetime import datetime, timedelta

import pytest

from taskq.schedule import Schedule, format_interval, next_run, parse_interval


@pytest.mark.parametrize(
    "spec, seconds",
    [
        ("90s", 90),
        ("1s", 1),
        ("1m30s", 90),
        ("2h15m10s", 2 * 3600 + 15 * 60 + 10),
        ("45 s", 45),
        ("90sec", 90),
        ("5m", 300),
    ],
)
def test_seconds_suffix(spec, seconds):
    assert parse_interval(spec) == timedelta(seconds=seconds)


def test_round_trip_with_format_interval():
    for seconds in (5, 59, 61, 3601, 90061):
        delta = timedelta(seconds=seconds)
        assert parse_interval(format_interval(delta)) == delta


def test_next_run_with_seconds_interval():
    last = datetime(2024, 1, 1, 12, 0, 0)
    assert next_run(last, "90s", datetime(2024, 1, 1, 12, 0, 30)) == datetime(2024, 1, 1, 12, 1, 30)


def test_schedule_with_seconds_interval_is_due():
    start = datetime(2024, 1, 1, 9, 0, 0)
    job = Schedule("health-check", "30s", last_run=start)
    assert not job.is_due(start + timedelta(seconds=29))
    assert job.is_due(start + timedelta(seconds=30))
