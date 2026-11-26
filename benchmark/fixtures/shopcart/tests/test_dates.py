from datetime import date, datetime

import pytest

from shopcart.utils.dates import date_range, days_until, end_of_day, is_expired, parse_date


@pytest.mark.parametrize("text", ["2024-03-01", "2024/03/01", "01.03.2024", " 2024-03-01 "])
def test_parse_date_formats(text):
    assert parse_date(text) == date(2024, 3, 1)


def test_parse_date_passes_dates_through():
    assert parse_date(datetime(2024, 3, 1, 15, 0)) == date(2024, 3, 1)
    assert parse_date(date(2024, 3, 1)) == date(2024, 3, 1)


def test_parse_date_rejects_garbage():
    with pytest.raises(ValueError):
        parse_date("next tuesday")


def test_end_of_day():
    assert end_of_day(date(2024, 3, 1)).hour == 23


def test_future_date_is_not_expired():
    assert not is_expired("2024-06-30", now=datetime(2024, 6, 1, 9, 0))


def test_expiry_date_itself_is_still_valid():
    assert not is_expired(date(2024, 6, 30), now=datetime(2024, 6, 30, 22, 15))


def test_days_until():
    assert days_until("2024-03-11", today=date(2024, 3, 1)) == 10


def test_date_range_inclusive():
    days = list(date_range("2024-02-28", "2024-03-01"))
    assert days == [date(2024, 2, 28), date(2024, 2, 29), date(2024, 3, 1)]
