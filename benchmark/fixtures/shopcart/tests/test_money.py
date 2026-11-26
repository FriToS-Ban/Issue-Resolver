from decimal import Decimal

import pytest

from shopcart.utils.money import (
    format_cents,
    format_money,
    from_cents,
    round_half_up,
    split_evenly,
    to_cents,
    to_decimal,
)


@pytest.mark.parametrize(
    "amount, cents",
    [(Decimal("19.99"), 1999), (5, 500), ("0.10", 10), (12.5, 1250), (Decimal("0"), 0)],
)
def test_to_cents(amount, cents):
    assert to_cents(amount) == cents


def test_from_cents():
    assert from_cents(1999) == Decimal("19.99")
    assert from_cents(5) == Decimal("0.05")


def test_round_half_up():
    assert round_half_up(Decimal("2.5")) == 3
    assert round_half_up(Decimal("2.4999")) == 2


def test_to_decimal_rejects_garbage():
    with pytest.raises(ValueError):
        to_decimal("twelve")
    with pytest.raises(TypeError):
        to_decimal(True)


@pytest.mark.parametrize(
    "amount, text",
    [(1234.5, "$1,234.50"), (0, "$0.00"), (Decimal("7"), "$7.00"), ("1000000", "$1,000,000.00")],
)
def test_format_money(amount, text):
    assert format_money(amount) == text


def test_format_money_custom_symbol():
    assert format_money(3, symbol="EUR ") == "EUR 3.00"


def test_format_cents():
    assert format_cents(499) == "$4.99"


def test_split_evenly_adds_up():
    shares = split_evenly(1000, 3)
    assert shares == [334, 333, 333]
    assert sum(shares) == 1000
