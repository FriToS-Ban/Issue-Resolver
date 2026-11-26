from datetime import datetime
from decimal import Decimal

import pytest

from shopcart import Cart, Checkout, Product, reports
from shopcart.inventory import Inventory
from shopcart.payments.gateway import FakeGateway
from shopcart.pricing import line_total_cents, subtotal_cents
from shopcart.utils.money import format_money

SCREW = Product("SCREW-M4", "M4 screw", Decimal("0.335"))
WASHER = Product("WASHER-M4", "M4 washer", Decimal("0.125"))
BOLT = Product("BOLT-M8", "M8 bolt", Decimal("1.255"))


def hardware_cart():
    cart = Cart()
    cart.add(SCREW, 3)
    cart.add(WASHER, 1)
    cart.add(BOLT, 1)
    return cart


def make_checkout(tax_rate="0"):
    stock = {"SCREW-M4": 100, "WASHER-M4": 100, "BOLT-M8": 100}
    return Checkout(Inventory(stock), FakeGateway(), tax_rate=tax_rate, clock=lambda: datetime(2024, 6, 3, 10, 0))


@pytest.mark.parametrize(
    "product, qty, cents",
    [(SCREW, 3, 101), (WASHER, 1, 13), (BOLT, 1, 126), (SCREW, 1, 34), (BOLT, 2, 251)],
)
def test_line_totals_round_half_up(product, qty, cents):
    cart = Cart()
    cart.add(product, qty)
    assert line_total_cents(cart.items[0]) == cents


def test_subtotal_is_sum_of_rounded_lines():
    cart = hardware_cart()
    assert subtotal_cents(cart.items) == 240
    assert subtotal_cents(cart.items) == sum(line_total_cents(line) for line in cart.items)


def test_two_half_cent_lines():
    other = Product("SCREW-M5", "M5 screw", Decimal("0.335"))
    cart = Cart()
    cart.add(SCREW, 3)
    cart.add(other, 3)
    assert make_checkout().quote(cart).subtotal_cents == 202


def test_quote_and_charge():
    checkout = make_checkout(tax_rate="0.10")
    quote = checkout.quote(hardware_cart())
    assert quote.subtotal_cents == 240
    assert quote.tax_cents == 24
    order = checkout.place_order(hardware_cart(), "tok_visa")
    assert checkout.gateway.charges[-1].amount_cents == order.total_cents == quote.total_cents


def test_receipt_adds_up():
    order = make_checkout().place_order(hardware_cart(), "tok_visa")
    lines = reports.receipt_lines(order)
    assert lines[1:5] == [
        "3 x M4 screw  $1.01",
        "1 x M4 washer  $0.13",
        "1 x M8 bolt  $1.26",
        "Subtotal  $2.40",
    ]


@pytest.mark.parametrize(
    "amount, text",
    [(Decimal("1.255"), "$1.26"), ("0.125", "$0.13"), ("1.005", "$1.01"), (Decimal("1234.565"), "$1,234.57"), (Decimal("2.675"), "$2.68"), (19.99, "$19.99")],
)
def test_format_money_rounds_half_up(amount, text):
    assert format_money(amount) == text
