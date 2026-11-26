from datetime import datetime
from decimal import Decimal

import pytest

from shopcart import reports
from shopcart.checkout import Checkout
from shopcart.discounts import Coupon, CouponBook
from shopcart.inventory import Inventory
from shopcart.models import Cart, Product
from shopcart.payments.gateway import FakeGateway
from shopcart.utils.money import format_cents, format_money


@pytest.mark.parametrize(
    "amount, text",
    [
        (-5, "-$5.00"),
        (-5.50, "-$5.50"),
        (Decimal("-0.25"), "-$0.25"),
        ("-0.01", "-$0.01"),
        (-1234.5, "-$1,234.50"),
    ],
)
def test_negative_amounts(amount, text):
    assert format_money(amount) == text


@pytest.mark.parametrize("amount, text", [(5, "$5.00"), (0, "$0.00"), (1234.56, "$1,234.56")])
def test_non_negative_amounts_unchanged(amount, text):
    assert format_money(amount) == text


def test_negative_cents():
    assert format_cents(-250) == "-$2.50"
    assert format_cents(-100_000) == "-$1,000.00"


def test_custom_symbol_with_negative():
    assert format_money(-3, symbol="EUR ") == "-EUR 3.00"


def test_receipt_discount_line():
    lamp = Product("LAMP-1", "Desk lamp", Decimal("25.00"), weight_kg=1.2)
    cart = Cart()
    cart.add(lamp, 2)
    checkout = Checkout(
        Inventory({"LAMP-1": 5}),
        FakeGateway(),
        coupons=CouponBook([Coupon("TENOFF", 10)]),
        clock=lambda: datetime(2024, 5, 1, 12, 0),
    )
    order = checkout.place_order(cart, "tok_visa", coupon_code="TENOFF")
    assert "Discount  -$5.00" in reports.receipt_lines(order)
