from datetime import datetime
from decimal import Decimal

import pytest

from shopcart import Cart, Checkout, Product
from shopcart.discounts import Coupon, CouponBook
from shopcart.inventory import Inventory, OutOfStock
from shopcart.payments.gateway import FakeGateway, GatewayUnavailable, PaymentDeclined, PaymentError

MUG = Product("MUG-01", "Enamel mug", Decimal("12.50"), weight_kg=0.4)
POSTER = Product("PST-A2", "A2 poster", Decimal("19.99"), weight_kg=0.2)


def make_checkout(stock=None):
    inventory = Inventory(stock if stock is not None else {"MUG-01": 3, "PST-A2": 2})
    gateway = FakeGateway()
    coupons = CouponBook([Coupon("WELCOME10", 10)])
    checkout = Checkout(inventory, gateway, coupons=coupons, clock=lambda: datetime(2024, 5, 1, 12, 0))
    return checkout, inventory, gateway, coupons


def two_line_cart():
    cart = Cart()
    cart.add(MUG, 2)
    cart.add(POSTER, 1)
    return cart


@pytest.mark.parametrize(
    "token, error",
    [("tok_declined", PaymentDeclined), ("tok_insufficient_funds", PaymentDeclined), ("tok_timeout", GatewayUnavailable)],
)
def test_failed_payment_releases_every_line(token, error):
    checkout, inventory, gateway, _ = make_checkout()
    with pytest.raises(error):
        checkout.place_order(two_line_cart(), token)
    assert inventory.available("MUG-01") == 3
    assert inventory.available("PST-A2") == 2
    assert inventory.reserved("MUG-01") == 0
    assert inventory.reserved("PST-A2") == 0
    assert inventory.on_hand("MUG-01") == 3
    assert gateway.charges == []


def test_error_is_still_a_payment_error():
    checkout, _, _, _ = make_checkout()
    with pytest.raises(PaymentError):
        checkout.place_order(two_line_cart(), "tok_expired_card")


def test_repeated_declines_do_not_exhaust_stock():
    checkout, inventory, gateway, _ = make_checkout()
    for _ in range(3):
        with pytest.raises(PaymentDeclined):
            checkout.place_order(two_line_cart(), "tok_declined")
    order = checkout.place_order(two_line_cart(), "tok_visa")
    assert gateway.charges[-1].transaction_id == order.transaction_id
    assert inventory.on_hand("MUG-01") == 1
    assert inventory.available("MUG-01") == 1
    assert inventory.reserved("MUG-01") == 0


def test_coupon_not_redeemed_when_payment_fails():
    checkout, _, _, coupons = make_checkout()
    with pytest.raises(PaymentDeclined):
        checkout.place_order(two_line_cart(), "tok_declined", coupon_code="WELCOME10")
    assert coupons.uses("WELCOME10") == 0


def test_out_of_stock_still_never_charges():
    checkout, inventory, gateway, _ = make_checkout({"MUG-01": 5, "PST-A2": 0})
    with pytest.raises(OutOfStock):
        checkout.place_order(two_line_cart(), "tok_visa")
    assert gateway.charges == []
    assert inventory.available("MUG-01") == 5


def test_successful_order_commits_stock_once():
    checkout, inventory, gateway, _ = make_checkout()
    checkout.place_order(two_line_cart(), "tok_visa")
    assert inventory.on_hand("MUG-01") == 1
    assert inventory.on_hand("PST-A2") == 1
    assert inventory.reserved("MUG-01") == 0
    assert len(gateway.charges) == 1
