from decimal import Decimal

import pytest

from shopcart.auth.permissions import PermissionDenied, Role, User
from shopcart.checkout import Checkout, CheckoutError
from shopcart.inventory import OutOfStock
from shopcart.models import Cart
from shopcart.payments.gateway import PaymentDeclined


def make_checkout(inventory, gateway, coupons, fixed_now, tax_rate="0"):
    return Checkout(inventory, gateway, coupons=coupons, tax_rate=tax_rate, clock=lambda: fixed_now)


def test_quote_without_discount_includes_tax(inventory, gateway, coupons, fixed_now, mug):
    checkout = make_checkout(inventory, gateway, coupons, fixed_now, tax_rate="0.08")
    cart = Cart()
    cart.add(mug, 2)
    quote = checkout.quote(cart)
    assert quote.subtotal_cents == 2500
    assert quote.tax_cents == 200
    assert quote.shipping_cents == 499
    assert quote.total_cents == 2500 + 200 + 499


def test_quote_with_coupon(inventory, gateway, coupons, fixed_now, kettle):
    checkout = make_checkout(inventory, gateway, coupons, fixed_now)
    cart = Cart()
    cart.add(kettle, 2)
    quote = checkout.quote(cart, "WELCOME10")
    assert quote.subtotal_cents == 7800
    assert quote.discount_cents == 780
    assert quote.shipping_cents == 899


def test_free_shipping_over_threshold(inventory, gateway, coupons, fixed_now, kettle):
    checkout = make_checkout(inventory, gateway, coupons, fixed_now)
    cart = Cart()
    cart.add(kettle, 2)
    assert checkout.quote(cart).shipping_cents == 0


def test_manager_manual_discount(inventory, gateway, coupons, fixed_now, kettle):
    checkout = make_checkout(inventory, gateway, coupons, fixed_now)
    cart = Cart()
    cart.add(kettle, 1)
    quote = checkout.quote(cart, staff=User("priya", Role.MANAGER), manual_discount=15)
    assert quote.discount_cents == 585


def test_manual_discount_needs_staff(inventory, gateway, coupons, fixed_now, kettle):
    checkout = make_checkout(inventory, gateway, coupons, fixed_now)
    cart = Cart()
    cart.add(kettle, 1)
    with pytest.raises(PermissionDenied):
        checkout.quote(cart, manual_discount=5)


def test_place_order_charges_and_commits(inventory, gateway, coupons, fixed_now, mug, poster):
    checkout = make_checkout(inventory, gateway, coupons, fixed_now, tax_rate="0.08")
    cart = Cart()
    cart.add(mug, 2)
    cart.add(poster, 1)
    order = checkout.place_order(cart, "tok_visa")
    assert order.order_id == "ORD-00001"
    assert gateway.charges[0].amount_cents == order.total_cents
    assert inventory.on_hand("MUG-01") == 8
    assert inventory.available("PST-A2") == 2
    assert order.placed_at == fixed_now


def test_coupon_is_redeemed_on_success(inventory, gateway, coupons, fixed_now, mug):
    checkout = make_checkout(inventory, gateway, coupons, fixed_now)
    cart = Cart()
    cart.add(mug, 1)
    order = checkout.place_order(cart, "tok_visa", coupon_code="welcome10")
    assert order.coupon_code == "WELCOME10"
    assert coupons.uses("WELCOME10") == 1


def test_empty_cart_rejected(inventory, gateway, coupons, fixed_now):
    checkout = make_checkout(inventory, gateway, coupons, fixed_now)
    with pytest.raises(CheckoutError):
        checkout.place_order(Cart(), "tok_visa")


def test_out_of_stock_is_not_charged(inventory, gateway, coupons, fixed_now, kettle):
    checkout = make_checkout(inventory, gateway, coupons, fixed_now)
    cart = Cart()
    cart.add(kettle, 3)
    with pytest.raises(OutOfStock):
        checkout.place_order(cart, "tok_visa")
    assert gateway.charges == []
    assert inventory.available("KTL-02") == 2


def test_declined_card_raises(inventory, gateway, coupons, fixed_now, mug):
    checkout = make_checkout(inventory, gateway, coupons, fixed_now)
    cart = Cart()
    cart.add(mug, 1)
    with pytest.raises(PaymentDeclined):
        checkout.place_order(cart, "tok_declined")
    assert gateway.charges == []
    assert coupons.uses("WELCOME10") == 0


def test_tax_rate_is_decimal(inventory, gateway, coupons, fixed_now):
    checkout = make_checkout(inventory, gateway, coupons, fixed_now, tax_rate=0.0725)
    assert checkout.tax_rate == Decimal("0.0725")
