from datetime import date, datetime
from decimal import Decimal

import pytest

from shopcart import Cart, Checkout, Product
from shopcart.discounts import Coupon, CouponBook, CouponExpired
from shopcart.inventory import Inventory
from shopcart.payments.gateway import FakeGateway

KETTLE = Product("KTL-02", "Stovetop kettle", Decimal("39.00"), weight_kg=1.3)


def book():
    return CouponBook(
        [
            Coupon("SPRING15", 15, expires_on="2024-04-30"),
            Coupon("FOREVER5", 5),
            Coupon("SUMMER25", 25, expires_on=date(2024, 8, 31)),
        ]
    )


def kettle_cart():
    cart = Cart()
    cart.add(KETTLE)
    return cart


def test_checkout_rejects_expired_coupon():
    checkout = Checkout(Inventory({"KTL-02": 10}), FakeGateway(), coupons=book(), clock=lambda: datetime(2024, 5, 20, 14, 0))
    with pytest.raises(CouponExpired):
        checkout.quote(kettle_cart(), "SPRING15")


def test_place_order_with_expired_coupon_charges_nothing():
    inventory = Inventory({"KTL-02": 10})
    gateway = FakeGateway()
    coupons = book()
    checkout = Checkout(inventory, gateway, coupons=coupons, clock=lambda: datetime(2024, 5, 1, 0, 0, 1))
    with pytest.raises(CouponExpired):
        checkout.place_order(kettle_cart(), "tok_visa", coupon_code="spring15")
    assert gateway.charges == []
    assert inventory.available("KTL-02") == 10
    assert coupons.uses("SPRING15") == 0


def test_coupon_still_valid_on_its_last_day():
    checkout = Checkout(Inventory({"KTL-02": 10}), FakeGateway(), coupons=book(), clock=lambda: datetime(2024, 4, 30, 23, 59))
    assert checkout.quote(kettle_cart(), "SPRING15").discount_cents == 585


def test_validate_one_day_after_expiry():
    with pytest.raises(CouponExpired):
        book().validate("SUMMER25", 5000, now=date(2024, 9, 1))
    assert book().validate("SUMMER25", 5000, now=date(2024, 8, 31)).code == "SUMMER25"


def test_validate_long_after_expiry():
    with pytest.raises(CouponExpired):
        book().validate("SPRING15", 5000, now=datetime(2025, 1, 15, 9, 0))


def test_active_excludes_expired_coupons():
    codes = [c.code for c in book().active(now=datetime(2024, 5, 20, 14, 0))]
    assert codes == ["FOREVER5", "SUMMER25"]
