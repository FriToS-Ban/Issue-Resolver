from datetime import date, datetime
from decimal import Decimal

import pytest

from shopcart.discounts import (
    Coupon,
    CouponBook,
    CouponNotApplicable,
    UnknownCoupon,
)


def test_coupon_code_is_normalised():
    assert Coupon(" spring5 ", 5).code == "SPRING5"


def test_coupon_expiry_parsed():
    assert Coupon("X", 5, expires_on="2024-12-31").expires_on == date(2024, 12, 31)


def test_coupon_percent_must_be_positive():
    with pytest.raises(ValueError):
        Coupon("ZERO", 0)


def test_validate_is_case_insensitive(coupons, fixed_now):
    assert coupons.validate("welcome10", 1000, now=fixed_now).percent_off == Decimal("10")


def test_validate_unknown_code(coupons, fixed_now):
    with pytest.raises(UnknownCoupon):
        coupons.validate("NOPE", 1000, now=fixed_now)


def test_coupon_before_expiry_is_accepted(coupons, fixed_now):
    assert coupons.validate("SUMMER25", 5000, now=fixed_now).code == "SUMMER25"


def test_minimum_subtotal(coupons, fixed_now):
    with pytest.raises(CouponNotApplicable):
        coupons.validate("BIGSPENDER", 9_999, now=fixed_now)
    assert coupons.validate("BIGSPENDER", 10_000, now=fixed_now).code == "BIGSPENDER"


def test_max_uses():
    book = CouponBook([Coupon("ONCE", 50, max_uses=1)])
    book.validate("ONCE", 100)
    book.redeem("ONCE")
    with pytest.raises(CouponNotApplicable):
        book.validate("ONCE", 100)


def test_duplicate_codes_rejected():
    with pytest.raises(ValueError):
        CouponBook([Coupon("A", 5), Coupon("a", 10)])


def test_active_lists_usable_coupons(coupons):
    codes = [c.code for c in coupons.active(now=datetime(2024, 5, 1))]
    assert codes == ["BIGSPENDER", "SUMMER25", "WELCOME10"]


def test_contains(coupons):
    assert "summer25" in coupons
    assert "WINTER" not in coupons
