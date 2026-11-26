from decimal import Decimal

import pytest

from shopcart import pricing
from shopcart.models import LineItem, Product


def test_line_total(poster):
    assert pricing.line_total_cents(LineItem(poster, 2)) == 3998


def test_subtotal(mug, poster, kettle):
    items = [LineItem(mug, 2), LineItem(poster, 1), LineItem(kettle, 1)]
    assert pricing.subtotal_cents(items) == 2500 + 1999 + 3900


def test_subtotal_of_nothing_is_zero():
    assert pricing.subtotal_cents([]) == 0


def test_discount_cents():
    assert pricing.discount_cents(10_000, 15) == 1500
    assert pricing.discount_cents(999, Decimal("12.5")) == 125


def test_discount_rejects_out_of_range():
    with pytest.raises(ValueError):
        pricing.discount_cents(1000, 120)


def test_tax_cents():
    assert pricing.tax_cents(10_000, "0.08") == 800
    assert pricing.tax_cents(0, "0.08") == 0


def test_breakdown_total():
    b = pricing.PriceBreakdown(subtotal_cents=5000, discount_cents=500, tax_cents=360, shipping_cents=499)
    assert b.total_cents == 5359
    assert b.as_dict()["total"] == 5359


def test_unit_price_after_discount():
    product = Product("X", "x", Decimal("20.00"))
    assert pricing.unit_price_after_discount(product, 25) == Decimal("15.0000")


def test_price_per_kg():
    product = Product("RICE-5", "Rice 5kg", Decimal("11.00"), weight_kg=5)
    assert pricing.price_per_kg(product) == Decimal("2.20")
    assert pricing.price_per_kg(Product("E", "ebook", Decimal("3"))) is None
