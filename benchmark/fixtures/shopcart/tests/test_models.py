from decimal import Decimal

import pytest

from shopcart.models import Cart, LineItem, Product, make_product


def test_product_normalises_price_to_decimal():
    product = Product("SKU-1", "Thing", "4.20")
    assert product.unit_price == Decimal("4.20")


def test_product_rejects_negative_price():
    with pytest.raises(ValueError):
        Product("SKU-1", "Thing", Decimal("-1"))


def test_product_requires_sku():
    with pytest.raises(ValueError):
        Product(" ", "Thing", Decimal("1"))


def test_line_item_requires_positive_quantity(mug):
    with pytest.raises(ValueError):
        LineItem(mug, 0)


def test_cart_merges_lines_for_same_sku(mug):
    cart = Cart()
    cart.add(mug, 2)
    cart.add(mug, 3)
    assert len(cart) == 1
    assert cart.items[0].quantity == 5


def test_cart_set_quantity_zero_removes_line(mug, poster):
    cart = Cart()
    cart.add(mug)
    cart.add(poster)
    cart.set_quantity("MUG-01", 0)
    assert "MUG-01" not in cart
    assert [line.sku for line in cart] == ["PST-A2"]


def test_cart_remove_unknown_sku_raises():
    with pytest.raises(KeyError):
        Cart().remove("NOPE")


def test_cart_totals(mug, kettle):
    cart = Cart()
    cart.add(mug, 3)
    cart.add(kettle, 1)
    assert cart.total_quantity == 4
    assert cart.total_weight_kg == pytest.approx(2.5)


def test_make_product_accepts_float():
    assert make_product("X", "x", 2.5).unit_price == Decimal("2.5")
