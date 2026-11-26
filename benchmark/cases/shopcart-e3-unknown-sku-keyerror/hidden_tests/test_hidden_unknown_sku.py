from decimal import Decimal

import pytest

from shopcart.checkout import Checkout
from shopcart.inventory import Inventory, OutOfStock
from shopcart.models import Cart, Product
from shopcart.payments.gateway import FakeGateway


def test_available_for_unknown_sku_is_zero():
    inventory = Inventory({"MUG-01": 4})
    assert inventory.available("TEE-XL") == 0
    assert inventory.available("MUG-01") == 4


def test_available_on_empty_inventory():
    assert Inventory().available("ANYTHING") == 0


def test_reserving_unknown_sku_is_out_of_stock():
    inventory = Inventory({"MUG-01": 4})
    with pytest.raises(OutOfStock) as excinfo:
        inventory.reserve("TEE-XL", 1)
    assert excinfo.value.available == 0
    assert inventory.reserved("TEE-XL") == 0


def test_new_sku_becomes_available_after_receiving_stock():
    inventory = Inventory()
    assert inventory.available("PRE-ORDER") == 0
    inventory.add_stock("PRE-ORDER", 3)
    assert inventory.available("PRE-ORDER") == 3


def test_checkout_of_never_stocked_product():
    tee = Product("TEE-XL", "T-shirt XL", Decimal("15.00"), weight_kg=0.2)
    inventory = Inventory({"MUG-01": 4})
    gateway = FakeGateway()
    cart = Cart()
    cart.add(tee, 1)
    with pytest.raises(OutOfStock):
        Checkout(inventory, gateway).place_order(cart, "tok_visa")
    assert gateway.charges == []
