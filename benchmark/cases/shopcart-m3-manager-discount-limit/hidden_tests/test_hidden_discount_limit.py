from decimal import Decimal

import pytest

from shopcart import Cart, Checkout, Product
from shopcart.auth.permissions import (
    PermissionDenied,
    Role,
    User,
    can_apply_discount,
    require_discount,
)
from shopcart.inventory import Inventory
from shopcart.payments.gateway import FakeGateway

MANAGER = User("priya", Role.MANAGER)
CLERK = User("sam", Role.CLERK)
ADMIN = User("root", Role.ADMIN)


@pytest.mark.parametrize(
    "user, percent",
    [(MANAGER, 20), (MANAGER, Decimal("20.00")), (CLERK, 10), (ADMIN, 100), (MANAGER, "19.99")],
)
def test_limit_is_inclusive(user, percent):
    assert can_apply_discount(user, percent)
    require_discount(user, percent)


@pytest.mark.parametrize(
    "user, percent",
    [(MANAGER, Decimal("20.01")), (MANAGER, 25), (CLERK, Decimal("10.5")), (User("c", Role.CUSTOMER), 1)],
)
def test_above_limit_still_refused(user, percent):
    assert not can_apply_discount(user, percent)
    with pytest.raises(PermissionDenied):
        require_discount(user, percent)


def test_inactive_manager_still_refused():
    assert not can_apply_discount(User("old", Role.MANAGER, active=False), 20)


def test_checkout_manager_twenty_percent():
    product = Product("KTL-02", "Stovetop kettle", Decimal("39.00"), weight_kg=1.3)
    cart = Cart()
    cart.add(product, 2)
    checkout = Checkout(Inventory({"KTL-02": 5}), FakeGateway())
    quote = checkout.quote(cart, staff=MANAGER, manual_discount=20)
    assert quote.discount_cents == 1_560
