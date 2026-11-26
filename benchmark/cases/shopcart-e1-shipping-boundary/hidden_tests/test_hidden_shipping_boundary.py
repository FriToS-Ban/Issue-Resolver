from decimal import Decimal

import pytest

from shopcart.checkout import Checkout
from shopcart.inventory import Inventory
from shopcart.models import Cart, Product
from shopcart.payments.gateway import FakeGateway
from shopcart.shipping import shipping_cost_cents, tier_price_cents


@pytest.mark.parametrize("weight, cents", [(1.0, 499), (5.0, 899), (20.0, 1499)])
def test_weight_exactly_at_tier_limit_uses_that_tier(weight, cents):
    assert tier_price_cents(weight) == cents
    assert shipping_cost_cents(weight) == cents


@pytest.mark.parametrize(
    "weight, cents",
    [(0.99, 499), (1.01, 899), (4.999, 899), (5.001, 1499), (19.5, 1499), (20.5, 1499 + 85)],
)
def test_weights_near_tier_limits(weight, cents):
    assert tier_price_cents(weight) == cents


def test_express_at_tier_limit():
    assert shipping_cost_cents(5.0, express=True) == 1349


def test_checkout_quote_for_exactly_five_kilos():
    coffee = Product("COF-2500", "Coffee beans 2.5 kg", Decimal("18.00"), weight_kg=2.5)
    cart = Cart()
    cart.add(coffee, 2)
    checkout = Checkout(Inventory({"COF-2500": 10}), FakeGateway())
    assert checkout.quote(cart).shipping_cents == 899
