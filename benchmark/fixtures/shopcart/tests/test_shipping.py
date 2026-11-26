import pytest

from shopcart.shipping import describe_tiers, shipping_cost_cents, tier_price_cents


@pytest.mark.parametrize(
    "weight, cents",
    [(0.3, 499), (2.5, 899), (12.0, 1499), (23.2, 1499 + 4 * 85)],
)
def test_tier_prices(weight, cents):
    assert tier_price_cents(weight) == cents


def test_weightless_orders_ship_free():
    assert shipping_cost_cents(0) == 0


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        tier_price_cents(-1)


def test_free_shipping_threshold():
    assert shipping_cost_cents(3.0, order_value_cents=7_500) == 0
    assert shipping_cost_cents(3.0, order_value_cents=7_499) == 899


def test_express_is_never_free():
    assert shipping_cost_cents(0.5, order_value_cents=20_000, express=True) == 749


def test_rate_card():
    assert describe_tiers() == [
        "up to 1 kg: $4.99",
        "up to 5 kg: $8.99",
        "up to 20 kg: $14.99",
        "each kg over 20 kg: +$0.85",
    ]
