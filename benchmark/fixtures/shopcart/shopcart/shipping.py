"""Shipping rates by parcel weight."""

from __future__ import annotations

import math
from decimal import Decimal

from shopcart.utils.money import format_cents, round_half_up

WEIGHT_TIERS_KG: tuple[tuple[float, int], ...] = (
    (1.0, 499),
    (5.0, 899),
    (20.0, 1499),
)
OVERWEIGHT_PER_KG_CENTS = 85
FREE_SHIPPING_THRESHOLD_CENTS = 7_500
EXPRESS_MULTIPLIER = Decimal("1.5")


def tier_price_cents(weight_kg: float) -> int:
    """Standard rate for a parcel of ``weight_kg`` kilograms."""
    if weight_kg < 0:
        raise ValueError("parcel weight cannot be negative")
    for limit, price in WEIGHT_TIERS_KG:
        if weight_kg <= limit:
            return price
    heaviest_limit, heaviest_price = WEIGHT_TIERS_KG[-1]
    extra_kg = math.ceil(weight_kg - heaviest_limit)
    return heaviest_price + extra_kg * OVERWEIGHT_PER_KG_CENTS


def shipping_cost_cents(
    weight_kg: float,
    *,
    order_value_cents: int | None = None,
    express: bool = False,
) -> int:
    """Shipping charge for a parcel.

    Standard shipping is free once ``order_value_cents`` reaches
    :data:`FREE_SHIPPING_THRESHOLD_CENTS`.  Express shipping is never free.
    """
    if weight_kg == 0:
        return 0
    if (
        not express
        and order_value_cents is not None
        and order_value_cents >= FREE_SHIPPING_THRESHOLD_CENTS
    ):
        return 0
    cost = tier_price_cents(weight_kg)
    if express:
        cost = round_half_up(Decimal(cost) * EXPRESS_MULTIPLIER)
    return cost


def describe_tiers() -> list[str]:
    """Human readable rate card, as printed on the shipping info page."""
    lines = [f"up to {limit:g} kg: {format_cents(price)}" for limit, price in WEIGHT_TIERS_KG]
    lines.append(f"each kg over {WEIGHT_TIERS_KG[-1][0]:g} kg: +{format_cents(OVERWEIGHT_PER_KG_CENTS)}")
    return lines
