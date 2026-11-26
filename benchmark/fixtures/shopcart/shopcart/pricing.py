"""Price calculations for carts and orders.

Every function here takes and returns integer cents.  Percentages are given as
numbers between 0 and 100; tax rates as fractions (``0.08`` for 8%).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from shopcart.models import LineItem, Product
from shopcart.utils.money import Number, round_half_up, to_cents, to_decimal


@dataclass(frozen=True)
class PriceBreakdown:
    """The money side of an order, all in cents."""

    subtotal_cents: int
    discount_cents: int = 0
    tax_cents: int = 0
    shipping_cents: int = 0

    @property
    def total_cents(self) -> int:
        return self.subtotal_cents - self.discount_cents + self.tax_cents + self.shipping_cents

    def as_dict(self) -> dict[str, int]:
        return {
            "subtotal": self.subtotal_cents,
            "discount": self.discount_cents,
            "tax": self.tax_cents,
            "shipping": self.shipping_cents,
            "total": self.total_cents,
        }


def line_total_cents(item: LineItem) -> int:
    """Price of one cart line in cents."""
    return to_cents(item.product.unit_price * item.quantity)


def subtotal_cents(items: Iterable[LineItem]) -> int:
    """Merchandise subtotal of ``items`` in cents."""
    return sum(line_total_cents(item) for item in items)


def discount_cents(subtotal: int, percent: Number) -> int:
    """Amount taken off ``subtotal`` by a ``percent`` discount."""
    pct = to_decimal(percent)
    if pct < 0 or pct > 100:
        raise ValueError(f"discount percent must be between 0 and 100, got {percent}")
    return min(subtotal, round_half_up(Decimal(subtotal) * pct / 100))


def tax_cents(taxable: int, rate: Number) -> int:
    """Sales tax due on ``taxable`` cents at ``rate``."""
    if taxable <= 0:
        return 0
    rate_value = to_decimal(rate)
    if rate_value < 0:
        raise ValueError("tax rate cannot be negative")
    return round_half_up(Decimal(taxable) * rate_value)


def unit_price_after_discount(product: Product, percent: Number) -> Decimal:
    """Catalogue price of ``product`` with a ``percent`` markdown, in dollars."""
    pct = to_decimal(percent)
    return (product.unit_price * (100 - pct) / 100).quantize(Decimal("0.0001"))


def price_per_kg(product: Product) -> Decimal | None:
    """Unit price divided by weight, shown on shelf labels for loose goods."""
    if not product.weight_kg:
        return None
    return (product.unit_price / to_decimal(product.weight_kg)).quantize(Decimal("0.01"))
