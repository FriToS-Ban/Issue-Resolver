"""Back-office reporting: sales summaries, receipts and best sellers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Iterable

from shopcart.pricing import line_total_cents
from shopcart.utils.dates import DateLike, parse_date
from shopcart.utils.money import format_cents

if TYPE_CHECKING:
    from shopcart.checkout import Order


@dataclass(frozen=True)
class SalesSummary:
    order_count: int
    gross_cents: int
    discount_cents: int
    tax_cents: int
    shipping_cents: int

    @property
    def net_revenue_cents(self) -> int:
        """Merchandise revenue after discounts, excluding tax and shipping."""
        return self.gross_cents - self.discount_cents

    @property
    def average_order_cents(self) -> int:
        if not self.order_count:
            return 0
        return (self.gross_cents - self.discount_cents + self.tax_cents + self.shipping_cents) // self.order_count


def summarize(orders: Iterable["Order"]) -> SalesSummary:
    count = gross = discount = tax = shipping = 0
    for order in orders:
        count += 1
        gross += order.breakdown.subtotal_cents
        discount += order.breakdown.discount_cents
        tax += order.breakdown.tax_cents
        shipping += order.breakdown.shipping_cents
    return SalesSummary(count, gross, discount, tax, shipping)


def orders_between(orders: Iterable["Order"], start: DateLike, end: DateLike) -> list["Order"]:
    """Orders placed between ``start`` and ``end`` (both inclusive)."""
    first, last = parse_date(start), parse_date(end)
    return [
        order
        for order in orders
        if order.placed_at is not None and first <= order.placed_at.date() <= last
    ]


def revenue_by_day(orders: Iterable["Order"]) -> dict[date, int]:
    totals: dict[date, int] = defaultdict(int)
    for order in orders:
        if order.placed_at is not None:
            totals[order.placed_at.date()] += order.total_cents
    return dict(sorted(totals.items()))


def top_products(orders: Iterable["Order"], n: int = 5) -> list[tuple[str, int]]:
    """Best selling SKUs by units sold."""
    units: Counter[str] = Counter()
    for order in orders:
        for line in order.lines:
            units[line.sku] += line.quantity
    return sorted(units.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def receipt_lines(order: "Order") -> list[str]:
    """Plain-text receipt for ``order``, one entry per printed line."""
    lines = [f"Order {order.order_id}"]
    for line in order.lines:
        lines.append(f"{line.quantity} x {line.product.name}  {format_cents(line_total_cents(line))}")
    b = order.breakdown
    lines.append(f"Subtotal  {format_cents(b.subtotal_cents)}")
    if b.discount_cents:
        lines.append(f"Discount  {format_cents(-b.discount_cents)}")
    lines.append(f"Tax  {format_cents(b.tax_cents)}")
    lines.append(f"Shipping  {format_cents(b.shipping_cents)}")
    lines.append(f"Total  {format_cents(b.total_cents)}")
    return lines


def format_summary(summary: SalesSummary) -> str:
    return "\n".join(
        [
            f"Orders: {summary.order_count}",
            f"Gross sales: {format_cents(summary.gross_cents)}",
            f"Discounts: {format_cents(summary.discount_cents)}",
            f"Tax collected: {format_cents(summary.tax_cents)}",
            f"Shipping: {format_cents(summary.shipping_cents)}",
            f"Net revenue: {format_cents(summary.net_revenue_cents)}",
        ]
    )
