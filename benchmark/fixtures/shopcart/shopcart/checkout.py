"""Checkout: turn a cart into a paid order.

The flow for :meth:`Checkout.place_order` is

1. price the cart (subtotal, discounts, tax, shipping),
2. reserve stock for every line,
3. charge the customer through the payment gateway,
4. commit the reserved stock and record coupon usage.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Protocol

from shopcart import pricing
from shopcart.auth.permissions import PermissionDenied, User, require_discount
from shopcart.discounts import CouponBook, normalize_code
from shopcart.inventory import Inventory, OutOfStock
from shopcart.models import Cart, LineItem
from shopcart.payments.gateway import ChargeReceipt, PaymentError
from shopcart.pricing import PriceBreakdown
from shopcart.shipping import shipping_cost_cents
from shopcart.utils.money import Number, to_decimal


class CheckoutError(Exception):
    """Raised when a cart cannot be checked out."""


class Gateway(Protocol):
    def charge(self, token: str, amount_cents: int, *, reference: str | None = None) -> ChargeReceipt:
        ...


@dataclass(frozen=True)
class Order:
    order_id: str
    lines: tuple[LineItem, ...]
    breakdown: PriceBreakdown
    transaction_id: str
    coupon_code: str | None = None
    placed_at: datetime | None = None

    @property
    def total_cents(self) -> int:
        return self.breakdown.total_cents


class Checkout:
    """Prices carts and places orders against an inventory and a gateway."""

    def __init__(
        self,
        inventory: Inventory,
        gateway: Gateway,
        *,
        coupons: CouponBook | None = None,
        tax_rate: Number = Decimal("0"),
        clock: Callable[[], datetime] = datetime.now,
        order_prefix: str = "ORD",
    ) -> None:
        self.inventory = inventory
        self.gateway = gateway
        self.coupons = coupons if coupons is not None else CouponBook()
        self.tax_rate = to_decimal(tax_rate)
        self.clock = clock
        self._order_numbers = itertools.count(1)
        self._prefix = order_prefix

    # -- pricing -----------------------------------------------------------

    def quote(
        self,
        cart: Cart,
        coupon_code: str | None = None,
        *,
        staff: User | None = None,
        manual_discount: Number = 0,
        express: bool = False,
    ) -> PriceBreakdown:
        """Price ``cart`` without reserving stock or charging anything."""
        subtotal = pricing.subtotal_cents(cart.items)
        percent = self._discount_percent(subtotal, coupon_code, staff, manual_discount)
        discount = pricing.discount_cents(subtotal, percent)
        tax = pricing.tax_cents(subtotal - discount, self.tax_rate)
        shipping = shipping_cost_cents(
            cart.total_weight_kg, order_value_cents=subtotal - discount, express=express
        )
        return PriceBreakdown(
            subtotal_cents=subtotal,
            discount_cents=discount,
            tax_cents=tax,
            shipping_cents=shipping,
        )

    def _discount_percent(
        self,
        subtotal: int,
        coupon_code: str | None,
        staff: User | None,
        manual_discount: Number,
    ) -> Decimal:
        percent = Decimal("0")
        if coupon_code:
            coupon = self.coupons.validate(coupon_code, subtotal, now=self.clock())
            percent += coupon.percent_off
        manual = to_decimal(manual_discount)
        if manual:
            if staff is None:
                raise PermissionDenied("a staff member must authorise manual discounts")
            require_discount(staff, manual)
            percent += manual
        return min(percent, Decimal("100"))

    # -- ordering ----------------------------------------------------------

    def place_order(
        self,
        cart: Cart,
        payment_token: str,
        coupon_code: str | None = None,
        *,
        staff: User | None = None,
        manual_discount: Number = 0,
        express: bool = False,
    ) -> Order:
        """Reserve stock, take payment and return the placed :class:`Order`."""
        if cart.is_empty():
            raise CheckoutError("cannot check out an empty cart")
        breakdown = self.quote(
            cart, coupon_code, staff=staff, manual_discount=manual_discount, express=express
        )
        order_id = f"{self._prefix}-{next(self._order_numbers):05d}"
        reserved = self._reserve_all(cart)
        try:
            receipt = self.gateway.charge(payment_token, breakdown.total_cents, reference=order_id)
        except PaymentError:
            self._release_all(reserved)
            raise
        self._commit_all(reserved)
        if coupon_code:
            self.coupons.redeem(coupon_code)
        return Order(
            order_id=order_id,
            lines=tuple(cart.items),
            breakdown=breakdown,
            transaction_id=receipt.transaction_id,
            coupon_code=normalize_code(coupon_code) if coupon_code else None,
            placed_at=self.clock(),
        )

    def _reserve_all(self, cart: Cart) -> list[tuple[str, int]]:
        reserved: list[tuple[str, int]] = []
        try:
            for line in cart.items:
                self.inventory.reserve(line.sku, line.quantity)
                reserved.append((line.sku, line.quantity))
        except OutOfStock:
            self._release_all(reserved)
            raise
        return reserved

    def _release_all(self, reserved: list[tuple[str, int]]) -> None:
        for sku, quantity in reserved:
            self.inventory.release(sku, quantity)

    def _commit_all(self, reserved: list[tuple[str, int]]) -> None:
        for sku, quantity in reserved:
            self.inventory.commit(sku, quantity)
