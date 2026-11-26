"""Role-based limits for staff actions at the till."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from shopcart.utils.money import Number, format_cents, to_decimal


class Role(str, Enum):
    CUSTOMER = "customer"
    CLERK = "clerk"
    MANAGER = "manager"
    ADMIN = "admin"


DISCOUNT_LIMITS: dict[Role, Decimal] = {
    Role.CUSTOMER: Decimal("0"),
    Role.CLERK: Decimal("10"),
    Role.MANAGER: Decimal("20"),
    Role.ADMIN: Decimal("100"),
}

REFUND_LIMITS_CENTS: dict[Role, int | None] = {
    Role.CUSTOMER: 0,
    Role.CLERK: 5_000,
    Role.MANAGER: 50_000,
    Role.ADMIN: None,
}


class PermissionDenied(Exception):
    """Raised when a user tries something their role does not allow."""


@dataclass(frozen=True)
class User:
    username: str
    role: Role = Role.CUSTOMER
    active: bool = True

    @property
    def is_staff(self) -> bool:
        return self.role is not Role.CUSTOMER


def max_discount_percent(user: User) -> Decimal:
    """Largest manual discount ``user`` may apply, in percent."""
    if not user.active:
        return Decimal("0")
    return DISCOUNT_LIMITS[user.role]


def can_apply_discount(user: User, percent: Number) -> bool:
    pct = to_decimal(percent)
    if pct < 0:
        return False
    return pct <= max_discount_percent(user)


def require_discount(user: User, percent: Number) -> None:
    """Raise :class:`PermissionDenied` unless ``user`` may apply ``percent``."""
    if not can_apply_discount(user, percent):
        raise PermissionDenied(
            f"{user.username} ({user.role.value}) may not apply a {percent}% discount; "
            f"the limit is {max_discount_percent(user)}%"
        )


def can_refund(user: User, amount_cents: int) -> bool:
    if not user.active or amount_cents < 0:
        return False
    limit = REFUND_LIMITS_CENTS[user.role]
    return limit is None or amount_cents <= limit


def require_refund(user: User, amount_cents: int) -> None:
    if not can_refund(user, amount_cents):
        raise PermissionDenied(
            f"{user.username} ({user.role.value}) may not refund {format_cents(amount_cents)}"
        )
