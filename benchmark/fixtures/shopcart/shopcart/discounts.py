"""Coupon codes and the rules for accepting them."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Iterable

from shopcart.utils.dates import DateLike, is_expired, parse_date
from shopcart.utils.money import format_cents, to_decimal


class CouponError(ValueError):
    """Base class for coupon problems that are shown to the customer."""


class UnknownCoupon(CouponError):
    pass


class CouponExpired(CouponError):
    pass


class CouponNotApplicable(CouponError):
    pass


@dataclass(frozen=True)
class Coupon:
    """A percentage-off coupon.

    ``expires_on`` is the last day on which the coupon can be used.
    """

    code: str
    percent_off: Decimal
    expires_on: date | None = None
    min_subtotal_cents: int = 0
    max_uses: int | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.code.strip():
            raise ValueError("coupon code cannot be blank")
        pct = to_decimal(self.percent_off)
        if not 0 < pct <= 100:
            raise ValueError(f"coupon {self.code}: percent_off must be in (0, 100]")
        object.__setattr__(self, "code", self.code.strip().upper())
        object.__setattr__(self, "percent_off", pct)
        if self.expires_on is not None:
            object.__setattr__(self, "expires_on", parse_date(self.expires_on))


def normalize_code(code: str) -> str:
    return code.strip().upper()


class CouponBook:
    """All coupons known to the shop, plus how often each has been redeemed."""

    def __init__(self, coupons: Iterable[Coupon] = ()) -> None:
        self._coupons: dict[str, Coupon] = {}
        self._uses: dict[str, int] = {}
        for coupon in coupons:
            self.add(coupon)

    def add(self, coupon: Coupon) -> None:
        if coupon.code in self._coupons:
            raise ValueError(f"duplicate coupon code {coupon.code}")
        self._coupons[coupon.code] = coupon

    def get(self, code: str) -> Coupon:
        try:
            return self._coupons[normalize_code(code)]
        except KeyError:
            raise UnknownCoupon(f"unknown coupon code {code!r}") from None

    def uses(self, code: str) -> int:
        return self._uses.get(normalize_code(code), 0)

    def validate(
        self,
        code: str,
        subtotal_cents: int,
        now: datetime | DateLike | None = None,
    ) -> Coupon:
        """Return the coupon for ``code`` if it may be applied to this order."""
        coupon = self.get(code)
        if coupon.expires_on is not None and is_expired(coupon.expires_on, now):
            raise CouponExpired(f"coupon {coupon.code} expired on {coupon.expires_on.isoformat()}")
        if coupon.max_uses is not None and self.uses(coupon.code) >= coupon.max_uses:
            raise CouponNotApplicable(f"coupon {coupon.code} has already been fully redeemed")
        if subtotal_cents < coupon.min_subtotal_cents:
            raise CouponNotApplicable(
                f"coupon {coupon.code} needs a subtotal of at least "
                f"{format_cents(coupon.min_subtotal_cents)}"
            )
        return coupon

    def redeem(self, code: str) -> None:
        """Record one use of ``code`` (called once the order is paid)."""
        coupon = self.get(code)
        self._uses[coupon.code] = self._uses.get(coupon.code, 0) + 1

    def active(self, now: datetime | DateLike | None = None) -> list[Coupon]:
        """Coupons that could currently be used, sorted by code."""
        result = []
        for code in sorted(self._coupons):
            coupon = self._coupons[code]
            if coupon.expires_on is not None and is_expired(coupon.expires_on, now):
                continue
            if coupon.max_uses is not None and self._uses.get(code, 0) >= coupon.max_uses:
                continue
            result.append(coupon)
        return result

    def __len__(self) -> int:
        return len(self._coupons)

    def __contains__(self, code: object) -> bool:
        return isinstance(code, str) and normalize_code(code) in self._coupons
