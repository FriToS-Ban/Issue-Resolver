"""Payment gateway client.

:class:`FakeGateway` mirrors the hosted gateway's behaviour closely enough for
development and tests.  Card tokens drive the outcome:

* ``tok_declined``, ``tok_insufficient_funds``, ``tok_expired_card`` -> :class:`PaymentDeclined`
* ``tok_timeout`` -> :class:`GatewayUnavailable`
* any other ``tok_...`` token is charged successfully
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


class PaymentError(Exception):
    """Base class for anything that stops a charge from going through."""


class PaymentDeclined(PaymentError):
    """The card issuer refused the charge."""

    def __init__(self, reason: str = "card_declined") -> None:
        super().__init__(f"payment declined: {reason}")
        self.reason = reason


class GatewayUnavailable(PaymentError):
    """The gateway could not be reached or timed out."""


@dataclass(frozen=True)
class ChargeReceipt:
    transaction_id: str
    amount_cents: int
    token: str
    reference: str | None = None


class FakeGateway:
    """In-memory stand-in for the hosted payment gateway."""

    DECLINED_TOKENS = frozenset({"tok_declined", "tok_insufficient_funds", "tok_expired_card"})
    UNAVAILABLE_TOKENS = frozenset({"tok_timeout"})

    def __init__(self) -> None:
        self.charges: list[ChargeReceipt] = []
        self.refunds: list[tuple[str, int]] = []
        self._ids = itertools.count(1)

    def charge(self, token: str, amount_cents: int, *, reference: str | None = None) -> ChargeReceipt:
        if amount_cents < 0:
            raise ValueError("cannot charge a negative amount")
        if token in self.DECLINED_TOKENS:
            raise PaymentDeclined(token.removeprefix("tok_"))
        if token in self.UNAVAILABLE_TOKENS:
            raise GatewayUnavailable("payment gateway timed out")
        if not token.startswith("tok_"):
            raise PaymentDeclined("invalid_token")
        receipt = ChargeReceipt(f"txn_{next(self._ids):06d}", amount_cents, token, reference)
        self.charges.append(receipt)
        return receipt

    def refund(self, transaction_id: str, amount_cents: int | None = None) -> None:
        for receipt in self.charges:
            if receipt.transaction_id == transaction_id:
                amount = receipt.amount_cents if amount_cents is None else amount_cents
                if amount > receipt.amount_cents:
                    raise ValueError("refund exceeds the original charge")
                self.refunds.append((transaction_id, amount))
                return
        raise KeyError(transaction_id)

    @property
    def total_charged_cents(self) -> int:
        return sum(r.amount_cents for r in self.charges) - sum(a for _, a in self.refunds)
