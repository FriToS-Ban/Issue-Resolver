"""Stock levels and reservations.

Stock moves through two stages during checkout: units are first *reserved*
(promised to an order that is being paid for) and then *committed* once the
payment succeeds, at which point they leave the warehouse.
"""

from __future__ import annotations

from typing import Mapping


class InventoryError(Exception):
    """Raised for invalid stock operations."""


class OutOfStock(InventoryError):
    """Raised when an order asks for more units than are available."""

    def __init__(self, sku: str, requested: int, available: int) -> None:
        super().__init__(f"{sku}: requested {requested}, only {available} available")
        self.sku = sku
        self.requested = requested
        self.available = available


def _check_quantity(quantity: int) -> None:
    if not isinstance(quantity, int) or quantity < 1:
        raise InventoryError(f"quantity must be a positive integer, got {quantity!r}")


class Inventory:
    """In-memory stock ledger keyed by SKU."""

    def __init__(self, stock: Mapping[str, int] | None = None) -> None:
        self._on_hand: dict[str, int] = dict(stock or {})
        self._reserved: dict[str, int] = {}

    def add_stock(self, sku: str, quantity: int) -> None:
        """Receive ``quantity`` new units of ``sku`` into the warehouse."""
        _check_quantity(quantity)
        self._on_hand[sku] = self._on_hand.get(sku, 0) + quantity

    def on_hand(self, sku: str) -> int:
        """Physical units in the warehouse, including reserved ones."""
        return self._on_hand.get(sku, 0)

    def reserved(self, sku: str) -> int:
        """Units currently held for orders that are being paid for."""
        return self._reserved.get(sku, 0)

    def available(self, sku: str) -> int:
        """Units that can still be promised to new orders."""
        return self._on_hand.get(sku, 0) - self._reserved.get(sku, 0)

    def reserve(self, sku: str, quantity: int) -> None:
        """Hold ``quantity`` units of ``sku`` for an order."""
        _check_quantity(quantity)
        free = self.available(sku)
        if quantity > free:
            raise OutOfStock(sku, quantity, free)
        self._reserved[sku] = self._reserved.get(sku, 0) + quantity

    def release(self, sku: str, quantity: int) -> None:
        """Give back units that were reserved but will not be sold."""
        _check_quantity(quantity)
        held = self._reserved.get(sku, 0)
        if quantity > held:
            raise InventoryError(f"cannot release {quantity} x {sku}: only {held} reserved")
        remaining = held - quantity
        if remaining:
            self._reserved[sku] = remaining
        else:
            self._reserved.pop(sku, None)

    def commit(self, sku: str, quantity: int) -> None:
        """Turn a reservation into a sale; the units leave the warehouse."""
        self.release(sku, quantity)
        self._on_hand[sku] = self._on_hand.get(sku, 0) - quantity

    def low_stock(self, threshold: int = 5) -> list[str]:
        """SKUs whose available quantity is at or below ``threshold``."""
        return sorted(sku for sku in self._on_hand if self.available(sku) <= threshold)

    def snapshot(self) -> dict[str, dict[str, int]]:
        """A copy of the ledger, handy for reports and debugging."""
        return {
            sku: {"on_hand": qty, "reserved": self._reserved.get(sku, 0)}
            for sku, qty in sorted(self._on_hand.items())
        }
