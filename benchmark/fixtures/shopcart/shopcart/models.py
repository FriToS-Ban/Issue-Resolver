"""Catalogue and cart data structures."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Iterator

from shopcart.utils.money import Number, to_decimal


@dataclass(frozen=True)
class Product:
    """A sellable catalogue item.

    ``unit_price`` is in dollars and may carry sub-cent precision for items sold
    in bulk (for example ``Decimal("0.335")`` per screw).
    """

    sku: str
    name: str
    unit_price: Decimal
    weight_kg: float = 0.0
    category: str = "general"

    def __post_init__(self) -> None:
        if not self.sku or not self.sku.strip():
            raise ValueError("a product needs a SKU")
        price = to_decimal(self.unit_price)
        if price < 0:
            raise ValueError(f"{self.sku}: price cannot be negative")
        if self.weight_kg < 0:
            raise ValueError(f"{self.sku}: weight cannot be negative")
        object.__setattr__(self, "unit_price", price)


@dataclass
class LineItem:
    """One product line in a cart."""

    product: Product
    quantity: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, int) or self.quantity < 1:
            raise ValueError(f"quantity must be a positive integer, got {self.quantity!r}")

    @property
    def sku(self) -> str:
        return self.product.sku

    @property
    def weight_kg(self) -> float:
        return self.product.weight_kg * self.quantity


class Cart:
    """An ordered collection of line items, one per SKU."""

    def __init__(self, items: Iterable[LineItem] = ()) -> None:
        self._lines: dict[str, LineItem] = {}
        for item in items:
            self.add(item.product, item.quantity)

    def add(self, product: Product, quantity: int = 1) -> LineItem:
        """Add ``quantity`` units of ``product``, merging with an existing line."""
        existing = self._lines.get(product.sku)
        if existing is None:
            line = LineItem(product, quantity)
        else:
            line = LineItem(product, existing.quantity + quantity)
        self._lines[product.sku] = line
        return line

    def remove(self, sku: str) -> None:
        if sku not in self._lines:
            raise KeyError(sku)
        del self._lines[sku]

    def set_quantity(self, sku: str, quantity: int) -> None:
        """Change a line's quantity; a quantity of zero removes the line."""
        if quantity == 0:
            self.remove(sku)
            return
        line = self._lines[sku]
        self._lines[sku] = LineItem(line.product, quantity)

    def clear(self) -> None:
        self._lines.clear()

    @property
    def items(self) -> list[LineItem]:
        return list(self._lines.values())

    @property
    def total_quantity(self) -> int:
        return sum(line.quantity for line in self._lines.values())

    @property
    def total_weight_kg(self) -> float:
        return round(sum(line.weight_kg for line in self._lines.values()), 3)

    def is_empty(self) -> bool:
        return not self._lines

    def __len__(self) -> int:
        return len(self._lines)

    def __iter__(self) -> Iterator[LineItem]:
        return iter(self.items)

    def __contains__(self, sku: object) -> bool:
        return sku in self._lines


def make_product(sku: str, name: str, price: Number, **extra: object) -> Product:
    """Convenience constructor accepting prices as str, int or float."""
    return Product(sku, name, to_decimal(price), **extra)  # type: ignore[arg-type]
