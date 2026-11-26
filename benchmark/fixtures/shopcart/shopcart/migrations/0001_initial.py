"""Initial schema: products, stock, coupons and orders."""

from __future__ import annotations

import copy

VERSION = 1

TABLES: dict[str, dict[str, str]] = {
    "products": {
        "sku": "TEXT PRIMARY KEY",
        "name": "TEXT NOT NULL",
        "unit_price": "NUMERIC(12, 4) NOT NULL",
        "weight_kg": "REAL NOT NULL DEFAULT 0",
        "category": "TEXT NOT NULL DEFAULT 'general'",
    },
    "stock": {
        "sku": "TEXT PRIMARY KEY REFERENCES products(sku)",
        "on_hand": "INTEGER NOT NULL DEFAULT 0",
        "reserved": "INTEGER NOT NULL DEFAULT 0",
    },
    "coupons": {
        "code": "TEXT PRIMARY KEY",
        "percent_off": "NUMERIC(5, 2) NOT NULL",
        "expires_on": "DATE",
        "min_subtotal_cents": "INTEGER NOT NULL DEFAULT 0",
        "max_uses": "INTEGER",
        "uses": "INTEGER NOT NULL DEFAULT 0",
    },
    "orders": {
        "order_id": "TEXT PRIMARY KEY",
        "placed_at": "TIMESTAMP NOT NULL",
        "subtotal_cents": "INTEGER NOT NULL",
        "discount_cents": "INTEGER NOT NULL DEFAULT 0",
        "tax_cents": "INTEGER NOT NULL DEFAULT 0",
        "shipping_cents": "INTEGER NOT NULL DEFAULT 0",
        "transaction_id": "TEXT NOT NULL",
        "coupon_code": "TEXT REFERENCES coupons(code)",
    },
}


def upgrade(schema: dict) -> dict:
    result = copy.deepcopy(schema)
    tables = result.setdefault("tables", {})
    for name, columns in TABLES.items():
        if name in tables:
            raise ValueError(f"table {name!r} already exists")
        tables[name] = dict(columns)
    result["version"] = VERSION
    return result


def downgrade(schema: dict) -> dict:
    result = copy.deepcopy(schema)
    tables = result.get("tables", {})
    for name in TABLES:
        tables.pop(name, None)
    result["version"] = VERSION - 1
    return result
