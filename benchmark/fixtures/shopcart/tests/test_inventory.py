import pytest

from shopcart.inventory import Inventory, InventoryError, OutOfStock


def test_available_counts_reservations():
    inv = Inventory({"A": 5})
    inv.reserve("A", 2)
    assert inv.available("A") == 3
    assert inv.on_hand("A") == 5
    assert inv.reserved("A") == 2


def test_reserve_more_than_available():
    inv = Inventory({"A": 1})
    with pytest.raises(OutOfStock) as excinfo:
        inv.reserve("A", 2)
    assert excinfo.value.available == 1


def test_release_returns_units():
    inv = Inventory({"A": 4})
    inv.reserve("A", 3)
    inv.release("A", 3)
    assert inv.available("A") == 4
    assert inv.reserved("A") == 0


def test_release_more_than_reserved_is_an_error():
    inv = Inventory({"A": 4})
    inv.reserve("A", 1)
    with pytest.raises(InventoryError):
        inv.release("A", 2)


def test_commit_removes_stock():
    inv = Inventory({"A": 4})
    inv.reserve("A", 3)
    inv.commit("A", 3)
    assert inv.on_hand("A") == 1
    assert inv.available("A") == 1


def test_add_stock_for_new_sku():
    inv = Inventory()
    inv.add_stock("NEW", 7)
    assert inv.on_hand("NEW") == 7


def test_quantity_validation():
    with pytest.raises(InventoryError):
        Inventory({"A": 1}).reserve("A", 0)


def test_low_stock():
    inv = Inventory({"A": 2, "B": 50, "C": 5})
    assert inv.low_stock() == ["A", "C"]


def test_snapshot():
    inv = Inventory({"B": 1, "A": 2})
    inv.reserve("A", 1)
    assert inv.snapshot() == {"A": {"on_hand": 2, "reserved": 1}, "B": {"on_hand": 1, "reserved": 0}}
