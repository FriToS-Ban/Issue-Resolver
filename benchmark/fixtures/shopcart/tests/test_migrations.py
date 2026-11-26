import importlib

import pytest

from shopcart import migrations


def test_initial_migration_creates_tables():
    schema = migrations.apply_all()
    assert schema["version"] == 1
    assert set(schema["tables"]) == {"products", "stock", "coupons", "orders"}
    assert schema["tables"]["coupons"]["expires_on"] == "DATE"


def test_upgrade_twice_fails():
    initial = importlib.import_module("shopcart.migrations.0001_initial")
    schema = initial.upgrade({})
    with pytest.raises(ValueError):
        initial.upgrade(schema)


def test_downgrade_removes_tables():
    initial = importlib.import_module("shopcart.migrations.0001_initial")
    schema = initial.downgrade(initial.upgrade({"tables": {"audit": {}}}))
    assert schema == {"tables": {"audit": {}}, "version": 0}
