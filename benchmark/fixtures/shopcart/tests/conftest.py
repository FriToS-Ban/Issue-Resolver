from datetime import datetime
from decimal import Decimal

import pytest

from shopcart.discounts import Coupon, CouponBook
from shopcart.inventory import Inventory
from shopcart.models import Product
from shopcart.payments.gateway import FakeGateway


@pytest.fixture
def mug():
    return Product("MUG-01", "Enamel mug", Decimal("12.50"), weight_kg=0.4)


@pytest.fixture
def poster():
    return Product("PST-A2", "A2 poster", Decimal("19.99"), weight_kg=0.2)


@pytest.fixture
def kettle():
    return Product("KTL-02", "Stovetop kettle", Decimal("39.00"), weight_kg=1.3)


@pytest.fixture
def inventory():
    return Inventory({"MUG-01": 10, "PST-A2": 3, "KTL-02": 2})


@pytest.fixture
def gateway():
    return FakeGateway()


@pytest.fixture
def fixed_now():
    return datetime(2024, 5, 14, 12, 30)


@pytest.fixture
def coupons():
    return CouponBook(
        [
            Coupon("WELCOME10", Decimal("10")),
            Coupon("SUMMER25", Decimal("25"), expires_on="2024-08-31"),
            Coupon("BIGSPENDER", Decimal("15"), min_subtotal_cents=10_000),
        ]
    )
