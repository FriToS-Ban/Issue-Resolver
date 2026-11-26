import pytest

from shopcart.auth.permissions import (
    PermissionDenied,
    Role,
    User,
    can_apply_discount,
    can_refund,
    max_discount_percent,
    require_discount,
)

clerk = User("sam", Role.CLERK)
manager = User("priya", Role.MANAGER)
admin = User("root", Role.ADMIN)


def test_clerk_small_discount():
    assert can_apply_discount(clerk, 5)


def test_clerk_cannot_exceed_limit():
    assert not can_apply_discount(clerk, 15)


def test_manager_discounts():
    assert can_apply_discount(manager, 15)
    assert not can_apply_discount(manager, 25)


def test_admin_can_discount_a_lot():
    assert can_apply_discount(admin, 75)


def test_negative_discount_never_allowed():
    assert not can_apply_discount(admin, -5)


def test_inactive_user_has_no_discount_rights():
    retired = User("old", Role.MANAGER, active=False)
    assert max_discount_percent(retired) == 0
    assert not can_apply_discount(retired, 5)


def test_require_discount_raises():
    with pytest.raises(PermissionDenied, match="limit is 10%"):
        require_discount(clerk, 12)


def test_refund_limits():
    assert can_refund(clerk, 5_000)
    assert not can_refund(clerk, 5_001)
    assert can_refund(admin, 10_000_000)
    assert not can_refund(User("c", Role.CUSTOMER), 100)
