from datetime import datetime
from decimal import Decimal

from shopcart import Cart, Checkout, Product
from shopcart.auth.permissions import Role, User
from shopcart.discounts import Coupon, CouponBook
from shopcart.inventory import Inventory
from shopcart.payments.gateway import FakeGateway

LAMP = Product("LAMP-1", "Desk lamp", Decimal("50.00"), weight_kg=1.2)
MUG = Product("MUG-01", "Enamel mug", Decimal("12.50"), weight_kg=0.4)


def make(tax_rate="0.08"):
    gateway = FakeGateway()
    checkout = Checkout(
        Inventory({"LAMP-1": 5, "MUG-01": 20}),
        gateway,
        coupons=CouponBook([Coupon("WELCOME10", 10), Coupon("FREEBIE", 100)]),
        tax_rate=tax_rate,
        clock=lambda: datetime(2024, 5, 1, 12, 0),
    )
    return checkout, gateway


def lamp_cart():
    cart = Cart()
    cart.add(LAMP, 2)
    return cart


def test_coupon_reduces_taxable_amount():
    checkout, _ = make()
    quote = checkout.quote(lamp_cart(), "WELCOME10")
    assert quote.subtotal_cents == 10_000
    assert quote.discount_cents == 1_000
    assert quote.tax_cents == 720
    assert quote.total_cents == 9_720


def test_manual_discount_reduces_taxable_amount():
    checkout, _ = make()
    quote = checkout.quote(lamp_cart(), staff=User("priya", Role.MANAGER), manual_discount=15)
    assert quote.discount_cents == 1_500
    assert quote.tax_cents == 680


def test_full_discount_means_no_tax():
    checkout, _ = make()
    quote = checkout.quote(lamp_cart(), "FREEBIE")
    assert quote.tax_cents == 0
    assert quote.total_cents == quote.shipping_cents


def test_undiscounted_order_unchanged():
    checkout, _ = make()
    assert checkout.quote(lamp_cart()).tax_cents == 800


def test_shipping_is_not_taxed_and_rounding():
    checkout, _ = make(tax_rate="0.0725")
    cart = Cart()
    cart.add(MUG, 3)
    quote = checkout.quote(cart, "WELCOME10")
    assert quote.subtotal_cents == 3_750
    assert quote.discount_cents == 375
    assert quote.tax_cents == 245
    assert quote.shipping_cents == 899


def test_charged_amount_uses_discounted_tax():
    checkout, gateway = make()
    order = checkout.place_order(lamp_cart(), "tok_visa", coupon_code="WELCOME10")
    assert order.breakdown.tax_cents == 720
    assert gateway.charges[-1].amount_cents == 9_720
