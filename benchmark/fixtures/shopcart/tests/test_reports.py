from datetime import date, datetime

from shopcart import reports
from shopcart.checkout import Checkout
from shopcart.models import Cart


def place(checkout, *lines):
    cart = Cart()
    for product, qty in lines:
        cart.add(product, qty)
    return checkout.place_order(cart, "tok_visa")


def test_summary_and_top_products(inventory, gateway, mug, poster):
    times = iter([datetime(2024, 5, 1, 10, 0), datetime(2024, 5, 2, 9, 30)])
    checkout = Checkout(inventory, gateway, tax_rate="0.1", clock=lambda: next(times))
    first = place(checkout, (mug, 2))
    second = place(checkout, (mug, 1), (poster, 1))
    summary = reports.summarize([first, second])
    assert summary.order_count == 2
    assert summary.gross_cents == 2500 + 1250 + 1999
    assert summary.tax_cents == first.breakdown.tax_cents + second.breakdown.tax_cents
    assert reports.top_products([first, second]) == [("MUG-01", 3), ("PST-A2", 1)]
    assert list(reports.revenue_by_day([first, second])) == [date(2024, 5, 1), date(2024, 5, 2)]
    assert reports.orders_between([first, second], "2024-05-02", "2024-05-31") == [second]


def test_receipt_lines(inventory, gateway, fixed_now, mug):
    checkout = Checkout(inventory, gateway, tax_rate="0.08", clock=lambda: fixed_now)
    order = place(checkout, (mug, 2))
    assert reports.receipt_lines(order) == [
        "Order ORD-00001",
        "2 x Enamel mug  $25.00",
        "Subtotal  $25.00",
        "Tax  $2.00",
        "Shipping  $4.99",
        "Total  $31.99",
    ]


def test_format_summary_empty():
    text = reports.format_summary(reports.summarize([]))
    assert "Orders: 0" in text
    assert "Net revenue: $0.00" in text
