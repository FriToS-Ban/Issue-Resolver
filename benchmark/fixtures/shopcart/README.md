# shopcart

Cart, pricing, coupon and checkout primitives for small web shops.

`shopcart` is deliberately framework-free: you bring the web layer, it brings the
arithmetic and the bookkeeping.

```python
from decimal import Decimal
from shopcart import Cart, Checkout, Product
from shopcart.discounts import Coupon, CouponBook
from shopcart.inventory import Inventory
from shopcart.payments.gateway import FakeGateway

mug = Product("MUG-01", "Enamel mug", Decimal("12.50"), weight_kg=0.4)
cart = Cart()
cart.add(mug, 2)

checkout = Checkout(
    Inventory({"MUG-01": 10}),
    FakeGateway(),
    coupons=CouponBook([Coupon("WELCOME10", 10)]),
    tax_rate="0.08",
)
order = checkout.place_order(cart, "tok_visa", coupon_code="welcome10")
print(order.total_cents)
```

## Money

All arithmetic is done in integer cents. Dollar amounts only appear at the edges
(catalogue prices, receipts, reports). `shopcart.utils.money` has the conversion
and formatting helpers.

## Pricing rules

* Each line is priced as `unit_price x quantity`.
* Coupon and manual (staff) discounts are percentages of the merchandise subtotal.
* Sales tax is charged on the merchandise subtotal after discounts. Shipping is
  not taxed.
* Orders of $75.00 or more (after discounts) ship free with standard shipping.

## Shipping rates

| Parcel weight      | Standard |
|--------------------|----------|
| up to 1 kg         | $4.99    |
| up to 5 kg         | $8.99    |
| up to 20 kg        | $14.99   |
| each kg over 20 kg | +$0.85   |

Express shipping costs 1.5x the standard rate and is never free.

## Coupons

Coupons are percentage discounts with an optional expiry date, minimum subtotal and
usage cap. A coupon stays valid through the whole of its expiry date.

## Staff discounts

Staff can apply a manual discount at the till, up to a limit set by their role:
clerks 10%, managers 20%, admins any amount.

## Running the tests

```
pip install -e .[test]
pytest
```
