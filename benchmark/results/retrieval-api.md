# Retrieval evaluation (top_n=5, 20 cases)

| Metric | keyword | layered-noembed | layered-local | layered-api |
|---|---|---|---|---|
| recall@1 | 0.25 | 0.60 | 0.60 | 0.60 |
| recall@3 | 0.70 | 0.80 | 0.80 | 0.80 |
| recall@5 | 0.82 | 0.93 | 0.93 | 0.93 |
| mrr | 0.50 | 0.71 | 0.71 | 0.71 |
| all_found | 0.80 | 0.90 | 0.90 | 0.90 |
| symbol_recall | 0.00 | 0.69 | 0.64 | 0.69 |
| recall@5_easy | 1.00 | 1.00 | 1.00 | 1.00 |
| recall@5_medium | 0.57 | 0.86 | 0.86 | 0.86 |
| recall@5_hard | 0.88 | 0.88 | 0.88 | 0.88 |
| mean_files_returned | 5.00 | 3.25 | 3.20 | 3.50 |
| mean_context_chars | 13924 | 11450 | 11221 | 12274 |
| errors | 0 | 0 | 0 | 0 |

## Misses (expected file not in top-n)

- `shopcart-h2-cent-rounding-line-items` [keyword] missing ['shopcart/utils/money.py']; got ['shopcart/checkout.py', 'tests/test_checkout.py', 'shopcart/reports.py', 'README.md', 'shopcart/pricing.py']
- `shopcart-h2-cent-rounding-line-items` [layered-noembed] missing ['shopcart/pricing.py']; got ['shopcart/payments/gateway.py', 'shopcart/inventory.py', 'shopcart/models.py', 'shopcart/checkout.py', 'shopcart/utils/money.py']
- `shopcart-h2-cent-rounding-line-items` [layered-local] missing ['shopcart/pricing.py']; got ['shopcart/payments/gateway.py', 'shopcart/inventory.py', 'shopcart/checkout.py', 'shopcart/models.py', 'shopcart/utils/money.py']
- `shopcart-h2-cent-rounding-line-items` [layered-api] missing ['shopcart/pricing.py']; got ['shopcart/payments/gateway.py', 'shopcart/inventory.py', 'shopcart/models.py', 'shopcart/checkout.py', 'shopcart/utils/money.py']
- `shopcart-m2-expired-coupon-accepted` [keyword] missing ['shopcart/utils/dates.py']; got ['shopcart/checkout.py', 'tests/test_checkout.py', 'shopcart/discounts.py', 'README.md', 'tests/test_discounts.py']
- `shopcart-m2-expired-coupon-accepted` [layered-noembed] missing ['shopcart/utils/dates.py']; got ['shopcart/discounts.py', 'shopcart/payments/gateway.py', 'shopcart/inventory.py', 'shopcart/checkout.py', 'shopcart/pricing.py']
- `shopcart-m2-expired-coupon-accepted` [layered-local] missing ['shopcart/utils/dates.py']; got ['shopcart/discounts.py', 'shopcart/payments/gateway.py', 'shopcart/inventory.py', 'shopcart/checkout.py', 'shopcart/pricing.py']
- `shopcart-m2-expired-coupon-accepted` [layered-api] missing ['shopcart/utils/dates.py']; got ['shopcart/discounts.py', 'shopcart/payments/gateway.py', 'shopcart/inventory.py', 'shopcart/checkout.py', 'shopcart/pricing.py']
- `taskq-m1-priority-fifo-ties` [keyword] missing ['taskq/queue.py']; got ['README.md', 'tests/test_api.py', 'tests/test_worker.py', 'taskq/api.py', 'taskq/worker.py']
- `textkit-m1-top-words-normalization` [keyword] missing ['textkit/tokenize.py']; got ['textkit/cli.py', 'textkit/stats.py', 'textkit/__init__.py', 'tests/test_stats.py', 'README.md']
