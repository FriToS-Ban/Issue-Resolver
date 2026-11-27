# Benchmark results — profile `baseline`

- Model: `nvidia/nemotron-3-ultra-550b-a55b` · critic: `mistralai/mistral-large-4-1t-a52b` · temperature 0.0
- Embeddings: `openai_compatible` nvidia/nemotron-3.5-embed-1b
- Cases: 23 (20 fix, 3 triage) · tool commit `f87c42d`
- Started 2026-11-27T22:13:59 · total 1674.5s · 66 LLM calls

| Metric | Value |
|---|---|
| Overall success | 17/23 (74%) |
| Fix-case success | 75% |
| Triage-case success | 67% |
| Classification accuracy | 96% |
| Retrieval recall@5 (initial retrieval, cases where it ran) | 82% |
| Patch application rate | 90% |
| Edit-round apply rate | 78% |
| Visible test pass rate | 100% |
| **Issue actually fixed (hidden tests)** | 80% |
| Tests pass but issue NOT fixed | 2 |
| First-attempt success | 75% |
| Scope respected | 95% |
| Mean attempts (fix cases) | 1.15 |
| Critique verdicts | {'none': 20} |
| Critique false approvals | 0 |
| Manual reviews pending | 0 |

## By difficulty

| Difficulty | Success |
|---|---|
| easy | 8/9 (89%) |
| medium | 4/7 (57%) |
| hard | 5/7 (71%) |

## Failure categories

- `tests_pass_but_not_fixed`: 2
- `cannot_fix+retrieval_miss`: 2
- `triage_misclassified`: 1
- `scope_not_respected+retrieval_miss`: 1

## Per case

| Case | Diff. | Class. | Retrieval | Patch | Visible | Hidden | Scope | Att. | Review | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| `shopcart-e1-shipping-boundary` | easy | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `shopcart-e2-negative-money-format` | easy | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `shopcart-e3-unknown-sku-keyerror` | easy | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `taskq-e1-interval-seconds-unit` | easy | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `taskq-e2-backoff-max-cap` | easy | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `taskq-e3-from-dict-missing-tags` | easy | fixable | 100% | ✅ | ✅ | ❌ | ✅ | 1 | — | ❌ tests_pass_but_not_fixed |
| `textkit-e1-slug-edge-separators` | easy | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `textkit-e2-truncate-length` | easy | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `textkit-e3-title-case-first-word` | easy | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `shopcart-m1-tax-after-discount` | medium | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `shopcart-m2-expired-coupon-accepted` | medium | fixable | 0% | ✅ | ✅ | ✅ | ❌ | 3 | — | ❌ scope_not_respected+retrieval_miss |
| `shopcart-m3-manager-discount-limit` | medium | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `taskq-m1-priority-fifo-ties` | medium | fixable | 0% | ❌ | ✅ | ❌ | ✅ | 1 | — | ❌ cannot_fix+retrieval_miss |
| `taskq-m2-storage-datetime-roundtrip` | medium | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `textkit-m1-top-words-normalization` | medium | fixable | 0% | ❌ | ✅ | ❌ | ✅ | 1 | — | ❌ cannot_fix+retrieval_miss |
| `textkit-m2-csv-quoted-fields` | medium | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `shopcart-h1-reservation-leak-on-payment-failure` | hard | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `shopcart-h2-cent-rounding-line-items` | hard | fixable | 50% | ✅ | ✅ | ❌ | ✅ | 2 | — | ❌ tests_pass_but_not_fixed |
| `shopcart-h3-oauth-feature-request` | hard | out_of_scope | — | ❌ | — | — | ✅ | 0 | — | ✅ |
| `shopcart-h4-duplicate-expired-coupon` | hard | fixable ⚠ | — | ✅ | — | — | ✅ | 1 | — | ❌ triage_misclassified |
| `taskq-h1-nested-config-defaults` | hard | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
| `taskq-h2-dst-wrong-time` | hard | needs_info | — | ❌ | — | — | ✅ | 0 | — | ✅ |
| `textkit-h1-template-escapes-and-missing-keys` | hard | fixable | 100% | ✅ | ✅ | ✅ | ✅ | 1 | — | ✅ |
