# Diagnostic runs (pre-fix, not the reported result)

These two runs were made on 2026-11-28 at tool commit `1bc964f`, before the fixes on this branch
from the code and security reviews. They are kept because they are the evidence for several
of those fixes. **The headline benchmark numbers are in `../full-20261128.md` and
`../baseline-20261128-final.md`**, run on the final code with the final grader.

| File | Profile | Success | Hidden tests pass (fix cases) |
|---|---|---|---|
| `baseline-20261128.json` | baseline | 16/23 | 80% |
| `full-20261128-r1.json` | full | 13/23 | 90% |

What these runs showed, and what changed because of it:

- **Grader counted regression tests against the fix-size cap.** 7 of the full run's 10 failures
  were `scope_not_respected` on cases whose hidden tests passed: the pipeline added a regression
  test (as its prompt asks) and the grader counted those lines against caps of 10–15 lines. The
  cap now applies to non-test lines (`fix(bench): apply the fix-size cap to source lines`).
- **Phantom blank line at end of file** in rendered context made SEARCH blocks miss the end of
  the file (shopcart-h2, shopcart-h4). Fixed in retrieval rendering, recovery excerpts and the
  edit matcher.
- **Triage forbade `tests/**`** for shopcart-h2 and then blocked the regression test → the run
  ended `scope_violation`. Triage-suggested forbidden globs no longer apply to test files.
- **A failed revision round discarded a reviewed patch** (taskq-e3: NEEDS_REVISION, then the
  revision's SEARCH missed and the run ended `exhausted:missing_search`). The reviewed candidate
  is now published for HUMAN_REVIEW instead.
- **The embedding API refused some code chunks** (NIM HTTP 422 "content safety screening"), which
  switched embeddings off for the whole repository in every case of these runs. The embedder now
  drops only the refused inputs.
- **shopcart-h4 (duplicate)** failed in both runs: the duplicate threshold (0.82) was unreachable.
  It was recalibrated on a separate labelled set; the benchmark pair still scores below the new
  threshold (0.46 vs 0.53 on NIM), so this case is still expected to fail.

The per-case run records referenced by `meta.runs_dir` are not committed (`benchmark/results/runs/`
is gitignored).
