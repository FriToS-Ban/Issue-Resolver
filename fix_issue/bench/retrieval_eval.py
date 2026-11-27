"""
bench/retrieval_eval.py — Measure retrieval on its own, independent of patch generation.

For every benchmark fix case (ground truth: ``[expected] files`` / ``symbols``) we
materialise the buggy repository and run several retrieval configurations on the issue text:

  keyword           the legacy file_selector (baseline)
  layered-noembed   mentions + symbols + graph + BM25, embeddings disabled (ablation)
  layered-local     + local n-gram vectors (default, offline)
  layered-api       + an OpenAI-compatible embedding model (when configured)

Metrics per configuration: recall@1/3/5, MRR, all-expected-found, symbol recall, mean
number of files returned and mean rendered-context size. No LLM is involved.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fix_issue.bench.cases import Case, materialise
from fix_issue.file_selector import select_relevant_files
from fix_issue.retrieval import EmbeddingCache, LocalNgramEmbedder, RepositoryRetriever, retrieval_metrics
from fix_issue.retrieval.retriever import RetrievalResult, RetrievedFile
from fix_issue.retrieval.mentions import Mentions


def _keyword_result(case: Case, root: Path, top_n: int) -> RetrievalResult:
    files = select_relevant_files(case.issue["title"], case.issue["body"], root, top_n=top_n)
    rfs = [RetrievedFile(f.path, f.score, ["keyword score"]) for f in files]
    return RetrievalResult(rfs, [], Mentions(), [], "none", False)


def evaluate_retrieval(cases: list[Case], configs: dict[str, Any], top_n: int = 5,
                       cache_path: Path | None = None) -> dict[str, Any]:
    """``configs``: name -> "keyword" | None (no embeddings) | EmbeddingBackend instance."""
    cache = EmbeddingCache(cache_path)
    per_case: list[dict[str, Any]] = []
    for case in cases:
        if not case.expected_files:
            continue
        tmp = Path(tempfile.mkdtemp(prefix=f"reteval-{case.id[:20]}-"))
        try:
            materialise(case, tmp)
            row: dict[str, Any] = {"case_id": case.id, "difficulty": case.difficulty,
                                   "expected_files": case.expected_files, "expected_symbols": case.expected_symbols}
            for name, backend in configs.items():
                t0 = time.monotonic()
                if backend == "keyword":
                    res = _keyword_result(case, tmp, top_n)
                    ctx_chars = sum(min(len((tmp / f.path).read_text(encoding="utf-8", errors="replace")), 50_000)
                                    for f in res.files)
                    error = None
                else:
                    # Classes are factories (fresh local embedder fitted per repo); instances are reused.
                    embedder = backend() if isinstance(backend, type) else backend
                    retriever = RepositoryRetriever(tmp, embedder=embedder, cache=cache,
                                                    use_embeddings=embedder is not None)
                    res = retriever.retrieve(case.issue["title"], case.issue["body"], top_n=top_n)
                    ctx_chars = len(retriever.render_context(res, 60_000))
                    error = retriever.embedding_error
                m = retrieval_metrics(res, case.expected_files, case.expected_symbols)
                m.update(seconds=round(time.monotonic() - t0, 2), context_chars=ctx_chars,
                         ranked=[f.path for f in res.files], error=error,
                         reasons={f.path: f.reasons[:3] for f in res.files if f.path in case.expected_files})
                row[name] = m
            per_case.append(row)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    summary: dict[str, Any] = {}
    for name in configs:
        rows = [r[name] for r in per_case if name in r]
        if not rows:
            continue

        def mean(key: str) -> float | None:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        summary[name] = {
            "n": len(rows),
            "recall@1": mean("recall@1"), "recall@3": mean("recall@3"), "recall@5": mean("recall@5"),
            "mrr": mean("mrr"),
            "all_found": round(sum(1 for r in rows if r.get("all_found")) / len(rows), 3),
            "symbol_recall": mean("symbol_recall"),
            "mean_files_returned": mean("returned"),
            "mean_context_chars": mean("context_chars"),
            "errors": sum(1 for r in rows if r.get("error")),
        }
        for d in ("easy", "medium", "hard"):
            drows = [r[name] for r in per_case if r["difficulty"] == d and name in r]
            if drows:
                summary[name][f"recall@5_{d}"] = round(
                    sum(x["recall@5"] for x in drows if x["recall@5"] is not None) / len(drows), 3)
    return {"summary": summary, "cases": per_case, "top_n": top_n}


def render_retrieval_markdown(data: dict[str, Any]) -> str:
    s = data["summary"]
    names = list(s)
    lines = [f"# Retrieval evaluation (top_n={data['top_n']}, {next(iter(s.values()))['n'] if s else 0} cases)", "",
             "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key in ("recall@1", "recall@3", "recall@5", "mrr", "all_found", "symbol_recall",
                "recall@5_easy", "recall@5_medium", "recall@5_hard", "mean_files_returned", "mean_context_chars",
                "errors"):
        vals = []
        for n in names:
            v = s[n].get(key)
            vals.append("—" if v is None else (f"{v:.0f}" if key in ("mean_context_chars", "errors") else f"{v:.2f}"))
        lines.append(f"| {key} | " + " | ".join(vals) + " |")
    failed = [n for n in names if s[n].get("errors")]
    if failed:
        lines += ["", "**Warning:** embedding errors in " + ", ".join(
            f"`{n}` ({s[n]['errors']}/{s[n]['n']} cases)" for n in failed)
            + " — those cases fell back to retrieval without embeddings; see `error` per case in the JSON."]
    lines += ["", "## Misses (expected file not in top-n)", ""]
    for row in data["cases"]:
        for n in names:
            m = row.get(n)
            if m and m.get("recall@5") is not None and m["recall@5"] < 1.0:
                missing = [e for e in row["expected_files"] if e not in m["ranked"]]
                lines.append(f"- `{row['case_id']}` [{n}] missing {missing}; got {m['ranked'][:5]}")
    return "\n".join(lines) + "\n"


def default_configs(api_embedder=None) -> dict[str, Any]:
    cfgs: dict[str, Any] = {
        "keyword": "keyword",
        "layered-noembed": None,
        "layered-local": LocalNgramEmbedder,
    }
    if api_embedder is not None:
        cfgs["layered-api"] = api_embedder
    return cfgs
