"""
duplicates.py — Semantic duplicate detection against recently resolved issues.

Candidates come from three sources (all plain dicts: number, title, body, state,
resolved_by, source):

  recent_resolved_from_github()  recently closed issues of the repo (pull requests skipped)
  resolved_from_history()        past fix-issue runs on the same repo that opened a PR
  "provided"                     anything the caller passes in directly (tests, benchmark)

``find_similar_issues`` ranks them by a blend of embedding cosine similarity (title weighted
twice + first 1500 chars of the body) and title-token Jaccard (so short, near-identical
titles still match). The result feeds the triage prompt as evidence, and
``triage.apply_duplicate_evidence`` only accepts a "duplicate" verdict when a listed issue
clears ``[pipeline] duplicate_threshold`` for the scorer that produced its similarity
(``SimilarIssue.scorer``: the scales of the local n-gram scorer and of API embedding models
differ, so thresholds are per scorer — see ``config.DUPLICATE_THRESHOLDS``). Lookups never
raise: failures yield [].

Scorer choice: a semantic (API) backend is used when given; otherwise — no backend, a local
backend, or a semantic one that fails — a fresh LocalNgramEmbedder is fitted on the query and
its candidates (local vectors depend on a fitted IDF, and a caller's shared embedder is fitted
on code, not issue text). An input the API refuses (empty vector) is scored locally instead of
as cosine 0, and its result says so in ``scorer``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import requests

from fix_issue.github_client import GitHubError
from fix_issue.retrieval.embeddings import EmbeddingBackend, EmbeddingCache, LocalNgramEmbedder, cosine
from fix_issue.retrieval.text import stem, tokenize

TITLE_WEIGHT = 0.25          # share of the score from title-token Jaccard
BODY_CHARS = 1500
EXCERPT_CHARS = 300


@dataclass
class SimilarIssue:
    number: int
    title: str
    similarity: float
    state: str
    source: str                      # "github" | "history" | "provided"
    resolved_by: str | None = None   # PR url (history) when known
    body_excerpt: str = ""
    scorer: str = ""                 # embedder that produced `similarity` (selects the threshold)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).strip().lstrip("#"))
    except (TypeError, ValueError):
        return None


def _doc_text(title: str, body: str) -> str:
    """Title repeated twice (weighting) + the first BODY_CHARS of the body."""
    return f"{title}\n{title}\n\n{(body or '')[:BODY_CHARS]}"


def _title_tokens(title: str) -> set[str]:
    return {stem(t) for t in tokenize(title or "")}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _excerpt(body: str) -> str:
    return re.sub(r"\s+", " ", body or "").strip()[:EXCERPT_CHARS].rstrip()


def _embed_local(texts: list[str], dim: int = 768) -> tuple[list[list[float]], str]:
    """A fresh LocalNgramEmbedder fitted on exactly these texts (never a caller's shared one)."""
    local = LocalNgramEmbedder(dim=dim)
    local.fit(texts)
    return local.embed(texts, "query"), local.name


def _embed_all(
    texts: list[str],
    embedder: EmbeddingBackend | None,
    cache: EmbeddingCache | None,
) -> tuple[list[list[float]], str]:
    """(vectors, scorer name). Semantic backends are used directly; anything else — or a failing
    semantic backend — gets a fresh LocalNgramEmbedder fitted on ``texts``."""
    if embedder is not None and getattr(embedder, "semantic", False):
        try:
            # Issue-to-issue similarity is symmetric: embed both sides with the same kind.
            if cache is not None:
                vecs = cache.embed(embedder, texts, kind="query")
            else:
                vecs = embedder.embed(texts, "query")
            if len(vecs) == len(texts):
                return vecs, str(getattr(embedder, "name", "semantic"))
        except Exception:  # network/API failure — fall back to the offline embedder
            pass
    dim = embedder.dim if isinstance(embedder, LocalNgramEmbedder) else 768
    return _embed_local(texts, dim)


def find_similar_issues(
    issue: dict[str, Any],
    candidates: list[dict[str, Any]],
    embedder: EmbeddingBackend | None = None,
    top_k: int = 3,
    min_similarity: float = 0.35,
    cache: EmbeddingCache | None = None,
) -> list[SimilarIssue]:
    """
    Rank ``candidates`` by similarity to ``issue``; return at most ``top_k`` with
    similarity >= ``min_similarity``, highest first.

    similarity = 0.75 * cosine(embed(query), embed(candidate)) + 0.25 * title Jaccard,
    clamped to [0, 1]. The issue itself (same number) and candidates without a number
    are excluded; duplicate numbers are merged (first wins, missing resolved_by filled).
    Each result records the scorer that produced its similarity; when a semantic backend
    refuses the query, every candidate is scored locally, and when it refuses a candidate,
    that candidate is.
    """
    own = _as_number(issue.get("number"))
    pool: dict[int, dict[str, Any]] = {}
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        num = _as_number(c.get("number"))
        if num is None or num == own:
            continue
        if not (c.get("title") or c.get("body")):
            continue
        if num in pool:
            if not pool[num].get("resolved_by") and c.get("resolved_by"):
                pool[num] = {**pool[num], "resolved_by": c["resolved_by"]}
            continue
        pool[num] = c
    if not pool or top_k <= 0:
        return []

    q_title = str(issue.get("title") or "")
    q_text = _doc_text(q_title, str(issue.get("body") or ""))
    items = list(pool.items())
    texts = [q_text] + [_doc_text(str(c.get("title") or ""), str(c.get("body") or "")) for _, c in items]
    vecs, scorer = _embed_all(texts, embedder, cache)
    # A semantic backend may refuse single inputs (empty vector, e.g. content screening):
    # score those on the local scale rather than as cosine 0 — and the whole pool locally
    # when the query itself was refused.
    local: tuple[list[list[float]], str] | None = None
    if not all(vecs):
        local = _embed_local(texts)
        if not vecs[0]:
            vecs, scorer = local
            local = None

    q_tokens = _title_tokens(q_title)
    out: list[SimilarIssue] = []
    for i, (num, c) in enumerate(items, start=1):
        if vecs[i] or local is None:
            cos, item_scorer = cosine(vecs[0], vecs[i]), scorer
        else:
            cos, item_scorer = cosine(local[0][0], local[0][i]), local[1]
        jac = _jaccard(q_tokens, _title_tokens(str(c.get("title") or "")))
        sim = max(0.0, min(1.0, (1 - TITLE_WEIGHT) * cos + TITLE_WEIGHT * jac))
        if sim < min_similarity:
            continue
        out.append(SimilarIssue(
            number=num,
            title=str(c.get("title") or ""),
            similarity=round(sim, 4),
            state=str(c.get("state") or "closed"),
            source=str(c.get("source") or "provided"),
            resolved_by=c.get("resolved_by"),
            body_excerpt=_excerpt(str(c.get("body") or "")),
            scorer=item_scorer,
        ))
    out.sort(key=lambda s: (-s.similarity, s.number))
    return out[:top_k]


def precision_first_threshold(scores: Iterable[float], is_duplicate: Iterable[bool]) -> float | None:
    """
    Calibration rule behind ``config.DUPLICATE_THRESHOLDS``: among thresholds with zero false
    positives on the labelled ``scores``, those with the highest recall form the interval
    (highest non-duplicate score, lowest duplicate score above it]; return its midpoint.
    None when no duplicate scores above every non-duplicate (or a class is missing).
    Pass the union of scores from several conditions (e.g. candidate-pool sizes) to get one
    threshold that holds in all of them.
    """
    pairs = list(zip(scores, is_duplicate))
    neg = [s for s, d in pairs if not d]
    if not neg or not any(d for _, d in pairs):
        return None
    top_neg = max(neg)
    above = [s for s, d in pairs if d and s > top_neg]
    return (top_neg + min(above)) / 2 if above else None


def recent_resolved_from_github(gh: Any, repo: str, limit: int = 50) -> list[dict[str, Any]]:
    """
    Recently closed issues of ``repo`` via ``gh._get`` (pull requests and issues closed as
    "not_planned" are skipped). Any API/network failure returns [].
    """
    try:
        items = gh._get(
            f"/repos/{repo}/issues",
            params={"state": "closed", "per_page": max(1, min(int(limit), 100)), "sort": "updated"},
        )
    except (GitHubError, requests.RequestException, ValueError):
        return []
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict) or "pull_request" in it:
            continue
        if it.get("state_reason") == "not_planned":
            continue
        num = _as_number(it.get("number"))
        if num is None:
            continue
        out.append({
            "number": num,
            "title": it.get("title") or "",
            "body": it.get("body") or "",
            "state": it.get("state") or "closed",
            "resolved_by": None,
            "source": "github",
            "url": it.get("html_url"),
            "closed_at": it.get("closed_at"),
        })
    return out


def resolved_from_history(records: list[dict[str, Any]], repo: str) -> list[dict[str, Any]]:
    """
    Issues of ``repo`` that past fix-issue runs resolved (outcome result "pr_opened").
    ``records`` are run-history documents (newest first, as ``run_history.list_runs``
    returns them); the first record per issue number wins.
    """
    want = (repo or "").lower()
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for r in records or []:
        if not isinstance(r, dict) or str(r.get("repo") or "").lower() != want:
            continue
        outcome = r.get("outcome") or {}
        if outcome.get("result") != "pr_opened":
            continue
        issue = r.get("issue") or {}
        num = _as_number(r.get("issue_number") if r.get("issue_number") is not None else issue.get("number"))
        if num is None or num in seen:
            continue
        seen.add(num)
        out.append({
            "number": num,
            "title": issue.get("title") or "",
            "body": issue.get("body") or "",
            "state": "pr_opened",
            "resolved_by": outcome.get("pr_url"),
            "source": "history",
        })
    return out
