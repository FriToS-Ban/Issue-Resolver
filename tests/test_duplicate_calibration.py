"""
Duplicate-threshold calibration (config.DUPLICATE_THRESHOLDS) against tests/data/duplicate_pairs.json:
hand-written (new issue, resolved candidate) pairs from four made-up projects — none taken from the
benchmark — labelled "duplicate" or "distinct" (hard negatives).

The local-scorer tests run offline. The NIM test needs the network and an API key and is opt-in:
    FIX_ISSUE_NIM_TESTS=1 pytest tests/test_duplicate_calibration.py
(the key comes from the environment or the gitignored .env via config.load; it is never printed).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import requests

from fix_issue import config
from fix_issue.config import DUPLICATE_THRESHOLDS, PipelineConfig, duplicate_threshold_for
from fix_issue.duplicates import _doc_text, find_similar_issues, precision_first_threshold
from fix_issue.triage import TriageResult, apply_duplicate_evidence

REPO = Path(__file__).resolve().parents[1]
DATA = json.loads((Path(__file__).parent / "data" / "duplicate_pairs.json").read_text(encoding="utf-8"))
PROJECTS = DATA["projects"]
ALL_RESOLVED = [c for proj in PROJECTS.values() for c in proj["resolved"]]
NEMOTRON = "api:nvidia/nemotron-3.5-embed-1b"


def _get(project: str, kind: str, number: int) -> dict:
    return next(x for x in PROJECTS[project][kind] if x["number"] == number)


def _labelled():
    """(id, project, issue, candidate, is_duplicate) for the listed pairs."""
    for p in DATA["pairs"]:
        yield (p["id"], p["project"], _get(p["project"], "issues", p["issue"]),
               _get(p["project"], "resolved", p["candidate"]), p["label"] == "duplicate")


def _implicit_negatives():
    """Unlisted issue x candidate combinations inside a project — distinct by construction."""
    listed = {(p["project"], p["issue"], p["candidate"]) for p in DATA["pairs"]}
    for name, proj in PROJECTS.items():
        for issue in proj["issues"]:
            for cand in proj["resolved"]:
                if (name, issue["number"], cand["number"]) not in listed:
                    yield f"{name}-{issue['number']}-{cand['number']}", name, issue, cand, False


def _score(issue: dict, pool: list[dict], target: int, **kw):
    res = find_similar_issues(issue, pool, top_k=len(pool), min_similarity=-1.0, **kw)
    return next(r for r in res if r.number == target)


def _background(project: str) -> list[dict]:
    """Other projects' new issues, renumbered, as unrelated extra candidates (a ~40-issue pool)."""
    return [{**x, "number": 10_000 + x["number"]}
            for name, proj in PROJECTS.items() if name != project for x in proj["issues"]]


@pytest.fixture(scope="module")
def local_scores() -> list[tuple[str, str, bool, float]]:
    """(pair id, pool, is_duplicate, similarity). The local IDF is fitted on the candidate pool,
    so every pair is scored in pools of four sizes: alone, its project's resolved issues, every
    resolved issue, and every resolved issue plus ~20 unrelated background issues."""
    rows = []
    for pid, project, issue, cand, dup in list(_labelled()) + list(_implicit_negatives()):
        pools = {"pair": [cand], "project": PROJECTS[project]["resolved"], "all": ALL_RESOLVED,
                 "big": ALL_RESOLVED + _background(project)}
        for pool_name, pool in pools.items():
            r = _score(issue, pool, cand["number"])
            assert r.scorer.startswith("local")
            rows.append((pid, pool_name, dup, r.similarity))
    return rows


# ---------------------------------------------------------------------------
# The labelled set itself
# ---------------------------------------------------------------------------

def test_calibration_set_shape():
    pairs = DATA["pairs"]
    assert 20 <= len(pairs) <= 40
    assert len({p["id"] for p in pairs}) == len(pairs)
    labels = [p["label"] for p in pairs]
    assert set(labels) == {"duplicate", "distinct"}
    assert min(labels.count("duplicate"), labels.count("distinct")) >= 10
    hard = {p["kind"] for p in pairs if p["label"] == "distinct"}
    assert {"same_module", "same_keywords", "feature_vs_bug"} <= hard
    dup_of: dict[tuple[str, int], int] = {}
    for pid, project, issue, cand, dup in _labelled():
        assert issue["title"] and cand["title"], pid
        if dup:
            assert (project, issue["number"]) not in dup_of, f"{pid}: an issue duplicates at most one candidate"
            dup_of[(project, issue["number"])] = cand["number"]


def test_calibration_set_is_not_taken_from_the_benchmark():
    cases = sorted((REPO / "benchmark" / "cases").glob("*/case.toml"))
    if not cases:
        pytest.skip("benchmark cases not present")
    bench = "\n".join(p.read_text(encoding="utf-8") for p in cases).lower()
    for proj in PROJECTS.values():
        for item in proj["resolved"] + proj["issues"]:
            assert item["title"].lower() not in bench, item["title"]


# ---------------------------------------------------------------------------
# Local scorer: the shipped threshold separates the labelled set
# ---------------------------------------------------------------------------

def test_local_threshold_has_no_false_positives_in_any_pool_size(local_scores):
    t = DUPLICATE_THRESHOLDS["local"]
    false_pos = [(pid, pool, s) for pid, pool, dup, s in local_scores if not dup and s >= t]
    assert false_pos == []
    for pool in ("pair", "project", "all", "big"):
        hits = [pid for pid, p, dup, s in local_scores if p == pool and dup and s >= t]
        assert len(hits) >= 2, f"threshold {t} accepts no duplicates in the {pool!r} pool"


def test_local_threshold_is_what_the_calibration_rule_gives(local_scores):
    rule = precision_first_threshold([s for *_, s in local_scores], [dup for _, _, dup, _ in local_scores])
    assert rule is not None
    assert abs(rule - DUPLICATE_THRESHOLDS["local"]) <= 0.005


def test_precision_first_threshold_rule():
    # gap between the top negative (0.5) and the next duplicate above it (0.7) -> 0.6
    assert precision_first_threshold([0.9, 0.7, 0.45, 0.5, 0.2], [True, True, True, False, False]) == pytest.approx(0.6)
    assert precision_first_threshold([0.4, 0.6], [True, False]) is None    # no duplicate above every negative
    assert precision_first_threshold([0.4, 0.6], [True, True]) is None     # no negatives to bound precision
    assert precision_first_threshold([], []) is None


# ---------------------------------------------------------------------------
# Threshold resolution + backward compatibility
# ---------------------------------------------------------------------------

def test_threshold_resolution_by_scorer():
    default = PipelineConfig().duplicate_threshold
    assert default == DUPLICATE_THRESHOLDS and default is not DUPLICATE_THRESHOLDS
    assert duplicate_threshold_for(default, "local-ngram-768") == DUPLICATE_THRESHOLDS["local"]
    assert duplicate_threshold_for(default, "local-ngram-256") == DUPLICATE_THRESHOLDS["local"]
    assert duplicate_threshold_for(default, NEMOTRON) == DUPLICATE_THRESHOLDS["nvidia/nemotron-3.5-embed-1b"]
    # an uncalibrated model, or no scorer at all, gets the conservative default
    assert duplicate_threshold_for(default, "api:text-embedding-3-small") == DUPLICATE_THRESHOLDS["default"]
    assert duplicate_threshold_for(default, None) == DUPLICATE_THRESHOLDS["default"]
    # partial tables fall back key by key; an exact scorer name beats the bare model name
    table = {"local": 0.5, NEMOTRON: 0.6, "nvidia/nemotron-3.5-embed-1b": 0.7}
    assert duplicate_threshold_for(table, "local-ngram-768") == 0.5
    assert duplicate_threshold_for(table, NEMOTRON) == 0.6
    assert duplicate_threshold_for({"local": 0.5}, NEMOTRON) == DUPLICATE_THRESHOLDS["nvidia/nemotron-3.5-embed-1b"]


def test_explicit_number_applies_to_every_scorer():
    for scorer in ("local-ngram-768", NEMOTRON, "api:other", None):
        assert duplicate_threshold_for(0.82, scorer) == 0.82
    assert duplicate_threshold_for(1, "local-ngram-768") == 1.0


def test_toml_number_and_table(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text("[pipeline]\nplanning = true\n", encoding="utf-8")
    assert config.load(toml, dotenv=False).pipeline.duplicate_threshold == DUPLICATE_THRESHOLDS

    toml.write_text("[pipeline]\nduplicate_threshold = 0.9\n", encoding="utf-8")
    cfg = config.load(toml, dotenv=False)
    assert cfg.pipeline.duplicate_threshold == 0.9                       # old-style key still works
    assert duplicate_threshold_for(cfg.pipeline.duplicate_threshold, "local-ngram-768") == 0.9
    assert not [e for e in config.validate(cfg) if "duplicate_threshold" in e]

    toml.write_text("[pipeline.duplicate_threshold]\nlocal = 0.5\n\"api:my-model\" = 0.6\n", encoding="utf-8")
    cfg = config.load(toml, dotenv=False)
    assert cfg.pipeline.duplicate_threshold == {**DUPLICATE_THRESHOLDS, "local": 0.5, "api:my-model": 0.6}
    assert not [e for e in config.validate(cfg) if "duplicate_threshold" in e]

    toml.write_text("[pipeline.duplicate_threshold]\nlocal = 1.5\ndefault = \"high\"\n", encoding="utf-8")
    errors = [e for e in config.validate(config.load(toml, dotenv=False)) if "duplicate_threshold" in e]
    assert len(errors) == 2


def _dup(**kw) -> TriageResult:
    base = dict(classification="duplicate", reason="r", confidence=0.9, suggested_question=None, raw={})
    return TriageResult(**{**base, **kw})


def test_evidence_uses_the_threshold_of_each_scorer():
    table = PipelineConfig().duplicate_threshold
    local_hit = [{"number": 7, "similarity": 0.45, "scorer": "local-ngram-768"}]
    assert apply_duplicate_evidence(_dup(duplicate_of=7), local_hit, table).classification == "duplicate"

    nim_low = [{"number": 7, "similarity": 0.45, "scorer": NEMOTRON}]
    out = apply_duplicate_evidence(_dup(duplicate_of=7), nim_low, table)
    assert out.classification == "fixable" and "below threshold 0.53" in out.warnings[-1]

    uncalibrated = [{"number": 7, "similarity": 0.7, "scorer": "api:some-other-model"}]
    assert apply_duplicate_evidence(_dup(duplicate_of=7), uncalibrated, table).classification == "fixable"

    mixed = [{"number": 7, "similarity": 0.3, "scorer": "local-ngram-768"},
             {"number": 8, "similarity": 0.5, "scorer": NEMOTRON}]
    out = apply_duplicate_evidence(_dup(), mixed, table)
    assert out.classification == "fixable" and "0.41/0.53" in out.warnings[-1]


# ---------------------------------------------------------------------------
# Which scorer find_similar_issues uses
# ---------------------------------------------------------------------------

ISSUE = {"number": 1, "title": "Search does not match accented words",
         "body": "Searching cafe does not find notes containing café."}
CANDS = [
    {"number": 2, "title": "Accent-insensitive search", "body": "cafe should match café in search"},
    {"number": 3, "title": "Export notes as PDF", "body": "Please add a PDF export."},
]


class _Semantic:
    name = "api:fake"
    semantic = True

    def __init__(self, refuse=(), fail=False):
        self.refuse, self.fail, self.calls = set(refuse), fail, 0

    def embed(self, texts, kind):
        self.calls += 1
        if self.fail:
            raise requests.ConnectionError("down")
        return [[] if any(r in t for r in self.refuse) else
                ([1.0, 0.0] if "caf" in t.lower() else [0.0, 1.0]) for t in texts]


class _NonSemanticBackend:
    """A non-neural backend that is not LocalNgramEmbedder: its vectors are not used directly."""
    name = "custom-lexical"
    semantic = False

    def __init__(self):
        self.calls = 0

    def embed(self, texts, kind):
        self.calls += 1
        return [[1.0] for _ in texts]


def test_scorer_is_recorded_per_result():
    assert {r.scorer for r in find_similar_issues(ISSUE, CANDS, min_similarity=0.0)} == {"local-ngram-768"}
    res = find_similar_issues(ISSUE, CANDS, embedder=_Semantic(), min_similarity=0.0)
    assert {r.scorer for r in res} == {"api:fake"} and res[0].number == 2


def test_non_semantic_backend_gets_a_pool_fitted_local_scorer():
    emb = _NonSemanticBackend()
    res = find_similar_issues(ISSUE, CANDS, embedder=emb, min_similarity=0.0)
    assert emb.calls == 0
    assert {r.scorer for r in res} == {"local-ngram-768"}


def test_failing_semantic_backend_is_scored_and_labelled_local():
    res = find_similar_issues(ISSUE, CANDS, embedder=_Semantic(fail=True), min_similarity=0.0)
    assert {r.scorer for r in res} == {"local-ngram-768"}


def test_refused_query_scores_the_whole_pool_locally():
    res = find_similar_issues(ISSUE, CANDS, embedder=_Semantic(refuse={ISSUE["title"]}), min_similarity=0.0)
    assert {r.scorer for r in res} == {"local-ngram-768"}
    assert res[0].number == 2 and res[0].similarity > 0.25      # not just the title-Jaccard share


def test_refused_candidate_is_scored_locally_others_semantically():
    res = find_similar_issues(ISSUE, CANDS, embedder=_Semantic(refuse={"Export notes"}), min_similarity=0.0)
    by_num = {r.number: r.scorer for r in res}
    assert by_num == {2: "api:fake", 3: "local-ngram-768"}


# ---------------------------------------------------------------------------
# NIM (opt-in: network + API key)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nim_embedder():
    if os.environ.get("FIX_ISSUE_NIM_TESTS") != "1":
        pytest.skip("set FIX_ISSUE_NIM_TESTS=1 to score the calibration set with NVIDIA NIM")
    before = set(os.environ)
    try:
        cfg = config.load(REPO / "benchmark" / "bench.toml")   # key from env or the gitignored .env
    finally:
        for k in set(os.environ) - before:                      # do not leak .env into other tests
            os.environ.pop(k, None)
    backend, base_url, key, model = config.embedding_settings(cfg)
    if backend != "openai_compatible" or not key:
        pytest.skip("no NIM embedding backend / API key configured")
    from fix_issue.retrieval.embeddings import EmbeddingCache, build_embedder

    emb = build_embedder(backend, base_url=base_url, api_key=key, model=model)
    cache = EmbeddingCache(None)
    texts = sorted({_doc_text(x["title"], x["body"]) for proj in PROJECTS.values()
                    for x in proj["resolved"] + proj["issues"]})
    try:
        cache.embed(emb, texts, kind="query")                   # one batched warm-up
    except requests.RequestException as exc:
        pytest.skip(f"NIM unreachable: {type(exc).__name__}")
    return emb, cache


def test_nim_threshold_has_no_false_positives_on_labelled_set(nim_embedder):
    emb, cache = nim_embedder
    t = duplicate_threshold_for(DUPLICATE_THRESHOLDS, emb.name)
    assert t == DUPLICATE_THRESHOLDS["nvidia/nemotron-3.5-embed-1b"]
    rows, refused = [], []
    for pid, _, issue, cand, dup in list(_labelled()) + list(_implicit_negatives()):
        r = _score(issue, [cand], cand["number"], embedder=emb, cache=cache)
        if r.scorer == emb.name:
            rows.append((pid, dup, r.similarity))
        else:                                                   # refused by NIM -> scored locally
            refused.append(pid)
    if len(refused) > 5:
        pytest.skip(f"{len(refused)} pairs were refused by NIM or failed to embed")
    assert [(pid, s) for pid, dup, s in rows if not dup and s >= t] == []
    assert sum(1 for _, dup, s in rows if dup and s >= t) >= 5
    rule = precision_first_threshold([s for *_, s in rows], [dup for _, dup, _ in rows])
    assert rule is not None and abs(rule - t) <= 0.01
