"""Tests for duplicates.py — similar-issue ranking and candidate sources (no network)."""
import requests

from fix_issue.duplicates import (
    SimilarIssue,
    find_similar_issues,
    recent_resolved_from_github,
    resolved_from_history,
)
from fix_issue.github_client import GitHubError
from fix_issue.retrieval.embeddings import LocalNgramEmbedder


ISSUE = {
    "number": 50,
    "title": "parse_date crashes on leap day 2024-02-29",
    "body": "Calling parse_date('2024-02-29') raises ValueError: day is out of range for month. "
            "Leap years should be supported.",
}

PARAPHRASE = {
    "number": 12,
    "title": "ValueError from parse_date for February 29 in leap years",
    "body": "parse_date fails with 'day is out of range for month' when given 2020-02-29. "
            "The leap year check is wrong.",
    "state": "closed",
    "source": "github",
}

DISTRACTORS = [
    {"number": 13, "title": "format_currency rounds half-cents incorrectly",
     "body": "format_currency(0.125) returns 0.12 instead of 0.13.", "source": "github"},
    {"number": 14, "title": "Add CSV export to the report command",
     "body": "It would be nice to export reports as CSV files.", "source": "github"},
    {"number": 15, "title": "Date picker shows wrong month name in French locale",
     "body": "The date picker renders 'Janvier' for February when locale=fr.", "source": "github"},
]


# ---------------------------------------------------------------------------
# find_similar_issues
# ---------------------------------------------------------------------------

def test_paraphrased_duplicate_ranks_first():
    results = find_similar_issues(ISSUE, DISTRACTORS + [PARAPHRASE], top_k=4, min_similarity=0.0)
    assert results[0].number == 12
    assert all(isinstance(r, SimilarIssue) for r in results)
    assert results[0].similarity > results[1].similarity + 0.2
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)
    assert results[0].state == "closed" and results[0].source == "github"


def test_paraphrase_with_explicit_local_embedder_is_not_refit():
    emb = LocalNgramEmbedder(dim=256)
    emb.fit(["completely unrelated corpus about kubernetes ingress"])
    before = dict(emb._idf)
    results = find_similar_issues(ISSUE, DISTRACTORS + [PARAPHRASE], embedder=emb, min_similarity=0.0)
    assert results[0].number == 12
    assert emb._idf == before  # caller's (shared) embedder left untouched


def test_identical_text_scores_one():
    twin = {"number": 99, "title": ISSUE["title"], "body": ISSUE["body"]}
    results = find_similar_issues(ISSUE, [twin] + DISTRACTORS)
    assert results[0].number == 99
    assert results[0].similarity == 1.0


def test_self_is_excluded():
    me = {"number": 50, "title": ISSUE["title"], "body": ISSUE["body"]}
    results = find_similar_issues(ISSUE, [me, PARAPHRASE] + DISTRACTORS, min_similarity=0.0, top_k=10)
    assert 50 not in [r.number for r in results]
    # string numbers are compared numerically too
    results = find_similar_issues({**ISSUE, "number": "50"}, [me], min_similarity=0.0)
    assert results == []


def test_min_similarity_filters_and_top_k_limits():
    results = find_similar_issues(ISSUE, DISTRACTORS + [PARAPHRASE], min_similarity=0.35)
    assert [r.number for r in results] == [12]
    assert find_similar_issues(ISSUE, DISTRACTORS + [PARAPHRASE], min_similarity=0.99) == []
    assert len(find_similar_issues(ISSUE, DISTRACTORS + [PARAPHRASE], top_k=2, min_similarity=0.0)) == 2


def test_short_title_match_gets_jaccard_boost():
    issue = {"number": 1, "title": "Login button broken", "body": ""}
    cands = [
        {"number": 2, "title": "login button broken", "body": "x"},
        {"number": 3, "title": "Export to PDF", "body": "x"},
    ]
    results = find_similar_issues(issue, cands, min_similarity=0.0)
    assert results[0].number == 2
    assert results[0].similarity >= 0.5


def test_empty_and_malformed_candidates():
    assert find_similar_issues(ISSUE, []) == []
    cands = [{"title": "no number"}, {"number": 7}, "not a dict", {"number": 8, "title": "", "body": ""}]
    assert find_similar_issues(ISSUE, cands, min_similarity=0.0) == []


def test_duplicate_numbers_merged_and_resolved_by_kept():
    a = {**PARAPHRASE, "source": "github"}
    b = {**PARAPHRASE, "source": "history", "resolved_by": "https://github.com/o/r/pull/3"}
    results = find_similar_issues(ISSUE, [a, b], min_similarity=0.0)
    assert len(results) == 1
    assert results[0].resolved_by == "https://github.com/o/r/pull/3"


def test_body_excerpt_and_to_dict():
    long_body = {"number": 21, "title": PARAPHRASE["title"], "body": "leap  year\n\n" * 200}
    r = find_similar_issues(ISSUE, [long_body], min_similarity=0.0)[0]
    assert len(r.body_excerpt) <= 300
    assert "\n" not in r.body_excerpt
    d = r.to_dict()
    assert set(d) == {"number", "title", "similarity", "state", "source", "resolved_by", "body_excerpt",
                      "scorer"}
    assert d["source"] == "provided"  # default when the candidate does not say
    assert d["scorer"] == "local-ngram-768"  # no embedder given -> pool-fitted local scorer


class _FakeSemantic:
    name = "fake-semantic"
    semantic = True

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def embed(self, texts, kind):
        self.calls += 1
        if self.fail:
            raise requests.ConnectionError("down")
        # 2-d vectors: "leap" texts point one way, everything else the other.
        return [[1.0, 0.0] if "leap" in t.lower() else [0.0, 1.0] for t in texts]


def test_custom_semantic_embedder_used():
    emb = _FakeSemantic()
    results = find_similar_issues(ISSUE, DISTRACTORS + [PARAPHRASE], embedder=emb, min_similarity=0.5)
    assert emb.calls == 1
    assert [r.number for r in results] == [12]


def test_failing_embedder_falls_back_to_local():
    emb = _FakeSemantic(fail=True)
    results = find_similar_issues(ISSUE, DISTRACTORS + [PARAPHRASE], embedder=emb, min_similarity=0.0)
    assert results[0].number == 12


# ---------------------------------------------------------------------------
# resolved_from_history
# ---------------------------------------------------------------------------

def _record(repo, number, result, pr_url=None, title="t", body="b"):
    return {
        "repo": repo,
        "issue_number": number,
        "issue": {"title": title, "body": body},
        "outcome": {"result": result, "pr_url": pr_url},
    }


def test_history_keeps_only_same_repo_pr_opened():
    records = [
        _record("o/r", 1, "pr_opened", "https://github.com/o/r/pull/10", title="Fix pager"),
        _record("o/r", 2, "needs_info"),
        _record("other/repo", 3, "pr_opened", "https://github.com/other/repo/pull/1"),
        _record("o/r", 4, "error"),
        {"repo": "o/r", "issue_number": 5, "issue": {}, "outcome": None},  # incomplete run
        _record("O/R", 6, "pr_opened", "https://github.com/o/r/pull/11"),
        _record("o/r", 1, "pr_opened", "https://github.com/o/r/pull/9"),   # older run, same issue
    ]
    out = resolved_from_history(records, "o/r")
    assert [d["number"] for d in out] == [1, 6]
    first = out[0]
    assert first["resolved_by"] == "https://github.com/o/r/pull/10"
    assert first["source"] == "history"
    assert first["title"] == "Fix pager"
    assert first["body"] == "b"


def test_history_feeds_find_similar():
    records = [_record("o/r", 12, "pr_opened", "https://github.com/o/r/pull/3",
                       title=PARAPHRASE["title"], body=PARAPHRASE["body"])]
    results = find_similar_issues(ISSUE, resolved_from_history(records, "o/r") + DISTRACTORS)
    assert results[0].number == 12
    assert results[0].source == "history"
    assert results[0].resolved_by == "https://github.com/o/r/pull/3"


def test_history_empty():
    assert resolved_from_history([], "o/r") == []


# ---------------------------------------------------------------------------
# recent_resolved_from_github
# ---------------------------------------------------------------------------

class _FakeGH:
    def __init__(self, payload=None, exc=None):
        self.payload = payload
        self.exc = exc
        self.calls = []

    def _get(self, path, params=None):
        self.calls.append((path, params))
        if self.exc:
            raise self.exc
        return self.payload


def test_github_helper_skips_pull_requests():
    gh = _FakeGH([
        {"number": 1, "title": "Bug A", "body": "body a", "state": "closed", "html_url": "u1"},
        {"number": 2, "title": "PR B", "body": "", "state": "closed", "pull_request": {"url": "x"}},
        {"number": 3, "title": "Wontfix C", "body": None, "state": "closed", "state_reason": "not_planned"},
        {"number": 4, "title": "Bug D", "body": None, "state": "closed", "state_reason": "completed"},
    ])
    out = recent_resolved_from_github(gh, "o/r", limit=50)
    assert [d["number"] for d in out] == [1, 4]
    assert out[0]["source"] == "github"
    assert out[1]["body"] == ""
    path, params = gh.calls[0]
    assert path == "/repos/o/r/issues"
    assert params == {"state": "closed", "per_page": 50, "sort": "updated"}


def test_github_helper_errors_return_empty():
    assert recent_resolved_from_github(_FakeGH(exc=GitHubError("GitHub API 404: Not Found")), "o/r") == []
    assert recent_resolved_from_github(_FakeGH(exc=requests.ConnectionError("offline")), "o/r") == []
    assert recent_resolved_from_github(_FakeGH(payload={"message": "weird"}), "o/r") == []
