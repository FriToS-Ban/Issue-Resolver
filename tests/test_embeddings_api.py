"""OpenAICompatibleEmbedder resilience against servers that refuse some inputs.

``requests.post`` is replaced by a fake HTTP endpoint (no network): it refuses blank inputs
with 400 like NIM does, and refuses any batch containing a "poison" text with 422, the way
NIM's content-safety screening does for an occasional ordinary code chunk.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import requests

from fix_issue.retrieval.embeddings import EmbeddingCache, OpenAICompatibleEmbedder


class _Resp:
    def __init__(self, status: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeEndpoint:
    def __init__(self, poison: set[str] = frozenset(), fail_status: int | None = None) -> None:
        self.poison = set(poison)
        self.fail_status = fail_status
        self.calls: list[list[str]] = []

    def __call__(self, url, json=None, timeout=None, headers=None):
        inputs = list(json["input"])
        self.calls.append(inputs)
        if self.fail_status:
            return _Resp(self.fail_status, text="server error")
        if any(not t.strip() for t in inputs):
            return _Resp(400, text='{"message":"input[0] must not be blank or empty"}')
        if any(t in self.poison for t in inputs):
            return _Resp(422, text='{"error":{"message":"input rejected by content safety screening"}}')
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(inputs)]
        return _Resp(200, {"data": list(reversed(data))})   # out of order on purpose


@pytest.fixture
def endpoint(monkeypatch):
    def install(**kw) -> FakeEndpoint:
        ep = FakeEndpoint(**kw)
        monkeypatch.setattr(requests, "post", ep)
        return ep
    return install


def _embedder(batch: int = 4) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder("https://example.invalid/v1", "test-key", "m", batch=batch)


def test_refused_inputs_get_empty_vectors_and_the_rest_are_embedded(endpoint):
    endpoint(poison={"bad"})
    emb = _embedder(batch=4)
    texts = ["aa", "bad", "cccc", "d", "eeeee"]
    vecs = emb.embed(texts, "passage")
    assert vecs[1] == []
    assert emb.rejected == 1
    for i in (0, 2, 3, 4):
        assert len(vecs[i]) == 2 and vecs[i][0] > 0      # order preserved despite shuffled response
    assert vecs[2][0] > vecs[0][0]


def test_blank_inputs_are_never_sent(endpoint):
    ep = endpoint()
    emb = _embedder()
    vecs = emb.embed(["x", "   ", "", "yy"], "query")
    assert vecs[1] == [] and vecs[2] == []
    assert all(t.strip() for call in ep.calls for t in call)
    assert emb.send_input_type is True                   # a blank 400 must not disable input_type


def test_other_http_errors_still_raise(endpoint):
    endpoint(fail_status=500)
    with pytest.raises(requests.HTTPError):
        _embedder().embed(["x"], "query")


def test_cache_does_not_persist_refused_inputs_and_survives_torn_lines(endpoint, tmp_path):
    endpoint(poison={"bad"})
    path = tmp_path / "emb.jsonl"
    cache = EmbeddingCache(path)
    emb = _embedder()
    cache.embed(emb, ["ok", "bad"], "passage")
    stored = [json.loads(ln)[0] for ln in path.read_text(encoding="utf-8").splitlines()]
    assert len(stored) == 1 and stored[0].endswith(hashlib.sha1(b"ok").hexdigest())
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('["torn", [0.1, \n')                     # simulated concurrent-append damage
    reloaded = EmbeddingCache(path)
    assert len(reloaded.vec) == 1                         # the good entry survives the torn line
