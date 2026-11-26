"""
retrieval/embeddings.py — Replaceable embedding backends + cache.

Backends implement ``embed(texts, kind) -> list[vector]`` (kind: "passage" | "query").

  LocalNgramEmbedder      (default, offline, dependency-free) signed feature hashing of
                          identifier sub-tokens and their character 3–5-grams with
                          corpus IDF. It captures morphology ("expired"/"expiry"/"is_expired")
                          and identifier overlap; it is NOT a neural semantic model and is
                          reported as ``semantic=False`` everywhere.
  OpenAICompatibleEmbedder any POST {base_url}/embeddings endpoint (OpenAI, NVIDIA NIM,
                          Ollama, LM Studio...). ``semantic=True``.

Vectors are cached on disk by (backend name, sha1(text)) so unchanged code is never
re-embedded across runs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Protocol

from fix_issue.retrieval.text import split_identifier, stem, tokenize


class EmbeddingBackend(Protocol):
    name: str
    semantic: bool

    def embed(self, texts: list[str], kind: str) -> list[list[float]]: ...


def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class LocalNgramEmbedder:
    semantic = False

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.name = f"local-ngram-{dim}"
        self._idf: dict[str, float] = {}

    @staticmethod
    def _features(text: str) -> Counter:
        feats: Counter = Counter()
        for tok in tokenize(text):
            base = stem(tok)
            feats[f"w:{base}"] += 1.0
            if len(tok) >= 5 and "_" not in tok:
                padded = f"<{tok}>"
                for n in (3, 4, 5):
                    for i in range(len(padded) - n + 1):
                        feats[f"g:{padded[i:i + n]}"] += 0.25
            for part in split_identifier(tok):
                if part != tok:
                    feats[f"w:{stem(part)}"] += 0.5
        return feats

    def fit(self, corpus: list[str]) -> None:
        df: Counter = Counter()
        for doc in corpus:
            df.update(self._features(doc).keys())
        n = max(len(corpus), 1)
        self._idf = {f: math.log(1 + n / (1 + c)) for f, c in df.items()}

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        out = []
        default_idf = math.log(1 + max(len(self._idf), 1)) if self._idf else 1.0
        for t in texts:
            v = [0.0] * self.dim
            for feat, tf in self._features(t).items():
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                w = (1 + math.log(tf)) * self._idf.get(feat, default_idf) if tf > 0 else 0.0
                v[h % self.dim] += w if (h >> 40) & 1 else -w
            out.append(unit(v))
        return out


class OpenAICompatibleEmbedder:
    semantic = True

    def __init__(self, base_url: str, api_key: str, model: str, batch: int = 32,
                 send_input_type: bool = True, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch = batch
        self.send_input_type = send_input_type
        self.timeout = timeout
        self.name = f"api:{model}"

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        import requests

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            body: dict = {"model": self.model, "input": texts[i:i + self.batch], "encoding_format": "float"}
            if self.send_input_type:
                body["input_type"] = kind
            r = requests.post(f"{self.base_url}/embeddings", json=body, timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.api_key}"})
            if r.status_code == 400 and self.send_input_type:
                # Some servers reject input_type; retry without it once and remember.
                self.send_input_type = False
                body.pop("input_type", None)
                r = requests.post(f"{self.base_url}/embeddings", json=body, timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
            out.extend(unit(d["embedding"]) for d in data)
        return out


class EmbeddingCache:
    """(backend, sha1(text)) -> unit vector, persisted as JSON lines."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.vec: dict[str, list[float]] = {}
        if path and path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    k, v = json.loads(line)
                    self.vec[k] = v
            except (OSError, ValueError):
                self.vec = {}

    def embed(self, backend: EmbeddingBackend, texts: list[str], kind: str = "passage") -> list[list[float]]:
        if not backend.semantic:  # local vectors depend on fitted IDF — never persist them
            return backend.embed(texts, kind)
        keys = [f"{backend.name}|{kind}|{hashlib.sha1(t.encode('utf-8', 'replace')).hexdigest()}" for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.vec]
        if missing:
            uniq = dict(missing)
            vecs = backend.embed(list(uniq.values()), kind)
            new = dict(zip(uniq.keys(), vecs))
            self.vec.update(new)
            if self.path:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.path, "a", encoding="utf-8") as fh:
                        for k, v in new.items():
                            fh.write(json.dumps([k, [round(x, 6) for x in v]]) + "\n")
                except OSError:
                    pass
        return [self.vec[k] for k in keys]


def build_embedder(kind: str, *, base_url: str = "", api_key: str = "", model: str = "") -> EmbeddingBackend | None:
    """Factory from config. Returns None for 'none'."""
    kind = (kind or "local").lower()
    if kind == "none":
        return None
    if kind == "openai_compatible":
        if not (base_url and api_key and model):
            raise ValueError("embedding_backend=openai_compatible needs embedding_base_url, api key and model")
        return OpenAICompatibleEmbedder(base_url, api_key, model)
    return LocalNgramEmbedder()
