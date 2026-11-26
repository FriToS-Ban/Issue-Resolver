"""retrieval/lexical.py — BM25 over documents of code-aware tokens (files or symbols)."""

from __future__ import annotations

import math
from collections import Counter

from fix_issue.retrieval.text import stem


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(stem(t) for t in d) for d in docs]
        self.lens = [sum(c.values()) for c in self.tfs]
        n = max(len(docs), 1)
        self.avg = (sum(self.lens) / n) or 1.0
        df: Counter = Counter()
        for c in self.tfs:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (n - m + 0.5) / (m + 0.5)) for t, m in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        q = Counter(stem(t) for t in query)
        out = []
        for tf, ln in zip(self.tfs, self.lens):
            s = 0.0
            for t, qn in q.items():
                f = tf.get(t)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg)) * min(qn, 3)
            out.append(s)
        return out

    def matched_terms(self, query: list[str], doc_index: int, limit: int = 6) -> list[str]:
        tf = self.tfs[doc_index]
        terms = sorted({stem(t) for t in query if stem(t) in tf}, key=lambda t: -self.idf.get(t, 0))
        return terms[:limit]
