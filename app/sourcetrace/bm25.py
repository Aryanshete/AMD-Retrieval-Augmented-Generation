"""A small, dependency-free BM25 implementation (Okapi, k1=1.2, b=0.75)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        self._avg = 0.0
        self._idf: dict[str, float] = {}

    def fit(self, docs: list[list[str]]) -> BM25:
        self._postings = defaultdict(list)
        self._lengths = [len(d) for d in docs]
        for doc_id, tokens in enumerate(docs):
            for term, tf in Counter(tokens).items():
                self._postings[term].append((doc_id, tf))
        n = len(docs)
        self._avg = (sum(self._lengths) / n) if n else 0.0
        self._idf = {
            term: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self._postings.items()
        }
        return self

    def __len__(self) -> int:
        return len(self._lengths)

    def score(self, query_tokens: list[str]) -> dict[int, float]:
        """Scores for every document sharing at least one term with the query."""
        scores: dict[int, float] = defaultdict(float)
        avg = self._avg or 1.0
        for term, qtf in Counter(query_tokens).items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            weight = 1.0 + 0.25 * (qtf - 1)
            for doc_id, tf in postings:
                norm = tf + self.k1 * (1.0 - self.b + self.b * self._lengths[doc_id] / avg)
                scores[doc_id] += weight * idf * tf * (self.k1 + 1.0) / norm
        return dict(scores)
