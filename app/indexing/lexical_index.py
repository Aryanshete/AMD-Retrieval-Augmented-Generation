"""BM25 and inverted lexical search index.

Provides:
- Inverted index supporting exact token and sublinear BM25 scoring.
- Exact identifier indexing (e.g. 'ORR-1847', 'TQ-40', 'THERM_ALERT#', 'REV-C2', '4.3.2', 'E7731').
- Fast persistence to disk and reloading.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.indexing.models import DocumentChunk


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase terms while preserving symbols in identifiers."""
    if not text:
        return []
    # Match alphanumeric sequences, plus hyphenated/underscored identifiers like ORR-1847, THERM_ALERT#
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-_#.][A-Za-z0-9]+)*|#[A-Za-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 0]


class BM25LexicalIndex:
    """Okapi BM25 implementation with specialized identifier matching."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.chunk_ids: List[str] = []
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0
        self.total_docs: int = 0
        
        # Inverted index: term -> {doc_idx: freq}
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        # Identifier index: identifier (uppercase) -> set of doc_idxs
        self.identifier_index: Dict[str, Set[int]] = defaultdict(set)
        # Term IDF cache
        self.idf_cache: Dict[str, float] = {}

    def build(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 and identifier inverted indexes from document chunks."""
        self.chunks = list(chunks)
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.total_docs = len(chunks)
        self.doc_lens = []
        self.inverted_index.clear()
        self.identifier_index.clear()
        self.idf_cache.clear()

        if self.total_docs == 0:
            self.avg_dl = 0.0
            return

        total_tokens = 0
        for doc_idx, chunk in enumerate(chunks):
            # Index text and metadata fields
            combined_text = f"{chunk.source_path} {chunk.text}"
            if chunk.sheet:
                combined_text += f" {chunk.sheet}"
            if chunk.section:
                combined_text += f" {chunk.section}"

            tokens = tokenize(combined_text)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_tokens += doc_len

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.inverted_index[term][doc_idx] = count

            # Index exact identifiers
            ids = chunk.extract_identifiers()
            for id_str in ids:
                self.identifier_index[id_str.upper()].add(doc_idx)
                # Also index normalized version without punctuation
                norm_id = re.sub(r"[-_#.]", "", id_str.upper())
                if norm_id:
                    self.identifier_index[norm_id].add(doc_idx)

        self.avg_dl = total_tokens / max(self.total_docs, 1)

        # Precalculate IDF for all indexed terms
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf_cache[term] = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(
        self, query: str, top_k: int = 20, boost_identifiers: bool = True
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform lexical BM25 search over the corpus chunks.
        
        Returns:
            List of (DocumentChunk, score) tuples sorted in descending order of score.
        """
        if self.total_docs == 0:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = defaultdict(float)

        # 1. Standard BM25 scoring across terms
        for term in q_tokens:
            if term not in self.inverted_index:
                continue

            idf = self.idf_cache.get(term, 0.0)
            postings = self.inverted_index[term]

            for doc_idx, tf in postings.items():
                dl = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                scores[doc_idx] += idf * (numerator / denominator)

        # Boost exact identifier matches
        if boost_identifiers:
            pattern = r"\b([A-Za-z0-9]+(?:[-_#.][A-Za-z0-9]+)*|#[A-Za-z0-9]+)\b"
            query_candidates = set(re.findall(pattern, query))
            for cand in query_candidates:
                cand_upper = cand.upper()
                if cand_upper in self.identifier_index:
                    for doc_idx in self.identifier_index[cand_upper]:
                        scores[doc_idx] += 15.0  # Substantial boost for exact domain identifier match
                cand_clean = re.sub(r"[-_#.]", "", cand_upper)
                if cand_clean and cand_clean in self.identifier_index:
                    for doc_idx in self.identifier_index[cand_clean]:
                        scores[doc_idx] += 5.0

        if not scores:
            return []

        ranked_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[idx], score) for idx, score in ranked_indices]

    def save(self, index_dir: Path) -> Path:
        """Serialize index to lexical.pkl."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        out_path = index_dir / "lexical.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({
                "k1": self.k1,
                "b": self.b,
                "chunks": [c.to_dict() for c in self.chunks],
                "doc_lens": self.doc_lens,
                "avg_dl": self.avg_dl,
                "total_docs": self.total_docs,
                "inverted_index": dict(self.inverted_index),
                "identifier_index": {k: list(v) for k, v in self.identifier_index.items()},
                "idf_cache": self.idf_cache,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out_path

    @classmethod
    def load(cls, index_dir: Path) -> Optional[BM25LexicalIndex]:
        """Deserialize index from lexical.pkl."""
        index_dir = Path(index_dir)
        pkl_path = index_dir / "lexical.pkl"
        if not pkl_path.exists():
            return None
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            idx = cls(k1=data["k1"], b=data["b"])
            idx.chunks = [DocumentChunk.from_dict(d) for d in data["chunks"]]
            idx.chunk_ids = [c.chunk_id for c in idx.chunks]
            idx.doc_lens = data["doc_lens"]
            idx.avg_dl = data["avg_dl"]
            idx.total_docs = data["total_docs"]
            idx.inverted_index = defaultdict(dict, data["inverted_index"])
            idx.identifier_index = defaultdict(set, {k: set(v) for k, v in data["identifier_index"].items()})
            idx.idf_cache = data["idf_cache"]
            return idx
        except Exception:
            return None
