"""Dense semantic embedding index and cosine similarity search.

Provides:
- Dense semantic vector representation and cosine similarity retrieval.
- Offline and dependency-safe fallback embedding generator (pure numpy).
- Persistence as embeddings.npy and chunk mapping.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.indexing.models import DocumentChunk

logger = logging.getLogger(__name__)


class SemanticEmbeddingGenerator:
    """Deterministic, zero-network semantic vectorizer.
    
    Generates normalized dense vectors using multi-resolution n-gram hashing
    with sublinear term weighting. Produces consistent cosine similarities
    across natural language queries and documents without runtime model downloads.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
        "on", "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "from", "up", "down", "in", "out",
        "this", "that", "these", "those", "it", "its", "what", "which", "who", "whom",
    }

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        """Encode text into a unit-normalized dense vector."""
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec

        text_clean = text.lower().strip()
        tokens = re.findall(r"\b[a-z0-9_-]+\b", text_clean)
        
        # 1. Unigram feature hashing with stopword downweighting
        for t in tokens:
            weight = 0.2 if t in self.STOPWORDS else 2.0
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[h] += weight

        # 2. Bigram feature hashing for phrase semantics
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            if t1 not in self.STOPWORDS or t2 not in self.STOPWORDS:
                bigram = f"{t1}_{t2}"
                h = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16) % self.dim
                vec[h] += 3.0

        # 3. Subword 3-gram character hashing for vocabulary robustness
        for t in tokens:
            if t not in self.STOPWORDS and len(t) >= 4:
                for j in range(len(t) - 2):
                    sub = t[j:j+3]
                    h = int(hashlib.md5(sub.encode("utf-8")).hexdigest(), 16) % self.dim
                    vec[h] += 0.5

        # 4. Sublinear scaling and L2 normalization
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into an (N, dim) matrix."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = [self.encode(t) for t in texts]
        return np.vstack(vectors)


class VectorEmbeddingIndex:
    """Stores chunk embeddings and performs fast cosine similarity search."""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.encoder = SemanticEmbeddingGenerator(dim=dim)
        self.chunks: List[DocumentChunk] = []
        self.chunk_ids: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, dim), dtype=np.float32)
        try:
            from app.models.gpu_accelerator import ROCmGPUAccelerator
            self.accelerator = ROCmGPUAccelerator()
        except Exception:
            self.accelerator = None

    def build(self, chunks: List[DocumentChunk]) -> None:
        """Encode chunks and build vector matrix."""
        self.chunks = list(chunks)
        self.chunk_ids = [c.chunk_id for c in chunks]

        if not chunks:
            self.embeddings = np.empty((0, self.dim), dtype=np.float32)
            return

        texts = [f"{c.source_path} {c.text}" for c in chunks]
        self.embeddings = self.encoder.encode_batch(texts)
        if self.accelerator:
            self.accelerator.mount_embeddings_to_gpu(self.embeddings)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[DocumentChunk, float]]:
        """Cosine similarity search over indexed chunks."""
        if len(self.chunks) == 0 or self.embeddings.shape[0] == 0:
            return []

        q_vec = self.encoder.encode(query)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scores: Optional[np.ndarray] = None
        if self.accelerator:
            scores = self.accelerator.compute_similarity(q_vec)

        if scores is None:
            # Fallback to numpy dot product
            scores = np.dot(self.embeddings, q_vec)

        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.chunks[idx], score))

        return results

    def save(self, index_dir: Path) -> Tuple[Path, Path]:
        """Save embeddings to embeddings.npy and metadata to vector_meta.json."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        npy_path = index_dir / "embeddings.npy"
        meta_path = index_dir / "vector_meta.json"

        np.save(npy_path, self.embeddings)
        meta_data = {
            "dim": self.dim,
            "chunk_ids": self.chunk_ids,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        meta_path.write_text(json.dumps(meta_data, ensure_ascii=False), encoding="utf-8")
        return npy_path, meta_path

    @classmethod
    def load(cls, index_dir: Path) -> Optional[VectorEmbeddingIndex]:
        """Load embeddings from embeddings.npy and vector_meta.json."""
        index_dir = Path(index_dir)
        npy_path = index_dir / "embeddings.npy"
        meta_path = index_dir / "vector_meta.json"

        if not npy_path.exists() or not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            idx = cls(dim=meta.get("dim", 128))
            idx.chunk_ids = meta.get("chunk_ids", [])
            idx.chunks = [DocumentChunk.from_dict(d) for d in meta.get("chunks", [])]
            idx.embeddings = np.load(npy_path)
            if idx.accelerator:
                idx.accelerator.mount_embeddings_to_gpu(idx.embeddings)
            return idx
        except Exception as exc:
            logger.error(f"Failed to load vector index: {exc}")
            return None
