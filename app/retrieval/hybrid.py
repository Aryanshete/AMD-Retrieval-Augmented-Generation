"""Hybrid retrieval engine combining lexical and vector search.

Orchestrates:
- Layer A: Lexical BM25 retrieval
- Layer B: Dense Vector Semantic retrieval
- Layer C: Multi-signal Reranking
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple

from app.indexing.lexical_index import BM25LexicalIndex
from app.indexing.models import DocumentChunk
from app.indexing.vector_index import VectorEmbeddingIndex
from app.retrieval.reranker import CandidateReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Integrated hybrid search engine combining BM25 lexical and dense semantic indexing."""

    def __init__(self, lexical_index: Optional[BM25LexicalIndex] = None, vector_index: Optional[VectorEmbeddingIndex] = None):
        self.lexical_index = lexical_index or BM25LexicalIndex()
        self.vector_index = vector_index or VectorEmbeddingIndex()
        self.reranker = CandidateReranker()
        self.all_chunks: List[DocumentChunk] = []

    def build(self, chunks: List[DocumentChunk]) -> None:
        """Build both lexical and vector indices over the given chunks."""
        self.all_chunks = list(chunks)
        self.lexical_index.build(chunks)
        self.vector_index.build(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool: int = 30,
        expected_file_types: Optional[Set[str]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform hybrid retrieval and reranking for a query.
        
        Returns:
            List of (DocumentChunk, score) tuples for top-k candidates.
        """
        # 1. Lexical retrieval (Layer A)
        lexical_candidates = self.lexical_index.search(query, top_k=candidate_pool)

        # 2. Semantic retrieval (Layer B)
        semantic_candidates = self.vector_index.search(query, top_k=candidate_pool)

        # 3. Ensemble reranking (Layer C)
        ranked = self.reranker.rerank(
            query=query,
            lexical_candidates=lexical_candidates,
            semantic_candidates=semantic_candidates,
            top_k=top_k,
            expected_file_types=expected_file_types,
        )
        return ranked

    def save(self, index_dir: Path) -> None:
        """Persist both indices to the target directory."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self.lexical_index.save(index_dir)
        self.vector_index.save(index_dir)

    @classmethod
    def load(cls, index_dir: Path) -> Optional[HybridRetriever]:
        """Load indices from the target directory."""
        index_dir = Path(index_dir)
        lex = BM25LexicalIndex.load(index_dir)
        vec = VectorEmbeddingIndex.load(index_dir)
        if lex is None and vec is None:
            return None
        inst = cls(lexical_index=lex, vector_index=vec)
        inst.all_chunks = lex.chunks if lex else (vec.chunks if vec else [])
        return inst
