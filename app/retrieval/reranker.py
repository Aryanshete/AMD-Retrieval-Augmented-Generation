"""Multi-signal candidate reranker.

Combines:
- BM25 Lexical Score
- Dense Semantic Score
- Exact Identifier Overlap
- Document-Type Relevance
- Demotion of superseded/withdrawn revisions

Selects the small, high-precision top-k candidate set.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.indexing.models import DocumentChunk


def compute_identifier_overlap(query: str, chunk: DocumentChunk) -> float:
    """Calculate exact identifier match score between query and chunk."""
    # Pattern matching ticket IDs, part numbers, version numbers, error codes, pin signals
    pattern = r"\b([A-Za-z0-9]+(?:[-_#.][A-Za-z0-9]+)*|#[A-Za-z0-9]+)\b"
    query_tokens = set(re.findall(pattern, query.upper()))
    chunk_ids = {i.upper() for i in chunk.extract_identifiers()}

    overlap = query_tokens.intersection(chunk_ids)
    if overlap:
        return 1.0 + 0.5 * len(overlap)

    # Check direct substring match of query identifiers in chunk text
    for q_tok in query_tokens:
        if len(q_tok) >= 4 and (q_tok in chunk.text.upper() or q_tok in chunk.source_path.upper()):
            return 0.8

    return 0.0


def compute_document_type_relevance(
    query: str, chunk: DocumentChunk, expected_file_types: Optional[Set[str]] = None
) -> float:
    """Calculate score boost based on expected vs chunk document types."""
    if not expected_file_types:
        return 0.0
    return 0.6 if chunk.file_type in expected_file_types else 0.0



class CandidateReranker:
    """Reranker implementing Layer C combining lexical, semantic, and structural signals."""

    def __init__(
        self,
        weight_lexical: float = 0.40,
        weight_semantic: float = 0.30,
        weight_identifier: float = 0.30,
    ):
        self.w_lex = weight_lexical
        self.w_sem = weight_semantic
        self.w_id = weight_identifier

    def rerank(
        self,
        query: str,
        lexical_candidates: List[Tuple[DocumentChunk, float]],
        semantic_candidates: List[Tuple[DocumentChunk, float]],
        top_k: int = 10,
        expected_file_types: Optional[Set[str]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Merge, normalize scores, apply penalties/boosts, and sort candidates."""
        # Map chunks by chunk_id
        chunk_map: Dict[str, DocumentChunk] = {}
        lex_scores: Dict[str, float] = {}
        sem_scores: Dict[str, float] = {}

        for chunk, score in lexical_candidates:
            chunk_map[chunk.chunk_id] = chunk
            lex_scores[chunk.chunk_id] = max(score, 0.0)

        for chunk, score in semantic_candidates:
            chunk_map[chunk.chunk_id] = chunk
            sem_scores[chunk.chunk_id] = max(score, 0.0)

        if not chunk_map:
            return []

        # Normalize lexical scores to [0, 1]
        max_lex = max(lex_scores.values()) if lex_scores else 1.0
        if max_lex > 0:
            norm_lex = {cid: score / max_lex for cid, score in lex_scores.items()}
        else:
            norm_lex = {cid: 0.0 for cid in lex_scores}

        # Normalize semantic scores to [0, 1]
        max_sem = max(sem_scores.values()) if sem_scores else 1.0
        if max_sem > 0:
            norm_sem = {cid: score / max_sem for cid, score in sem_scores.items()}
        else:
            norm_sem = {cid: 0.0 for cid in sem_scores}

        q_lower = query.lower()
        query_explicitly_requests_withdrawn = "withdrawn" in q_lower or "r1" in q_lower or "obsolete" in q_lower

        final_scores: List[Tuple[DocumentChunk, float]] = []

        for cid, chunk in chunk_map.items():
            s_lex = norm_lex.get(cid, 0.0)
            s_sem = norm_sem.get(cid, 0.0)
            s_id = compute_identifier_overlap(query, chunk)
            s_type = compute_document_type_relevance(query, chunk, expected_file_types)

            combined_score = (
                self.w_lex * s_lex +
                self.w_sem * s_sem +
                self.w_id * s_id +
                0.20 * s_type
            )

            # Demote superseded documents:
            # If document is withdrawn, apply significant penalty UNLESS specifically requested
            if chunk.is_superseded_or_withdrawn() and not query_explicitly_requests_withdrawn:
                combined_score -= 2.0

            final_scores.append((chunk, combined_score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]
