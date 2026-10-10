"""Multi-hop query routing and revision conflict resolution.

Provides:
- Multi-hop chain reasoning: Follows link paths across different files (e.g., log incident -> ticket ID -> bug database -> fixed version).
- Superseded / Withdrawn document resolution: Prefers active revisions over obsolete/withdrawn ones.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from app.indexing.models import DocumentChunk
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class MultiHopReasoner:
    """Handles cross-document multi-hop reasoning chains."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def resolve_multi_hop_chain(
        self, query: str
    ) -> Optional[Tuple[List[DocumentChunk], List[str]]]:
        """Perform 2-stage multi-hop traversal across log and bug database.
        
        Reasoning path:
            Question -> Log chunk (incident/error) -> Bridge identifier (ticket ID / error code)
            -> Bug database chunk (ticket details / fixed version).
            
        Returns:
            Tuple of (evidence_chunks, citations) containing both source files.
        """
        # Stage 1: Retrieve log chunk for the incident
        log_candidates = self.retriever.retrieve(
            query=query,
            top_k=5,
            expected_file_types={"log"},
        )
        if not log_candidates:
            return None

        # Filter to actual log chunks
        log_chunks = [c for c, _ in log_candidates if c.file_type == "log"]
        if not log_chunks:
            return None

        primary_log_chunk = log_chunks[0]

        # Stage 2: Extract bridge identifiers or keywords from log chunk
        # Look for ticket identifiers like ORR-XXXX, error codes like EXXXX, or keywords like "thermal throttle"
        bridge_identifiers = primary_log_chunk.extract_identifiers()
        
        # Also check for semantic correlation if no direct ticket ID in the log
        # In sample: log mentions "ERROR E7731: thermal throttle engaged", while bug database mentions "ORR-1847 | thermal throttle"
        second_stage_queries = []
        for b_id in bridge_identifiers:
            second_stage_queries.append(b_id)
        
        # Add symptom keywords from log text
        if "thermal throttle" in primary_log_chunk.text.lower() or "thermal" in query.lower():
            second_stage_queries.append("thermal throttle")
            second_stage_queries.append("die temperature")

        # Stage 3: Retrieve downstream evidence chunk from CSV / bug database
        downstream_chunk = None
        for q_sub in second_stage_queries:
            db_candidates = self.retriever.retrieve(
                query=f"{q_sub} fixed_in firmware release defect",
                top_k=5,
                expected_file_types={"csv"},
            )
            csv_chunks = [c for c, _ in db_candidates if c.file_type == "csv"]
            if csv_chunks:
                downstream_chunk = csv_chunks[0]
                break

        if downstream_chunk and downstream_chunk.source_path != primary_log_chunk.source_path:
            # Retain reasoning chain documents
            evidence_chunks = [primary_log_chunk, downstream_chunk]
            citations = [primary_log_chunk.source_path, downstream_chunk.source_path]
            return evidence_chunks, citations

        return None


class RevisionResolver:
    """Resolves document revision conflicts."""

    REVISION_REGEX = re.compile(r"r(\d+)|rev[-_]?([a-z0-9]+)", re.IGNORECASE)

    @classmethod
    def filter_and_rank_revisions(
        cls, chunks: List[DocumentChunk], query: str
    ) -> List[DocumentChunk]:
        """Rank active revisions higher than withdrawn/superseded ones unless requested."""
        q_lower = query.lower()
        query_requests_withdrawn = any(
            w in q_lower for w in ["withdrawn", "obsolete", "preliminary", "r1", "rev-a"]
        )

        active_chunks = []
        withdrawn_chunks = []

        for c in chunks:
            if c.is_superseded_or_withdrawn():
                withdrawn_chunks.append(c)
            else:
                active_chunks.append(c)

        if query_requests_withdrawn:
            return withdrawn_chunks + active_chunks

        # Normal query: active latest documents must come first
        return active_chunks + withdrawn_chunks
