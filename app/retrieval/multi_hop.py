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
        # Stage 1: Retrieve candidate chunks describing the initial incident / symptom
        candidates = self.retriever.retrieve(
            query=query,
            top_k=8,
        )
        if not candidates:
            return None

        # Prefer log chunk if query explicitly mentions log, otherwise general incident chunks
        if "log" in query.lower():
            log_only = [c for c, _ in candidates if c.file_type == "log"]
            primary_chunk = log_only[0] if log_only else candidates[0][0]
        else:
            primary_chunks = [c for c, _ in candidates if c.file_type in ["log", "text"] or "incident" in c.text.lower() or "error" in c.text.lower() or "throttle" in c.text.lower()]
            primary_chunk = primary_chunks[0] if primary_chunks else candidates[0][0]


        # Stage 2: Extract bridge identifiers or keywords from primary chunk and query
        bridge_queries: List[str] = []

        # 2a. Ticket numbers (e.g. ORR-1847, BUG-102)
        ticket_matches = re.findall(r"\b([A-Z]{2,5}-\d+)\b", primary_chunk.text, re.IGNORECASE)
        for t in ticket_matches:
            if t.upper() not in bridge_queries:
                bridge_queries.append(t.upper())

        # 2b. Error codes (e.g. E7731)
        err_matches = re.findall(r"\b(E\d{4,5})\b", primary_chunk.text)
        for e in err_matches:
            if e not in bridge_queries:
                bridge_queries.append(e)

        # 2c. Extracted identifiers from model chunk
        for b_id in primary_chunk.extract_identifiers():
            if b_id not in bridge_queries:
                bridge_queries.append(b_id)

        # 2d. Incident symptoms from text
        symptoms = ["thermal throttle", "memory leak", "sensor reports high", "hang", "crash", "clocks reduced"]
        for s in symptoms:
            if s in primary_chunk.text.lower() or s in query.lower():
                bridge_queries.append(s)

        # Stage 3: Retrieve downstream resolution chunk from another document
        downstream_chunk = None
        for q_sub in bridge_queries:
            db_candidates = self.retriever.retrieve(
                query=f"{q_sub} fixed_in firmware version release defect resolution",
                top_k=8,
            )
            for c, _ in db_candidates:
                if c.source_path != primary_chunk.source_path:
                    # Verify downstream chunk mentions resolution or fix
                    c_lower = c.text.lower()
                    if any(k in c_lower for k in ["fixed", "version", "release", "closed", "resolved", "patch", "4."]):
                        downstream_chunk = c
                        break
            if downstream_chunk:
                break

        if downstream_chunk and downstream_chunk.source_path != primary_chunk.source_path:
            # Retain reasoning chain documents
            evidence_chunks = [primary_chunk, downstream_chunk]
            citations = [primary_chunk.source_path, downstream_chunk.source_path]
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
