"""Minimal citation extraction and path normalization.

Enforces the core challenge scoring rule:
"A question receives credit only when BOTH the answer and the citation set are exactly correct."
"Cite a file only when removing that file would make the answer impossible."
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Set

from app.indexing.models import DocumentChunk

logger = logging.getLogger(__name__)


class CitationSelector:
    """Selects and validates the minimal necessary citation set."""

    @staticmethod
    def normalize_path(path_str: str) -> str:
        """Ensure corpus-relative POSIX format with no leading slashes or corpus prefix."""
        norm = str(path_str).replace("\\", "/").strip()
        if norm.startswith("/app/corpus/"):
            norm = norm[len("/app/corpus/"):]
        elif norm.startswith("app/corpus/"):
            norm = norm[len("app/corpus/"):]
        elif norm.startswith("/"):
            norm = norm.lstrip("/")
        return norm

    @classmethod
    def select_minimal_citations(
        cls,
        evidence_chunks: List[DocumentChunk],
        is_multi_hop: bool = False,
        valid_corpus_files: Optional[Set[str]] = None,
    ) -> List[str]:
        """Determine the minimal necessary source set from evidence chunks.
        
        Algorithm:
        1. Group chunks by source file.
        2. If single-hop: pick the single primary source file containing the answer.
        3. If multi-hop: pick the minimal set of files spanning the reasoning chain (e.g., log + csv).
        4. Eliminate redundant contextual files (e.g. general release notes when csv has direct answer).
        5. Verify every citation corresponds to a file actually parsed and accessible.
        """
        if not evidence_chunks:
            return []

        # Deduplicate source files while preserving order of first appearance in evidence
        unique_sources: List[str] = []
        seen = set()

        for c in evidence_chunks:
            norm_path = cls.normalize_path(c.source_path)
            if norm_path not in seen:
                seen.add(norm_path)
                unique_sources.append(norm_path)

        # Validate against known corpus files if provided (Check C)
        if valid_corpus_files is not None:
            norm_valid = {cls.normalize_path(p) for p in valid_corpus_files}
            unique_sources = [s for s in unique_sources if s in norm_valid]

        if not is_multi_hop:
            # Retain only the single essential source file
            # Do NOT include multiple files just because they are topically related
            return unique_sources[:1]

        # Multi-hop: Return minimal set representing each unique file type in the reasoning chain
        # (e.g., one log file and one csv file)
        selected: List[str] = []
        file_types_covered: Set[str] = set()

        for c in evidence_chunks:
            norm_path = cls.normalize_path(c.source_path)
            if norm_path in unique_sources and c.file_type not in file_types_covered:
                selected.append(norm_path)
                file_types_covered.add(c.file_type)

        return selected if selected else unique_sources[:2]

    @classmethod
    def validate_citations(
        cls, answer: str, citations: List[str]
    ) -> Tuple[str, List[str], float]:
        """Run citation consistency checks.
        
        Check A: if answer != "", citations must not be empty.
        Check B: every citation must be corpus-relative POSIX.
        Check D: remove redundant citations.
        """
        clean_ans = answer.strip()

        # Check A: If answer produced without citations, refuse to prevent scoring penalty
        if clean_ans and not citations:
            logger.warning("Answer produced without citations; refusing to protect score.")
            return "", [], 0.0

        # If answer is empty, citations must also be empty
        if not clean_ans:
            return "", [], 0.0

        # Check B & D: Clean paths and deduplicate preserving order
        deduped: List[str] = []
        seen = set()
        for c in citations:
            norm = cls.normalize_path(c)
            if norm and norm not in seen:
                seen.add(norm)
                deduped.append(norm)

        if not deduped:
            return "", [], 0.0

        return clean_ans, deduped, 0.95
