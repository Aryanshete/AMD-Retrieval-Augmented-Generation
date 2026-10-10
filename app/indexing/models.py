"""Canonical document and chunk data structures.

Provides uniform interfaces across all file types: PDF, DOCX, XLSX, CSV, TXT, LOG, PY, and IMAGES.
Ensures schema adherence, validation, serialization, and metadata standardization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Set


@dataclass
class DocumentChunk:
    """Canonical representation of a single retrievable document fragment.
    
    Canonical chunk interface:
        DocumentChunk(
            chunk_id,
            source_path,
            file_type,
            page=None,
            sheet=None,
            row=None,
            section=None,
            text,
            metadata,
            content_type
        )
    """

    chunk_id: str
    source_path: str  # Corpus-relative POSIX path (e.g. 'specs/tq40_datasheet_r2.pdf')
    file_type: str    # 'pdf', 'docx', 'xlsx', 'csv', 'text', 'log', 'source', 'image'
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    page: Optional[int] = None
    sheet: Optional[str] = None
    row: Optional[int] = None
    section: Optional[str] = None
    content_type: str = "text"  # 'text', 'table_row', 'code', 'log_event', 'image_relationships', 'image_ocr', 'image_metadata'

    def __post_init__(self):
        # Enforce POSIX relative paths
        self.source_path = str(self.source_path).replace("\\", "/").lstrip("/")
        self.text = self.text.strip()
        self.validate()

    def validate(self) -> None:
        """Validate that all mandatory fields are present and properly typed."""
        if not self.chunk_id or not isinstance(self.chunk_id, str):
            raise ValueError(f"Invalid chunk_id: {self.chunk_id!r}")
        if not self.source_path or not isinstance(self.source_path, str):
            raise ValueError(f"Invalid source_path: {self.source_path!r}")
        if not self.file_type or not isinstance(self.file_type, str):
            raise ValueError(f"Invalid file_type: {self.file_type!r}")
        if not isinstance(self.text, str):
            raise ValueError(f"Invalid text in chunk {self.chunk_id}: must be string")
        if not isinstance(self.metadata, dict):
            raise ValueError(f"Invalid metadata in chunk {self.chunk_id}: must be dict")

    def is_superseded_or_withdrawn(self) -> bool:
        """Determine if this chunk represents superseded or withdrawn information."""
        if self.metadata.get("is_withdrawn") or self.metadata.get("is_superseded"):
            return True
        path_lower = self.source_path.lower()
        if "withdrawn" in path_lower or "superseded" in path_lower:
            return True
        return False

    def extract_identifiers(self) -> Set[str]:
        """Extract alphanumeric identifiers, ticket IDs, part numbers, and error codes."""
        # Finds patterns like: ORR-1847, ORR-FAN-2214-B, E7731, TQ-40, REV-C2, 4.3.2
        pattern = r"\b([A-Z]{2,4}-[A-Z0-9-]+|E\d{4,5}|[A-Z]\d{1,3}|REV-[A-Z0-9]+|\d+\.\d+\.\d+)\b"
        tokens = set(re.findall(pattern, self.text))
        tokens.update(re.findall(pattern, self.source_path))
        return tokens

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "file_type": self.file_type,
            "page": self.page,
            "sheet": self.sheet,
            "row": self.row,
            "section": self.section,
            "text": self.text,
            "metadata": self.metadata,
            "content_type": self.content_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentChunk:
        """Deserialize from a dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            source_path=data["source_path"],
            file_type=data["file_type"],
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            page=data.get("page"),
            sheet=data.get("sheet"),
            row=data.get("row"),
            section=data.get("section"),
            content_type=data.get("content_type", "text"),
        )

    def __hash__(self) -> int:
        return hash(self.chunk_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DocumentChunk):
            return False
        return self.chunk_id == other.chunk_id


@dataclass
class ParsedDocument:
    """Canonical representation of a parsed document comprising multiple chunks."""

    source_path: str  # Corpus-relative POSIX path
    file_type: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self):
        self.source_path = str(self.source_path).replace("\\", "/").lstrip("/")

    @property
    def is_valid(self) -> bool:
        return self.error is None and len(self.chunks) > 0

    def add_chunk(self, chunk: DocumentChunk) -> None:
        self.chunks.append(chunk)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "file_type": self.file_type,
            "title": self.title,
            "metadata": self.metadata,
            "error": self.error,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ParsedDocument:
        doc = cls(
            source_path=data["source_path"],
            file_type=data["file_type"],
            title=data.get("title"),
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )
        doc.chunks = [DocumentChunk.from_dict(c) for c in data.get("chunks", [])]
        return doc
