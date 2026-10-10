"""PDF document parser with page tracking and encryption detection.

Extracts text, page numbers, titles, and identifies superseded/withdrawn statuses.
Defensively catches encrypted and corrupt PDFs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from app.config import SUPERSEDED_MARKERS
from app.indexing.models import DocumentChunk, ParsedDocument


class EncryptedPDFError(Exception):
    """Raised when a PDF is password-protected or encrypted."""
    pass


def parse_pdf(file_path: Path, rel_path: str) -> ParsedDocument:
    """Parse a PDF file into canonical ParsedDocument and DocumentChunks."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            fitz = None

    if fitz is None:
        return _parse_pdf_fallback(file_path, rel_path)

    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        return ParsedDocument(
            source_path=rel_path,
            file_type="pdf",
            error=f"Corrupt or unreadable PDF: {type(exc).__name__}: {exc}",
        )

    # Check for encryption
    if getattr(doc, "is_encrypted", False):
        doc.close()
        return ParsedDocument(
            source_path=rel_path,
            file_type="pdf",
            error="encrypted",
            metadata={"encrypted": True},
        )

    title = None
    if doc.metadata:
        title = doc.metadata.get("title") or None

    # Detect revision indicators in filename
    rel_lower = rel_path.lower()
    is_withdrawn_filename = any(f"_{m}" in rel_lower or f"-{m}" in rel_lower for m in ["withdrawn", "superseded", "obsolete", "retired"])
    rev_match = re.search(r"\b(r\d+|rev-?[a-z0-9]+)\b", rel_lower)
    doc_revision = rev_match.group(1).lower() if rev_match else None

    chunks = []
    full_text = []

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text() or ""
            page_num = page_idx + 1

            if not page_text.strip():
                continue

            full_text.append(page_text)
            page_lower = page_text.lower()

            # Check page header (first 5 lines) for withdrawn status of this document
            first_lines = "\n".join(page_lower.splitlines()[:6])
            
            # If the header specifically declares THIS revision is withdrawn / preliminary
            is_withdrawn_content = False
            if "withdrawn" in first_lines or "superseded by" in first_lines:
                is_withdrawn_content = True
            elif "preliminary silicon" in first_lines or "do not design" in first_lines:
                is_withdrawn_content = True

            # If header says "supersedes revision 1" or "revision 2", it is active!
            if "supersedes" in first_lines and not ("superseded by" in first_lines or "withdrawn" in first_lines):
                is_withdrawn_content = False

            page_withdrawn = is_withdrawn_filename or is_withdrawn_content

            # Detect page-level revision
            page_rev_match = re.search(r"revision\s+(\d+|[a-z0-9]+)", first_lines)
            page_rev = f"r{page_rev_match.group(1)}" if page_rev_match else doc_revision

            chunk = DocumentChunk(
                chunk_id=f"{rel_path}#p{page_num}",
                source_path=rel_path,
                file_type="pdf",
                page=page_num,
                text=page_text.strip(),
                metadata={
                    "revision": page_rev,
                    "is_withdrawn": page_withdrawn,
                    "is_superseded": page_withdrawn,
                    "page_number": page_num,
                },
                content_type="text",
            )
            chunks.append(chunk)

    except Exception as exc:
        doc.close()
        if "encrypted" in str(exc).lower() or "password" in str(exc).lower():
            return ParsedDocument(
                source_path=rel_path,
                file_type="pdf",
                error="encrypted",
                metadata={"encrypted": True},
            )
        return ParsedDocument(
            source_path=rel_path,
            file_type="pdf",
            error=f"Error reading PDF pages: {type(exc).__name__}: {exc}",
        )
    finally:
        doc.close()

    # Determine overall document status
    doc_is_withdrawn = is_withdrawn_filename or any(c.metadata.get("is_withdrawn") for c in chunks)

    return ParsedDocument(
        source_path=rel_path,
        file_type="pdf",
        title=title,
        chunks=chunks,
        metadata={
            "page_count": len(chunks),
            "revision": doc_revision,
            "is_withdrawn": doc_is_withdrawn,
            "is_superseded": doc_is_withdrawn,
        },
    )


def _parse_pdf_fallback(file_path: Path, rel_path: str) -> ParsedDocument:
    """Fallback PDF parsing using pypdf if PyMuPDF is not installed."""
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        if reader.is_encrypted:
            return ParsedDocument(
                source_path=rel_path,
                file_type="pdf",
                error="encrypted",
                metadata={"encrypted": True},
            )
        chunks = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{rel_path}#p{i+1}",
                        source_path=rel_path,
                        file_type="pdf",
                        page=i + 1,
                        text=text.strip(),
                        content_type="text",
                    )
                )
        return ParsedDocument(source_path=rel_path, file_type="pdf", chunks=chunks)
    except Exception as exc:
        return ParsedDocument(
            source_path=rel_path,
            file_type="pdf",
            error=f"Fallback PDF parsing failed: {type(exc).__name__}: {exc}",
        )
