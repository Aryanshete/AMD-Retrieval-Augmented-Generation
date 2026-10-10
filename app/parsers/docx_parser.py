"""DOCX document parser with table extraction.

Extracts paragraphs, headings, and tables (cells, rows, columns).
Preserves table relationships and heading hierarchy.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from app.indexing.models import DocumentChunk, ParsedDocument


def parse_docx(file_path: Path, rel_path: str) -> ParsedDocument:
    """Parse a DOCX file, extracting paragraphs, headings, and table rows."""
    try:
        import docx
    except ImportError:
        return ParsedDocument(
            source_path=rel_path,
            file_type="docx",
            error="python-docx is not installed",
        )

    try:
        doc = docx.Document(file_path)
    except Exception as exc:
        return ParsedDocument(
            source_path=rel_path,
            file_type="docx",
            error=f"Corrupt or unreadable DOCX: {type(exc).__name__}: {exc}",
        )

    chunks: List[DocumentChunk] = []
    current_heading = "General"
    chunk_index = 0

    # 1. Parse paragraphs and headings
    for p_idx, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        if not text:
            continue

        style_name = p.style.name if p.style else ""
        if "heading" in style_name.lower() or "title" in style_name.lower():
            current_heading = text

        chunk_index += 1
        chunks.append(
            DocumentChunk(
                chunk_id=f"{rel_path}#p{chunk_index}",
                source_path=rel_path,
                file_type="docx",
                section=current_heading,
                text=text,
                metadata={
                    "paragraph_index": p_idx,
                    "style": style_name,
                    "heading": current_heading,
                },
                content_type="text",
            )
        )

    # 2. Parse tables (CRITICAL: Tables must be extracted with row/column context)
    for t_idx, table in enumerate(doc.tables):
        if not table.rows:
            continue

        # Extract table headers if available in row 0
        header_cells = [cell.text.strip() for cell in table.rows[0].cells]
        has_header = any(header_cells)
        start_row = 1 if has_header and len(table.rows) > 1 else 0

        for r_idx in range(start_row, len(table.rows)):
            row = table.rows[r_idx]
            row_cells = [cell.text.strip() for cell in row.cells]
            if not any(row_cells):
                continue

            if has_header and len(header_cells) == len(row_cells):
                row_str_parts = [
                    f"{header_cells[c]}: {row_cells[c]}" for c in range(len(row_cells)) if row_cells[c]
                ]
                row_text = " | ".join(row_str_parts)
            else:
                row_text = " | ".join(filter(None, row_cells))

            chunk_index += 1
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{rel_path}#t{t_idx}r{r_idx}",
                    source_path=rel_path,
                    file_type="docx",
                    row=r_idx,
                    section=f"Table {t_idx} ({current_heading})",
                    text=row_text,
                    metadata={
                        "table_index": t_idx,
                        "row_index": r_idx,
                        "heading": current_heading,
                    },
                    content_type="table_row",
                )
            )

    return ParsedDocument(
        source_path=rel_path,
        file_type="docx",
        chunks=chunks,
        metadata={
            "paragraph_count": len(doc.paragraphs),
            "table_count": len(doc.tables),
            "total_chunks": len(chunks),
        },
    )
