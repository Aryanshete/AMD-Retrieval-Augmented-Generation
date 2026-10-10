"""CSV tabular parser.

Extracts headers and rows, preserving original values exactly.
Supports multiple encodings and delimiters defensively.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List

from app.indexing.models import DocumentChunk, ParsedDocument


def parse_csv(file_path: Path, rel_path: str) -> ParsedDocument:
    """Parse a CSV file, preserving exact raw values and field associations."""
    content_bytes = None
    try:
        content_bytes = file_path.read_bytes()
    except Exception as exc:
        return ParsedDocument(
            source_path=rel_path,
            file_type="csv",
            error=f"Unreadable file: {type(exc).__name__}: {exc}",
        )

    # Decode with fallback encodings
    text = None
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            text = content_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        return ParsedDocument(
            source_path=rel_path,
            file_type="csv",
            error="EncodingDecodeError: unable to decode file content",
        )

    chunks: List[DocumentChunk] = []

    try:
        # Detect delimiter
        sample = text[:2048]
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = list(reader)

        if not rows:
            return ParsedDocument(
                source_path=rel_path,
                file_type="csv",
                chunks=[],
                metadata={"total_rows": 0},
            )

        # First non-empty row is header
        headers = [h.strip() for h in rows[0]]
        for r_idx, row_cells in enumerate(rows[1:], start=2):
            cleaned_cells = [c.strip() for c in row_cells]
            if not any(cleaned_cells):
                continue

            field_dict = {}
            parts = []
            for col_idx, val in enumerate(cleaned_cells):
                col_name = headers[col_idx] if col_idx < len(headers) and headers[col_idx] else f"Col{col_idx+1}"
                field_dict[col_name] = val
                parts.append(f"{col_name}: {val}")

            raw_line = delimiter.join(cleaned_cells)
            row_text = f"{', '.join(parts)}\nRaw: {raw_line}"

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{rel_path}#r{r_idx}",
                    source_path=rel_path,
                    file_type="csv",
                    row=r_idx,
                    text=row_text,
                    metadata={
                        "file": rel_path,
                        "row": r_idx,
                        "fields": field_dict,
                        "raw_line": raw_line,
                    },
                    content_type="table_row",
                )
            )

    except Exception as exc:
        return ParsedDocument(
            source_path=rel_path,
            file_type="csv",
            error=f"Error parsing CSV rows: {type(exc).__name__}: {exc}",
        )

    return ParsedDocument(
        source_path=rel_path,
        file_type="csv",
        chunks=chunks,
        metadata={"total_rows": len(rows), "total_chunks": len(chunks)},
    )
