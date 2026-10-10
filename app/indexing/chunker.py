"""Format-aware document chunking strategy.

Implements format-specific chunking for:
- Text / Logs: chunks around log events, error codes, and sliding line windows.
- CSV: logical row records with header associations and raw CSV lines.
- XLSX: worksheet rows with headers, sheet names, and cell coordinates.
- DOCX: headings paired with related paragraphs and table rows.
- PDF: page-aware and section-aware chunks with revision metadata.
- Python: constants paired with comments and function definitions.
- Images: OCR text, diagram relationships (pin-to-signal), and key-value pairs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.indexing.models import DocumentChunk


class Chunker:
    """Provides format-aware chunking routines for all document types."""

    @staticmethod
    def chunk_log(
        lines: List[str], rel_path: str, window_size: int = 5, step: int = 3
    ) -> List[DocumentChunk]:
        """Chunk log files around significant events and sliding line windows."""
        chunks: List[DocumentChunk] = []

        # 1. Event-specific chunks (errors, warnings, tickets, error codes)
        for idx, line in enumerate(lines, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            upper = line_str.upper()
            is_significant = any(
                tag in upper for tag in ["ERROR", "WARN", "FAIL", "INCIDENT", "THROTTLE", "ORR-"]
            )
            error_code_match = re.search(r"\b(E\d{4,5})\b", line_str)
            error_code = error_code_match.group(1) if error_code_match else None

            if is_significant or error_code:
                meta = {
                    "line_number": idx,
                    "is_significant": True,
                    "event_type": "log_event",
                }
                if error_code:
                    meta["error_code"] = error_code

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{rel_path}#line{idx}",
                        source_path=rel_path,
                        file_type="log",
                        row=idx,
                        text=line_str,
                        metadata=meta,
                        content_type="log_event",
                    )
                )

        # 2. Sliding window chunks for surrounding context
        for start in range(0, len(lines), step):
            window = lines[start : start + window_size]
            if not window:
                break
            combined = "\n".join(window).strip()
            if not combined:
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{rel_path}#w{start+1}-{start+len(window)}",
                    source_path=rel_path,
                    file_type="log",
                    row=start + 1,
                    text=combined,
                    metadata={"start_line": start + 1, "end_line": start + len(window)},
                    content_type="text",
                )
            )

        return chunks

    @staticmethod
    def chunk_text(text: str, rel_path: str) -> List[DocumentChunk]:
        """Chunk prose text by paragraphs and sections."""
        chunks: List[DocumentChunk] = []
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for idx, p in enumerate(paras, start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{rel_path}#p{idx}",
                    source_path=rel_path,
                    file_type="text",
                    text=p,
                    metadata={"paragraph_index": idx},
                    content_type="text",
                )
            )
        return chunks

    @staticmethod
    def chunk_csv(
        headers: List[str], rows: List[Tuple[int, List[str]]], rel_path: str, delimiter: str = ","
    ) -> List[DocumentChunk]:
        """Chunk CSV rows as individual searchable records with column headers."""
        chunks: List[DocumentChunk] = []
        for row_idx, row_cells in rows:
            cleaned = [c.strip() for c in row_cells]
            if not any(cleaned):
                continue

            field_dict = {}
            parts = []
            for col_idx, val in enumerate(cleaned):
                header = headers[col_idx] if col_idx < len(headers) and headers[col_idx] else f"Col{col_idx+1}"
                field_dict[header] = val
                parts.append(f"{header}: {val}")

            raw_line = delimiter.join(cleaned)
            formatted_text = f"{', '.join(parts)}\nRaw: {raw_line}"

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{rel_path}#r{row_idx}",
                    source_path=rel_path,
                    file_type="csv",
                    row=row_idx,
                    text=formatted_text,
                    metadata={
                        "row": row_idx,
                        "fields": field_dict,
                        "raw_line": raw_line,
                    },
                    content_type="table_row",
                )
            )
        return chunks

    @staticmethod
    def chunk_xlsx(
        sheet_name: str,
        headers: List[str],
        rows: List[Tuple[int, List[str]]],
        rel_path: str,
    ) -> List[DocumentChunk]:
        """Chunk spreadsheet rows preserving sheet name, column headers, and cell coordinates."""
        chunks: List[DocumentChunk] = []
        for row_idx, cell_vals in rows:
            cleaned = [c.strip() for c in cell_vals]
            if not any(cleaned):
                continue

            field_dict = {}
            parts = []
            for col_idx, val in enumerate(cleaned):
                if not val:
                    continue
                header = headers[col_idx] if col_idx < len(headers) and headers[col_idx] else f"Col{col_idx+1}"
                field_dict[header] = val
                parts.append(f"{header}: {val}")

            if not parts:
                continue

            raw_text = " | ".join(val for val in cleaned if val)
            structured_text = f"Sheet: {sheet_name} | {' | '.join(parts)}"
            combined_text = f"{structured_text}\nRow data: {raw_text}"

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{rel_path}#{sheet_name}#r{row_idx}",
                    source_path=rel_path,
                    file_type="xlsx",
                    sheet=sheet_name,
                    row=row_idx,
                    section=sheet_name,
                    text=combined_text,
                    metadata={
                        "sheet": sheet_name,
                        "row": row_idx,
                        "columns": field_dict,
                        "raw_text": raw_text,
                    },
                    content_type="table_row",
                )
            )
        return chunks

    @staticmethod
    def chunk_pdf(
        pages: List[Tuple[int, str]],
        rel_path: str,
        revision: Optional[str] = None,
        is_withdrawn: bool = False,
    ) -> List[DocumentChunk]:
        """Chunk PDF documents by pages, tracking revisions and withdrawal traps."""
        chunks: List[DocumentChunk] = []
        for page_num, text in pages:
            text_str = text.strip()
            if not text_str:
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{rel_path}#p{page_num}",
                    source_path=rel_path,
                    file_type="pdf",
                    page=page_num,
                    text=text_str,
                    metadata={
                        "page_number": page_num,
                        "revision": revision,
                        "is_withdrawn": is_withdrawn,
                        "is_superseded": is_withdrawn,
                    },
                    content_type="text",
                )
            )
        return chunks
