"""Plaintext and log file parser.

Extracts structured log events, timestamps, error codes, and text paragraphs.
Preserves exact verbatim content.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from app.indexing.models import DocumentChunk, ParsedDocument


def parse_text(file_path: Path, rel_path: str) -> ParsedDocument:
    """Parse text and log files, preserving structured events and exact text."""
    try:
        content_bytes = file_path.read_bytes()
    except Exception as exc:
        return ParsedDocument(
            source_path=rel_path,
            file_type="log" if rel_path.endswith(".log") else "text",
            error=f"Unreadable file: {type(exc).__name__}: {exc}",
        )

    # Decode defensively
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
            file_type="text",
            error="EncodingDecodeError: unable to decode file content",
        )

    file_type = "log" if rel_path.endswith(".log") else "text"
    chunks: List[DocumentChunk] = []

    if file_type == "log":
        # Group log lines by events or small line blocks (e.g. 5 lines per chunk with 1 line overlap)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # 1. Individual significant log lines (e.g. containing ERROR, WARN, ticket, code)
        for idx, line in enumerate(lines, start=1):
            is_significant = any(k in line.upper() for k in ["ERROR", "WARN", "FAIL", "INCIDENT", "THROTTLE", "ORR-"])
            error_code_match = re.search(r"\b(E\d{4,5})\b", line)
            error_code = error_code_match.group(1) if error_code_match else None
            
            chunk_metadata = {
                "line_number": idx,
                "is_significant": is_significant,
            }
            if error_code:
                chunk_metadata["error_code"] = error_code

            # If it's a significant log line, create a high-priority event chunk
            if is_significant or error_code:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{rel_path}#line{idx}",
                        source_path=rel_path,
                        file_type="log",
                        row=idx,
                        text=line,
                        metadata=chunk_metadata,
                        content_type="log_event",
                    )
                )

        # 2. Windowed chunks for context
        window_size = 5
        step = 3
        for start in range(0, len(lines), step):
            window_lines = lines[start : start + window_size]
            if not window_lines:
                break
            combined_window = "\n".join(window_lines)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{rel_path}#w{start+1}-{start+len(window_lines)}",
                    source_path=rel_path,
                    file_type="log",
                    row=start + 1,
                    text=combined_window,
                    metadata={"start_line": start + 1, "end_line": start + len(window_lines)},
                    content_type="text",
                )
            )

    else:
        # Standard .txt files: paragraph based splitting
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for p_idx, para in enumerate(paragraphs, start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{rel_path}#p{p_idx}",
                    source_path=rel_path,
                    file_type="text",
                    text=para,
                    metadata={"paragraph_index": p_idx},
                    content_type="text",
                )
            )

    return ParsedDocument(
        source_path=rel_path,
        file_type=file_type,
        chunks=chunks,
        metadata={"total_chunks": len(chunks)},
    )
