"""Chunking helpers and document-level flags."""

from __future__ import annotations

import re

SINGLE_CHUNK_CHARS = 3200
WINDOW_CHARS = 1500
OVERLAP_LINES = 2

# A revision that says it is no longer valid. Only the file name and the top of the
# document are inspected: a current datasheet's revision history may legitimately mention
# that an older revision "is withdrawn".
_SUPERSEDED_RE = re.compile(
    r"\b(withdrawn|superseded\s+by|obsolete|deprecated|do\s+not\s+use|no\s+longer\s+valid"
    r"|replaced\s+by)\b",
    re.IGNORECASE,
)
_HEAD_CHARS = 400


def is_superseded(rel_path: str, text: str) -> bool:
    name = rel_path.replace("_", " ").replace("-", " ")
    return bool(_SUPERSEDED_RE.search(name) or _SUPERSEDED_RE.search(text[:_HEAD_CHARS]))


def split_windows(text: str, label: str = "lines") -> list[tuple[str, str]]:
    """Split text into ``(loc, chunk_text)`` windows along line boundaries."""
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return []
    if len(text) <= SINGLE_CHUNK_CHARS:
        return [(f"{label} 1-{len(lines)}", text.strip())]
    out: list[tuple[str, str]] = []
    start = 0
    while start < len(lines):
        size = 0
        end = start
        while end < len(lines) and (size < WINDOW_CHARS or end == start):
            size += len(lines[end]) + 1
            end += 1
        body = "\n".join(lines[start:end]).strip()
        if body:
            out.append((f"{label} {start + 1}-{end}", body))
        if end >= len(lines):
            break
        start = max(start + 1, end - OVERLAP_LINES)
    return out


def split_paragraphs(paragraphs: list[str], label: str = "section") -> list[tuple[str, str]]:
    """Group paragraphs into windows of about ``WINDOW_CHARS`` characters."""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    size = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if buf and size + len(para) > WINDOW_CHARS:
            out.append((f"{label} {len(out) + 1}", "\n".join(buf)))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 1
    if buf:
        out.append((f"{label} {len(out) + 1}", "\n".join(buf)))
    return out


def render_rows(
    header: list[str], rows: list[list[str]], sheet: str = "", first_row_no: int = 2
) -> list[tuple[str, str]]:
    """Render table rows as ``column=value | ...`` and group them into chunks."""
    names = [h.strip() or f"col{i + 1}" for i, h in enumerate(header)]
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    first = first_row_no
    size = 0
    prefix = f"[sheet: {sheet}]\n" if sheet else ""
    for offset, row in enumerate(rows):
        cells = [
            f"{names[i] if i < len(names) else f'col{i + 1}'}={value.strip()}"
            for i, value in enumerate(row)
            if value.strip()
        ]
        if not cells:
            continue
        line = " | ".join(cells)
        if buf and (size + len(line) > 1200 or len(buf) >= 8):
            last = first_row_no + offset - 1
            out.append((f"rows {first}-{last}", prefix + "\n".join(buf)))
            buf, size, first = [], 0, first_row_no + offset
        buf.append(line)
        size += len(line) + 1
    if buf:
        out.append((f"rows {first}-{first_row_no + len(rows) - 1}", prefix + "\n".join(buf)))
    return out


def looks_like_header(row: list[str]) -> bool:
    cells = [c.strip() for c in row if c.strip()]
    if len(cells) < 2:
        return False
    if len(set(cells)) != len(cells):
        return False
    numeric = sum(1 for c in cells if re.fullmatch(r"[-+]?[\d.,]+%?", c))
    return numeric <= len(cells) // 3
