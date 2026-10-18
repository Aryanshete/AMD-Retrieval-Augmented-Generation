"""Plain text, logs, source code and CSV."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from ..models import KIND_CODE, KIND_TABLE, KIND_TEXT, Chunk, ParsedDoc, Skip
from .chunking import is_superseded, looks_like_header, render_rows, split_windows

MAX_TEXT_BYTES = 40 * 1024 * 1024

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".c", ".h", ".cc", ".cpp", ".hpp", ".go", ".rs", ".rb",
    ".sh", ".bash", ".sql", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".json",
    ".xml", ".html", ".css", ".proto", ".tf", ".cu", ".hip", ".ipynb",
}  # fmt: skip
TEXT_EXTENSIONS = {
    ".txt",
    ".log",
    ".md",
    ".rst",
    ".text",
    ".out",
    ".err",
    ".readme",
    ".tex",
}


def read_text(path: Path) -> str:
    size = path.stat().st_size
    if size > MAX_TEXT_BYTES:
        raise Skip(f"too large ({size} bytes)")
    raw = path.read_bytes()
    if b"\x00" in raw[:4096]:
        raise Skip("binary content")
    for enc in ("utf-8-sig", "utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def printable_ratio(sample: bytes) -> float:
    if not sample:
        return 0.0
    ok = sum(1 for b in sample if b in (9, 10, 13) or 32 <= b < 127 or b >= 0xC2)
    return ok / len(sample)


def parse_text(path: Path, rel: str) -> ParsedDoc:
    text = read_text(path)
    if not text.strip():
        raise Skip("empty file")
    kind = KIND_CODE if path.suffix.lower() in CODE_EXTENSIONS else KIND_TEXT
    flagged = is_superseded(rel, text)
    doc = ParsedDoc(file=rel, superseded=flagged)
    for loc, body in split_windows(text):
        doc.chunks.append(Chunk(rel, loc, body, kind, flagged))
    return doc


def parse_csv(path: Path, rel: str) -> ParsedDoc:
    text = read_text(path)
    if not text.strip():
        raise Skip("empty file")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        pass
    rows = [
        r for r in csv.reader(io.StringIO(text), delimiter=delimiter) if any(c.strip() for c in r)
    ]
    if not rows:
        raise Skip("no rows")
    flagged = is_superseded(rel, text)
    doc = ParsedDoc(file=rel, superseded=flagged)
    if len(rows) > 1 and looks_like_header(rows[0]):
        for loc, body in render_rows(rows[0], rows[1:]):
            doc.chunks.append(Chunk(rel, loc, body, KIND_TABLE, flagged))
    else:
        for loc, body in split_windows("\n".join(" | ".join(r) for r in rows), "rows"):
            doc.chunks.append(Chunk(rel, loc, body, KIND_TABLE, flagged))
    return doc
