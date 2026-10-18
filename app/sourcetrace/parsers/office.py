"""Word and Excel files, read straight from the OOXML zip (standard library only)."""

from __future__ import annotations

import posixpath
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree as ET

from ..models import KIND_IMAGE, KIND_TABLE, KIND_TEXT, Chunk, ParsedDoc, Skip
from .chunking import is_superseded, looks_like_header, render_rows, split_paragraphs

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"

MAX_XML_BYTES = 120 * 1024 * 1024
ImageReader = Callable[[bytes, str], str]


def _read(zf: zipfile.ZipFile, name: str) -> bytes:
    info = zf.getinfo(name)
    if info.file_size > MAX_XML_BYTES:
        raise Skip(f"{name} is too large")
    return zf.read(name)


# ------------------------------------------------------------------------------- docx
def _para_text(p: ET.Element) -> str:
    out: list[str] = []
    for node in p.iter():
        if node.tag == f"{W}t" and node.text:
            out.append(node.text)
        elif node.tag in (f"{W}tab",):
            out.append(" ")
        elif node.tag in (f"{W}br", f"{W}cr"):
            out.append("\n")
    return "".join(out).strip()


def _table_rows(tbl: ET.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in tbl.findall(f"{W}tr"):
        cells = []
        for tc in tr.findall(f"{W}tc"):
            cells.append(" ".join(t for t in (_para_text(p) for p in tc.iter(f"{W}p")) if t))
        if any(c.strip() for c in cells):
            rows.append(cells)
    return rows


def _docx_blocks(root: ET.Element) -> list[tuple[str, object]]:
    body = root.find(f"{W}body")
    if body is None:
        return []
    blocks: list[tuple[str, object]] = []
    for child in body:
        if child.tag == f"{W}p":
            text = _para_text(child)
            if text:
                blocks.append(("p", text))
        elif child.tag == f"{W}tbl":
            rows = _table_rows(child)
            if rows:
                blocks.append(("t", rows))
        elif child.tag == f"{W}sdt":
            for p in child.iter(f"{W}p"):
                text = _para_text(p)
                if text:
                    blocks.append(("p", text))
    return blocks


def parse_docx(path: Path, rel: str, read_image: ImageReader | None = None) -> ParsedDoc:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise Skip("not a valid docx (or encrypted)") from exc
    with zf:
        names = set(zf.namelist())
        if "word/document.xml" not in names:
            raise Skip("no word/document.xml (encrypted or not a docx)")
        root = ET.fromstring(_read(zf, "word/document.xml"))
        blocks = _docx_blocks(root)
        extras: list[str] = []
        for name in sorted(names):
            if re.fullmatch(r"word/(header|footer|footnotes|endnotes|comments)\d*\.xml", name):
                try:
                    part = ET.fromstring(_read(zf, name))
                    extras.extend(t for t in (_para_text(p) for p in part.iter(f"{W}p")) if t)
                except (ET.ParseError, Skip):
                    continue
        head = " ".join(str(b[1]) for b in blocks[:3] if b[0] == "p")
        flagged = is_superseded(rel, head)
        doc = ParsedDoc(file=rel, superseded=flagged)

        paragraphs: list[str] = []
        pending_tables: list[tuple[str, str]] = []
        n_table = 0
        for kind, payload in blocks:
            if kind == "p":
                paragraphs.append(str(payload))
                continue
            rows: list[list[str]] = payload  # type: ignore[assignment]
            n_table += 1
            if len(rows) > 1 and looks_like_header(rows[0]):
                rendered = render_rows(rows[0], rows[1:], sheet=f"table {n_table}")
            else:
                rendered = [(f"table {n_table}", "\n".join(" | ".join(r) for r in rows))]
            pending_tables.extend(rendered)
        for loc, body_text in split_paragraphs(paragraphs):
            doc.chunks.append(Chunk(rel, loc, body_text, KIND_TEXT, flagged))
        for loc, body_text in pending_tables:
            doc.chunks.append(Chunk(rel, f"table {loc}", body_text, KIND_TABLE, flagged))
        if extras:
            doc.chunks.append(Chunk(rel, "headers/notes", "\n".join(extras), KIND_TEXT, flagged))

        if read_image is not None:
            media = [n for n in sorted(names) if n.startswith("word/media/")]
            for name in media[:12]:
                try:
                    text = read_image(zf.read(name), f"{rel}::{posixpath.basename(name)}")
                except Exception:  # noqa: BLE001 - one bad picture must not lose the document
                    continue
                if text.strip():
                    doc.chunks.append(
                        Chunk(
                            rel,
                            f"image {posixpath.basename(name)}",
                            text,
                            KIND_IMAGE,
                            flagged,
                        )
                    )
    if not doc.chunks:
        raise Skip("no readable text")
    return doc


# ------------------------------------------------------------------------------- xlsx
def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Za-z]+", ref)
    n = 0
    for ch in (letters.group(0) if letters else "A").upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(_read(zf, "xl/sharedStrings.xml"))
    out = []
    for si in root.findall(f"{S}si"):
        out.append("".join(t.text or "" for t in si.iter(f"{S}t")))
    return out


def _sheet_targets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    wb = ET.fromstring(_read(zf, "xl/workbook.xml"))
    rels: dict[str, str] = {}
    if "xl/_rels/workbook.xml.rels" in zf.namelist():
        for rel in ET.fromstring(_read(zf, "xl/_rels/workbook.xml.rels")):
            rels[rel.get("Id", "")] = rel.get("Target", "")
    out = []
    sheets = wb.find(f"{S}sheets")
    for i, sheet in enumerate(sheets if sheets is not None else [], start=1):
        target = rels.get(sheet.get(f"{R}id", ""), f"worksheets/sheet{i}.xml")
        target = target.lstrip("/")
        path = target if target.startswith("xl/") else posixpath.join("xl", target)
        out.append((sheet.get("name", f"Sheet{i}"), posixpath.normpath(path)))
    return out


def _fmt_number(value: str) -> str:
    try:
        f = float(value)
    except ValueError:
        return value
    return str(int(f)) if f.is_integer() and abs(f) < 1e15 else repr(f)


def _read_sheet(zf: zipfile.ZipFile, name: str, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(_read(zf, name))
    rows: list[list[str]] = []
    data = root.find(f"{S}sheetData")
    for row in data if data is not None else []:
        cells: dict[int, str] = {}
        for c in row.findall(f"{S}c"):
            kind = c.get("t", "n")
            v = c.find(f"{S}v")
            if kind == "inlineStr":
                text = "".join(t.text or "" for t in c.iter(f"{S}t"))
            elif v is None or v.text is None:
                continue
            elif kind == "s":
                idx = int(v.text)
                text = shared[idx] if 0 <= idx < len(shared) else ""
            elif kind == "b":
                text = "TRUE" if v.text.strip() == "1" else "FALSE"
            elif kind in ("str", "e"):
                text = v.text
            else:
                text = _fmt_number(v.text)
            cells[_col_index(c.get("r", "A1"))] = text
        if cells:
            width = max(cells) + 1
            rows.append([cells.get(i, "") for i in range(width)])
    return rows


def parse_xlsx(path: Path, rel: str) -> ParsedDoc:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise Skip("not a valid xlsx (or encrypted)") from exc
    with zf:
        if "xl/workbook.xml" not in zf.namelist():
            raise Skip("no xl/workbook.xml (encrypted or not an xlsx)")
        shared = _shared_strings(zf)
        flagged = is_superseded(rel, "")
        doc = ParsedDoc(file=rel, superseded=flagged)
        for sheet_name, target in _sheet_targets(zf):
            try:
                rows = _read_sheet(zf, target, shared)
            except (KeyError, ET.ParseError, Skip, ValueError):
                continue
            if not rows:
                continue
            if len(rows) > 1 and looks_like_header(rows[0]):
                first_row = 2
                rendered = render_rows(rows[0], rows[1:], sheet=sheet_name, first_row_no=first_row)
            else:
                lines = " | ".join
                text = "\n".join(lines(c for c in r if c.strip()) for r in rows)
                rendered = [(f"rows 1-{len(rows)}", f"[sheet: {sheet_name}]\n{text}")]
            for loc, body in rendered:
                doc.chunks.append(Chunk(rel, f"{sheet_name} {loc}", body, KIND_TABLE, flagged))
    if not doc.chunks:
        raise Skip("no cells")
    return doc
