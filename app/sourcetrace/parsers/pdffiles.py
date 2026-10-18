"""PDF text (and embedded pictures) via ``pypdf``. Encrypted files are skipped."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

from ..models import KIND_IMAGE, KIND_TEXT, Chunk, ParsedDoc, Skip
from .chunking import is_superseded, split_windows

ImageReader = Callable[[bytes, str], str]
MAX_PAGES = 400
MAX_IMAGE_PAGES = 30


def parse_pdf(path: Path, rel: str, read_image: ImageReader | None = None) -> ParsedDoc:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is in requirements.txt
        raise Skip("pypdf is not installed") from exc

    reader = PdfReader(str(path), strict=False)
    if reader.is_encrypted:
        # Nothing inside an encrypted file is ever the answer: never try to open it.
        raise Skip("encrypted PDF")

    pages_text: list[str] = []
    for page in list(reader.pages)[:MAX_PAGES]:
        try:
            pages_text.append((page.extract_text() or "").strip())
        except Exception:  # noqa: BLE001 - a broken page must not lose the others
            pages_text.append("")

    flagged = is_superseded(rel, "\n".join(pages_text[:1]))
    doc = ParsedDoc(file=rel, superseded=flagged)
    for number, text in enumerate(pages_text, start=1):
        for loc, body in split_windows(text, label=f"p{number} lines"):
            doc.chunks.append(Chunk(rel, loc, body, KIND_TEXT, flagged))

    if read_image is not None:
        for number, page in enumerate(list(reader.pages)[:MAX_IMAGE_PAGES], start=1):
            try:
                images = list(page.images)[:3]
            except Exception:  # noqa: BLE001
                continue
            for k, img in enumerate(images, start=1):
                try:
                    text = read_image(img.data, f"{rel}::p{number}-img{k}")
                except Exception:  # noqa: BLE001
                    continue
                if text.strip():
                    doc.chunks.append(Chunk(rel, f"p{number} image {k}", text, KIND_IMAGE, flagged))
    if not doc.chunks:
        raise Skip("no extractable text")
    return doc


def pdf_page_count(path: Path) -> int:  # pragma: no cover - convenience for scripts
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(path.read_bytes())).pages)
