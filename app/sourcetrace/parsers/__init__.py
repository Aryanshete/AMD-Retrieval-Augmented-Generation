"""File parsers. Every parser raises on trouble; the walker catches per file."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..models import ParsedDoc, Skip
from .imagefiles import load_image, parse_image
from .office import parse_docx, parse_xlsx
from .pdffiles import parse_pdf
from .textfiles import (
    CODE_EXTENSIONS,
    TEXT_EXTENSIONS,
    parse_csv,
    parse_text,
    printable_ratio,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
Describer = Callable[[Image.Image, str], str]


@dataclass
class ParseContext:
    """Optional capabilities handed to parsers."""

    describe: Describer | None = None  # vision model: image -> text
    read_embedded: bool = True  # also read pictures inside PDF / DOCX files
    max_image_side: int = 1600

    def embedded_reader(self) -> Callable[[bytes, str], str] | None:
        if self.describe is None or not self.read_embedded:
            return None
        describe, side = self.describe, self.max_image_side

        def read(data: bytes, label: str) -> str:
            try:
                img = load_image(data, side)
            except Exception:  # noqa: BLE001
                return ""
            if min(img.size) < 120:  # icons and rules are not worth a model call
                return ""
            return describe(img, label) or ""

        return read


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def parse_file(path: Path, rel: str, ctx: ParseContext | None = None) -> ParsedDoc:
    """Parse one file or raise (``Skip`` for 'leave it out', anything else for a real error)."""
    ctx = ctx or ParseContext()
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return parse_image(path, rel, ctx.describe, ctx.max_image_side)
    if ext == ".pdf":
        return parse_pdf(path, rel, ctx.embedded_reader())
    if ext in (".docx", ".docm"):
        return parse_docx(path, rel, ctx.embedded_reader())
    if ext in (".xlsx", ".xlsm"):
        return parse_xlsx(path, rel)
    if ext in (".csv", ".tsv"):
        return parse_csv(path, rel)
    if ext in TEXT_EXTENSIONS or ext in CODE_EXTENSIONS:
        return parse_text(path, rel)
    # Unknown extension: take it only if it really is text, otherwise leave it out.
    with path.open("rb") as handle:
        head = handle.read(4096)
    if head.startswith(b"%PDF"):
        return parse_pdf(path, rel, ctx.embedded_reader())
    if head and b"\x00" not in head and printable_ratio(head) > 0.97:
        return parse_text(path, rel)
    raise Skip(f"unknown file type {ext or '(none)'}")
