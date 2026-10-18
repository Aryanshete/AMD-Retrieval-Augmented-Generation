"""Images: the text printed inside them is read by the vision-language model."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageOps

from ..models import KIND_IMAGE, Chunk, ParsedDoc, Skip

Describer = Callable[[Image.Image, str], str]
MAX_PIXELS = 120_000_000


def load_image(source: Path | bytes, max_side: int = 1600) -> Image.Image:
    """Open an image as upright RGB, downscaled so its longest side is ``max_side``."""
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    img = Image.open(io.BytesIO(source) if isinstance(source, bytes) else str(source))
    img.load()
    img = ImageOps.exif_transpose(img) or img
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        base = Image.new("RGB", rgba.size, (255, 255, 255))
        base.paste(rgba, mask=rgba.split()[-1])
        img = base
    elif img.mode != "RGB":
        img = img.convert("RGB")
    longest = max(img.size)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    elif longest < 448:
        scale = 448 / longest
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


def stub_text(rel: str) -> str:
    return f"[image file] {rel}"


def parse_image(
    path: Path, rel: str, describe: Describer | None = None, max_side: int = 1600
) -> ParsedDoc:
    try:
        img = load_image(path, max_side)
    except Exception as exc:
        raise Skip(f"cannot open image: {exc}") from exc
    caption = ""
    if describe is not None:
        caption = (describe(img, rel) or "").strip()
    text = stub_text(rel) + (f"\n{caption}" if caption else "")
    doc = ParsedDoc(file=rel, note="captioned" if caption else "stub")
    doc.chunks.append(Chunk(rel, "image", text, KIND_IMAGE, False))
    return doc
