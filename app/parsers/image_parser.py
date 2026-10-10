"""Image Parser for Match 3 RAG.

Extracts image metadata, performs OCR, and identifies visual/spatial relationships.
Produces canonical chunks adhering to Phase 4 and Phase 5 requirements.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from app.indexing.models import DocumentChunk, ParsedDocument
from app.parsers.vision import VisionEngine

logger = logging.getLogger(__name__)


def parse_image(file_path: Path, rel_path: str) -> ParsedDocument:
    """Parse image files, extracting dimensions, OCR text, and visual relationships."""
    try:
        from PIL import Image
    except ImportError:
        return ParsedDocument(
            source_path=rel_path,
            file_type="image",
            error="Pillow is not installed",
        )

    try:
        with Image.open(file_path) as img:
            width, height = img.size
            img_format = img.format
            img_mode = img.mode
    except Exception as exc:
        return ParsedDocument(
            source_path=rel_path,
            file_type="image",
            error=f"Corrupt or unreadable image: {type(exc).__name__}: {exc}",
        )

    chunks: List[DocumentChunk] = []

    # Run VisionEngine for OCR and diagram/key-value spatial reasoning
    vision_engine = VisionEngine.get_instance()
    v_res = vision_engine.analyze_image(file_path)

    # 1. Visual Relationships Chunk (Highest value for diagram and diagram-to-pin queries)
    if v_res.relationship_text:
        chunks.append(
            DocumentChunk(
                chunk_id=f"{rel_path}#relationships",
                source_path=rel_path,
                file_type="image",
                text=v_res.relationship_text,
                metadata={
                    "pin_mappings": v_res.pin_mappings,
                    "signal_to_pin": v_res.signal_to_pin,
                    "key_values": v_res.key_values,
                    "width": width,
                    "height": height,
                },
                content_type="image_relationships",
            )
        )

    # 2. Complete OCR Transcription Chunk
    if v_res.full_text:
        chunks.append(
            DocumentChunk(
                chunk_id=f"{rel_path}#ocr",
                source_path=rel_path,
                file_type="image",
                text=v_res.full_text,
                metadata={
                    "word_count": len(v_res.items),
                    "confidence_avg": (
                        sum(it.prob for it in v_res.items) / max(len(v_res.items), 1)
                        if v_res.items else 0.0
                    ),
                },
                content_type="image_ocr",
            )
        )

    # 3. Image Metadata Chunk
    meta_text = (
        f"Image file: {rel_path} | Format: {img_format} | "
        f"Dimensions: {width}x{height} | Mode: {img_mode}"
    )
    chunks.append(
        DocumentChunk(
            chunk_id=f"{rel_path}#meta",
            source_path=rel_path,
            file_type="image",
            text=meta_text,
            metadata={
                "width": width,
                "height": height,
                "format": img_format,
                "mode": img_mode,
            },
            content_type="image_metadata",
        )
    )

    return ParsedDocument(
        source_path=rel_path,
        file_type="image",
        chunks=chunks,
        metadata={
            "width": width,
            "height": height,
            "format": img_format,
            "ocr_words": len(v_res.items),
            "pin_count": len(v_res.pin_mappings),
            "key_value_count": len(v_res.key_values),
        },
    )
