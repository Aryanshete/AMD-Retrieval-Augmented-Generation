"""Dispatcher for routing files to their respective parsers based on extension."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

from app.config import SUPPORTED_EXTENSIONS
from app.indexing.models import ParsedDocument

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = set(SUPPORTED_EXTENSIONS.keys())


def is_supported(file_path: Path) -> bool:
    """Check if the given file has a supported extension."""
    return file_path.suffix.lower() in SUPPORTED_SUFFIXES


def dispatch_file(file_path: Path, rel_path: str) -> Optional[ParsedDocument]:
    """Route a file to the appropriate parser using lazy imports to avoid circular dependencies."""
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_SUFFIXES:
        logger.info(f"Skipping unsupported file type: {rel_path} ({ext})")
        return None

    try:
        if ext == ".pdf":
            from app.parsers.pdf_parser import parse_pdf
            return parse_pdf(file_path, rel_path)
        elif ext == ".docx":
            from app.parsers.docx_parser import parse_docx
            return parse_docx(file_path, rel_path)
        elif ext == ".xlsx":
            from app.parsers.xlsx_parser import parse_xlsx
            return parse_xlsx(file_path, rel_path)
        elif ext == ".csv":
            from app.parsers.csv_parser import parse_csv
            return parse_csv(file_path, rel_path)
        elif ext in [".txt", ".log"]:
            from app.parsers.text_parser import parse_text
            return parse_text(file_path, rel_path)
        elif ext == ".py":
            from app.parsers.source_parser import parse_source
            return parse_source(file_path, rel_path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            from app.parsers.image_parser import parse_image
            return parse_image(file_path, rel_path)
        else:
            return None

    except PermissionError as exc:
        logger.warning(f"Permission denied for {rel_path}: {exc}")
        return ParsedDocument(
            source_path=rel_path,
            file_type=SUPPORTED_EXTENSIONS.get(ext, "unknown"),
            error="permission_denied",
            metadata={"error_type": "PermissionError", "message": str(exc)},
        )
    except Exception as exc:
        logger.warning(f"Error parsing {rel_path}: {type(exc).__name__}: {exc}")
        return ParsedDocument(
            source_path=rel_path,
            file_type=SUPPORTED_EXTENSIONS.get(ext, "unknown"),
            error=type(exc).__name__,
            metadata={"error_type": type(exc).__name__, "message": str(exc)},
        )
