"""Visual diagram analysis and spatial relation extraction.

Extracts text, diagram connections (e.g. pin assignments, signal mappings),
and key-value pairs (e.g. board revision, model names) from images.
Defensively falls back if OCR models are missing or image is corrupt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VisionItem:
    text: str
    box: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    prob: float
    cx: float
    cy: float


@dataclass
class VisionResult:
    full_text: str = ""
    items: List[VisionItem] = field(default_factory=list)
    pin_mappings: Dict[str, str] = field(default_factory=dict)
    signal_to_pin: Dict[str, str] = field(default_factory=dict)
    key_values: Dict[str, str] = field(default_factory=dict)
    relationship_text: str = ""
    error: Optional[str] = None


class VisionEngine:
    _instance: Optional[VisionEngine] = None
    _reader: Any = None
    _init_attempted: bool = False

    @classmethod
    def get_instance(cls) -> VisionEngine:
        if cls._instance is None:
            cls._instance = VisionEngine()
        return cls._instance

    def _get_reader(self) -> Any:
        if self._init_attempted:
            return self._reader

        self._init_attempted = True
        try:
            # Set UTF-8 encoding for progress reporting safety
            os.environ.setdefault("PYTHONIOENCODING", "utf-8")
            import easyocr
            import torch
            has_gpu = torch.cuda.is_available()

            # Check offline storage paths
            ocr_dir = Path("/models/ocr")
            if not ocr_dir.exists():
                ocr_dir = Path(__file__).resolve().parent.parent.parent / "models" / "ocr"
            storage_dir = str(ocr_dir) if ocr_dir.exists() else None
            download_ok = not (ocr_dir.exists() or os.environ.get("MC3_OFFLINE", ""))

            self._reader = easyocr.Reader(
                ["en"],
                model_storage_directory=storage_dir,
                download_enabled=download_ok,
                gpu=has_gpu,
                verbose=False,
            )
            logger.info(f"Initialized EasyOCR reader (gpu={has_gpu}, storage={storage_dir})")
        except Exception as exc:
            logger.warning(f"EasyOCR reader could not be initialized: {exc}")
            self._reader = None

        return self._reader

    def analyze_image(self, file_path: Path) -> VisionResult:
        """Run OCR and visual relationship extraction on an image file."""
        reader = self._get_reader()
        if reader is None:
            return VisionResult(error="OCR engine unavailable")

        try:
            raw_results = reader.readtext(str(file_path))
        except Exception as exc:
            logger.error(f"OCR failed for {file_path}: {exc}")
            return VisionResult(error=f"OCR error: {type(exc).__name__}: {exc}")

        items: List[VisionItem] = []
        for bbox, text, prob in raw_results:
            text_str = str(text).strip()
            if not text_str:
                continue
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            items.append(
                VisionItem(
                    text=text_str,
                    box=(float(xmin), float(ymin), float(xmax), float(ymax)),
                    prob=float(prob),
                    cx=float((xmin + xmax) / 2),
                    cy=float((ymin + ymax) / 2),
                )
            )

        full_text = " ".join(item.text for item in items)
        pin_mappings: Dict[str, str] = {}
        signal_to_pin: Dict[str, str] = {}
        key_values: Dict[str, str] = {}

        # 1. Diagram Pinout Analysis
        # Detect pin identifiers (e.g. B11, B12, B14, J1, PIN1)
        pin_items = [it for it in items if re.match(r"^[A-Z]\d{1,3}$", it.text)]
        if pin_items:
            for pin in pin_items:
                # Find signal vertically aligned below the pin (cx close, cy > pin.cy)
                candidates = []
                for it in items:
                    if it == pin or it.text in [p.text for p in pin_items]:
                        continue
                    x_dist = abs(it.cx - pin.cx)
                    y_diff = it.cy - pin.cy
                    # Vertically aligned and within vertical span
                    if 10 < y_diff < 180 and x_dist < 60:
                        candidates.append((x_dist, y_diff, it.text))

                if candidates:
                    candidates.sort(key=lambda c: (c[0], c[1]))
                    signal = candidates[0][2]
                    pin_mappings[pin.text] = signal
                    signal_to_pin[signal] = pin.text
                    # Also map normalized signal (e.g. THERM_ALERT without trailing #)
                    signal_norm = signal.rstrip("#")
                    if signal_norm != signal:
                        signal_to_pin[signal_norm] = pin.text

        # 2. Key-Value Extraction (Horizontal alignment + single-line label extraction)
        for i, it in enumerate(items):
            # Check single-box key-values (e.g. "BOARD REVISION REV-C2", "MODEL TQ-40")
            m_kv = re.match(r"^(MODEL|BOARD\s+REVISION|REVISION|REV|SN|SERIAL\s+NUMBER|PART\s+NUMBER|PN|LOT|BATCH|MFG|DATE)[\s:]+(.+)$", it.text, re.IGNORECASE)
            if m_kv:
                k_clean = m_kv.group(1).strip().upper()
                v_clean = m_kv.group(2).strip()
                key_values[k_clean] = v_clean
                if "REVISION" in k_clean:
                    key_values["BOARD REVISION"] = v_clean

            # Check direct revision pattern in item text
            m_rev = re.search(r"\b(REV-[A-Z0-9]+)\b", it.text, re.IGNORECASE)
            if m_rev:
                key_values["BOARD REVISION"] = m_rev.group(1).upper()
                key_values["REVISION"] = m_rev.group(1).upper()

            # Check horizontal alignment across adjacent boxes
            for j, other in enumerate(items):
                if i == j:
                    continue
                y_diff = abs(it.cy - other.cy)
                x_gap = other.box[0] - it.box[2]
                if y_diff < 30 and -10 < x_gap < 200:
                    key = it.text.strip().rstrip(":")
                    val = other.text.strip()
                    key_values[key] = val
                    if "revision" in key.lower():
                        key_values["BOARD REVISION"] = val

        # Also inspect full OCR text for standard label patterns
        m_full_rev = re.search(r"BOARD\s+REVISION[:\s]+(REV-[A-Z0-9]+)", full_text, re.IGNORECASE)
        if m_full_rev:
            key_values["BOARD REVISION"] = m_full_rev.group(1).upper()

        m_full_mod = re.search(r"MODEL[:\s]+([A-Z0-9-]+)", full_text, re.IGNORECASE)
        if m_full_mod:
            key_values["MODEL"] = m_full_mod.group(1).upper()

        # Compose structured relationship description text
        rel_lines = []
        if pin_mappings:
            rel_lines.append("Diagram Pin Assignments:")
            for pin, sig in sorted(pin_mappings.items()):
                rel_lines.append(f"Pin {pin} -> {sig} (Signal {sig} is connected to Pin {pin})")

        if key_values:
            rel_lines.append("\nKey-Value Pairs:")
            for k, v in sorted(key_values.items()):
                rel_lines.append(f"{k}: {v}")

        rel_text = "\n".join(rel_lines)

        return VisionResult(
            full_text=full_text,
            items=items,
            pin_mappings=pin_mappings,
            signal_to_pin=signal_to_pin,
            key_values=key_values,
            relationship_text=rel_text,
        )
