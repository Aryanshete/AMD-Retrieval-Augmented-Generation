"""Writing ``/app/output/<query-id>_output.json`` - atomically, and always valid."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path

log = logging.getLogger("sourcetrace.output")


def output_path_for(query_id: str, output_dir: str | Path) -> Path:
    """The file is named after the query id the harness GAVE us, nothing else."""
    safe = re.sub(r"[\\/\x00]", "_", query_id.strip()) or "query"
    return Path(output_dir) / f"{safe}_output.json"


def _sanitize(answer: object, citations: object, confidence: object) -> dict:
    text = " ".join(str(answer if answer is not None else "").split())
    cites: list[str] = []
    if isinstance(citations, (list, tuple)):
        for c in citations:
            c = str(c).strip()
            if c and c not in cites:
                cites.append(c)
    if not text:  # a refusal is an empty answer AND an empty citation list
        cites = []
    try:
        conf = float(confidence)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        conf = 0.0
    if not text:
        conf = 0.0
    return {
        "answer": text,
        "citations": cites,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
    }


class ResultWriter:
    """Thread-safe, first-final-write-wins writer for one answer file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._final = False

    def _atomic_write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.chmod(tmp, 0o644)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    def write_provisional(self) -> None:
        """An empty (refusing) but valid answer, so a file exists from the first moment."""
        with self._lock:
            if self._final:
                return
            try:
                self._atomic_write(_sanitize("", [], 0.0))
            except OSError as exc:
                log.error("cannot write provisional answer to %s: %s", self.path, exc)

    def finalize(self, answer: object, citations: object, confidence: object) -> bool:
        with self._lock:
            if self._final:
                return False
            self._final = True
            try:
                self._atomic_write(_sanitize(answer, citations, confidence))
            except OSError as exc:
                log.error("cannot write answer to %s: %s", self.path, exc)
                return False
            return True
