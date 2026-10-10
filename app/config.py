"""System configuration and runtime defaults."""

from __future__ import annotations

import os
from pathlib import Path

# Paths configured via environment variables with defaults matching the contract
OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", os.environ.get("MC3_OUTPUT_DIR", "/app/output")))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
CORPUS_DIR = Path(os.environ.get("MC3_CORPUS_DIR", "/app/corpus"))
RUN_DIR = Path(os.environ.get("MC3_RUN_DIR", "/app/run"))
SOCKET_PATH = Path(os.environ.get("MC3_SOCKET_PATH", str(RUN_DIR / "rag.sock")))

# File extensions supported by the system
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".csv": "csv",
    ".txt": "text",
    ".log": "log",
    ".py": "source",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
}

# Tokens indicating superseded / withdrawn status
SUPERSEDED_MARKERS = [
    "withdrawn",
    "superseded",
    "obsolete",
    "retired",
    "preliminary silicon",
    "do not design",
]

# Standard refusal response adhering to contract
REFUSAL_RESPONSE = {
    "answer": "",
    "citations": [],
    "confidence": 0.0,
}
