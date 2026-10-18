#!/usr/bin/env python3
"""Evaluation entry point for AMD RAG System.

Supports the following harness invocations:
1. Indexing pass:
   python3 /app/app.py --index /app/corpus
2. Per-query evaluation:
   python3 /app/app.py --corpus /app/corpus --query-id <id> --query "..."
3. Resident model server (started by container CMD):
   python3 /app/app.py --serve
4. Docker container healthcheck:
   python3 /app/app.py --healthcheck

Outputs to {args.query_id}_output.json with valid JSON (written atomically via .tmp_ pattern) containing:
- "answer": direct concise string without conversational filler
- "citations": minimal necessary relative paths (e.g., ["specs/tq40.pdf"])
- "confidence": float in [0.0, 1.0]
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, Dict, Optional

_STARTED_AT = time.monotonic()

# Ensure app package and parent root are in sys.path
_current_dir = Path(__file__).resolve().parent
_repo_root = _current_dir.parent
for _p in [str(_current_dir), str(_repo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from app.config import SOCKET_PATH
except Exception:
    SOCKET_PATH = Path("/app/run/rag.sock")

logger = logging.getLogger("app")


def send_to_worker(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Backward compatibility helper for communication over Unix domain socket."""
    if not hasattr(socket, "AF_UNIX") or not SOCKET_PATH.exists():
        return None

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(20.0)
        client.connect(str(SOCKET_PATH))
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        client.sendall(data)

        resp_bytes = b""
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            resp_bytes += chunk
            if b"\n" in chunk:
                break
        client.close()

        if resp_bytes:
            return json.loads(resp_bytes.decode("utf-8").strip())
    except Exception as exc:
        logger.debug(f"Worker socket communication failed ({exc})")

    return None


def index(corpus: Path) -> None:
    """Build and persist index over the corpus."""
    from sourcetrace.config import Settings
    from sourcetrace.cli import run_index
    settings = Settings.from_env()
    run_index(corpus, settings)


def execute_query(corpus: Path, query_id: str, query_text: str) -> Dict[str, Any]:
    """Execute a query and return normalized result payload."""
    from sourcetrace.config import Settings
    from sourcetrace.cli import run_query
    from sourcetrace.output import output_path_for
    settings = Settings.from_env()
    run_query(corpus, query_id, query_text, settings)
    out_file = output_path_for(query_id, settings.output_dir)
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    return {"answer": "", "citations": [], "confidence": 0.0}


from sourcetrace.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], started_at=_STARTED_AT))
