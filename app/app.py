#!/usr/bin/env python3
"""CLI interface for corpus indexing and question answering.

Supports:
1. Indexing pass:
   python3 /app/app.py --index /app/corpus
2. Per-query evaluation:
   python3 /app/app.py --corpus /app/corpus --query-id <id> --query "..."

Communicates with long-running worker daemon via Unix domain socket /app/run/rag.sock
when available, with seamless in-process fallback for standalone / local execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import socket
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

# Robust module path resolution for both local development and Docker runtime
_current_dir = Path(__file__).resolve().parent
if _current_dir.name == "app":
    _repo_root = _current_dir.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
else:
    if str(_current_dir) not in sys.path:
        sys.path.insert(0, str(_current_dir))

if "app" not in sys.modules:
    try:
        import app
    except ImportError:
        _app_mod = types.ModuleType("app")
        _app_mod.__path__ = [str(_current_dir)]
        sys.modules["app"] = _app_mod

from app.config import INDEX_DIR, OUTPUT_DIR, REFUSAL_RESPONSE, SOCKET_PATH
from app.indexing.crawler import CrawlResult, crawl_corpus, load_persisted_index, persist_index
from app.indexing.models import DocumentChunk

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [app] %(message)s",
)
logger = logging.getLogger(__name__)


def send_to_worker(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attempt communication with worker daemon over Unix domain socket."""
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
        logger.debug(f"Worker socket communication failed ({exc}), falling back to direct execution")

    return None


def index(corpus: Path) -> None:
    """Build and persist the complete corpus index."""
    resp = send_to_worker({"command": "index", "corpus": str(corpus.resolve())})
    if resp and resp.get("status") == "ok":
        logger.info("Indexed corpus via worker daemon")
        return

    # In-process execution fallback
    logger.info(f"Indexing corpus directly in-process: {corpus}")
    crawl_result = crawl_corpus(corpus)
    persist_index(crawl_result, INDEX_DIR)
    logger.info(f"Direct indexing complete: {crawl_result.summary()}")


def _normalize_answer(ans: str) -> str:
    """Normalize answer to value-only, preserving qualifiers."""
    ans = ans.strip()
    ans = ans.strip("\"' \t\r\n")
    return ans


def execute_query(
    corpus: Path, query: str, index_data: Optional[CrawlResult] = None
) -> Tuple[str, List[str], float]:
    """Execute retrieval and answer extraction across parsed chunks via QAEngine."""
    from app.generation.qa import QAEngine
    from app.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever.load(INDEX_DIR)
    if retriever is None or not retriever.all_chunks:
        if index_data is None:
            index_data = load_persisted_index(INDEX_DIR)
            if index_data is None:
                index_data = crawl_corpus(corpus)
                persist_index(index_data, INDEX_DIR)
        retriever = HybridRetriever()
        retriever.build(index_data.all_chunks)

    valid_files = {c.source_path for c in retriever.all_chunks}
    qa_engine = QAEngine(retriever)
    return qa_engine.answer_query(query, valid_corpus_files=valid_files)


def answer(corpus: Path, query: str) -> Tuple[str, List[str], float]:
    """Answer a single query adhering to contract."""
    resp = send_to_worker({
        "command": "query",
        "corpus": str(corpus.resolve()),
        "query": query,
        "query_id": "ephemeral",
    })
    if resp and resp.get("status") == "ok":
        return resp.get("answer", ""), resp.get("citations", []), resp.get("confidence", 0.0)

    return execute_query(corpus, query)


def main() -> int:
    ap = argparse.ArgumentParser(description="Corpus indexing and query answering engine")
    ap.add_argument("--index", type=Path, help="Build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="Corpus root directory for answering a query")
    ap.add_argument("--query-id", help="Harness-assigned output stem, e.g. query_01")
    ap.add_argument("--query", help="Question text to answer")
    args = ap.parse_args()

    # 1. Index mode
    if args.index is not None:
        index(args.index)
        return 0

    # 2. Query mode
    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("A query requires --corpus, --query-id, and --query")

    ans_text, citations, confidence = answer(args.corpus, args.query)

    # Citation validation and consistency check
    # Check A: if answer != "", citations must not be empty
    if ans_text and not citations:
        logger.warning("Answer produced without citations; refusing to prevent penalty")
        ans_text, citations, confidence = "", [], 0.0

    # Check B: all citations must be relative posix paths
    clean_citations = []
    for c in citations:
        c_clean = str(c).replace("\\", "/").strip()
        if c_clean.startswith("/app/corpus/"):
            c_clean = c_clean[len("/app/corpus/"):]
        elif c_clean.startswith("/"):
            c_clean = c_clean.lstrip("/")
        clean_citations.append(c_clean)

    # Check C & D: Deduplicate while preserving order
    seen = set()
    deduped_citations = []
    for c in clean_citations:
        if c not in seen:
            seen.add(c)
            deduped_citations.append(c)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{args.query_id}_output.json"

    payload = {
        "answer": ans_text,
        "citations": deduped_citations,
        "confidence": round(confidence, 4),
    }

    payload_json = json.dumps(payload, ensure_ascii=False) + "\n"
    tmp_file = OUTPUT_DIR / f"{args.query_id}_output.json.tmp_{os.getpid()}"
    tmp_file.write_text(payload_json, encoding="utf-8")
    tmp_file.replace(out_file)
    logger.info(f"Wrote atomically {out_file} -> answer={ans_text!r}, citations={deduped_citations}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
