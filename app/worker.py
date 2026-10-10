"""Background worker daemon for persistent in-memory indexing and querying.

Maintains resident index, models, and retrieval pipeline in memory to satisfy
the < 30-second per-query evaluation budget and prevent re-indexing or model reload.

Listens on Unix domain socket:
    /app/run/rag.sock
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import types
from typing import Any, Dict, Optional, Tuple

# Module path resolution
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

from app.config import INDEX_DIR, RUN_DIR, SOCKET_PATH
from app.generation.qa import QAEngine
from app.indexing.crawler import CrawlResult, crawl_corpus, load_persisted_index, persist_index
from app.retrieval.hybrid import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [worker] %(message)s",
)
logger = logging.getLogger(__name__)

_RESIDENT_INDEX: Optional[CrawlResult] = None
_RESIDENT_RETRIEVER: Optional[HybridRetriever] = None
_RESIDENT_QA_ENGINE: Optional[QAEngine] = None
_INDEX_LOCK = threading.Lock()
_SHUTDOWN_EVENT = threading.Event()


def get_resident_pipeline(corpus_path: Optional[Path] = None) -> Tuple[Optional[HybridRetriever], Optional[QAEngine]]:
    """Retrieve or initialize resident HybridRetriever and QAEngine in memory."""
    global _RESIDENT_INDEX, _RESIDENT_RETRIEVER, _RESIDENT_QA_ENGINE

    with _INDEX_LOCK:
        if _RESIDENT_QA_ENGINE is not None and _RESIDENT_RETRIEVER is not None:
            return _RESIDENT_RETRIEVER, _RESIDENT_QA_ENGINE

        # 1. Try loading persisted index and hybrid retriever from disk
        retriever = HybridRetriever.load(INDEX_DIR)
        persisted = load_persisted_index(INDEX_DIR)

        if retriever is not None and retriever.all_chunks:
            _RESIDENT_RETRIEVER = retriever
            _RESIDENT_INDEX = persisted
            _RESIDENT_QA_ENGINE = QAEngine(retriever)
            logger.info(f"Loaded resident pipeline from {INDEX_DIR} ({len(retriever.all_chunks)} chunks)")
            return _RESIDENT_RETRIEVER, _RESIDENT_QA_ENGINE

        # 2. If corpus path provided and unindexed, index it now
        if corpus_path and Path(corpus_path).exists():
            logger.info(f"Building resident pipeline on corpus: {corpus_path}")
            res = crawl_corpus(Path(corpus_path))
            persist_index(res, INDEX_DIR)
            _RESIDENT_INDEX = res
            retriever = HybridRetriever()
            retriever.build(res.all_chunks)
            retriever.save(INDEX_DIR)
            _RESIDENT_RETRIEVER = retriever
            _RESIDENT_QA_ENGINE = QAEngine(retriever)
            logger.info(f"Initialized resident pipeline ({len(res.all_chunks)} chunks)")
            return _RESIDENT_RETRIEVER, _RESIDENT_QA_ENGINE

        return None, None


def handle_index(corpus_path: str) -> Dict[str, Any]:
    """Execute complete corpus indexing and update resident pipeline."""
    global _RESIDENT_INDEX, _RESIDENT_RETRIEVER, _RESIDENT_QA_ENGINE
    c_path = Path(corpus_path).resolve()
    logger.info(f"Indexing corpus pass: {c_path}")

    result = crawl_corpus(c_path)
    persist_index(result, INDEX_DIR)

    retriever = HybridRetriever()
    retriever.build(result.all_chunks)
    retriever.save(INDEX_DIR)

    with _INDEX_LOCK:
        _RESIDENT_INDEX = result
        _RESIDENT_RETRIEVER = retriever
        _RESIDENT_QA_ENGINE = QAEngine(retriever)

    logger.info(f"Worker indexing pass complete: {result.summary()}")
    return {"status": "ok", "summary": result.summary()}


def handle_query(corpus_path: str, query: str, query_id: str) -> Dict[str, Any]:
    """Process query using resident pipeline in memory (sub-second latency)."""
    c_path = Path(corpus_path).resolve() if corpus_path else None
    retriever, qa_engine = get_resident_pipeline(c_path)

    if qa_engine is None or retriever is None:
        return {
            "status": "ok",
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "warning": "No index available",
        }

    valid_files = {c.source_path for c in retriever.all_chunks}
    ans, citations, conf = qa_engine.answer_query(query, valid_corpus_files=valid_files)

    return {
        "status": "ok",
        "answer": ans,
        "citations": citations,
        "confidence": conf,
    }


def process_client_connection(conn: socket.socket):
    """Handle client request over Unix domain socket."""
    with conn:
        try:
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in chunk:
                    break

            if not data:
                return

            req = json.loads(data.decode("utf-8").strip())
            cmd = req.get("command")

            if cmd == "index":
                resp = handle_index(req["corpus"])
            elif cmd == "query":
                resp = handle_query(req.get("corpus", ""), req["query"], req.get("query_id", ""))
            elif cmd == "ping":
                resp = {"status": "ok", "message": "pong"}
            elif cmd == "shutdown":
                resp = {"status": "ok", "message": "shutting down"}
                _SHUTDOWN_EVENT.set()
            else:
                resp = {"status": "error", "message": f"Unknown command: {cmd}"}

            out_data = json.dumps(resp, ensure_ascii=False) + "\n"
            conn.sendall(out_data.encode("utf-8"))
        except Exception as exc:
            logger.error(f"Worker client error: {exc}", exc_info=True)
            err_resp = json.dumps({"status": "error", "message": str(exc)}) + "\n"
            try:
                conn.sendall(err_resp.encode("utf-8"))
            except Exception:
                pass


def run_worker_server(socket_path: Optional[Path] = None, run_once: bool = False):
    """Start the Unix domain socket server and keep process alive."""
    target_socket = Path(socket_path) if socket_path else SOCKET_PATH
    target_socket.parent.mkdir(parents=True, exist_ok=True)

    if target_socket.exists():
        try:
            target_socket.unlink()
        except OSError:
            pass

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(target_socket))
        server.listen(32)
        server.settimeout(1.0)
        logger.info(f"Worker listening on Unix domain socket: {target_socket}")

        # Warm up resident pipeline if index already exists
        get_resident_pipeline()

        while not _SHUTDOWN_EVENT.is_set():
            try:
                conn, _ = server.accept()
                t = threading.Thread(target=process_client_connection, args=(conn,), daemon=True)
                t.start()
                if run_once:
                    t.join()
                    break
            except socket.timeout:
                continue
            except (KeyboardInterrupt, SystemExit):
                break
    except Exception as exc:
        logger.error(f"Failed running socket server: {exc}", exc_info=True)
    finally:
        server.close()
        if target_socket.exists():
            try:
                target_socket.unlink()
            except OSError:
                pass
        logger.info("Worker socket server terminated cleanly.")


if __name__ == "__main__":
    run_worker_server()
