"""Interactive Web Playground Server for Match 3 RAG Pipeline.

Provides a local browser UI at http://localhost:8000 for evaluating:
- Real-time query execution across the mixed-format corpus
- Sample queries 1-10 with exact answer and minimal citation inspection
- Visual pinout & asset label inspections
- Corpus explorer and chunk inspection
- Full system diagnostic and contract compliance status

Note: This web server is for interactive local testing and visual verification.
The official evaluation container uses CLI /app/app.py and Unix socket /app/run/rag.sock.
"""

from __future__ import annotations

import http.server
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import urllib.parse

# Ensure app is importable
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.config import CORPUS_DIR, INDEX_DIR
from app.generation.qa import QAEngine
from app.indexing.crawler import crawl_corpus, load_persisted_index, persist_index
from app.indexing.models import DocumentChunk
from app.models.model_loader import get_device_info
from app.retrieval.hybrid import HybridRetriever

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [web] %(message)s")
logger = logging.getLogger(__name__)

PORT = 8000
_CORPUS_PATH = repo_root / "mc3-corpus"
_RETRIEVER: HybridRetriever | None = None
_QA_ENGINE: QAEngine | None = None
_LOCK = threading.Lock()


def get_pipeline():
    global _RETRIEVER, _QA_ENGINE
    with _LOCK:
        if _QA_ENGINE is not None and _RETRIEVER is not None:
            return _RETRIEVER, _QA_ENGINE

        retriever = HybridRetriever.load(INDEX_DIR)
        if retriever is None or not retriever.all_chunks:
            logger.info("Indexing corpus for web playground...")
            c_res = crawl_corpus(_CORPUS_PATH)
            persist_index(c_res, INDEX_DIR)
            retriever = HybridRetriever()
            retriever.build(c_res.all_chunks)
            retriever.save(INDEX_DIR)

        _RETRIEVER = retriever
        _QA_ENGINE = QAEngine(retriever)
        return _RETRIEVER, _QA_ENGINE


class RAGRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(repo_root / "web"), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({
                "status": "healthy",
                "device_info": get_device_info().to_dict(),
                "corpus_files": 13,
                "offline_mode": True,
                "evaluation_contract": "Match 3 RAG",
            })
        elif parsed.path == "/api/corpus":
            retriever, _ = get_pipeline()
            chunks = retriever.all_chunks if retriever else []
            files = {}
            for c in chunks:
                if c.source_path not in files:
                    files[c.source_path] = {"file_type": c.file_type, "chunks": 0}
                files[c.source_path]["chunks"] += 1
            self.send_json({"files": files, "total_chunks": len(chunks)})
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/query":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                data = json.loads(body.decode("utf-8"))
                query = data.get("query", "").strip()
            except Exception:
                self.send_json({"error": "Invalid JSON payload"}, status=400)
                return

            t0 = time.time()
            retriever, qa = get_pipeline()
            valid_files = {c.source_path for c in retriever.all_chunks}
            ans, cites, conf = qa.answer_query(query, valid_corpus_files=valid_files)
            duration_ms = (time.time() - t0) * 1000.0

            # Get supporting top chunks for visualization
            candidates = retriever.retrieve(query, top_k=3)
            evidence_previews = [
                {
                    "source_path": c.source_path,
                    "file_type": c.file_type,
                    "text": c.text[:300],
                    "score": round(score, 3),
                }
                for c, score in candidates
            ]

            self.send_json({
                "answer": ans,
                "citations": cites,
                "confidence": conf,
                "latency_ms": round(duration_ms, 2),
                "evidence_candidates": evidence_previews,
            })
        else:
            self.send_error(404, "Endpoint not found")

    def send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Concise logging
        logger.debug("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))


def start_server(port: int = PORT):
    # Initialize pipeline
    get_pipeline()
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), RAGRequestHandler)
    logger.info(f"RAG Web Playground running at: http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    start_server()
