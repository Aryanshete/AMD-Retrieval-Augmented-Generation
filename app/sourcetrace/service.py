"""The resident service: owns the model, the embedder and the index."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from .answering import Answer, Reasoner
from .backends import create_backend
from .backends.base import LLMBackend
from .config import Settings
from .embed import try_load_embedder
from .parsers import ParseContext
from .prompts import IMAGE_INDEX_PROMPT, SYSTEM
from .store import Index, corpus_fingerprint
from .timing import Deadline
from .walker import list_files, walk_corpus

log = logging.getLogger("sourcetrace.service")

BackendFactory = Callable[[Settings], LLMBackend]
EmbedderFactory = Callable[[str], Any]


def same_path(a: str, b: str) -> bool:
    return os.path.realpath(a) == os.path.realpath(b)


def make_describer(
    backend: LLMBackend, settings: Settings, budget: Deadline
) -> Callable[[Image.Image, str], str]:
    """A picture-to-text function for indexing, bounded by the index time budget."""

    def describe(img: Image.Image, label: str) -> str:
        left = budget.remaining()
        if left < 8.0:
            return ""
        try:
            return backend.generate(
                IMAGE_INDEX_PROMPT,
                system=SYSTEM,
                image=img,
                max_new_tokens=420,
                timeout_s=min(45.0, left - 2.0),
            ).strip()
        except Exception:
            log.exception("could not describe %s", label)
            return ""

    return describe


def build_index(
    root: Path,
    settings: Settings,
    backend: LLMBackend | None,
    embedder: Any,
    budget: Deadline,
    *,
    with_images: bool = True,
) -> Index:
    files, _ = list_files(root)
    describe = make_describer(backend, settings, budget) if (backend and with_images) else None
    ctx = ParseContext(describe=describe, max_image_side=settings.max_image_side)
    result = walk_corpus(
        root,
        ctx,
        file_timeout_s=settings.file_timeout_s,
        images=with_images,
        image_time_left=budget.remaining,
    )
    index = Index.build(
        result.docs,
        result.skipped,
        str(root),
        corpus_fingerprint(files),
        embedder=embedder,
    )
    return index


class SourceTraceService:
    def __init__(
        self,
        settings: Settings,
        backend_factory: BackendFactory | None = None,
        embedder_factory: EmbedderFactory | None = None,
    ) -> None:
        self.settings = settings
        self._backend_factory = backend_factory or create_backend
        self._embedder_factory = embedder_factory or try_load_embedder
        self._lock = threading.Lock()  # one GPU job at a time
        self.backend: LLMBackend | None = None
        self.embedder: Any = None
        self.index: Index | None = None
        self.ready = threading.Event()

    # ------------------------------------------------------------------- lifecycle
    def load(self) -> None:
        backend = self._backend_factory(self.settings)
        backend.load()
        backend.warmup()
        self.backend = backend
        self.embedder = self._embedder_factory(self.settings.embed_path)
        try:
            self.index = Index.load(self.settings.index_dir)
            log.info("loaded persisted index: %d chunks", len(self.index))
        except Exception:  # noqa: BLE001 - normal on first start
            self.index = None
        self.ready.set()

    # -------------------------------------------------------------------- requests
    def handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "ping":
            return {
                "ok": True,
                "ready": self.ready.is_set(),
                "indexed": self.index is not None,
            }
        if op == "index":
            return self._op_index(request)
        if op == "query":
            return self._op_query(request)
        return {"ok": False, "error": f"unknown op: {op!r}"}

    def _wait_ready(self, seconds: float) -> bool:
        return self.ready.wait(timeout=max(0.0, seconds))

    def _op_index(self, request: dict) -> dict:
        started = time.monotonic()
        wait = float(request.get("deadline_s", self.settings.index_client_wait_s))
        corpus = request.get("corpus")
        if not isinstance(corpus, str) or not corpus:
            return {"ok": False, "error": "missing 'corpus'"}
        if not self._wait_ready(wait):
            return {"ok": False, "error": "model not ready in time"}
        with self._lock:
            # Time already spent waiting for the model counts against the 10 minute startup.
            budget = Deadline(
                max(
                    30.0,
                    min(
                        self.settings.index_budget_s,
                        wait - (time.monotonic() - started),
                    ),
                )
            )
            try:
                index = build_index(
                    Path(corpus), self.settings, self.backend, self.embedder, budget
                )
            except Exception as exc:
                log.exception("indexing failed")
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self.index = index
            try:
                index.save(self.settings.index_dir)
            except OSError as exc:
                log.warning("could not persist the index (%s); keeping it in memory", exc)
        return {
            "ok": True,
            "files": len(index.files),
            "chunks": len(index),
            "skipped": [list(s) for s in index.skipped],
            "seconds": round(time.monotonic() - started, 2),
        }

    def _index_for(self, corpus: Path, deadline: Deadline) -> Index | None:
        if self.index is not None and same_path(self.index.corpus_root, str(corpus)):
            return self.index
        try:
            loaded = Index.load(self.settings.index_dir)
            if same_path(loaded.corpus_root, str(corpus)):
                self.index = loaded
                return loaded
        except Exception:  # noqa: BLE001
            pass
        if deadline.remaining() < 6.0:
            return None
        log.warning("no usable index for %s; building a text-only one on demand", corpus)
        index = build_index(
            corpus, self.settings, self.backend, self.embedder,
            Deadline(min(15.0, deadline.remaining() - 4.0)), with_images=False,
        )  # fmt: skip
        self.index = index
        return index

    def _op_query(self, request: dict) -> dict:
        try:
            budget = float(request.get("deadline_s", self.settings.request_budget_s))
        except (TypeError, ValueError):
            budget = self.settings.request_budget_s
        deadline = Deadline(max(2.0, min(budget, self.settings.request_budget_s + 2.0) - 1.0))
        question = request.get("question")
        corpus = request.get("corpus")
        if not isinstance(question, str) or not isinstance(corpus, str):
            return {"ok": False, "error": "missing 'question' or 'corpus'"}
        if not self._wait_ready(deadline.remaining() - 1.0):
            return {"ok": False, "error": "model not ready yet"}
        if not self._lock.acquire(timeout=max(0.0, deadline.remaining() - 1.0)):
            return {"ok": False, "error": "busy"}
        try:
            index = self._index_for(Path(corpus), deadline)
            if index is None or self.backend is None:
                return {"ok": False, "error": "no index available"}
            result = Reasoner(self.backend, index, self.settings, self.embedder).answer(
                question, deadline, Path(corpus)
            )
        finally:
            self._lock.release()
        return _answer_reply(result)


def _answer_reply(result: Answer) -> dict:
    return {
        "ok": True,
        "answer": result.answer,
        "citations": result.citations,
        "confidence": result.confidence,
        "trace": result.trace,
    }
