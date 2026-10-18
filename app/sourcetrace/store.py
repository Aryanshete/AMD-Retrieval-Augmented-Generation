"""The searchable index: chunks, BM25 and (optionally) dense vectors, persisted on disk."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .bm25 import BM25
from .models import Chunk, ParsedDoc
from .textutil import query_identifiers, tokenize

log = logging.getLogger("sourcetrace.store")

FORMAT_VERSION = 2
RRF_K = 60.0
SUPERSEDED_PENALTY = 0.6
IDENTIFIER_BOOST = 1.5


def corpus_fingerprint(files: list[tuple[Any, str]]) -> str:
    h = hashlib.sha1()
    for path, rel in files:
        try:
            size = os.stat(path).st_size
        except OSError:
            size = -1
        h.update(f"{rel}\0{size}\n".encode())
    return h.hexdigest()


class Index:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.skipped: list[tuple[str, str]] = []
        self.corpus_root: str = ""
        self.fingerprint: str = ""
        self.built_at: float = 0.0
        self._bm25 = BM25()
        self._dense: np.ndarray | None = None

    # -------------------------------------------------------------------- building
    @classmethod
    def build(
        cls,
        docs: list[ParsedDoc],
        skipped: list[tuple[str, str]],
        corpus_root: str,
        fingerprint: str = "",
        embedder: Any = None,
    ) -> Index:
        idx = cls()
        idx.corpus_root = corpus_root
        idx.fingerprint = fingerprint
        idx.skipped = list(skipped)
        for doc in docs:
            for chunk in doc.chunks:
                chunk.id = len(idx.chunks)
                idx.chunks.append(chunk)
        idx._fit(embedder)
        idx.built_at = time.time()
        return idx

    @staticmethod
    def _tokens_for(chunk: Chunk) -> list[str]:
        return tokenize(chunk.text) + tokenize(chunk.file.replace("/", " "))

    def _fit(self, embedder: Any = None) -> None:
        self._bm25 = BM25().fit([self._tokens_for(c) for c in self.chunks])
        self._dense = None
        if embedder is not None and self.chunks:
            try:
                self._dense = embedder.encode(
                    [f"{c.file}\n{c.text}"[:2000] for c in self.chunks]
                ).astype(np.float32)
            except Exception as exc:  # noqa: BLE001
                log.warning("dense encoding failed, using BM25 only: %s", exc)
                self._dense = None

    @property
    def files(self) -> list[str]:
        return sorted({c.file for c in self.chunks})

    def __len__(self) -> int:
        return len(self.chunks)

    def chunks_of(self, rel: str) -> list[Chunk]:
        return [c for c in self.chunks if c.file == rel]

    # ------------------------------------------------------------------- searching
    def search(
        self,
        query: str,
        *,
        k: int = 10,
        per_file: int = 3,
        embedder: Any = None,
        exclude: set[int] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Rank chunks for ``query`` (BM25 + dense via reciprocal-rank fusion)."""
        if not self.chunks:
            return []
        exclude = exclude or set()
        lexical = self._bm25.score(tokenize(query))
        ranked_lex = sorted(lexical, key=lambda i: -lexical[i])[:200]
        fused: dict[int, float] = {}
        for rank, doc_id in enumerate(ranked_lex):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        if self._dense is not None and embedder is not None:
            try:
                qv = embedder.encode([query], query=True)[0]
                sims = self._dense @ qv
                for rank, doc_id in enumerate(np.argsort(-sims)[:100].tolist()):
                    fused[doc_id] = fused.get(doc_id, 0.0) + 0.7 / (RRF_K + rank)
            except Exception as exc:  # noqa: BLE001
                log.debug("dense search failed: %s", exc)

        ids = [i.lower() for i in query_identifiers(query)]
        scored: list[tuple[float, int]] = []
        for doc_id, score in fused.items():
            if doc_id in exclude:
                continue
            chunk = self.chunks[doc_id]
            if ids:
                low = chunk.text.lower()
                if any(i in low for i in ids):
                    score *= IDENTIFIER_BOOST
            if chunk.superseded:
                score *= SUPERSEDED_PENALTY
            scored.append((score, doc_id))
        scored.sort(key=lambda item: (-item[0], item[1]))

        picked: list[tuple[Chunk, float]] = []
        per: dict[str, int] = {}
        for score, doc_id in scored:
            chunk = self.chunks[doc_id]
            if per.get(chunk.file, 0) >= per_file:
                continue
            per[chunk.file] = per.get(chunk.file, 0) + 1
            picked.append((chunk, score))
            if len(picked) >= k:
                break
        return picked

    # ----------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": FORMAT_VERSION,
            "corpus_root": self.corpus_root,
            "fingerprint": self.fingerprint,
            "built_at": self.built_at,
            "skipped": self.skipped,
            "chunks": [c.to_json() for c in self.chunks],
        }
        _atomic_write_bytes(d / "index.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        _atomic_write_bytes(d / "manifest.json", json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))
        chunks_bytes = "\n".join(json.dumps(c.to_json(), ensure_ascii=False) for c in self.chunks).encode("utf-8")
        _atomic_write_bytes(d / "chunks.jsonl", chunks_bytes)
        if self._dense is not None:
            tmp = d / "dense.npy.tmp"
            with tmp.open("wb") as handle:
                np.save(handle, self._dense)
            os.replace(tmp, d / "dense.npy")
        else:
            with contextlib.suppress(FileNotFoundError):
                (d / "dense.npy").unlink()

    @classmethod
    def load(cls, directory: str | Path) -> Index:
        d = Path(directory)
        meta = json.loads((d / "index.json").read_text(encoding="utf-8"))
        if meta.get("version") != FORMAT_VERSION:
            raise ValueError("index format version mismatch")
        idx = cls()
        idx.corpus_root = meta.get("corpus_root", "")
        idx.fingerprint = meta.get("fingerprint", "")
        idx.built_at = float(meta.get("built_at", 0.0))
        idx.skipped = [tuple(x) for x in meta.get("skipped", [])]  # type: ignore[misc]
        idx.chunks = [Chunk.from_json(c, i) for i, c in enumerate(meta.get("chunks", []))]
        idx._bm25 = BM25().fit([cls._tokens_for(c) for c in idx.chunks])
        dense_path = d / "dense.npy"
        if dense_path.exists():
            dense = np.load(dense_path)
            if dense.shape[0] == len(idx.chunks):
                idx._dense = dense
        return idx


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
