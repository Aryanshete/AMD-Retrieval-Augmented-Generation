"""Optional dense retrieval with a small sentence-embedding model.

Everything here is best-effort: if the model is missing or fails, retrieval silently
falls back to BM25 only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("sourcetrace.embed")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, path: str, *, device: str | None = None, pooling: str = "cls") -> None:
        self.path = path
        self.pooling = pooling
        self._device = device
        self._tok: Any = None
        self._model: Any = None
        self._torch: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.path)
        self._model = AutoModel.from_pretrained(self.path).to(device).eval()
        self._device = device
        log.info("embedding model ready on %s", device)

    def encode(self, texts: list[str], *, query: bool = False, batch: int = 32) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("embedder not loaded")
        torch = self._torch
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch):
            enc = self._tok(
                texts[i : i + batch],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                hidden = self._model(**enc).last_hidden_state
            if self.pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                vec = hidden[:, 0]
            vec = torch.nn.functional.normalize(vec.float(), dim=-1)
            out.append(vec.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


def try_load_embedder(path: str) -> Embedder | None:
    if not path:
        return None
    try:
        emb = Embedder(path)
        emb.load()
        return emb
    except Exception as exc:  # noqa: BLE001 - dense retrieval is optional
        log.warning("dense retrieval disabled (%s: %s)", type(exc).__name__, exc)
        return None
