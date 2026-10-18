"""The interface every model backend implements."""

from __future__ import annotations

from typing import Protocol

from PIL import Image


class LLMBackend(Protocol):
    name: str

    def load(self) -> None:
        """Load weights onto the GPU (slow; called once by the server)."""

    def warmup(self) -> None:
        """Run one tiny generation so the first real request is fast."""

    def generate(
        self,
        user: str,
        *,
        system: str | None = None,
        image: Image.Image | None = None,
        max_new_tokens: int = 128,
        timeout_s: float = 20.0,
    ) -> str:
        """Greedy decoding of one reply. ``image`` is optional."""
