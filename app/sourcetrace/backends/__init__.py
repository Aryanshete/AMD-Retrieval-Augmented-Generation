"""Model backends."""

from __future__ import annotations

from ..config import Settings
from .base import LLMBackend


def create_backend(settings: Settings) -> LLMBackend:
    if settings.backend == "mock":
        from .mock import MockBackend

        return MockBackend()
    from .qwen import QwenVLBackend

    return QwenVLBackend(settings)
