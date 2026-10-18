"""A scripted backend for tests and for exercising the container contract without a GPU."""

from __future__ import annotations

import time
from collections.abc import Callable

from PIL import Image

Responder = Callable[[str, "Image.Image | None"], str]

REFUSAL_JSON = '{"why": "mock", "answer": "", "evidence": [], "lookup": null}'


def _default_responder(user: str, image: Image.Image | None) -> str:
    return "" if image is not None else REFUSAL_JSON


class MockBackend:
    name = "mock"

    def __init__(self, responder: Responder | None = None, delay_s: float = 0.0) -> None:
        self.responder = responder or _default_responder
        self.delay_s = delay_s
        self.calls: list[tuple[str, bool]] = []

    def load(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def generate(
        self,
        user: str,
        *,
        system: str | None = None,
        image: Image.Image | None = None,
        max_new_tokens: int = 128,
        timeout_s: float = 20.0,
    ) -> str:
        self.calls.append((user, image is not None))
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.responder(user, image)
