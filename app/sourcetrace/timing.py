"""Time-budget helpers: a monotonic deadline and a hard watchdog."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class Deadline:
    """A point in time after which work must stop."""

    def __init__(self, seconds: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._end = clock() + max(0.0, seconds)

    @classmethod
    def from_start(
        cls,
        started_at: float,
        total_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> Deadline:
        """Deadline ``total_s`` seconds after ``started_at`` (a ``clock()`` value)."""
        return cls(total_s - (clock() - started_at), clock=clock)

    def remaining(self) -> float:
        return max(0.0, self._end - self._clock())

    def expired(self) -> bool:
        return self.remaining() <= 0.0


class Watchdog:
    """Run ``on_expire`` once if not cancelled within ``seconds``.

    The harness scores a whole run as zero if a single image exceeds its time
    limit, so the CLI uses this to guarantee a valid answer file is on disk
    and the process exits before the limit, whatever the model is doing.
    """

    def __init__(self, seconds: float, on_expire: Callable[[], None]) -> None:
        self._timer = threading.Timer(max(0.0, seconds), on_expire)
        self._timer.daemon = True

    def start(self) -> None:
        self._timer.start()

    def cancel(self) -> None:
        self._timer.cancel()
