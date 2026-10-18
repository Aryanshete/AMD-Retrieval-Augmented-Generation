"""Defensive corpus walk: one bad file never costs the rest of the corpus."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .models import ParsedDoc, Skip
from .parsers import ParseContext, is_image, parse_file

log = logging.getLogger("sourcetrace.walker")

_JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}


@dataclass
class WalkResult:
    docs: list[ParsedDoc] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (relative path, reason)


def list_files(root: Path) -> tuple[list[tuple[Path, str]], list[tuple[str, str]]]:
    """All regular files below ``root`` as ``(path, relative_posix_path)``, sorted.

    Directories that cannot be listed are recorded and skipped, not raised.
    """
    problems: list[tuple[str, str]] = []

    def on_error(err: OSError) -> None:
        problems.append((str(err.filename), f"cannot list directory: {err.strerror}"))

    found: list[tuple[Path, str]] = []
    for base, dirs, files in os.walk(root, onerror=on_error, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.lower() in _JUNK_NAMES or name.startswith("~$") or name.startswith(".~"):
                continue
            full = Path(base) / name
            rel = full.relative_to(root).as_posix()
            found.append((full, rel))
    found.sort(key=lambda item: item[1])
    return found, problems


def _parse_with_timeout(path: Path, rel: str, ctx: ParseContext, timeout_s: float) -> ParsedDoc:
    box: dict[str, object] = {}

    def work() -> None:
        try:
            box["doc"] = parse_file(path, rel, ctx)
        except BaseException as exc:  # noqa: BLE001 - reported to the caller below
            box["err"] = exc

    thread = threading.Thread(target=work, name=f"parse-{rel}", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"parsing took longer than {timeout_s:.0f}s")
    if "err" in box:
        raise box["err"]  # type: ignore[misc]
    return box["doc"]  # type: ignore[return-value]


def walk_corpus(
    root: Path,
    ctx: ParseContext | None = None,
    *,
    file_timeout_s: float = 60.0,
    images: bool = True,
    image_time_left: Callable[[], float] | None = None,
) -> WalkResult:
    """Parse every file below ``root``.

    Text-like files go first; images last, so a slow vision model can only ever use the
    time that is left over. Images that miss the time budget still get a searchable stub.
    """
    result = WalkResult()
    files, problems = list_files(root)
    result.skipped.extend(problems)
    plain_ctx = ParseContext(describe=None, max_image_side=(ctx.max_image_side if ctx else 1600))

    ordered = sorted(files, key=lambda item: (is_image(item[0]), item[1]))
    for path, rel in ordered:
        use = ctx if (ctx and ctx.describe and images) else plain_ctx
        if use is not plain_ctx and image_time_left is not None and image_time_left() <= 5.0:
            use = plain_ctx  # out of time for the vision model: stubs only
        # A file with pictures may legitimately spend several model calls on them.
        limit = file_timeout_s * (4 if use is not plain_ctx else 1)
        try:
            doc = _parse_with_timeout(path, rel, use, limit)
        except Skip as why:
            result.skipped.append((rel, f"skipped: {why}"))
        except PermissionError:
            result.skipped.append((rel, "no permission to read"))
        except Exception as exc:  # noqa: BLE001 - per-file isolation is the whole point
            log.warning("could not parse %s: %s: %s", rel, type(exc).__name__, exc)
            result.skipped.append((rel, f"error: {type(exc).__name__}: {exc}"[:200]))
        else:
            if doc.chunks:
                result.docs.append(doc)
            else:
                result.skipped.append((rel, "no content"))
    return result
