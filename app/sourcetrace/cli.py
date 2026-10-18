"""Command line interface.

python3 /app/app.py --serve                                   # container CMD: resident server
python3 /app/app.py --index /app/corpus                       # once, before any question
python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."   # per question
python3 /app/app.py --healthcheck                             # Docker HEALTHCHECK
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from . import __version__, client
from .answering import Reasoner
from .backends import create_backend
from .config import BACKENDS, MODES, Settings
from .embed import try_load_embedder
from .output import ResultWriter, output_path_for
from .server import run_server
from .service import BackendFactory, EmbedderFactory, build_index, same_path
from .store import Index
from .timing import Deadline, Watchdog

log = logging.getLogger("sourcetrace.cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sourcetrace",
        description="Citation-exact question answering over a mixed document corpus.",
    )
    p.add_argument("--index", type=Path, metavar="CORPUS", help="index this corpus, then exit")
    p.add_argument("--corpus", type=Path, help="corpus root for a query")
    p.add_argument("--query-id", help="output stem assigned by the harness, e.g. query_01")
    p.add_argument("--query", help="the question")
    p.add_argument("--serve", action="store_true", help="run the resident model server")
    p.add_argument("--healthcheck", action="store_true", help="exit 0 once the server is ready")
    p.add_argument("--output-dir", type=Path, help="where to write <query-id>_output.json")
    p.add_argument("--mode", choices=MODES, help="how to obtain the answer")
    p.add_argument("--backend", choices=BACKENDS, help="model backend")
    p.add_argument("--deadline", type=float, help="seconds before a refusal is written")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"sourcetrace {__version__}")
    return p


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    changes: dict = {}
    if args.output_dir is not None:
        changes["output_dir"] = str(args.output_dir)
    if args.mode:
        changes["mode"] = args.mode
    if args.backend:
        changes["backend"] = args.backend
    if args.deadline is not None:
        changes["client_deadline_s"] = args.deadline
    if args.verbose:
        changes["log_level"] = "DEBUG"
    return dataclasses.replace(settings, **changes) if changes else settings


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _resolve_mode(settings: Settings) -> str:
    if settings.mode != "auto":
        return settings.mode
    return "server" if client.ping(settings.socket_path) is not None else "standalone"


# ----------------------------------------------------------------------------- index
def run_index(
    corpus: Path,
    settings: Settings,
    *,
    started_at: float | None = None,
    backend_factory: BackendFactory | None = None,
    embedder_factory: EmbedderFactory | None = None,
) -> int:
    """Index the corpus. Always exits 0: a query can still rebuild lazily if this fails."""
    started = started_at if started_at is not None else time.monotonic()
    deadline = Deadline.from_start(started, settings.index_client_wait_s)
    mode = _resolve_mode(settings)
    try:
        if mode == "server":
            if not client.ensure_server(settings, deadline):
                raise client.ServerUnavailable("no server could be reached")
            reply = client.server_index(settings, str(corpus), deadline)
            log.info(
                "indexed %s files / %s chunks in %ss",
                reply.get("files"), reply.get("chunks"), reply.get("seconds"),
            )  # fmt: skip
            for rel, why in reply.get("skipped", []):
                log.info("  left out %s: %s", rel, why)
        else:
            backend = None
            try:
                backend = (backend_factory or create_backend)(settings)
                backend.load()
            except Exception as exc:
                log.warning("could not load backend during standalone index (%s); proceeding with text index", exc)
                backend = None
            embedder = (embedder_factory or try_load_embedder)(settings.embed_path)
            budget = Deadline(settings.index_budget_s)
            index = build_index(corpus, settings, backend, embedder, budget)
            index.save(settings.index_dir)
            log.info("indexed %d files / %d chunks", len(index.files), len(index))
    except Exception:
        log.exception("indexing failed; queries will fall back to an on-demand index")
    return 0


# ----------------------------------------------------------------------------- query
def _answer_in_process(
    corpus: Path,
    question: str,
    settings: Settings,
    deadline: Deadline,
    backend_factory: BackendFactory | None,
    embedder_factory: EmbedderFactory | None,
) -> dict:
    """Load everything in this process (slow: for development, tests and fallback)."""
    backend = (backend_factory or create_backend)(settings)
    backend.load()
    embedder = (embedder_factory or try_load_embedder)(settings.embed_path)
    index: Index | None = None
    try:
        loaded = Index.load(settings.index_dir)
        if same_path(loaded.corpus_root, str(corpus)):
            index = loaded
    except Exception:  # noqa: BLE001
        index = None
    if index is None:
        index = build_index(corpus, settings, backend, embedder, Deadline(10.0), with_images=False)
    result = Reasoner(backend, index, settings, embedder).answer(question, deadline, corpus)
    return {
        "answer": result.answer,
        "citations": result.citations,
        "confidence": result.confidence,
        "trace": result.trace,
    }


def _write_trace(settings: Settings, query_id: str, result: dict) -> None:
    if not settings.trace_dir:
        return
    try:
        target = Path(settings.trace_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{query_id}_trace.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        log.warning("could not write trace: %s", exc)


def run_query(
    corpus: Path,
    query_id: str,
    question: str,
    settings: Settings,
    *,
    started_at: float | None = None,
    backend_factory: BackendFactory | None = None,
    embedder_factory: EmbedderFactory | None = None,
    exit_fn: Callable[[int], None] = os._exit,
) -> int:
    """Answer one question and write ``<query-id>_output.json``. Always exits 0."""
    started = started_at if started_at is not None else time.monotonic()
    deadline = Deadline.from_start(started, settings.client_deadline_s)
    writer = ResultWriter(output_path_for(query_id, settings.output_dir))
    writer.write_provisional()

    def on_expire() -> None:
        log.error("hard deadline reached; refusing rather than timing out")
        writer.finalize("", [], 0.0)
        exit_fn(0)

    watchdog = Watchdog(deadline.remaining(), on_expire)
    watchdog.start()
    try:
        mode = _resolve_mode(settings)
        if mode == "server":
            if not client.ensure_server(settings, deadline):
                raise client.ServerUnavailable("server socket never appeared")
            result = client.server_query(settings, str(corpus), question, deadline)
        else:
            result = _answer_in_process(
                corpus, question, settings, deadline, backend_factory, embedder_factory
            )
        _write_trace(settings, query_id, result)
        writer.finalize(
            result.get("answer", ""),
            result.get("citations", []),
            result.get("confidence", 0.0),
        )
    except client.ServerUnavailable as exc:
        log.error("no answer for %s: %s", query_id, exc)
        writer.finalize("", [], 0.0)
    except Exception:
        log.exception("unexpected failure for %s", query_id)
        writer.finalize("", [], 0.0)
    finally:
        watchdog.cancel()
    return 0


# ------------------------------------------------------------------------------ main
def main(
    argv: list[str] | None = None,
    *,
    started_at: float | None = None,
    backend_factory: BackendFactory | None = None,
    embedder_factory: EmbedderFactory | None = None,
    exit_fn: Callable[[int], None] = os._exit,
    settings: Settings | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = _apply_overrides(settings or Settings.from_env(), args)
    _configure_logging(settings.log_level)

    if args.serve:
        return run_server(settings, backend_factory, embedder_factory)
    if args.healthcheck:
        reply = client.ping(settings.socket_path)
        return 0 if reply and reply.get("ready") else 1
    if args.index is not None:
        return run_index(
            args.index,
            settings,
            started_at=started_at,
            backend_factory=backend_factory,
            embedder_factory=embedder_factory,
        )
    if args.corpus is None or args.query is None or not args.query_id:
        parser.error("a query needs --corpus, --query-id and --query (or use --index / --serve)")
    return run_query(
        args.corpus,
        args.query_id,
        args.query,
        settings,
        started_at=started_at,
        backend_factory=backend_factory,
        embedder_factory=embedder_factory,
        exit_fn=exit_fn,
    )
