"""Resident server on a Unix socket (one JSON object per line, one request per connection).

    {"op": "ping"}                                        -> {"ok": true, "ready": bool, ...}
    {"op": "index", "corpus": PATH, "deadline_s": N}      -> {"ok": true, "files": ..., ...}
    {"op": "query", "corpus": PATH, "question": STR, "deadline_s": N} -> {"ok": true, "answer": ...}

The socket is bound *before* the model loads, so clients can connect immediately and
simply wait for ``ready``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import socketserver
import threading

from .config import Settings
from .service import BackendFactory, EmbedderFactory, SourceTraceService

log = logging.getLogger("sourcetrace.server")
_MAX_REQUEST_BYTES = 1 << 20


def _make_handler(
    service: SourceTraceService,
) -> type[socketserver.StreamRequestHandler]:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            try:
                line = self.rfile.readline(_MAX_REQUEST_BYTES)
                response = service.handle(json.loads(line))
            except Exception as exc:
                log.exception("request failed")
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            payload = (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
            with contextlib.suppress(OSError):  # client gave up (its deadline); nothing to do
                self.wfile.write(payload)

    return Handler


_ServerBase = getattr(socketserver, "UnixStreamServer", socketserver.TCPServer)


class _UnixServer(socketserver.ThreadingMixIn, _ServerBase):
    daemon_threads = True


class SourceTraceServer:
    def __init__(
        self,
        settings: Settings,
        backend_factory: BackendFactory | None = None,
        embedder_factory: EmbedderFactory | None = None,
    ) -> None:
        self.settings = settings
        self.service = SourceTraceService(settings, backend_factory, embedder_factory)
        self._server: _UnixServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        path = self.settings.socket_path
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        self._server = _UnixServer(path, _make_handler(self.service))
        os.chmod(path, 0o666)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="sourcetrace-accept", daemon=True
        )
        self._thread.start()
        log.info("listening on %s", path)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.settings.socket_path)


def run_server(
    settings: Settings,
    backend_factory: BackendFactory | None = None,
    embedder_factory: EmbedderFactory | None = None,
) -> int:
    """Entry point for ``app.py --serve`` (the container's CMD)."""
    server = SourceTraceServer(settings, backend_factory, embedder_factory)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    server.start()
    try:
        server.service.load()
    except Exception:
        log.exception("model failed to load; exiting")
        server.stop()
        return 1
    log.info("ready")
    stop.wait()
    server.stop()
    return 0
