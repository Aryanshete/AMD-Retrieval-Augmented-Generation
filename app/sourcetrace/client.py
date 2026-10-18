"""Thin client used by ``app.py`` to talk to the resident server."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .config import Settings
from .timing import Deadline

log = logging.getLogger("sourcetrace.client")


class ServerUnavailable(RuntimeError):
    """The server could not be reached or did not answer in time."""


def request(socket_path: str, payload: dict, deadline: Deadline) -> dict:
    remaining = deadline.remaining()
    if remaining <= 0:
        raise ServerUnavailable("deadline reached before connecting")
    if not hasattr(socket, "AF_UNIX"):
        raise ServerUnavailable("AF_UNIX socket not supported on this platform")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(remaining)
            sock.connect(socket_path)
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            buffer = b""
            while not buffer.endswith(b"\n"):
                sock.settimeout(max(0.05, deadline.remaining()))
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buffer += chunk
    except (OSError, TimeoutError) as exc:
        raise ServerUnavailable(f"{type(exc).__name__}: {exc}") from exc
    try:
        return json.loads(buffer.decode("utf-8"))
    except ValueError as exc:
        raise ServerUnavailable("server sent an invalid reply") from exc


def ping(socket_path: str, timeout_s: float = 2.0) -> dict | None:
    if not hasattr(socket, "AF_UNIX") or not os.path.exists(socket_path):
        return None
    try:
        return request(socket_path, {"op": "ping"}, Deadline(timeout_s))
    except (ServerUnavailable, AttributeError):
        return None


def wait_for_socket(socket_path: str, deadline: Deadline, poll_s: float = 0.1) -> bool:
    if not hasattr(socket, "AF_UNIX"):
        return False
    while ping(socket_path, 1.0) is None:
        if deadline.expired():
            return False
        time.sleep(poll_s)
    return True


def ensure_server(settings: Settings, deadline: Deadline) -> bool:
    """Make sure a server is listening, starting one if the container CMD did not.

    The container's CMD normally starts it; spawning here is a safety net for runners that
    override the CMD. Returns True once a socket answers a ping.
    """
    if not hasattr(socket, "AF_UNIX"):
        return False
    if ping(settings.socket_path) is not None:
        return True
    lock_path = Path(os.environ.get("TMPDIR", "/tmp")) / "sourcetrace.spawn.lock"
    try:
        import fcntl

        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if ping(settings.socket_path) is None:
                app = Path(__file__).resolve().parent.parent / "app.py"
                log_path = Path(os.environ.get("TMPDIR", "/tmp")) / "sourcetrace-server.log"
                with open(log_path, "ab") as log_file:
                    subprocess.Popen(
                        [sys.executable, str(app), "--serve"],
                        stdin=subprocess.DEVNULL,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=os.environ.copy(),
                    )
                log.warning("no server was running; started one")
                # Hold the lock until the socket is bound (it is bound before the model
                # loads), so concurrent clients never start a second server.
                wait_for_socket(settings.socket_path, Deadline(15.0))
            fcntl.flock(lock, fcntl.LOCK_UN)
    except (OSError, ImportError) as exc:
        log.error("could not start a server: %s", exc)
    return wait_for_socket(settings.socket_path, deadline)


def wait_ready(settings: Settings, deadline: Deadline, poll_s: float = 0.5) -> bool:
    while True:
        reply = ping(settings.socket_path, 2.0)
        if reply and reply.get("ready"):
            return True
        if deadline.expired():
            return False
        time.sleep(poll_s)


def server_index(settings: Settings, corpus: str, deadline: Deadline) -> dict:
    reply = request(
        settings.socket_path,
        {"op": "index", "corpus": corpus, "deadline_s": deadline.remaining()},
        deadline,
    )
    if not reply.get("ok"):
        raise ServerUnavailable(str(reply.get("error", "server error")))
    return reply


def server_query(settings: Settings, corpus: str, question: str, deadline: Deadline) -> dict:
    reply = request(
        settings.socket_path,
        {
            "op": "query",
            "corpus": corpus,
            "question": question,
            "deadline_s": deadline.remaining(),
        },
        deadline,
    )
    if not reply.get("ok"):
        raise ServerUnavailable(str(reply.get("error", "server error")))
    return reply


def shutdown_quietly(path: str) -> None:  # pragma: no cover - test helper
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
