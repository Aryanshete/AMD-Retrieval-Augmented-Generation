"""Tests for Phase 10: Long-Running Worker Daemon & Unix Socket IPC."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import socket
import tempfile
import threading
import time
import unittest

from app.app import send_to_worker
from app.worker import run_worker_server


class TestPhase10WorkerSocket(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.socket_path = self.temp_dir / "test_rag.sock"
        self.server_thread = None

    def tearDown(self):
        # Clean up
        if hasattr(socket, "AF_UNIX") and self.socket_path.exists():
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(str(self.socket_path))
                s.sendall(b'{"command": "shutdown"}\n')
                s.close()
            except Exception:
                pass
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Requires AF_UNIX support")
    def test_worker_socket_ping_and_query(self):
        """Worker daemon must handle socket requests and respond with valid JSON."""
        # Start server in background thread
        self.server_thread = threading.Thread(
            target=run_worker_server,
            kwargs={"socket_path": self.socket_path},
            daemon=True,
        )
        self.server_thread.start()

        # Wait for socket to become active
        ready = False
        for _ in range(30):
            if self.socket_path.exists():
                ready = True
                break
            time.sleep(0.1)
        self.assertTrue(ready, "Worker socket failed to bind")

        # 1. Ping command
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(self.socket_path))
        client.sendall(b'{"command": "ping"}\n')
        resp = json.loads(client.recv(4096).decode("utf-8").strip())
        client.close()

        self.assertEqual(resp.get("status"), "ok")
        self.assertEqual(resp.get("message"), "pong")

        # 2. Query command on resident pipeline
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(self.socket_path))
        query_cmd = {
            "command": "query",
            "query": "What is the unit price of the TQ-40 at 10,000 unit volume?",
            "query_id": "test_q10",
        }
        client.sendall((json.dumps(query_cmd) + "\n").encode("utf-8"))
        resp_q = json.loads(client.recv(4096).decode("utf-8").strip())
        client.close()

        self.assertEqual(resp_q.get("status"), "ok")
        # Refusal expected for unaccessible encrypted query
        self.assertEqual(resp_q.get("answer"), "")
        self.assertEqual(resp_q.get("citations"), [])

    def test_worker_direct_handlers(self):
        """Worker handle_index and handle_query must manage resident pipeline properly."""
        from app.worker import handle_index, handle_query

        # Query on refusal query
        res = handle_query("", "What is the unit price of the TQ-40 at 10,000 unit volume?", "q_test")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["answer"], "")
        self.assertEqual(res["citations"], [])


if __name__ == "__main__":
    unittest.main()
