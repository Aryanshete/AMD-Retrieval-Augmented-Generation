import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

import _path  # noqa: F401
from _util import (
    SAMPLE_QUESTIONS,
    is_verify,
    make_corpus_copy,
    passages_in,
    reply,
    test_settings,
    verdict,
)

from sourcetrace import cli, client
from sourcetrace.backends.mock import MockBackend
from sourcetrace.server import SourceTraceServer
from sourcetrace.timing import Deadline


def oracle(prompt: str, image) -> str:
    """A model that knows the sample answers, but only 'sees' what the prompt shows it.

    It cites every passage that contains the answer's file, so these tests also prove that
    the pipeline - not the model - is what makes the citation set exact.
    """
    if image is not None:
        return (
            "TEXT: B14 THERM_ALERT#\nREV-C2" if "Question:" not in prompt else "B14 -> THERM_ALERT#"
        )
    if is_verify(prompt):
        return verdict(True)
    seen = passages_in(prompt)
    question = prompt.split("Question:", 1)[1].split("\n", 1)[0].strip()
    for q in SAMPLE_QUESTIONS:
        if q["query"] != question:
            continue
        if q.get("unanswerable"):
            return reply("")
        needed = q["expected_citations"]
        if not all(f in seen for f in needed):
            return reply("")
        extra = [
            f
            for f in seen
            if f not in needed and f.endswith(("release_notes.txt", "r1_WITHDRAWN.pdf"))
        ]
        ids = [i for f in [*needed, *extra] for i in seen[f]]  # over-cites on purpose
        return reply(q["expected_answer"], ids)
    return reply("")


def oracle_backend(settings):
    return MockBackend(oracle)


class QueryCliTests(unittest.TestCase):
    def read(self, settings, qid):
        return json.loads((Path(settings.output_dir) / f"{qid}_output.json").read_text())

    def test_standalone_query_writes_the_contract_file(self):
        settings = test_settings()
        corpus = make_corpus_copy()
        rc = cli.run_query(
            corpus,
            "query_01",
            SAMPLE_QUESTIONS[0]["query"],
            settings,
            backend_factory=oracle_backend,
        )
        self.assertEqual(rc, 0)
        out = self.read(settings, "query_01")
        self.assertEqual(set(out), {"answer", "citations", "confidence"})
        self.assertEqual(out["answer"], "94")
        self.assertEqual(out["citations"], ["specs/tq40_datasheet_r2.pdf"])

    def test_output_name_comes_from_the_given_query_id(self):
        settings = test_settings()
        cli.run_query(
            make_corpus_copy(), "q-42", "What is the unit price?", settings,
            backend_factory=oracle_backend,
        )  # fmt: skip
        self.assertTrue((Path(settings.output_dir) / "q-42_output.json").exists())

    def test_every_sample_question_scores(self):
        """The full ten-question sample, answer AND exact citation set."""
        settings = test_settings()
        corpus = make_corpus_copy()
        cli.run_index(corpus, settings, backend_factory=oracle_backend)
        for q in SAMPLE_QUESTIONS:
            qid = f"query_{q['n']:02d}"
            cli.run_query(corpus, qid, q["query"], settings, backend_factory=oracle_backend)
            out = self.read(settings, qid)
            with self.subTest(n=q["n"], query=q["query"]):
                self.assertEqual(out["answer"], q["expected_answer"])
                self.assertEqual(sorted(out["citations"]), sorted(q["expected_citations"]))

    def test_crash_still_leaves_a_valid_refusal(self):
        def broken(settings):
            raise RuntimeError("no GPU")

        settings = test_settings()
        cli.run_query(make_corpus_copy(), "query_01", "q?", settings, backend_factory=broken)
        out = self.read(settings, "query_01")
        self.assertEqual(out, {"answer": "", "citations": [], "confidence": 0.0})

    def test_watchdog_writes_a_refusal_before_the_deadline(self):
        settings = test_settings(client_deadline_s=3.0)
        exits: list[int] = []
        done = threading.Event()

        def slow(settings):
            return MockBackend(lambda p, i: time.sleep(5) or reply("94", [1]))

        def exit_fn(code):
            exits.append(code)
            done.set()

        t = threading.Thread(
            target=cli.run_query,
            args=(make_corpus_copy(), "query_01", "max temp?", settings),
            kwargs={"backend_factory": slow, "exit_fn": exit_fn},
            daemon=True,
        )
        t.start()
        self.assertTrue(done.wait(6))
        self.assertEqual(exits, [0])
        self.assertEqual(
            self.read(settings, "query_01"), {"answer": "", "citations": [], "confidence": 0.0}
        )

    def test_refusal_never_carries_citations(self):
        from sourcetrace.output import ResultWriter, output_path_for

        d = tempfile.mkdtemp()
        w = ResultWriter(output_path_for("query_09", d))
        w.finalize("", ["specs/x.pdf"], 0.9)
        out = json.loads((Path(d) / "query_09_output.json").read_text())
        self.assertEqual(out, {"answer": "", "citations": [], "confidence": 0.0})

    def test_first_final_write_wins_and_duplicates_are_removed(self):
        from sourcetrace.output import ResultWriter, output_path_for

        d = tempfile.mkdtemp()
        w = ResultWriter(output_path_for("query_09", d))
        self.assertTrue(w.finalize("94", ["a.pdf", "a.pdf", "b.pdf"], 7))
        self.assertFalse(w.finalize("", [], 0))
        out = json.loads((Path(d) / "query_09_output.json").read_text())
        self.assertEqual(out, {"answer": "94", "citations": ["a.pdf", "b.pdf"], "confidence": 1.0})

    def test_index_survives_missing_corpus(self):
        settings = test_settings()
        rc = cli.run_index(
            Path(tempfile.mkdtemp()) / "nope", settings, backend_factory=oracle_backend
        )
        self.assertEqual(rc, 0)

    def test_query_without_index_builds_on_demand(self):
        settings = test_settings()
        cli.run_query(
            make_corpus_copy(), "query_05", SAMPLE_QUESTIONS[4]["query"], settings,
            backend_factory=oracle_backend,
        )  # fmt: skip
        out = self.read(settings, "query_05")
        self.assertEqual(out["answer"], "E7731")
        self.assertEqual(out["citations"], ["logs/prod_inference_2026-09-02.log"])

    def test_parser_requires_query_arguments(self):
        with self.assertRaises(SystemExit):
            cli.main(["--corpus", "/x"], settings=test_settings())


import socket


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Requires AF_UNIX support")
class ServerTests(unittest.TestCase):
    def setUp(self):
        self.settings = test_settings(mode="server")
        self.server = SourceTraceServer(self.settings, oracle_backend, lambda path: None)
        self.server.start()
        self.addCleanup(self.server.stop)

    def test_ping_before_and_after_ready(self):
        self.assertFalse(client.ping(self.settings.socket_path)["ready"])
        self.server.service.load()
        self.assertTrue(client.ping(self.settings.socket_path)["ready"])

    def test_index_then_query_through_the_socket(self):
        corpus = make_corpus_copy()
        threading.Thread(target=self.server.service.load, daemon=True).start()
        deadline = Deadline(10)
        info = client.server_index(self.settings, str(corpus), deadline)
        self.assertEqual(info["files"], 11)
        self.assertIn(
            ["vendor/supplier_agreement_ENCRYPTED.pdf", "skipped: encrypted PDF"], info["skipped"]
        )
        reply_ = client.server_query(
            self.settings, str(corpus), SAMPLE_QUESTIONS[5]["query"], Deadline(10)
        )
        self.assertEqual(reply_["answer"], "180")
        self.assertEqual(reply_["citations"], ["engineering/ingest_service.py"])

    def test_cli_in_server_mode(self):
        self.server.service.load()
        corpus = make_corpus_copy()
        cli.run_index(corpus, self.settings)
        cli.run_query(corpus, "query_02", SAMPLE_QUESTIONS[1]["query"], self.settings)
        out = json.loads((Path(self.settings.output_dir) / "query_02_output.json").read_text())
        self.assertEqual(out["answer"], "Q3 FY27")
        self.assertEqual(out["citations"], ["planning/roadmap_fy27.docx"])

    def test_query_waits_for_the_model(self):
        corpus = make_corpus_copy()
        threading.Timer(0.5, self.server.service.load).start()
        t0 = time.monotonic()
        reply_ = client.server_query(
            self.settings, str(corpus), SAMPLE_QUESTIONS[4]["query"], Deadline(10)
        )
        self.assertEqual(reply_["answer"], "E7731")
        self.assertGreaterEqual(time.monotonic() - t0, 0.4)

    def test_bad_requests_do_not_kill_the_server(self):
        self.server.service.load()
        for payload in ({"op": "nope"}, {"op": "query"}, {"op": "index"}):
            reply_ = client.request(self.settings.socket_path, payload, Deadline(5))
            self.assertFalse(reply_["ok"])
        self.assertTrue(client.ping(self.settings.socket_path)["ready"])

    def test_healthcheck_exit_codes(self):
        self.assertEqual(cli.main(["--healthcheck"], settings=self.settings), 1)
        self.server.service.load()
        self.assertEqual(cli.main(["--healthcheck"], settings=self.settings), 0)

    def test_client_without_server_refuses_cleanly(self):
        self.server.stop()
        settings = test_settings(mode="server", client_deadline_s=1.0)
        # Do not let the safety net spawn a real server during tests.
        import unittest.mock as mock

        with mock.patch.object(client, "ensure_server", return_value=False):
            cli.run_query(Path("/nonexistent"), "query_01", "q?", settings)
        out = json.loads((Path(settings.output_dir) / "query_01_output.json").read_text())
        self.assertEqual(out["answer"], "")


class UnreadableFileTests(unittest.TestCase):
    @unittest.skipIf(
        os.name == "nt" or getattr(os, "geteuid", lambda: -1)() == 0,
        "Windows or root bypasses POSIX chmod; the walker test simulates DAC_OVERRIDE",
    )
    def test_real_chmod_000(self):
        corpus = make_corpus_copy()
        path = corpus / "engineering/ingest_service.py"
        path.chmod(0)
        self.addCleanup(path.chmod, 0o644)
        settings = test_settings()
        cli.run_index(corpus, settings, backend_factory=oracle_backend)
        idx = __import__("sourcetrace.store", fromlist=["Index"]).Index.load(settings.index_dir)
        self.assertNotIn("engineering/ingest_service.py", idx.files)
        self.assertIn("planning/roadmap_fy27.docx", idx.files)

    def tearDown(self):
        shutil.rmtree(tempfile.gettempdir() + "/st-none", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
