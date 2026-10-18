import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _path  # noqa: F401
from _util import SAMPLE_CORPUS, make_corpus_copy

from sourcetrace import walker
from sourcetrace.parsers import ParseContext


class WalkerTests(unittest.TestCase):
    def test_sample_corpus_indexes_everything_readable(self):
        res = walker.walk_corpus(make_corpus_copy())
        files = {d.file for d in res.docs}
        self.assertEqual(
            files,
            {
                "engineering/ingest_service.py",
                "engineering/meridian_release_notes.txt",
                "logs/prod_inference_2026-09-02.log",
                "planning/roadmap_fy27.docx",
                "specs/backplane_pinout.png",
                "specs/tq40_datasheet_r1_WITHDRAWN.pdf",
                "specs/tq40_datasheet_r2.pdf",
                "support/asset_label.jpg",
                "support/bug_database.csv",
                "support/rma_parts.xlsx",
                "vendor/internal_audit.txt",
            },
        )
        skipped = dict(res.skipped)
        self.assertIn("vendor/supplier_agreement_ENCRYPTED.pdf", skipped)
        self.assertIn("vendor/telemetry_capture.dat", skipped)

    def test_empty_directory_and_empty_corpus(self):
        root = Path(tempfile.mkdtemp())
        (root / "archive").mkdir()
        res = walker.walk_corpus(root)
        self.assertEqual((res.docs, res.skipped), ([], []))

    def test_missing_corpus_does_not_raise(self):
        res = walker.walk_corpus(Path(tempfile.mkdtemp()) / "nope")
        self.assertEqual(res.docs, [])

    def test_unreadable_file_does_not_stop_the_walk(self):
        """Simulates ``docker run --cap-drop DAC_OVERRIDE`` (chmod 000 even blocks root)."""
        root = make_corpus_copy()
        real = walker.parse_file

        def fake(path, rel, ctx=None):
            if rel == "engineering/ingest_service.py":  # sorts FIRST among the text files
                raise PermissionError(13, "Permission denied", str(path))
            return real(path, rel, ctx)

        with mock.patch.object(walker, "parse_file", fake):
            res = walker.walk_corpus(root)
        files = {d.file for d in res.docs}
        self.assertNotIn("engineering/ingest_service.py", files)
        self.assertEqual(len(files), 10)  # everything after it is still indexed
        self.assertEqual(
            dict(res.skipped)["engineering/ingest_service.py"], "no permission to read"
        )

    def test_any_parser_crash_is_isolated(self):
        root = make_corpus_copy()
        real = walker.parse_file

        def fake(path, rel, ctx=None):
            if rel.endswith(".pdf"):
                raise RuntimeError("boom")
            return real(path, rel, ctx)

        with mock.patch.object(walker, "parse_file", fake):
            res = walker.walk_corpus(root)
        self.assertEqual(len(res.docs), 9)
        self.assertTrue(any("RuntimeError" in why for _, why in res.skipped))

    def test_hung_parser_times_out(self):
        root = make_corpus_copy()
        real = walker.parse_file

        def fake(path, rel, ctx=None):
            if rel.endswith(".csv"):
                import time

                time.sleep(5)
            return real(path, rel, ctx)

        with mock.patch.object(walker, "parse_file", fake):
            res = walker.walk_corpus(root, file_timeout_s=0.3)
        self.assertNotIn("support/bug_database.csv", {d.file for d in res.docs})
        self.assertTrue(any("longer than" in why for _, why in res.skipped))

    def test_images_come_last_and_respect_the_time_budget(self):
        root = make_corpus_copy()
        order: list[str] = []

        def describe(img, label):
            order.append(label)
            return "TEXT: x"

        res = walker.walk_corpus(root, ParseContext(describe=describe), image_time_left=lambda: 0.0)
        self.assertEqual(order, [])  # out of time: stubs only, no model calls
        stubs = [d for d in res.docs if d.file.endswith((".png", ".jpg"))]
        self.assertEqual(len(stubs), 2)

    def test_output_is_independent_of_listing_order(self):
        root = make_corpus_copy()
        a = [d.file for d in walker.walk_corpus(root).docs]
        shutil.rmtree(root / "archive")
        b = [d.file for d in walker.walk_corpus(root).docs]
        self.assertEqual(a, b)
        self.assertEqual(a[:-2], sorted(a[:-2]))  # text files sorted, images last

    def test_sample_fixture_untouched(self):
        self.assertTrue((SAMPLE_CORPUS / "specs/tq40_datasheet_r2.pdf").exists())


if __name__ == "__main__":
    unittest.main()
