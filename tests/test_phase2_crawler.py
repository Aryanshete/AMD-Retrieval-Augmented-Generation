"""Tests for Phase 2: Defensive Corpus Crawler and Robustness."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from app.indexing.crawler import crawl_corpus, load_persisted_index, persist_index
from app.indexing.dispatcher import is_supported


class TestPhase2Crawler(unittest.TestCase):
    def setUp(self):
        self.corpus_dir = Path("mc3-corpus").resolve()
        self.test_dir = tempfile.TemporaryDirectory()
        self.index_dir = Path(self.test_dir.name) / "index"

    def tearDown(self):
        self.test_dir.cleanup()

    def test_crawl_corpus_completes_safely(self):
        """Crawler should index the sample corpus without crashing."""
        result = crawl_corpus(self.corpus_dir)
        summary = result.summary()
        self.assertGreater(summary["valid_documents"], 0)
        self.assertGreater(summary["total_chunks"], 0)

    def test_unknown_file_is_skipped(self):
        """Unknown file extension .dat should be recorded in skipped_files, not cause crash."""
        result = crawl_corpus(self.corpus_dir)
        dat_files = [f for f in result.skipped_files if f.endswith(".dat")]
        self.assertTrue(len(dat_files) > 0, "telemetry_capture.dat should be in skipped_files")

    def test_encrypted_file_is_handled_defensively(self):
        """Encrypted PDF should be caught and recorded in failed_files, not abort indexing."""
        result = crawl_corpus(self.corpus_dir)
        encrypted_files = [
            f for f, reason in result.failed_files.items() if "encrypted" in f.lower() or "encrypted" in reason.lower()
        ]
        self.assertTrue(len(encrypted_files) > 0, "supplier_agreement_ENCRYPTED.pdf should be recorded as failed/encrypted")

    def test_empty_directory_survives(self):
        """Empty directory should not cause errors."""
        empty_sub = self.corpus_dir / "archive"
        self.assertTrue(empty_sub.exists() and empty_sub.is_dir())
        result = crawl_corpus(self.corpus_dir)
        self.assertIsNotNone(result)

    def test_posix_relative_paths(self):
        """All chunk source paths must be relative POSIX paths (no backslashes, no leading slash)."""
        result = crawl_corpus(self.corpus_dir)
        for chunk in result.all_chunks:
            self.assertFalse(chunk.source_path.startswith("/"), f"Leading slash in {chunk.source_path}")
            self.assertFalse("\\" in chunk.source_path, f"Backslash in {chunk.source_path}")

    def test_index_persistence_and_reloading(self):
        """Index should serialize to disk and reload accurately."""
        result = crawl_corpus(self.corpus_dir)
        persist_index(result, self.index_dir)

        self.assertTrue((self.index_dir / "manifest.json").exists())
        self.assertTrue((self.index_dir / "chunks.jsonl").exists())
        self.assertTrue((self.index_dir / "documents.jsonl").exists())

        loaded = load_persisted_index(self.index_dir)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.all_chunks), len(result.all_chunks))


if __name__ == "__main__":
    unittest.main()
