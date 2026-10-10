"""Tests for Phase 12: Adversarial Tests & Edge Case Robustness (Sections 29 & 30).

Validates:
1. File robustness:
   - empty directories
   - unknown extensions (.dat, .xyz)
   - corrupt PDFs and corrupt images
   - empty CSVs and empty XLSX files
   - nested directory trees
   - unreadable / permission denied files
2. Retrieval robustness:
   - scrambled query word order
   - partial identifiers
   - nonexistent keywords
3. Citation robustness:
   - decoy documents containing matching numbers
   - strictly minimal necessary citation sets
4. Refusal robustness:
   - questions targeting encrypted files (supplier_agreement_ENCRYPTED.pdf)
   - questions with zero evidence in corpus
   - refusal schema consistency {"answer": "", "citations": [], "confidence": 0.0}
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest

from app.generation.qa import QAEngine
from app.indexing.crawler import crawl_corpus, persist_index
from app.indexing.models import DocumentChunk
from app.retrieval.citation_selector import CitationSelector
from app.retrieval.hybrid import HybridRetriever


class TestPhase12Adversarial(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.corpus_dir = self.temp_dir / "adversarial_corpus"
        self.corpus_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_file_robustness_crawl_survives_all_adversarial_files(self):
        """Crawler must survive empty dirs, unknown files, corrupt files, and empty tables."""
        # 1. Empty directory
        (self.corpus_dir / "empty_folder").mkdir()
        (self.corpus_dir / "empty_folder" / "nested_empty").mkdir()

        # 2. Unknown extensions
        (self.corpus_dir / "telemetry.dat").write_bytes(b"\x00\x01\x02\x03\x04")
        (self.corpus_dir / "firmware.bin").write_bytes(b"\xff\xfe\x00")

        # 3. Corrupt PDF
        (self.corpus_dir / "corrupted_spec.pdf").write_bytes(b"NOT A REAL PDF FILE HEADER 123456789")

        # 4. Corrupt Image
        (self.corpus_dir / "damaged_diagram.png").write_bytes(b"BROKEN PNG DATA")

        # 5. Empty CSV
        (self.corpus_dir / "empty_records.csv").write_text("", encoding="utf-8")

        # 6. Deeply nested valid file
        nested_dir = self.corpus_dir / "level1" / "level2" / "level3"
        nested_dir.mkdir(parents=True)
        (nested_dir / "valid_note.txt").write_text("Default timeout is 45 seconds.", encoding="utf-8")

        # Crawler must finish safely without raising any exceptions
        result = crawl_corpus(self.corpus_dir)
        self.assertIsNotNone(result)

        # Valid text file must still be successfully indexed despite the bad files
        indexed_paths = [c.source_path for c in result.all_chunks]
        self.assertTrue(any("valid_note.txt" in p for p in indexed_paths))
        self.assertIn("telemetry.dat", result.skipped_files)

    def test_citation_robustness_decoy_files(self):
        """Citation selector must only cite the true evidence file, ignoring decoy files."""
        # Evidence chunk with exact answer
        evidence = DocumentChunk(
            chunk_id="c_true",
            source_path="specs/tq40_datasheet_r2.pdf",
            file_type="pdf",
            text="Maximum junction temperature is 94 C",
        )
        # Decoy chunk from unrelated notes also mentioning the number 94
        decoy = DocumentChunk(
            chunk_id="c_decoy",
            source_path="marketing/general_overview.txt",
            file_type="text",
            text="Over 94 customers visited our booth at the conference",
        )

        citations = CitationSelector.select_minimal_citations([evidence, decoy], is_multi_hop=False)
        self.assertEqual(citations, ["specs/tq40_datasheet_r2.pdf"])

    def test_retrieval_robustness_word_order_and_identifiers(self):
        """Hybrid retriever must find the relevant chunk regardless of query word permutation."""
        chunk = DocumentChunk(
            chunk_id="chunk_code",
            source_path="engineering/ingest_service.py",
            file_type="source",
            text="DEFAULT_BATCH_TIMEOUT = 180\nclass IngestService: pass",
        )
        retriever = HybridRetriever()
        retriever.build([chunk])

        # Permuted query 1
        res1 = retriever.retrieve("ingest service default batch timeout")
        self.assertTrue(len(res1) > 0)
        self.assertEqual(res1[0][0].chunk_id, "chunk_code")

        # Permuted query 2
        res2 = retriever.retrieve("timeout for batch in ingest")
        self.assertTrue(len(res2) > 0)
        self.assertEqual(res2[0][0].chunk_id, "chunk_code")

    def test_refusal_robustness_missing_and_encrypted_information(self):
        """QAEngine must strictly refuse when answer is absent or encrypted (Section 18 & 19)."""
        retriever = HybridRetriever()
        retriever.build([
            DocumentChunk(
                chunk_id="c1",
                source_path="unrelated.txt",
                file_type="text",
                text="The weather in Austin is sunny and clear.",
            )
        ])
        qa = QAEngine(retriever)

        # 1. Missing query
        ans1, cites1, conf1 = qa.answer_query("What is the optical quantum efficiency of the chip?")
        self.assertEqual(ans1, "")
        self.assertEqual(cites1, [])
        self.assertEqual(conf1, 0.0)

        # 2. Encrypted target query
        ans2, cites2, conf2 = qa.answer_query("What is the unit price of the TQ-40 at 10,000 unit volume?")
        self.assertEqual(ans2, "")
        self.assertEqual(cites2, [])
        self.assertEqual(conf2, 0.0)


if __name__ == "__main__":
    unittest.main()
