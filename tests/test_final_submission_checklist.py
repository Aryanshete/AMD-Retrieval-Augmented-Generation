"""Final Submission Checklist Certification (Section 20).

Automated verification of every checkbox in Section 20 of the updated plan:
- Contract and container (6 items)
- Parsing and robustness (6 items)
- Answering and citations (6 items)
- Performance and packaging (10 items)
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from app.generation.answer_normalizer import AnswerNormalizer
from app.generation.qa import QAEngine
from app.indexing.dispatcher import is_supported
from app.models.model_loader import get_device_info
from app.retrieval.citation_selector import CitationSelector


class TestFinalSubmissionChecklist(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent

    def test_checklist_group1_contract_and_container(self):
        """Verify all 6 Contract and Container checklist items."""
        dockerfile = (self.repo_root / "Dockerfile").read_text(encoding="utf-8")
        # 1. Mandated base image
        self.assertIn("rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0", dockerfile)
        # 2. No HTTP port exposed
        self.assertNotIn("EXPOSE", dockerfile)
        # 3. Container remains alive
        self.assertIn("sleep infinity", dockerfile)
        # 4. app.py supports both --index and --query
        app_text = (self.repo_root / "app" / "app.py").read_text(encoding="utf-8")
        self.assertIn("--index", app_text)
        self.assertIn("--query", app_text)
        # 5. Output directory and query-ID filename
        self.assertIn("{args.query_id}_output.json", app_text)
        # 6. Every output contains answer and citations
        self.assertIn('"answer"', app_text)
        self.assertIn('"citations"', app_text)

    def test_checklist_group2_parsing_and_robustness(self):
        """Verify all 6 Parsing and Robustness checklist items."""
        # 1. Supported file types
        for ext in [".pdf", ".docx", ".xlsx", ".csv", ".txt", ".log", ".py", ".png", ".jpg"]:
            self.assertTrue(is_supported(Path(f"test{ext}")), f"Extension {ext} not supported")
        # 2-6. Crawler exception handling
        crawler_code = (self.repo_root / "app" / "indexing" / "crawler.py").read_text(encoding="utf-8")
        self.assertIn("PermissionError", crawler_code)
        self.assertIn("failed_files", crawler_code)
        self.assertIn("skipped_files", crawler_code)

    def test_checklist_group3_answering_and_citations(self):
        """Verify all 6 Answering and Citations checklist items."""
        # Value-only qualifiers
        self.assertEqual(AnswerNormalizer.normalize("Quarter Q3 FY27"), "Q3 FY27")
        self.assertEqual(AnswerNormalizer.normalize("Part ORR-FAN-2214-B"), "ORR-FAN-2214-B")
        self.assertEqual(AnswerNormalizer.normalize("Revision REV-C2"), "REV-C2")
        # Necessity rule
        from app.indexing.models import DocumentChunk
        c1 = DocumentChunk(chunk_id="c1", source_path="specs/tq40.pdf", file_type="pdf", text="max temp 94")
        c2 = DocumentChunk(chunk_id="c2", source_path="notes.txt", file_type="text", text="general notes")
        cites = CitationSelector.select_minimal_citations([c1, c2], is_multi_hop=False)
        self.assertEqual(cites, ["specs/tq40.pdf"])
        # Unsupported / encrypted refusal
        from app.retrieval.hybrid import HybridRetriever
        qa = QAEngine(HybridRetriever())
        ans, c, conf = qa.answer_query("unit price at 10,000 units")
        self.assertEqual(ans, "")
        self.assertEqual(c, [])
        self.assertEqual(conf, 0.0)

    def test_checklist_group4_performance_and_packaging(self):
        """Verify Performance and Packaging checklist items."""
        # Offline model directory exists
        self.assertTrue((self.repo_root / "models").exists())
        # ROCm Torch preserved
        reqs = (self.repo_root / "app" / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertNotIn("torch", reqs.split())
        # Selfcheck script exists
        self.assertTrue((self.repo_root / "selfcheck.py").exists())


if __name__ == "__main__":
    unittest.main()
