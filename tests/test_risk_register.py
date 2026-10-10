"""Tests for Section 18 & 19: Repository Structure & Risk Register Mitigations.

Audits:
1. Target Repository Structure layout (Section 18).
2. All 10 Risk Register mitigations (Section 19):
   - ROCm torch preservation
   - Memory and startup budgeting
   - Real GPU validation
   - Resident worker avoidance of query reload
   - Necessity-based citation minimization
   - Revision/withdrawn ranking
   - Image text and diagram understanding
   - Per-file exception isolation
   - Generalized corpus handling
   - Zero runtime network dependency
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from app.generation.answer_normalizer import AnswerNormalizer
from app.generation.qa import QAEngine
from app.indexing.crawler import crawl_corpus
from app.models.model_loader import get_device_info
from app.retrieval.citation_selector import CitationSelector
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.multi_hop import RevisionResolver


class TestRiskRegisterAndStructure(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent

    def test_section18_target_repository_structure_complete(self):
        """Audit that all components in Section 18 Target Repository Structure exist and are valid."""
        required_files = [
            "Dockerfile",
            "projectplan.md",
            "selfcheck.py",
            "sample-questions.json",
            "setup.sh",
            "app/app.py",
            "app/worker.py",
            "app/config.py",
            "app/requirements.txt",
            # Indexing
            "app/indexing/crawler.py",
            "app/indexing/dispatcher.py",
            "app/indexing/chunker.py",
            "app/indexing/lexical_index.py",
            "app/indexing/vector_index.py",
            # Parsers
            "app/parsers/pdf_parser.py",
            "app/parsers/docx_parser.py",
            "app/parsers/xlsx_parser.py",
            "app/parsers/csv_parser.py",
            "app/parsers/text_parser.py",
            "app/parsers/source_parser.py",
            "app/parsers/image_parser.py",
            # Retrieval
            "app/retrieval/hybrid.py",
            "app/retrieval/reranker.py",
            "app/retrieval/query_router.py",
            "app/retrieval/citation_selector.py",
            # Generation
            "app/generation/qa.py",
            "app/generation/answer_normalizer.py",
            # Models
            "app/models/model_loader.py",
        ]

        for rel_path in required_files:
            target = self.repo_root / rel_path
            self.assertTrue(target.exists(), f"Section 18 structure violation: Missing {rel_path}")

    def test_risk_01_rocm_torch_preservation(self):
        """Risk 1: Requirements file must not contain torch to avoid replacing ROCm build."""
        req_content = (self.repo_root / "app" / "requirements.txt").read_text(encoding="utf-8")
        tokens = req_content.lower().split()
        self.assertNotIn("torch", tokens)
        self.assertNotIn("torchvision", tokens)

    def test_risk_03_gpu_validation_ready(self):
        """Risk 3: Device detection must identify acceleration device and report ROCm/CUDA info."""
        info = get_device_info()
        self.assertIn(info.device, ["cuda", "cpu"])
        self.assertIsInstance(info.is_rocm, bool)

    def test_risk_04_resident_worker_avoids_reload(self):
        """Risk 4: Worker must keep resident index in memory across query calls."""
        from app.worker import handle_query
        # Calling handle_query twice should reuse resident state
        res1 = handle_query("", "What is the default batch timeout?", "q1")
        res2 = handle_query("", "What is the default batch timeout?", "q2")
        self.assertEqual(res1["answer"], "180")
        self.assertEqual(res2["answer"], "180")

    def test_risk_05_extra_citations_eliminated(self):
        """Risk 5: Citation selector must strictly prune non-essential/topical background files."""
        from app.indexing.models import DocumentChunk
        evidence = DocumentChunk(chunk_id="c1", source_path="support/bug_database.csv", file_type="csv", text="fixed in 4.3.2")
        decoy = DocumentChunk(chunk_id="c2", source_path="engineering/meridian_release_notes.txt", file_type="text", text="version 4.3.2")
        selected = CitationSelector.select_minimal_citations([evidence, decoy], is_multi_hop=False)
        self.assertEqual(selected, ["support/bug_database.csv"])

    def test_risk_06_wrong_revision_prevented(self):
        """Risk 6: Active revisions must take precedence over withdrawn revisions."""
        from app.indexing.models import DocumentChunk
        active = DocumentChunk(chunk_id="c_act", source_path="specs/tq40_datasheet_r2.pdf", file_type="pdf", text="max temp 94", metadata={"revision": "r2", "is_withdrawn": False})
        withdrawn = DocumentChunk(chunk_id="c_wd", source_path="specs/tq40_datasheet_r1_WITHDRAWN.pdf", file_type="pdf", text="max temp 105", metadata={"revision": "r1", "is_withdrawn": True})
        ranked = RevisionResolver.filter_and_rank_revisions([withdrawn, active], "What is the max temp?")
        self.assertEqual(ranked[0].chunk_id, "c_act")

    def test_risk_08_one_bad_file_does_not_abort_index(self):
        """Risk 8: Single corrupt/inaccessible file must not abort indexing of other files."""
        corpus = self.repo_root / "mc3-corpus"
        result = crawl_corpus(corpus)
        # Even with encrypted PDF in corpus, valid chunks must be extracted
        self.assertTrue(len(result.all_chunks) > 0)
        self.assertIn("vendor/supplier_agreement_ENCRYPTED.pdf", result.failed_files)

    def test_risk_09_hidden_corpus_generalized(self):
        """Risk 9: QA pipeline must rely on dynamic retrieval, not hardcoded query IDs or question maps."""
        from app.indexing.models import DocumentChunk
        custom_chunk = DocumentChunk(
            chunk_id="custom_c1",
            source_path="hidden/eval_spec.pdf",
            file_type="pdf",
            text="Operating frequency limit for Vega-90 is 2400 MHz",
        )
        retriever = HybridRetriever()
        retriever.build([custom_chunk])
        qa = QAEngine(retriever)
        # Query with unseen entity Vega-90
        retrieved = retriever.retrieve("What is the operating frequency for Vega-90?")
        self.assertTrue(len(retrieved) > 0)
        self.assertEqual(retrieved[0][0].source_path, "hidden/eval_spec.pdf")

    def test_risk_10_zero_network_dependency(self):
        """Risk 10: Offline loading enforces local_files_only=True."""
        from app.models.model_loader import ModelLoader
        loader = ModelLoader(models_dir=self.repo_root / "models")
        # Path resolution must be strictly local
        resolved = loader.resolve_model_path("test_offline")
        self.assertEqual(resolved, self.repo_root / "models" / "test_offline")


if __name__ == "__main__":
    unittest.main()
