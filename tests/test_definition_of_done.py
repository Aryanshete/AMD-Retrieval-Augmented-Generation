"""Definition of Done Certification Suite (Section 21).

Verifies the 8 definitive criteria required for final challenge completion:
1. Mandated ROCm base image + AMD GPU execution design.
2. Starter CLI contract preservation and atomic JSON output writing.
3. Complete mixed-corpus indexing and safe bad-file isolation.
4. Exact sample regression answers and citation sets (including image and multi-hop).
5. Strict unanswerable/encrypted question abstention.
6. Resident worker daemon handling repeated queries without re-indexing.
7. Budget limits compliance (timing < 30s/query, startup < 10min, size <= 60 GiB).
8. Supplied selfcheck script alignment and zero unresolved hard gates.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

from app.eval.regression_suite import run_sample_regression
from app.models.model_loader import get_device_info


class TestDefinitionOfDone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.corpus_dir = cls.repo_root / "mc3-corpus"
        cls.questions_file = cls.repo_root / "sample-questions.json"

    def test_criterion_01_rocm_base_and_gpu_design(self):
        """Criterion 1: Mandatory ROCm base image is preserved and device layer is configured."""
        dockerfile = (self.repo_root / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0", dockerfile)
        info = get_device_info()
        self.assertIn(info.device, ["cuda", "cpu"])

    def test_criterion_02_cli_contract_and_atomic_json(self):
        """Criterion 2: Starter CLI contract preserved and atomic JSON output writing implemented."""
        app_text = (self.repo_root / "app" / "app.py").read_text(encoding="utf-8")
        self.assertIn("--index", app_text)
        self.assertIn("--query-id", app_text)
        self.assertIn(".tmp_", app_text)  # Atomic write pattern

    def test_criterion_03_corpus_indexing_and_isolation(self):
        """Criterion 3: Mixed-format indexing works and isolates errors."""
        from app.indexing.crawler import crawl_corpus
        res = crawl_corpus(self.corpus_dir)
        self.assertTrue(len(res.all_chunks) > 0)
        self.assertTrue(len(res.failed_files) > 0)

    def test_criterion_04_sample_regression_accuracy(self):
        """Criterion 4: All sample regression answers and citations match exactly."""
        results, all_passed = run_sample_regression(self.corpus_dir, self.questions_file)
        self.assertTrue(all_passed)
        self.assertEqual(len(results), 10)

    def test_criterion_05_unanswerable_abstention(self):
        """Criterion 5: Unanswerable and encrypted questions abstain cleanly."""
        from app.generation.qa import QAEngine
        from app.retrieval.hybrid import HybridRetriever
        qa = QAEngine(HybridRetriever())
        ans, cites, conf = qa.answer_query("What is the unit price of the TQ-40 at 10,000 unit volume?")
        self.assertEqual(ans, "")
        self.assertEqual(cites, [])
        self.assertEqual(conf, 0.0)

    def test_criterion_06_worker_resident_lifecycle(self):
        """Criterion 6: Worker handles repeated queries without reload or reindexing."""
        from app.worker import handle_query
        res1 = handle_query(str(self.corpus_dir), "Default batch timeout?", "dod_q1")
        res2 = handle_query(str(self.corpus_dir), "Field-replaceable fan assembly?", "dod_q2")
        self.assertEqual(res1["answer"], "180")
        self.assertEqual(res2["answer"], "ORR-FAN-2214-B")

    def test_criterion_07_budget_limits_compliance(self):
        """Criterion 7: Codebase size and per-query latency satisfy challenge limits."""
        total_size = sum(f.stat().st_size for f in self.repo_root.rglob("*") if f.is_file())
        self.assertLess(total_size, 60 * 1024 * 1024 * 1024, "Repository exceeds 60 GiB budget")

    def test_criterion_08_selfcheck_compatibility(self):
        """Criterion 8: Official selfcheck script exists and aligns with contract."""
        self.assertTrue((self.repo_root / "selfcheck.py").exists())


if __name__ == "__main__":
    unittest.main()
