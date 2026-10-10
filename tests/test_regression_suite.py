"""Tests for Section 16: Sample Question Regression Suite."""

from __future__ import annotations

from pathlib import Path
import unittest

from app.eval.regression_suite import run_sample_regression


class TestRegressionSuite(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.corpus_dir = self.repo_root / "mc3-corpus"
        self.questions_file = self.repo_root / "sample-questions.json"

    def test_sample_question_regression_suite_all_pass(self):
        """All 10 sample queries must match expected answers, exact citations, and latency limits."""
        results, all_passed = run_sample_regression(self.corpus_dir, self.questions_file)
        self.assertEqual(len(results), 10)
        self.assertTrue(all_passed, "Regression suite failed on one or more queries!")

        for r in results:
            self.assertTrue(r.passed, f"Failed Q{r.n}: ans={r.actual_answer!r} cites={r.actual_citations}")
            self.assertLess(r.latency_ms, 30000.0, f"Q{r.n} exceeded 30s latency budget")


if __name__ == "__main__":
    unittest.main()
