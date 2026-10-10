"""Tests for Phase 14: Selfcheck Simulation & Final Packaging Validation (Sections 35, 36, 38).

Simulates the exact external evaluation harness run by selfcheck.py:
1. Two-phase invocation: separate processes for --index and per-question queries.
2. Latency verification (< 30 seconds per query budget).
3. Output schema verification (/app/output/<query-id>_output.json).
4. Citation exactness (minimal, corpus-relative POSIX).
5. Refusal compliance (empty answer and citations for inaccessible info).
6. Index reuse without re-crawling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


class TestPhase14SelfcheckSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.corpus_dir = cls.repo_root / "mc3-corpus"
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.output_dir = cls.temp_dir / "output"
        cls.index_dir = cls.temp_dir / "index"

        cls.env = os.environ.copy()
        cls.env["MC3_OUTPUT_DIR"] = str(cls.output_dir)
        cls.env["MC2_OUTPUT_DIR"] = str(cls.output_dir)
        cls.env["MC3_INDEX_DIR"] = str(cls.index_dir)
        cls.env["PYTHONPATH"] = str(cls.repo_root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_index_pass_execution_and_persistence(self):
        """Phase 1: Harness runs python3 /app/app.py --index /app/corpus once."""
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "app/app.py", "--index", str(self.corpus_dir)],
            cwd=str(self.repo_root),
            env=self.env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        duration = time.time() - t0
        self.assertEqual(proc.returncode, 0, f"Indexing failed: {proc.stderr}\n{proc.stdout}")
        self.assertLess(duration, 300, "Indexing exceeded time budget")

        # Verify index persistence artifacts (Section 24)
        self.assertTrue((self.index_dir / "manifest.json").exists())
        self.assertTrue((self.index_dir / "chunks.jsonl").exists())
        self.assertTrue((self.index_dir / "lexical.pkl").exists())
        self.assertTrue((self.index_dir / "embeddings.npy").exists())

    def test_02_query_execution_and_schema_compliance(self):
        """Phase 2: Harness runs a query in a fresh process, expecting output JSON."""
        out_file = self.output_dir / "query_01_output.json"
        if out_file.exists():
            out_file.unlink()

        t0 = time.time()
        proc = subprocess.run(
            [
                sys.executable,
                "app/app.py",
                "--corpus",
                str(self.corpus_dir),
                "--query-id",
                "query_01",
                "--query",
                "What is the maximum junction temperature of the TQ-40?",
            ],
            cwd=str(self.repo_root),
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        duration = time.time() - t0
        self.assertEqual(proc.returncode, 0, f"Query execution failed: {proc.stderr}\n{proc.stdout}")
        self.assertLess(duration, 30.0, f"Query took {duration:.2f}s, exceeding 30s budget")

        # Verify output file exists
        self.assertTrue(out_file.exists(), f"Expected output file missing: {out_file}")

        # Verify exact JSON schema
        data = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertIn("answer", data)
        self.assertIn("citations", data)
        self.assertIn("confidence", data)

        self.assertEqual(data["answer"], "94")
        self.assertEqual(data["citations"], ["specs/tq40_datasheet_r2.pdf"])
        self.assertIsInstance(data["confidence"], float)

    def test_03_refusal_query_execution(self):
        """Phase 2: Inaccessible or encrypted questions must strictly refuse without guessing."""
        out_file = self.output_dir / "query_10_output.json"
        if out_file.exists():
            out_file.unlink()

        proc = subprocess.run(
            [
                sys.executable,
                "app/app.py",
                "--corpus",
                str(self.corpus_dir),
                "--query-id",
                "query_10",
                "--query",
                "What is the unit price of the TQ-40 at 10,000 unit volume?",
            ],
            cwd=str(self.repo_root),
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(out_file.exists())

        data = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertEqual(data["answer"], "")
        self.assertEqual(data["citations"], [])
        self.assertEqual(data["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
