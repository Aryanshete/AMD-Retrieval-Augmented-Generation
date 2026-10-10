"""Tests for Phase 1: Contract Enforcement and Invocation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestPhase1Contract(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.corpus_dir = Path("mc3-corpus").resolve()
        self.output_dir = Path(self.test_dir.name) / "output"
        self.index_dir = Path(self.test_dir.name) / "index"

        self.env = os.environ.copy()
        self.env["MC3_OUTPUT_DIR"] = str(self.output_dir)
        self.env["MC2_OUTPUT_DIR"] = str(self.output_dir)
        self.env["MC3_INDEX_DIR"] = str(self.index_dir)
        self.env["PYTHONPATH"] = str(Path(".").resolve())

    def tearDown(self):
        self.test_dir.cleanup()

    def test_index_contract(self):
        """python app/app.py --index <corpus> must exit 0 and write index artifacts."""
        proc = subprocess.run(
            [sys.executable, "app/app.py", "--index", str(self.corpus_dir)],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"--index failed: {proc.stderr}\n{proc.stdout}")
        self.assertTrue((self.index_dir / "manifest.json").exists())
        self.assertTrue((self.index_dir / "chunks.jsonl").exists())

    def test_query_contract_output_schema(self):
        """python app/app.py --corpus ... --query-id ... --query ... must output valid JSON schema."""
        # Run index first
        subprocess.run(
            [sys.executable, "app/app.py", "--index", str(self.corpus_dir)],
            env=self.env,
            capture_output=True,
            check=True,
        )

        query_id = "test_q01"
        proc = subprocess.run(
            [
                sys.executable,
                "app/app.py",
                "--corpus",
                str(self.corpus_dir),
                "--query-id",
                query_id,
                "--query",
                "What is the maximum junction temperature of the TQ-40?",
            ],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"--query failed: {proc.stderr}\n{proc.stdout}")

        out_file = self.output_dir / f"{query_id}_output.json"
        self.assertTrue(out_file.exists(), f"Missing output file: {out_file}")

        data = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertIn("answer", data)
        self.assertIn("citations", data)
        self.assertIn("confidence", data)

        self.assertIsInstance(data["answer"], str)
        self.assertIsInstance(data["citations"], list)
        self.assertIsInstance(data["confidence"], (int, float))

        # Check citations are relative
        for c in data["citations"]:
            self.assertIsInstance(c, str)
            self.assertFalse(c.startswith("/"), f"Citation must be relative: {c}")
            self.assertFalse("\\" in c, f"Citation must use forward slashes: {c}")

        # If answer is non-empty, citations must not be empty
        if data["answer"]:
            self.assertTrue(len(data["citations"]) > 0)

    def test_refusal_contract(self):
        """Unanswerable query must output exact refusal schema."""
        subprocess.run(
            [sys.executable, "app/app.py", "--index", str(self.corpus_dir)],
            env=self.env,
            capture_output=True,
            check=True,
        )

        query_id = "test_refusal_10"
        proc = subprocess.run(
            [
                sys.executable,
                "app/app.py",
                "--corpus",
                str(self.corpus_dir),
                "--query-id",
                query_id,
                "--query",
                "What is the unit price of the TQ-40 at 10,000 unit volume?",
            ],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)

        out_file = self.output_dir / f"{query_id}_output.json"
        data = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertEqual(data["answer"], "")
        self.assertEqual(data["citations"], [])


if __name__ == "__main__":
    unittest.main()
