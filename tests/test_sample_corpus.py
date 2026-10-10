"""End-to-End verification against sample-questions.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class TestSampleCorpusEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_dir = Path("mc3-corpus").resolve()
        cls.output_dir = Path("test_run_output").resolve()
        cls.index_dir = Path("test_run_index").resolve()

        cls.env = os.environ.copy()
        cls.env["MC3_OUTPUT_DIR"] = str(cls.output_dir)
        cls.env["MC2_OUTPUT_DIR"] = str(cls.output_dir)
        cls.env["MC3_INDEX_DIR"] = str(cls.index_dir)
        cls.env["PYTHONPATH"] = str(Path(".").resolve())

        # Clean old test run
        for p in [cls.output_dir, cls.index_dir]:
            if p.exists():
                for f in p.glob("*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass

        # 1. Run index pass
        proc = subprocess.run(
            [sys.executable, "app/app.py", "--index", str(cls.corpus_dir)],
            env=cls.env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Indexing failed: {proc.stderr}\n{proc.stdout}"

    def test_all_10_sample_questions(self):
        """Verify queries 1-10 match expected answers and citations."""
        questions_file = Path("sample-questions.json")
        data = json.loads(questions_file.read_text(encoding="utf-8"))
        queries = data["queries"]

        passed_count = 0
        results = []

        for q in queries:
            qid = f"query_{q['n']:02d}"
            query_text = q["query"]
            expected_ans = q["expected_answer"]
            aliases = [expected_ans] + q.get("answer_aliases", [])
            expected_citations = set(q["expected_citations"])

            proc = subprocess.run(
                [
                    sys.executable,
                    "app/app.py",
                    "--corpus",
                    str(self.corpus_dir),
                    "--query-id",
                    qid,
                    "--query",
                    query_text,
                ],
                env=self.env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"Query {qid} failed: {proc.stderr}")

            out_file = self.output_dir / f"{qid}_output.json"
            self.assertTrue(out_file.exists(), f"Output file missing for {qid}")

            out_json = json.loads(out_file.read_text(encoding="utf-8"))
            actual_ans = out_json.get("answer", "").strip()
            actual_citations = set(out_json.get("citations", []))

            # Normalize for comparison
            ans_match = (actual_ans in aliases) or (actual_ans.lower() in [a.lower() for a in aliases])
            cit_match = (actual_citations == expected_citations)

            match = ans_match and cit_match
            if match:
                passed_count += 1
            results.append({
                "n": q["n"],
                "query": query_text,
                "expected_answer": expected_ans,
                "actual_answer": actual_ans,
                "expected_citations": list(expected_citations),
                "actual_citations": list(actual_citations),
                "passed": match,
            })

        print(f"\nSample Corpus Evaluation: {passed_count}/{len(queries)} questions passed exactly.")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"[{status}] Q{r['n']}: ans={r['actual_answer']!r} (expected {r['expected_answer']!r}) citations={r['actual_citations']} (expected {r['expected_citations']})")

        self.assertEqual(passed_count, len(queries), "All sample queries must pass")


if __name__ == "__main__":
    unittest.main()
