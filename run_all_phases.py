"""End-to-end test execution suite."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

TEST_SUITES = [
    ("Contract", "CLI Contract & Execution Harness", "tests/test_phase1_contract.py"),
    ("Crawler", "Corpus Traversal & Error Recovery", "tests/test_phase2_crawler.py"),
    ("Parsers", "Multi-Format Parsers (PDF/DOCX/XLSX/CSV/TXT/PY)", "tests/test_phase3_parsers.py"),
    ("Vision", "Image Understanding & Spatial OCR", "tests/test_phase4_vision.py"),
    ("Models", "Canonical Document Representation", "tests/test_phase5_canonical_representation.py"),
    ("Chunking", "Format-Aware Chunking Strategy", "tests/test_phase6_chunking.py"),
    ("Retrieval", "Hybrid Search (BM25, Vectors, Reranker)", "tests/test_phase7_retrieval.py"),
    ("Routing", "Query Classification & Multi-Hop Traversal", "tests/test_phase8_routing_multihop.py"),
    ("QA", "Exact Citation Selection & Normalization", "tests/test_phase9_citation_qa.py"),
    ("Daemon", "Persistent Worker Process & Socket IPC", "tests/test_phase10_worker_socket.py"),
    ("Hardware", "ROCm GPU Loader & Hardware Validation", "tests/test_phase11_rocm_gpu.py"),
    ("Robustness", "Adversarial Inputs & Edge Case Safety", "tests/test_phase12_adversarial.py"),
    ("Container", "Dockerfile & Environment Contract", "tests/test_phase13_dockerfile_and_env.py"),
    ("Harness", "Multi-Process Simulation & Packaging", "tests/test_phase14_selfcheck_simulation.py"),
    ("Matrix", "Acceptance Test Matrix", "tests/test_acceptance_matrix.py"),
    ("Regression", "Sample Question Regression Suite", "tests/test_regression_suite.py"),
    ("Milestones", "Milestone Exit Gates", "tests/test_milestone_gates.py"),
    ("Architecture", "Repository Layout & Risk Controls", "tests/test_risk_register.py"),
    ("Checklist", "Submission Checklist Verification", "tests/test_final_submission_checklist.py"),
    ("Certification", "Definition of Done Certification Suite", "tests/test_definition_of_done.py"),
]


def main() -> int:
    print("=" * 80)
    print("MATCH 3 RAG — COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 80)
    total_start = time.time()
    results = []

    for suite_id, title, test_file in TEST_SUITES:
        t0 = time.time()
        print(f"\n[{suite_id}] Running {title} ({test_file})...")
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", test_file],
            capture_output=True,
            text=True,
            check=False,
        )
        duration = time.time() - t0
        passed = proc.returncode == 0
        status_str = "PASS" if passed else "FAIL"
        results.append((suite_id, title, status_str, duration, proc.stderr if not passed else ""))
        print(f"[{status_str}] Finished in {duration:.2f}s")
        if not passed:
            print(f"Error output:\n{proc.stderr}\n{proc.stdout}")

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"VERIFICATION SUMMARY (Total time: {total_time:.2f}s)")
    print("=" * 80)
    print(f"{'Suite':<14} | {'Description':<45} | {'Status':<6} | {'Time':<8}")
    print("-" * 80)
    all_passed = True
    for suite_id, title, status_str, duration, _ in results:
        if status_str != "PASS":
            all_passed = False
        print(f"{suite_id:<14} | {title:<45} | {status_str:<6} | {duration:>6.2f}s")
    print("=" * 80)

    if all_passed:
        print("ALL TEST SUITES SUCCESSFULLY VERIFIED AND COMPLIANT WITH THE CONTRACT!\n")
        return 0
    else:
        print("SOME TESTS FAILED! Review the logs above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
