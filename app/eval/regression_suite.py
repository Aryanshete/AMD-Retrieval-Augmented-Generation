"""Sample question regression suite runner.

Executes all 10 sample queries, compares exact answers and exact citation sets,
records per-query latency, and ensures zero time-limit or scoring violations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QueryRegressionResult:
    n: int
    query: str
    expected_answer: str
    actual_answer: str
    expected_citations: Set[str]
    actual_citations: Set[str]
    answer_match: bool
    citations_match: bool
    latency_ms: float
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "query": self.query,
            "expected_answer": self.expected_answer,
            "actual_answer": self.actual_answer,
            "expected_citations": sorted(list(self.expected_citations)),
            "actual_citations": sorted(list(self.actual_citations)),
            "answer_match": self.answer_match,
            "citations_match": self.citations_match,
            "latency_ms": round(self.latency_ms, 2),
            "passed": self.passed,
        }


def run_sample_regression(
    corpus_dir: Path,
    sample_questions_path: Path,
    index_data: Optional[Any] = None,
) -> Tuple[List[QueryRegressionResult], bool]:
    """Execute regression evaluation against sample questions."""
    from app.app import execute_query
    from app.indexing.crawler import crawl_corpus

    data = json.loads(sample_questions_path.read_text(encoding="utf-8"))
    queries = data.get("queries", [])

    results = []
    all_passed = True

    for q in queries:
        n = q["n"]
        query_text = q["query"]
        expected_ans = q["expected_answer"]
        aliases = {expected_ans}
        aliases.update(q.get("answer_aliases", []))
        expected_citations = set(q["expected_citations"])

        t0 = time.time()
        ans, actual_citations_list, conf = execute_query(corpus_dir, query_text, index_data)
        latency_ms = (time.time() - t0) * 1000.0

        actual_citations = set(actual_citations_list)
        ans_match = (ans in aliases) or (ans == expected_ans)
        cit_match = (actual_citations == expected_citations)
        passed = ans_match and cit_match and (latency_ms < 30000.0)

        if not passed:
            all_passed = False

        results.append(
            QueryRegressionResult(
                n=n,
                query=query_text,
                expected_answer=expected_ans,
                actual_answer=ans,
                expected_citations=expected_citations,
                actual_citations=actual_citations,
                answer_match=ans_match,
                citations_match=cit_match,
                latency_ms=latency_ms,
                passed=passed,
            )
        )

    return results, all_passed
