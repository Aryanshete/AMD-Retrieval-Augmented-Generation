"""Tests for Phase 9: Exact Citation Selection, Normalization & QA Refusal."""

from __future__ import annotations

import unittest

from app.generation.answer_normalizer import AnswerNormalizer
from app.generation.qa import QAEngine
from app.indexing.models import DocumentChunk
from app.retrieval.citation_selector import CitationSelector
from app.retrieval.hybrid import HybridRetriever


class TestPhase9CitationQA(unittest.TestCase):
    def test_answer_normalizer_preserves_qualifiers(self):
        """AnswerNormalizer must strip boilerplate while preserving all essential qualifiers."""
        # 1. Fiscal year preserved
        self.assertEqual(
            AnswerNormalizer.normalize("The answer is Q3 FY27."),
            "Q3 FY27",
        )
        # 2. Revision prefix preserved
        self.assertEqual(
            AnswerNormalizer.normalize("The board revision is REV-C2.", query="What board revision?"),
            "REV-C2",
        )
        # 3. Part-number suffix preserved
        self.assertEqual(
            AnswerNormalizer.normalize("Field replaceable fan: ORR-FAN-2214-B", query="What part number?"),
            "ORR-FAN-2214-B",
        )
        # 4. Temperature unit stripped for pure value
        self.assertEqual(
            AnswerNormalizer.normalize("The maximum junction temperature is 94°C", query="What is the maximum junction temperature?"),
            "94",
        )
        # 5. Version preserved
        self.assertEqual(
            AnswerNormalizer.normalize("Fixed in release 4.3.2", query="Which firmware version fixed ticket?"),
            "4.3.2",
        )
        # 6. Ignorance / refusal phrases mapped to empty string
        self.assertEqual(
            AnswerNormalizer.normalize("I don't know the answer"),
            "",
        )

    def test_citation_selector_minimal_sources(self):
        """CitationSelector must enforce minimal necessary sources (Section 13)."""
        c1 = DocumentChunk(chunk_id="c1", source_path="/app/corpus/support/bug_database.csv", file_type="csv", text="ORR-1847 fixed in 4.3.2")
        c2 = DocumentChunk(chunk_id="c2", source_path="engineering/meridian_release_notes.txt", file_type="text", text="Firmware 4.3.2 overview")

        # Single-hop: must drop redundant context file c2
        citations_single = CitationSelector.select_minimal_citations([c1, c2], is_multi_hop=False)
        self.assertEqual(citations_single, ["support/bug_database.csv"])

        # Multi-hop: must keep log and csv
        c_log = DocumentChunk(chunk_id="c_log", source_path="logs/prod.log", file_type="log", text="incident E7731")
        citations_multi = CitationSelector.select_minimal_citations([c_log, c1], is_multi_hop=True)
        self.assertEqual(len(citations_multi), 2)
        self.assertIn("logs/prod.log", citations_multi)
        self.assertIn("support/bug_database.csv", citations_multi)

    def test_citation_validation_consistency(self):
        """CitationSelector.validate_citations enforces Check A and path sanitization."""
        # Answer with citations
        ans, cites, conf = CitationSelector.validate_citations("94", ["\\app\\corpus\\specs\\tq40.pdf"])
        self.assertEqual(ans, "94")
        self.assertEqual(cites, ["specs/tq40.pdf"])

        # Answer with empty citations must refuse to protect score
        ans_ref, cites_ref, conf_ref = CitationSelector.validate_citations("94", [])
        self.assertEqual(ans_ref, "")
        self.assertEqual(cites_ref, [])

        # Empty answer must produce empty citations
        ans_empty, cites_empty, _ = CitationSelector.validate_citations("", ["some/file.pdf"])
        self.assertEqual(ans_empty, "")
        self.assertEqual(cites_empty, [])

    def test_qa_engine_refusal_on_encrypted_file(self):
        """QAEngine must refuse questions whose answer is only in encrypted files (Section 19)."""
        retriever = HybridRetriever()
        qa_engine = QAEngine(retriever)

        ans, cites, conf = qa_engine.answer_query("What is the unit price of the TQ-40 at 10,000 unit volume?")
        self.assertEqual(ans, "")
        self.assertEqual(cites, [])
        self.assertEqual(conf, 0.0)


if __name__ == "__main__":
    unittest.main()
