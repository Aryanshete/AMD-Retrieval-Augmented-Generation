"""Tests for Phase 8: Query Routing & Multi-Hop Reasoning."""

from __future__ import annotations

import unittest

from app.indexing.models import DocumentChunk
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.multi_hop import MultiHopReasoner, RevisionResolver
from app.retrieval.query_router import QueryCategory, QueryRouter


class TestPhase8RoutingMultiHop(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()
        self.log_chunk = DocumentChunk(
            chunk_id="log1",
            source_path="logs/prod_inference_2026-09-02.log",
            file_type="log",
            text="2026-09-02T03:14:41Z meridian[2211]: ERROR E7731: thermal throttle engaged on die 0",
            metadata={"error_code": "E7731"},
        )
        self.bug_chunk = DocumentChunk(
            chunk_id="csv1",
            source_path="support/bug_database.csv",
            file_type="csv",
            row=2,
            text="ticket: ORR-1847 | component: power | fixed_in: 4.3.2 | notes: resolve thermal throttle",
            metadata={"ticket": "ORR-1847", "fixed_in": "4.3.2"},
        )
        self.active_pdf = DocumentChunk(
            chunk_id="pdf_r2",
            source_path="specs/tq40_datasheet_r2.pdf",
            file_type="pdf",
            text="Maximum junction temperature 94 C",
            metadata={"revision": "r2", "is_withdrawn": False},
        )
        self.withdrawn_pdf = DocumentChunk(
            chunk_id="pdf_r1",
            source_path="specs/tq40_datasheet_r1_WITHDRAWN.pdf",
            file_type="pdf",
            text="Maximum junction temperature 105 C",
            metadata={"revision": "r1", "is_withdrawn": True},
        )

    def test_query_routing_categories(self):
        """QueryRouter should accurately identify domain categories and target file types."""
        # 1. Specification
        q1 = self.router.route("What is the maximum junction temperature of the TQ-40?")
        self.assertEqual(q1.category, QueryCategory.SPECIFICATION)
        self.assertIn("pdf", q1.prioritized_file_types)
        self.assertIn("TQ-40", q1.identifiers)

        # 2. Ticket / Bug
        q2 = self.router.route("Which firmware version fixed ticket ORR-1847?")
        self.assertEqual(q2.category, QueryCategory.TICKET_BUG)
        self.assertIn("csv", q2.prioritized_file_types)
        self.assertIn("ORR-1847", q2.identifiers)

        # 3. Image / Diagram
        q3 = self.router.route("Which backplane pin carries THERM_ALERT# on the TQ-40?")
        self.assertEqual(q3.category, QueryCategory.IMAGE_DIAGRAM)
        self.assertIn("image", q3.prioritized_file_types)

        # 4. Source code
        q4 = self.router.route("What is the default batch timeout in the ingest service?")
        self.assertEqual(q4.category, QueryCategory.SOURCE_CODE)
        self.assertIn("source", q4.prioritized_file_types)

        # 5. Multi-hop
        q5 = self.router.route("The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?")
        self.assertEqual(q5.category, QueryCategory.MULTI_HOP)
        self.assertTrue(q5.is_multi_hop)
        self.assertIn("log", q5.prioritized_file_types)
        self.assertIn("csv", q5.prioritized_file_types)

    def test_revision_resolver(self):
        """RevisionResolver should rank active r2 ahead of withdrawn r1 for normal queries."""
        chunks = [self.withdrawn_pdf, self.active_pdf]
        resolved = RevisionResolver.filter_and_rank_revisions(chunks, "What is the maximum junction temperature?")
        self.assertEqual(resolved[0].chunk_id, "pdf_r2")

        # Explicitly asking for withdrawn
        resolved_withdrawn = RevisionResolver.filter_and_rank_revisions(chunks, "What was the temperature in the withdrawn r1 spec?")
        self.assertEqual(resolved_withdrawn[0].chunk_id, "pdf_r1")

    def test_multi_hop_chain_resolution(self):
        """MultiHopReasoner should bridge the log incident to the bug database."""
        retriever = HybridRetriever()
        retriever.build([self.log_chunk, self.bug_chunk, self.active_pdf])

        reasoner = MultiHopReasoner(retriever)
        res = reasoner.resolve_multi_hop_chain("The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?")
        self.assertIsNotNone(res)
        evidence, citations = res
        self.assertEqual(len(citations), 2)
        self.assertIn("logs/prod_inference_2026-09-02.log", citations)
        self.assertIn("support/bug_database.csv", citations)


if __name__ == "__main__":
    unittest.main()
