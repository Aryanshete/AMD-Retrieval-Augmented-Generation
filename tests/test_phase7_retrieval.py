"""Tests for Phase 7: Hybrid Retrieval (Lexical, Semantic, Reranker)."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from app.indexing.lexical_index import BM25LexicalIndex
from app.indexing.models import DocumentChunk
from app.indexing.vector_index import VectorEmbeddingIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CandidateReranker


class TestPhase7HybridRetrieval(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_chunks = [
            DocumentChunk(
                chunk_id="chunk1",
                source_path="specs/tq40_datasheet_r2.pdf",
                file_type="pdf",
                page=1,
                text="The maximum junction temperature for TQ-40 processor is 94 C under full load.",
                metadata={"revision": "r2", "is_withdrawn": False},
            ),
            DocumentChunk(
                chunk_id="chunk2_withdrawn",
                source_path="specs/tq40_datasheet_r1_WITHDRAWN.pdf",
                file_type="pdf",
                page=1,
                text="The maximum junction temperature for TQ-40 processor is 105 C (withdrawn preliminary spec).",
                metadata={"revision": "r1", "is_withdrawn": True},
            ),
            DocumentChunk(
                chunk_id="chunk3",
                source_path="support/bug_database.csv",
                file_type="csv",
                row=2,
                text="ticket: ORR-1847 | component: power | fixed_in: 4.3.2 | notes: resolve thermal throttle",
                metadata={"ticket": "ORR-1847", "fixed_in": "4.3.2"},
            ),
            DocumentChunk(
                chunk_id="chunk4",
                source_path="logs/prod_inference_2026-09-02.log",
                file_type="log",
                text="meridian[2211]: ERROR E7731: thermal throttle engaged on die 0 above threshold",
                metadata={"error_code": "E7731"},
            ),
            DocumentChunk(
                chunk_id="chunk5",
                source_path="support/asset_label.jpg",
                file_type="image",
                text="ORRERY SYSTEMS MODEL: TQ-40 BOARD REVISION: REV-C2 SERIAL: 99120-A",
                metadata={"revision": "REV-C2"},
            ),
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bm25_lexical_search_and_identifier_boost(self):
        """BM25 search should retrieve relevant chunks and boost exact identifiers."""
        index = BM25LexicalIndex()
        index.build(self.sample_chunks)

        # Query with exact identifier ORR-1847
        results = index.search("Which firmware version fixed ticket ORR-1847?")
        self.assertTrue(len(results) > 0)
        top_chunk, score = results[0]
        self.assertEqual(top_chunk.chunk_id, "chunk3")
        self.assertIn("ORR-1847", top_chunk.text)

        # Query with error code E7731
        results_err = index.search("What is the error code E7731?")
        self.assertEqual(results_err[0][0].chunk_id, "chunk4")

    def test_vector_similarity_search(self):
        """Vector index should retrieve semantically similar chunks."""
        v_index = VectorEmbeddingIndex(dim=64)
        v_index.build(self.sample_chunks)

        results = v_index.search("maximum junction temperature processor limit", top_k=2)
        self.assertTrue(len(results) > 0)
        c_ids = [c.chunk_id for c, _ in results]
        self.assertIn("chunk1", c_ids)

    def test_reranker_prefers_active_over_withdrawn(self):
        """Reranker should prioritize active r2 over withdrawn r1 for general questions."""
        reranker = CandidateReranker()
        query = "What is the maximum junction temperature of TQ-40?"

        lexical_cands = [
            (self.sample_chunks[1], 10.0),  # Withdrawn r1 had high lexical score
            (self.sample_chunks[0], 9.5),   # Active r2
        ]
        semantic_cands = [
            (self.sample_chunks[1], 0.9),
            (self.sample_chunks[0], 0.88),
        ]

        reranked = reranker.rerank(query, lexical_cands, semantic_cands, top_k=2)
        # Active r2 must rank ahead of withdrawn r1
        self.assertEqual(reranked[0][0].chunk_id, "chunk1")
        self.assertFalse(reranked[0][0].is_superseded_or_withdrawn())

    def test_hybrid_retriever_save_and_load(self):
        """HybridRetriever should persist to disk and reload seamlessly."""
        retriever = HybridRetriever()
        retriever.build(self.sample_chunks)
        retriever.save(self.temp_dir)

        # Verify files exist
        self.assertTrue((self.temp_dir / "lexical.pkl").exists())
        self.assertTrue((self.temp_dir / "embeddings.npy").exists())
        self.assertTrue((self.temp_dir / "vector_meta.json").exists())

        loaded = HybridRetriever.load(self.temp_dir)
        self.assertIsNotNone(loaded)
        results = loaded.retrieve("What board revision is on the asset label?")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0][0].chunk_id, "chunk5")


if __name__ == "__main__":
    unittest.main()
