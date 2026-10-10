"""Tests for Phase 5: Canonical Document Representation."""

from __future__ import annotations

import unittest

from app.indexing.models import DocumentChunk, ParsedDocument


class TestPhase5CanonicalRepresentation(unittest.TestCase):
    def test_document_chunk_mandatory_fields_and_validation(self):
        """DocumentChunk must strictly enforce schema and POSIX path normalization."""
        chunk = DocumentChunk(
            chunk_id="support/bug_database.csv#r18",
            source_path="support\\bug_database.csv",  # Windows backslash
            file_type="csv",
            text="ORR-1847 ... fixed in 4.3.2",
            metadata={"row": 18},
            row=18,
            content_type="table_row",
        )

        # Enforces POSIX relative path
        self.assertEqual(chunk.source_path, "support/bug_database.csv")
        self.assertEqual(chunk.row, 18)
        self.assertEqual(chunk.content_type, "table_row")

        # Invalid chunk test
        with self.assertRaises(ValueError):
            DocumentChunk(
                chunk_id="",
                source_path="test.txt",
                file_type="txt",
                text="content",
            )

    def test_chunk_serialization_and_deserialization(self):
        """to_dict and from_dict round-trip should be exact."""
        original = DocumentChunk(
            chunk_id="specs/tq40_datasheet_r2.pdf#p1",
            source_path="specs/tq40_datasheet_r2.pdf",
            file_type="pdf",
            page=1,
            section="Electrical and thermal",
            text="Maximum junction temperature .......... 94 C",
            metadata={"revision": "r2", "is_withdrawn": False},
            content_type="text",
        )

        data = original.to_dict()
        self.assertEqual(data["chunk_id"], original.chunk_id)
        self.assertEqual(data["source_path"], original.source_path)
        self.assertEqual(data["page"], 1)

        restored = DocumentChunk.from_dict(data)
        self.assertEqual(original, restored)
        self.assertEqual(restored.metadata["revision"], "r2")

    def test_superseded_detection_on_chunk(self):
        """is_superseded_or_withdrawn must accurately flag withdrawn documents."""
        active = DocumentChunk(
            chunk_id="specs/tq40_datasheet_r2.pdf#p1",
            source_path="specs/tq40_datasheet_r2.pdf",
            file_type="pdf",
            text="active",
            metadata={"is_withdrawn": False},
        )
        self.assertFalse(active.is_superseded_or_withdrawn())

        withdrawn = DocumentChunk(
            chunk_id="specs/tq40_datasheet_r1_WITHDRAWN.pdf#p1",
            source_path="specs/tq40_datasheet_r1_WITHDRAWN.pdf",
            file_type="pdf",
            text="withdrawn",
            metadata={"is_withdrawn": True},
        )
        self.assertTrue(withdrawn.is_superseded_or_withdrawn())

    def test_identifier_extraction(self):
        """extract_identifiers must find ticket IDs, part numbers, and error codes."""
        chunk = DocumentChunk(
            chunk_id="c1",
            source_path="support/bug_database.csv",
            file_type="csv",
            text="Ticket ORR-1847 fixed in version 4.3.2 with part ORR-FAN-2214-B and error E7731",
        )
        identifiers = chunk.extract_identifiers()
        self.assertIn("ORR-1847", identifiers)
        self.assertIn("4.3.2", identifiers)
        self.assertIn("ORR-FAN-2214-B", identifiers)
        self.assertIn("E7731", identifiers)

    def test_parsed_document_representation(self):
        """ParsedDocument should contain canonical chunks and serialize cleanly."""
        doc = ParsedDocument(
            source_path="support/bug_database.csv",
            file_type="csv",
            title="Bug Database",
            metadata={"rows": 5},
        )
        doc.add_chunk(
            DocumentChunk(
                chunk_id="c1",
                source_path="support/bug_database.csv",
                file_type="csv",
                text="row1",
            )
        )
        self.assertTrue(doc.is_valid)

        d_dict = doc.to_dict()
        restored = ParsedDocument.from_dict(d_dict)
        self.assertEqual(len(restored.chunks), 1)
        self.assertEqual(restored.source_path, "support/bug_database.csv")


if __name__ == "__main__":
    unittest.main()
