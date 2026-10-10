"""Tests for Phase 6: Format-Aware Chunking Strategy."""

from __future__ import annotations

import unittest

from app.indexing.chunker import Chunker


class TestPhase6Chunking(unittest.TestCase):
    def test_log_chunking_strategy(self):
        """Log chunking must generate event-specific chunks for ERROR and E7731."""
        lines = [
            "2026-09-02T03:11:04Z meridian[2211]: INFO  scheduler: batch accepted",
            "2026-09-02T03:14:22Z meridian[2211]: WARN  thermal: approaching limit",
            "2026-09-02T03:14:41Z meridian[2211]: ERROR E7731: thermal throttle engaged on die 0",
            "2026-09-02T03:15:00Z meridian[2211]: INFO  scheduler: recovering",
        ]
        chunks = Chunker.chunk_log(lines, "logs/prod.log", window_size=3, step=2)

        event_chunks = [c for c in chunks if c.content_type == "log_event"]
        self.assertTrue(len(event_chunks) >= 1)
        error_chunk = next(c for c in event_chunks if "E7731" in c.text)
        self.assertEqual(error_chunk.metadata.get("error_code"), "E7731")

    def test_csv_chunking_strategy(self):
        """CSV chunking must format rows with column headers and preserve raw values."""
        headers = ["ticket", "component", "fixed_in"]
        rows = [
            (2, ["ORR-1847", "power", "4.3.2"]),
            (3, ["ORR-1802", "pcie", "4.3.1"]),
        ]
        chunks = Chunker.chunk_csv(headers, rows, "support/bug_database.csv")
        self.assertEqual(len(chunks), 2)
        self.assertIn("fixed_in: 4.3.2", chunks[0].text)
        self.assertIn("Raw: ORR-1847,power,4.3.2", chunks[0].text)
        self.assertEqual(chunks[0].metadata["fields"]["ticket"], "ORR-1847")

    def test_xlsx_chunking_strategy(self):
        """XLSX chunking must preserve sheet name and structured row data."""
        headers = ["part_number", "description", "compatible_with"]
        rows = [
            (2, ["ORR-FAN-2214-B", "Fan assembly, dual-rotor", "TQ-40"]),
        ]
        chunks = Chunker.chunk_xlsx("RMA Parts", headers, rows, "support/rma_parts.xlsx")
        self.assertEqual(len(chunks), 1)
        self.assertIn("Sheet: RMA Parts", chunks[0].text)
        self.assertIn("part_number: ORR-FAN-2214-B", chunks[0].text)
        self.assertEqual(chunks[0].sheet, "RMA Parts")

    def test_pdf_chunking_strategy(self):
        """PDF chunking must be page-aware and track revisions."""
        pages = [
            (1, "Orrery Systems TQ-40 Datasheet revision 2"),
        ]
        chunks = Chunker.chunk_pdf(pages, "specs/tq40_datasheet_r2.pdf", revision="r2", is_withdrawn=False)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page, 1)
        self.assertEqual(chunks[0].metadata["revision"], "r2")
        self.assertFalse(chunks[0].metadata["is_withdrawn"])


if __name__ == "__main__":
    unittest.main()
