"""Tests for Phase 3: Document Parsers."""

from __future__ import annotations

from pathlib import Path
import unittest

from app.parsers import (
    parse_csv,
    parse_docx,
    parse_image,
    parse_pdf,
    parse_source,
    parse_text,
    parse_xlsx,
)


class TestPhase3Parsers(unittest.TestCase):
    def setUp(self):
        self.corpus = Path("mc3-corpus")

    def test_pdf_active_and_withdrawn_detection(self):
        """Active PDF should parse cleanly; withdrawn PDF should be flagged."""
        # Active R2
        r2_path = self.corpus / "specs" / "tq40_datasheet_r2.pdf"
        r2_doc = parse_pdf(r2_path, "specs/tq40_datasheet_r2.pdf")
        self.assertTrue(r2_doc.is_valid)
        self.assertIn("94", r2_doc.chunks[0].text)
        self.assertFalse(r2_doc.metadata.get("is_withdrawn", False))

        # Withdrawn R1
        r1_path = self.corpus / "specs" / "tq40_datasheet_r1_WITHDRAWN.pdf"
        r1_doc = parse_pdf(r1_path, "specs/tq40_datasheet_r1_WITHDRAWN.pdf")
        self.assertTrue(r1_doc.is_valid)
        self.assertTrue(r1_doc.metadata.get("is_withdrawn", False))

        # Encrypted PDF
        enc_path = self.corpus / "vendor" / "supplier_agreement_ENCRYPTED.pdf"
        enc_doc = parse_pdf(enc_path, "vendor/supplier_agreement_ENCRYPTED.pdf")
        self.assertEqual(enc_doc.error, "encrypted")

    def test_docx_parser(self):
        """DOCX should parse headings, paragraphs, and tables."""
        docx_path = self.corpus / "planning" / "roadmap_fy27.docx"
        doc = parse_docx(docx_path, "planning/roadmap_fy27.docx")
        self.assertTrue(doc.is_valid)
        found_tq60 = any("TQ-60" in c.text and "Q3 FY27" in c.text for c in doc.chunks)
        self.assertTrue(found_tq60, "Should find TQ-60 and Q3 FY27 in docx chunks")

    def test_xlsx_parser(self):
        """XLSX parser should extract rows as structured records with headers."""
        xlsx_path = self.corpus / "support" / "rma_parts.xlsx"
        doc = parse_xlsx(xlsx_path, "support/rma_parts.xlsx")
        self.assertTrue(doc.is_valid)
        found_fan = any("ORR-FAN-2214-B" in c.text for c in doc.chunks)
        self.assertTrue(found_fan, "Should extract fan assembly part number from xlsx")

    def test_csv_parser(self):
        """CSV parser should preserve exact values and column headers."""
        csv_path = self.corpus / "support" / "bug_database.csv"
        doc = parse_csv(csv_path, "support/bug_database.csv")
        self.assertTrue(doc.is_valid)
        found_bug = any("ORR-1847" in c.text and "4.3.2" in c.text for c in doc.chunks)
        self.assertTrue(found_bug, "Should preserve ORR-1847 and 4.3.2 in csv chunk")

    def test_text_and_log_parser(self):
        """Text/log parser should extract log events and timestamps."""
        log_path = self.corpus / "logs" / "prod_inference_2026-09-02.log"
        doc = parse_text(log_path, "logs/prod_inference_2026-09-02.log")
        self.assertTrue(doc.is_valid)
        found_e7731 = any("E7731" in c.text and "thermal throttle" in c.text for c in doc.chunks)
        self.assertTrue(found_e7731, "Should extract E7731 thermal throttle event")

    def test_source_parser(self):
        """Python source parser should extract constants and comments."""
        src_path = self.corpus / "engineering" / "ingest_service.py"
        doc = parse_source(src_path, "engineering/ingest_service.py")
        self.assertTrue(doc.is_valid)
        found_const = any("180" in c.text and "DEFAULT_BATCH_TIMEOUT" in c.text for c in doc.chunks)
        self.assertTrue(found_const, "Should extract DEFAULT_BATCH_TIMEOUT and value 180")

    def test_image_parser(self):
        """Image parser should validate format and dimensions."""
        img_path = self.corpus / "specs" / "backplane_pinout.png"
        doc = parse_image(img_path, "specs/backplane_pinout.png")
        self.assertTrue(doc.is_valid)
        self.assertGreater(doc.metadata["width"], 0)
        self.assertGreater(doc.metadata["height"], 0)


if __name__ == "__main__":
    unittest.main()
