import tempfile
import unittest
import zipfile
from pathlib import Path

import _path  # noqa: F401
from _util import SAMPLE_CORPUS
from PIL import Image

from sourcetrace.models import Skip
from sourcetrace.parsers import ParseContext, parse_file


def parse(rel: str, ctx: ParseContext | None = None):
    return parse_file(SAMPLE_CORPUS / rel, rel, ctx)


def text_of(doc) -> str:
    return "\n".join(c.text for c in doc.chunks)


class SampleCorpusParsers(unittest.TestCase):
    def test_pdf_current_revision(self):
        doc = parse("specs/tq40_datasheet_r2.pdf")
        self.assertIn("94 C", text_of(doc))
        self.assertFalse(doc.superseded)

    def test_pdf_withdrawn_revision_is_flagged(self):
        doc = parse("specs/tq40_datasheet_r1_WITHDRAWN.pdf")
        self.assertIn("105 C", text_of(doc))
        self.assertTrue(doc.superseded)
        self.assertTrue(all(c.superseded for c in doc.chunks))

    def test_encrypted_pdf_is_skipped(self):
        with self.assertRaises(Skip):
            parse("vendor/supplier_agreement_ENCRYPTED.pdf")

    def test_docx_text(self):
        self.assertIn("Q3 FY27", text_of(parse("planning/roadmap_fy27.docx")))

    def test_xlsx_rows_keep_their_headers(self):
        text = text_of(parse("support/rma_parts.xlsx"))
        self.assertIn("part_number=ORR-FAN-2214-B", text)
        self.assertIn("compatible_with=TQ-40", text)
        self.assertIn("unit_cost_usd=84.5", text)
        self.assertIn("Lead times are supplier-quoted", text)  # second sheet

    def test_csv_rows(self):
        text = text_of(parse("support/bug_database.csv"))
        self.assertIn("ticket=ORR-1847", text)
        self.assertIn("fixed_in=4.3.2", text)

    def test_python_source(self):
        doc = parse("engineering/ingest_service.py")
        self.assertIn("DEFAULT_BATCH_TIMEOUT_S = 180", text_of(doc))
        self.assertEqual(doc.chunks[0].kind, "code")

    def test_log_and_txt(self):
        self.assertIn("E7731", text_of(parse("logs/prod_inference_2026-09-02.log")))
        self.assertIn("4.3.1", text_of(parse("engineering/meridian_release_notes.txt")))

    def test_unknown_binary_type_is_skipped(self):
        with self.assertRaises(Skip):
            parse("vendor/telemetry_capture.dat")

    def test_image_gets_a_searchable_stub_without_a_model(self):
        doc = parse("specs/backplane_pinout.png")
        self.assertEqual(doc.chunks[0].kind, "image")
        self.assertIn("backplane_pinout.png", doc.chunks[0].text)
        self.assertEqual(doc.note, "stub")

    def test_image_caption_comes_from_the_vision_model(self):
        seen = []

        def describe(img: Image.Image, label: str) -> str:
            seen.append((img.mode, label))
            return "TEXT: B14 THERM_ALERT#"

        doc = parse("specs/backplane_pinout.png", ParseContext(describe=describe))
        self.assertIn("THERM_ALERT#", doc.chunks[0].text)
        self.assertEqual(seen, [("RGB", "specs/backplane_pinout.png")])


class EdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="st-parse-"))

    def write(self, name: str, data: bytes | str) -> Path:
        path = self.tmp / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return path

    def test_empty_file_is_skipped(self):
        with self.assertRaises(Skip):
            parse_file(self.write("a.txt", ""), "a.txt")

    def test_corrupt_docx_is_skipped(self):
        with self.assertRaises(Skip):
            parse_file(self.write("a.docx", b"not a zip"), "a.docx")

    def test_extensionless_text_is_read(self):
        doc = parse_file(self.write("NOTES", "limit is 42\n"), "NOTES")
        self.assertIn("42", text_of(doc))

    def test_extensionless_binary_is_skipped(self):
        with self.assertRaises(Skip):
            parse_file(self.write("blob", bytes(range(256)) * 20), "blob")

    def test_pdf_with_wrong_extension_is_still_recognised(self):
        data = (SAMPLE_CORPUS / "specs/tq40_datasheet_r2.pdf").read_bytes()
        doc = parse_file(self.write("mystery", data), "mystery")
        self.assertIn("94 C", text_of(doc))

    def test_latin1_text(self):
        doc = parse_file(self.write("a.txt", "température 94 °C".encode("latin-1")), "a.txt")
        self.assertIn("94", text_of(doc))

    def test_long_text_is_split_with_overlap(self):
        lines = "\n".join(f"line {i} " + "x" * 60 for i in range(300))
        doc = parse_file(self.write("big.log", lines), "big.log")
        self.assertGreater(len(doc.chunks), 3)
        self.assertTrue(all(len(c.text) < 2200 for c in doc.chunks))
        joined = "\n".join(c.text for c in doc.chunks)
        for i in (0, 150, 299):
            self.assertIn(f"line {i} ", joined)

    def test_docx_table_rows_are_labelled(self):
        body = (
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
            'wordprocessingml/2006/main"><w:body>'
            "<w:p><w:r><w:t>Plan</w:t></w:r></w:p>"
            "<w:tbl>"
            "<w:tr><w:tc><w:p><w:r><w:t>part</w:t></w:r></w:p></w:tc>"
            "<w:tc><w:p><w:r><w:t>milestone</w:t></w:r></w:p></w:tc></w:tr>"
            "<w:tr><w:tc><w:p><w:r><w:t>TQ-60</w:t></w:r></w:p></w:tc>"
            "<w:tc><w:p><w:r><w:t>Q4 FY27</w:t></w:r></w:p></w:tc></w:tr>"
            "</w:tbl></w:body></w:document>"
        )
        path = self.tmp / "t.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("word/document.xml", body)
        text = text_of(parse_file(path, "t.docx"))
        self.assertIn("part=TQ-60", text)
        self.assertIn("milestone=Q4 FY27", text)

    def test_inline_string_xlsx(self):
        sheet = (
            '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main"><sheetData>'
            '<row r="1"><c r="A1" t="inlineStr"><is><t>name</t></is></c>'
            '<c r="B1" t="inlineStr"><is><t>value</t></is></c></row>'
            '<row r="2"><c r="A2" t="inlineStr"><is><t>alpha</t></is></c><c r="B2"><v>7</v></c></row>'
            '<row r="3"><c r="B3"><v>9</v></c></row>'
            "</sheetData></worksheet>"
        )
        wb = (
            '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main"><sheets><sheet name="Data" sheetId="1"/></sheets></workbook>'
        )
        path = self.tmp / "t.xlsx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("xl/workbook.xml", wb)
            zf.writestr("xl/worksheets/sheet1.xml", sheet)
        text = text_of(parse_file(path, "t.xlsx"))
        self.assertIn("name=alpha | value=7", text)
        self.assertIn("value=9", text)  # a row with a missing first cell stays aligned


if __name__ == "__main__":
    unittest.main()
