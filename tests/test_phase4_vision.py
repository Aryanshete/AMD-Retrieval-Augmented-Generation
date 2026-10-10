"""Tests for Phase 4: Image Understanding and Diagram Reasoning."""

from __future__ import annotations

from pathlib import Path
import unittest

from app.parsers.image_parser import parse_image
from app.parsers.vision import VisionEngine, VisionResult


class TestPhase4Vision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Path("mc3-corpus")
        cls.pinout_path = cls.corpus / "specs" / "backplane_pinout.png"
        cls.asset_label_path = cls.corpus / "support" / "asset_label.jpg"
        cls.engine = VisionEngine.get_instance()

    def test_backplane_pinout_spatial_relationships(self):
        """Diagram analyzer must correctly map Pin B14 to THERM_ALERT#."""
        res: VisionResult = self.engine.analyze_image(self.pinout_path)
        self.assertIsNone(res.error)
        self.assertTrue(len(res.items) > 0)

        # Pin to Signal mapping verification
        self.assertIn("B14", res.pin_mappings)
        self.assertIn("THERM_ALERT", res.pin_mappings["B14"])

        # Signal to Pin reverse mapping
        self.assertEqual(res.signal_to_pin.get("THERM_ALERT#", res.signal_to_pin.get("THERM_ALERT")), "B14")

        # Verify other pins in diagram
        self.assertIn("B11", res.pin_mappings)
        self.assertIn("GND", res.pin_mappings["B11"])

    def test_asset_label_key_value_relationships(self):
        """Asset label parser must identify BOARD REVISION: REV-C2 and MODEL: TQ-40."""
        res: VisionResult = self.engine.analyze_image(self.asset_label_path)
        self.assertIsNone(res.error)

        found_board_rev = False
        for k, v in res.key_values.items():
            if "BOARD" in k and "REV-C2" in v:
                found_board_rev = True
                break
        self.assertTrue(found_board_rev or "REV-C2" in res.full_text)

    def test_parse_image_generates_canonical_chunks(self):
        """parse_image must generate canonical chunks with relationship metadata."""
        doc = parse_image(self.pinout_path, "specs/backplane_pinout.png")
        self.assertTrue(doc.is_valid)

        content_types = [c.content_type for c in doc.chunks]
        self.assertIn("image_relationships", content_types)
        self.assertIn("image_ocr", content_types)
        self.assertIn("image_metadata", content_types)

        rel_chunk = next(c for c in doc.chunks if c.content_type == "image_relationships")
        self.assertIn("B14", rel_chunk.text)
        self.assertIn("THERM_ALERT", rel_chunk.text)

    def test_corrupt_or_missing_image_resilience(self):
        """Vision engine and parser must safely handle missing or corrupt images."""
        doc = parse_image(Path("non_existent_image.png"), "specs/non_existent_image.png")
        self.assertFalse(doc.is_valid)
        self.assertIsNotNone(doc.error)


if __name__ == "__main__":
    unittest.main()
