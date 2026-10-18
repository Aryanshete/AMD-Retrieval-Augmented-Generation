import unittest

import _path  # noqa: F401
from sourcetrace.textutil import (
    contains_value,
    normalize,
    query_identifiers,
    same_answer,
    tokenize,
)


class NormalizeTests(unittest.TestCase):
    def test_grader_rule(self):
        self.assertEqual(normalize("q3 fy27"), "Q3FY27")
        self.assertEqual(normalize("ORR-FAN-2214-B"), "ORRFAN2214B")
        self.assertEqual(normalize("94.5"), "945")
        self.assertEqual(normalize("a_b·c"), "ABC")

    def test_same_answer(self):
        self.assertTrue(same_answer("Q3 FY27", "Q3FY27"))
        self.assertFalse(same_answer("Q3", "Q3 FY27"))


class TokenizeTests(unittest.TestCase):
    def test_compound_identifiers_keep_parts(self):
        toks = tokenize("Ticket ORR-1847 failed")
        self.assertIn("orr-1847", toks)
        self.assertIn("orr", toks)
        self.assertIn("1847", toks)

    def test_stopwords_and_plurals(self):
        toks = tokenize("What are the temperatures?")
        self.assertEqual(toks, ["temperature"])

    def test_path_like(self):
        self.assertTrue({"tq40", "datasheet", "r2"} <= set(tokenize("specs/tq40_datasheet_r2.pdf")))


class IdentifierTests(unittest.TestCase):
    def test_finds_identifiers(self):
        ids = query_identifiers("Which firmware fixed ticket ORR-1847 on the TQ-40 (E7731)?")
        self.assertIn("orr-1847", ids)
        self.assertIn("tq-40", ids)
        self.assertIn("e7731", ids)

    def test_plain_words_are_not_identifiers(self):
        self.assertEqual(query_identifiers("What is the maximum temperature?"), [])


class ContainsValueTests(unittest.TestCase):
    def test_whole_token_match(self):
        self.assertTrue(contains_value("94", "Maximum junction temperature .... 94 C"))
        self.assertFalse(contains_value("94", "Power 194 W"))

    def test_multi_token_values(self):
        self.assertTrue(contains_value("REV-C2", "BOARD REVISION   REV-C2"))
        self.assertTrue(contains_value("Q3 FY27", "enters sampling in Q3 FY27."))
        self.assertFalse(contains_value("Q3 FY27", "Q3 FY26 and FY27"))

    def test_empty(self):
        self.assertFalse(contains_value("", "anything"))
        self.assertFalse(contains_value("---", "anything"))


if __name__ == "__main__":
    unittest.main()
