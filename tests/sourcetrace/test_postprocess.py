import unittest

import _path  # noqa: F401
from sourcetrace.postprocess import clean_answer, parse_bool, parse_ids, parse_reply

CLEAN = [
    ("94", "94"),
    ("94 C", "94"),
    ("94°C", "94"),
    ("94 degrees Celsius", "94"),
    ("180 seconds", "180"),
    ("180s", "180"),
    ("410 W", "410"),
    ("56 GiB", "56"),
    ("$6412", "6412"),
    ("Q3 FY27", "Q3 FY27"),
    ("REV-C2", "REV-C2"),
    ("ORR-FAN-2214-B", "ORR-FAN-2214-B"),
    ("12V-2x6", "12V-2x6"),
    ("4.3.2", "4.3.2"),
    ("4.1 TB/s", "4.1"),
    ("E7731", "E7731"),
    ("B14", "B14"),
    ('"B14"', "B14"),
    ("Answer: 94", "94"),
    ("  94.  ", "94"),
    ("", ""),
    (None, ""),
    ("NONE", ""),
    ("N/A", ""),
    ("unknown", ""),
    ("Not specified in the passages", ""),
    ("The passages do not state the price.", ""),
    ("I could not find that", ""),
    ("No information available", ""),
]


class CleanAnswerTests(unittest.TestCase):
    def test_cases(self):
        for raw, want in CLEAN:
            with self.subTest(raw=raw):
                self.assertEqual(clean_answer(raw), want)

    def test_keeps_identifiers_that_look_like_units(self):
        self.assertEqual(clean_answer("A100"), "A100")
        self.assertEqual(clean_answer("MI300X"), "MI300X")


class ParseReplyTests(unittest.TestCase):
    def test_plain_json(self):
        r = parse_reply('{"why": "x", "answer": "94", "evidence": [1], "lookup": null}')
        self.assertEqual(r["answer"], "94")

    def test_fenced_and_chatty(self):
        r = parse_reply('Sure!\n```json\n{"answer": "180", "evidence": [2]}\n```')
        self.assertEqual(r["answer"], "180")

    def test_trailing_comma(self):
        self.assertEqual(parse_reply('{"answer": "1", "evidence": [1,],}')["answer"], "1")

    def test_broken_json_falls_back_to_regex(self):
        r = parse_reply('{"answer": "B14", "evidence": [3, 4] oops')
        self.assertEqual(r["answer"], "B14")
        self.assertEqual(parse_ids(r["evidence"]), [3, 4])

    def test_garbage(self):
        self.assertEqual(parse_reply("no json here"), {})
        self.assertEqual(parse_reply(""), {})


class ParseIdsTests(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(parse_ids([1, 2]), [1, 2])
        self.assertEqual(parse_ids(["1", "[2]"]), [1, 2])
        self.assertEqual(parse_ids("[1],[2], 2"), [1, 2])
        self.assertEqual(parse_ids(3), [3])
        self.assertEqual(parse_ids(None), [])


class ParseBoolTests(unittest.TestCase):
    def test_values(self):
        self.assertIs(parse_bool(True), True)
        self.assertIs(parse_bool("No"), False)
        self.assertIsNone(parse_bool("maybe"))


if __name__ == "__main__":
    unittest.main()
