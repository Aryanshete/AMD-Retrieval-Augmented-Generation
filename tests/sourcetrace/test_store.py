import tempfile
import unittest

import _path  # noqa: F401
from _util import SAMPLE_QUESTIONS, make_corpus_copy, make_index

from sourcetrace import walker
from sourcetrace.store import Index


class SearchTests(unittest.TestCase):
    def test_identifier_lookup_finds_the_exact_row(self):
        idx = make_index(
            {
                "a.csv": "ticket=ORR-1847 | fixed_in=4.3.2",
                "b.csv": "ticket=ORR-1802 | fixed_in=4.3.1",
                "c.txt": "tickets are tracked in the bug database",
            }
        )
        top = idx.search("ORR-1847", k=3)
        self.assertEqual(top[0][0].file, "a.csv")

    def test_superseded_chunks_rank_lower(self):
        idx = make_index({"r2.pdf": "max junction temperature 94"})
        old = make_index({"r1.pdf": "max junction temperature 105"}, superseded=True)
        idx.chunks.extend(old.chunks)
        idx.chunks[1].id = 1
        idx._fit()
        top = idx.search("maximum junction temperature", k=2)
        self.assertEqual([c.file for c, _ in top], ["r2.pdf", "r1.pdf"])

    def test_per_file_cap_gives_other_files_a_turn(self):
        idx = make_index({f"f{i}.txt": "thermal throttle" for i in range(5)})
        idx.chunks.extend(make_index({"big.txt": "thermal throttle thermal"}).chunks * 5)
        for i, c in enumerate(idx.chunks):
            c.id = i
        idx._fit()
        files = [c.file for c, _ in idx.search("thermal throttle", k=8, per_file=2)]
        self.assertLessEqual(files.count("big.txt"), 2)

    def test_exclude(self):
        idx = make_index({"a.txt": "alpha beta", "b.txt": "alpha gamma"})
        top = idx.search("alpha", k=5, exclude={0})
        self.assertEqual([c.file for c, _ in top], ["b.txt"])

    def test_empty_index(self):
        self.assertEqual(Index().search("anything"), [])

    def test_file_name_words_are_searchable(self):
        idx = make_index({"support/asset_label.jpg": "[image file]", "x.txt": "other words"})
        self.assertEqual(idx.search("asset label", k=1)[0][0].file, "support/asset_label.jpg")


class PersistenceTests(unittest.TestCase):
    def test_roundtrip(self):
        idx = make_index({"a.txt": "alpha beta", "b.txt": "gamma delta"})
        idx.skipped = [("c.bin", "skipped: unknown file type .bin")]
        d = tempfile.mkdtemp()
        idx.save(d)
        loaded = Index.load(d)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded.skipped, idx.skipped)
        self.assertEqual(loaded.corpus_root, "/corpus")
        self.assertEqual(loaded.search("gamma", k=1)[0][0].file, "b.txt")

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            Index.load(tempfile.mkdtemp())


class SampleCorpusRecall(unittest.TestCase):
    """With no model at all, the right file must be among the first few hits."""

    @classmethod
    def setUpClass(cls):
        res = walker.walk_corpus(make_corpus_copy())
        cls.index = Index.build(res.docs, res.skipped, "/c")

    def test_text_questions_retrieve_their_source(self):
        for q in SAMPLE_QUESTIONS:
            if not q["expected_citations"] or q["n"] in (7, 8):  # 7, 8 need picture captions
                continue
            with self.subTest(n=q["n"]):
                files = [c.file for c, _ in self.index.search(q["query"], k=4)]
                for needed in q["expected_citations"]:
                    self.assertIn(needed, files)

    def test_withdrawn_datasheet_ranks_below_current(self):
        files = [c.file for c, _ in self.index.search("maximum junction temperature TQ-40", k=6)]
        self.assertLess(
            files.index("specs/tq40_datasheet_r2.pdf"),
            files.index("specs/tq40_datasheet_r1_WITHDRAWN.pdf"),
        )


if __name__ == "__main__":
    unittest.main()
