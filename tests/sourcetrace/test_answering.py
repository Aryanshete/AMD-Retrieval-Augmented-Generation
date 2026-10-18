import shutil
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401
from _util import (
    SAMPLE_CORPUS,
    is_verify,
    make_index,
    passages_in,
    reply,
    test_settings,
    verdict,
)

from sourcetrace.answering import Reasoner
from sourcetrace.backends.mock import MockBackend
from sourcetrace.models import Chunk, ParsedDoc
from sourcetrace.store import Index
from sourcetrace.timing import Deadline

Q_TEMP = "What is the maximum junction temperature of the TQ-40?"

FILES = {
    "specs/r2.pdf": "TQ-40 datasheet revision 2. Maximum junction temperature 94 C.",
    "specs/r1.pdf": "TQ-40 datasheet revision 1 WITHDRAWN. Maximum junction temperature 105 C.",
    "notes/misc.txt": "General notes about the TQ-40 and its junction temperature sensors.",
}


def run(responder, files=None, question=Q_TEMP, seconds=20.0, root=None, index=None, **cfg):
    backend = MockBackend(responder)
    idx = index or make_index(files or FILES)
    settings = test_settings(**cfg)
    result = Reasoner(backend, idx, settings).answer(question, Deadline(seconds), root)
    return result, backend


def ids_of(prompt: str, name: str) -> list[int]:
    return passages_in(prompt).get(name, [])


class BasicAnswerTests(unittest.TestCase):
    def test_single_source_answer(self):
        def responder(prompt, image):
            if is_verify(prompt):
                return verdict(True)
            return reply("94", ids_of(prompt, "specs/r2.pdf"))

        result, _ = run(responder)
        self.assertEqual(result.answer, "94")
        self.assertEqual(result.citations, ["specs/r2.pdf"])
        self.assertGreater(result.confidence, 0.5)

    def test_units_are_dropped_from_the_answer(self):
        def responder(prompt, image):
            if is_verify(prompt):
                return verdict(True)
            return reply("94 C", ids_of(prompt, "specs/r2.pdf"))

        result, _ = run(responder)
        self.assertEqual(result.answer, "94")

    def test_model_refusal_gives_empty_answer_and_citations(self):
        result, _ = run(lambda p, i: reply("", []))
        self.assertEqual((result.answer, result.citations, result.confidence), ("", [], 0.0))

    def test_ungrounded_answer_is_refused(self):
        """A value that is in no passage came from the model's head: fictional corpus, so wrong."""

        def responder(prompt, image):
            return verdict(True) if is_verify(prompt) else reply("6412", [1])

        result, _ = run(responder)
        self.assertEqual(result.answer, "")
        self.assertIn("verbatim", result.trace["refused"])

    def test_verification_can_veto(self):
        def responder(prompt, image):
            if is_verify(prompt):
                return verdict(False)
            return reply("94", ids_of(prompt, "specs/r2.pdf"))

        result, _ = run(responder)
        self.assertEqual((result.answer, result.citations), ("", []))
        self.assertIn("verification", result.trace["refused"])

    def test_unparsable_verification_does_not_veto_a_grounded_answer(self):
        def responder(prompt, image):
            if is_verify(prompt):
                return "hmm"
            return reply("94", ids_of(prompt, "specs/r2.pdf"))

        result, _ = run(responder)
        self.assertEqual(result.answer, "94")

    def test_missing_evidence_is_recovered_from_grounding(self):
        def responder(prompt, image):
            return verdict(True) if is_verify(prompt) else reply("94", [])

        result, _ = run(responder)
        self.assertEqual(result.answer, "94")
        self.assertEqual(result.citations, ["specs/r2.pdf"])

    def test_grounding_prefers_current_over_superseded(self):
        files = {
            "specs/r1.pdf": "revision 1 WITHDRAWN max junction 94 C",
            "specs/r2.pdf": "max junction 94 C",
        }
        idx = make_index({"specs/r2.pdf": files["specs/r2.pdf"]})
        old = make_index({"specs/r1.pdf": files["specs/r1.pdf"]}, superseded=True)
        idx.chunks.extend(old.chunks)
        idx.chunks[1].id = 1
        idx._fit()

        def responder(prompt, image):
            return verdict(True) if is_verify(prompt) else reply("94", [])

        result, _ = run(responder, index=idx)
        self.assertEqual(result.citations, ["specs/r2.pdf"])

    def test_empty_question_and_empty_index(self):
        result, _ = run(lambda p, i: reply("x"), question="  ")
        self.assertEqual(result.answer, "")
        result, _ = run(lambda p, i: reply("x"), index=Index())
        self.assertEqual(result.answer, "")

    def test_backend_exception_becomes_a_refusal(self):
        def responder(prompt, image):
            raise RuntimeError("GPU on fire")

        result, _ = run(responder)
        self.assertEqual((result.answer, result.citations), ("", []))

    def test_expired_deadline_makes_no_model_call(self):
        result, backend = run(lambda p, i: reply("94", [1]), seconds=0.0)
        self.assertEqual(result.answer, "")
        self.assertEqual(backend.calls, [])


class CitationPruningTests(unittest.TestCase):
    def test_redundant_file_is_dropped(self):
        files = {
            "specs/r2.pdf": "Maximum junction temperature 94 C (datasheet).",
            "notes/copy.txt": "Copy of the datasheet: maximum junction temperature 94 C.",
        }

        def responder(prompt, image):
            if is_verify(prompt):
                return verdict(True)
            seen = passages_in(prompt)
            return reply("94", [i for ids in seen.values() for i in ids])  # cites everything

        result, backend = run(responder, files=files)
        self.assertEqual(result.answer, "94")
        self.assertEqual(len(result.citations), 1)  # exact-set grading: one of them must go
        self.assertEqual(len(result.trace["pruned"]), 1)

    def test_needed_file_is_kept(self):
        files = {
            "support/bugs.csv": "ticket=ORR-1847 | fixed_in=4.3.2",
            "notes/release.txt": "Release 4.3.2 contains many fixes.",
        }

        def responder(prompt, image):
            if is_verify(prompt):
                return verdict(True)
            seen = passages_in(prompt)
            if "support/bugs.csv" not in seen:  # without the table there is no answer
                return reply("")
            return reply("4.3.2", [i for ids in seen.values() for i in ids])

        result, _ = run(responder, files=files, question="Which firmware version fixed ORR-1847?")
        self.assertEqual(result.citations, ["support/bugs.csv"])

    def test_pruning_can_be_disabled(self):
        files = {"a.txt": "limit 94 here", "b.txt": "limit 94 there"}

        def responder(prompt, image):
            if is_verify(prompt):
                return verdict(True)
            return reply("94", [i for ids in passages_in(prompt).values() for i in ids])

        result, _ = run(responder, files=files, question="limit?", prune_citations=False)
        self.assertEqual(sorted(result.citations), ["a.txt", "b.txt"])

    def test_pruning_stops_when_time_runs_out(self):
        files = {"a.txt": "limit 94 here", "b.txt": "limit 94 there"}

        def responder(prompt, image):  # every call "takes" 8 s on the fake clock
            if is_verify(prompt):
                return verdict(True)
            return reply("94", [i for ids in passages_in(prompt).values() for i in ids])

        backend = MockBackend(responder)
        deadline = Deadline(20.0, clock=lambda: _tick(backend))
        result = Reasoner(backend, make_index(files), test_settings()).answer("limit?", deadline)
        # Whatever happens, a grounded answer with its citations is still returned.
        self.assertEqual(result.answer, "94")
        self.assertTrue(result.citations)


def _tick(backend) -> float:
    return float(len(backend.calls)) * 8.0


class MultiHopTests(unittest.TestCase):
    FILES = {
        "logs/prod.log": "ERROR E7731: thermal throttle engaged. incident logged against ORR-1847",
        "support/bugs.csv": "ticket=ORR-1847 | summary=Die temperature sensor | fixed_in=4.3.2",
        "engineering/notes.txt": "4.3.0 scheduler. 4.3.1 PCIe fix. 4.3.3 packaging only.",
    }
    Q = "The production log shows a thermal throttle incident. Which firmware release fixed the defect?"

    def responder(self, prompt, image):
        if is_verify(prompt):
            return verdict(True)
        seen = passages_in(prompt)
        if "support/bugs.csv" in seen and "logs/prod.log" in seen:
            return reply("4.3.2", seen["support/bugs.csv"])  # forgets to cite the log
        if "logs/prod.log" in seen:
            return reply("", seen["logs/prod.log"], lookup="ORR-1847")
        return reply("")

    def test_two_hop_chain_cites_both_files(self):
        result, backend = run(
            self.responder, files=self.FILES, question=self.Q, top_chunks=1, max_chunks_per_file=1
        )
        self.assertEqual(result.answer, "4.3.2")
        self.assertEqual(sorted(result.citations), ["logs/prod.log", "support/bugs.csv"])
        self.assertEqual(result.trace["lookup"], "ORR-1847")

    def test_chain_source_is_never_pruned(self):
        result, backend = run(
            self.responder, files=self.FILES, question=self.Q, top_chunks=1, max_chunks_per_file=1
        )
        self.assertIn("logs/prod.log", result.citations)
        self.assertNotIn("engineering/notes.txt", result.citations)

    def test_no_hop_when_disabled(self):
        result, _ = run(
            self.responder, files=self.FILES, question=self.Q, top_chunks=1,
            max_chunks_per_file=1, max_hops=1,
        )  # fmt: skip
        self.assertEqual((result.answer, result.citations), ("", []))

    def test_lookup_without_cited_evidence_still_protects_the_source(self):
        def responder(prompt, image):
            if is_verify(prompt):
                return verdict(True)
            seen = passages_in(prompt)
            if "support/bugs.csv" in seen:
                return reply("4.3.2", seen["support/bugs.csv"])
            return reply("", [], lookup="ORR-1847")  # forgot to cite the log in hop 1

        result, _ = run(
            responder, files=self.FILES, question=self.Q, top_chunks=1, max_chunks_per_file=1
        )
        self.assertEqual(sorted(result.citations), ["logs/prod.log", "support/bugs.csv"])


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()) / "corpus"
        (self.root / "specs").mkdir(parents=True)
        shutil.copy(SAMPLE_CORPUS / "specs/backplane_pinout.png", self.root / "specs/")
        doc = ParsedDoc("specs/backplane_pinout.png")
        doc.chunks.append(
            Chunk("specs/backplane_pinout.png", "image", "[image file] specs/backplane_pinout.png", "image")
        )  # fmt: skip
        other = ParsedDoc("notes.txt")
        other.chunks.append(Chunk("notes.txt", "lines 1-1", "TQ-40 backplane connector notes"))
        self.index = Index.build([doc, other], [], str(self.root))

    def test_picture_is_read_with_the_question(self):
        def responder(prompt, image):
            if image is not None:
                self.assertIn("Question:", prompt)
                return "B14 -> THERM_ALERT#"
            if is_verify(prompt):
                return verdict(True)
            seen = passages_in(prompt)
            if "reading of this picture" in prompt:
                return reply("B14", seen["specs/backplane_pinout.png"])
            return reply("")

        result, backend = run(
            responder, index=self.index, root=self.root,
            question="Which backplane pin carries THERM_ALERT# on the TQ-40?",
        )  # fmt: skip
        self.assertEqual(result.answer, "B14")
        self.assertEqual(result.citations, ["specs/backplane_pinout.png"])
        self.assertEqual(sum(1 for _, has_image in backend.calls if has_image), 1)

    def test_not_visible_adds_nothing(self):
        def responder(prompt, image):
            if image is not None:
                return "NOT VISIBLE"
            return reply("")

        result, _ = run(responder, index=self.index, root=self.root, question="backplane pin?")
        self.assertEqual(result.answer, "")

    def test_vision_lookup_can_be_disabled(self):
        result, backend = run(
            lambda p, i: reply(""), index=self.index, root=self.root,
            question="backplane pin?", vision_lookup=False,
        )  # fmt: skip
        self.assertFalse(any(has_image for _, has_image in backend.calls))

    def test_no_vision_call_when_time_is_short(self):
        result, backend = run(
            lambda p, i: reply(""), index=self.index, root=self.root,
            question="backplane pin?", seconds=10.0,
        )  # fmt: skip
        self.assertFalse(any(has_image for _, has_image in backend.calls))


if __name__ == "__main__":
    unittest.main()


class BridgeTests(unittest.TestCase):
    """The model cites only the table; the log that identifies the entry is still needed."""

    FILES = MultiHopTests.FILES
    Q = MultiHopTests.Q

    def make(self, needed: bool, question: str | None = None, answer_ids=("support/bugs.csv",)):
        def responder(prompt, image):
            if "Does the question REFER" in prompt:
                return json_bool(needed)
            if is_verify(prompt):
                return verdict(True)
            seen = passages_in(prompt)
            if "support/bugs.csv" not in seen:
                return reply("")
            return reply("4.3.2", [i for f in answer_ids for i in seen.get(f, [])])

        return run(responder, files=self.FILES, question=question or self.Q)

    def test_log_is_added_when_the_question_refers_to_it(self):
        result, _ = self.make(True)
        self.assertEqual(sorted(result.citations), ["logs/prod.log", "support/bugs.csv"])
        self.assertTrue(result.trace["bridge"]["logs/prod.log"])

    def test_not_added_when_the_check_says_no(self):
        result, _ = self.make(False)
        self.assertEqual(result.citations, ["support/bugs.csv"])

    def test_identifier_named_in_the_question_is_never_bridged(self):
        result, backend = self.make(True, question="Which firmware version fixed ticket ORR-1847?")
        self.assertEqual(result.citations, ["support/bugs.csv"])
        self.assertFalse(any("Does the question REFER" in p for p, _ in backend.calls))


def json_bool(value: bool) -> str:
    import json

    return json.dumps({"needed": value})
