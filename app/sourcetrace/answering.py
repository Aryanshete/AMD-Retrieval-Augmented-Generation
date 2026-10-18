"""Question answering with citations that survive the grader's exact-set comparison.

Pipeline (each model call is guarded by the per-question deadline):

1. retrieve passages (BM25 + dense, superseded revisions demoted);
2. for pictures near the top, let the vision model look at the picture with the question;
3. ask for ``{answer, evidence, lookup}``; if the answer needs an identifier that is not
   yet answered (ticket -> fixed-in version), retrieve again with that identifier (a hop);
4. deterministic grounding: the answer must literally occur in a cited passage, else refuse;
5. an independent yes/no verification of the answer against its evidence, else refuse;
6. citation pruning: re-ask without each cited file; if the answer is unchanged, that file
   was not necessary, so it is not cited (the grader's test is necessity, not relevance).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backends.base import LLMBackend
from .config import Settings
from .models import KIND_IMAGE, Chunk
from .parsers.imagefiles import load_image
from .postprocess import clean_answer, parse_bool, parse_ids, parse_reply
from .prompts import (
    IMAGE_FOCUS_PROMPT,
    SYSTEM,
    build_answer_prompt,
    build_bridge_prompt,
    build_verify_prompt,
    format_passage,
)
from .store import Index
from .textutil import contains_value, same_answer, tokenize
from .timing import Deadline

log = logging.getLogger("sourcetrace.answer")

MAX_PROMPT_CHARS = 18000
_IDENT_RE = re.compile(r"\b(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z][A-Za-z0-9_-]{2,}\b")
MAX_BRIDGE_CHECKS = 3
MIN_TIME_FOR_CALL = 4.0


@dataclass
class Passage:
    pid: int
    chunk: Chunk
    text: str

    def render(self) -> str:
        c = self.chunk
        return format_passage(self.pid, c.file, c.loc, self.text, c.superseded)


@dataclass
class Answer:
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def refusal(cls, reason: str, trace: dict[str, Any] | None = None) -> Answer:
        t = dict(trace or {})
        t["refused"] = reason
        return cls("", [], 0.0, t)


class Reasoner:
    def __init__(
        self,
        backend: LLMBackend,
        index: Index,
        settings: Settings,
        embedder: Any = None,
    ) -> None:
        self.backend = backend
        self.index = index
        self.settings = settings
        self.embedder = embedder

    # ----------------------------------------------------------------- model calls
    def _generate(
        self,
        prompt: str,
        deadline: Deadline,
        max_new_tokens: int | None = None,
        image=None,
    ) -> str:
        budget = min(16.0, deadline.remaining() - 1.0)
        if budget < 1.0:
            return ""
        try:
            return self.backend.generate(
                prompt,
                system=SYSTEM,
                image=image,
                max_new_tokens=max_new_tokens or self.settings.max_new_tokens,
                timeout_s=budget,
            )
        except Exception:
            log.exception("model call failed")
            return ""

    def _ask(self, question: str, passages: list[Passage], deadline: Deadline) -> dict:
        shown, used = [], 0
        for p in passages:
            text = p.render()
            if used + len(text) > MAX_PROMPT_CHARS and shown:
                break
            shown.append(text)
            used += len(text)
        raw = self._generate(build_answer_prompt(question, shown), deadline)
        reply = parse_reply(raw)
        log.debug("answer reply: %r -> %r", raw, reply)
        return reply

    # --------------------------------------------------------------------- stages
    def _retrieve(self, query: str, k: int, per_file: int, exclude: set[int]) -> list[Chunk]:
        hits = self.index.search(
            query, k=k, per_file=per_file, embedder=self.embedder, exclude=exclude
        )
        return [chunk for chunk, _ in hits]

    def _vision_focus(
        self,
        question: str,
        passages: list[Passage],
        deadline: Deadline,
        corpus_root: Path | None,
        trace: dict,
    ) -> None:
        if not self.settings.vision_lookup or corpus_root is None:
            return
        done = 0
        for p in passages[:6]:
            c = p.chunk
            if c.kind != KIND_IMAGE or c.loc != "image" or done >= 2:
                continue
            if deadline.remaining() < 14.0:
                break
            try:
                img = load_image(corpus_root / c.file, self.settings.max_image_side)
            except Exception as exc:  # noqa: BLE001
                trace.setdefault("vision_errors", []).append(f"{c.file}: {exc}")
                continue
            done += 1
            text = self._generate(
                IMAGE_FOCUS_PROMPT.format(question=question),
                deadline,
                max_new_tokens=140,
                image=img,
            ).strip()
            trace.setdefault("vision", {})[c.file] = text
            if text and "NOT VISIBLE" not in text.upper():
                p.text = f"{p.text}\n[reading of this picture for the question] {text}"

    def _verify(
        self, question: str, answer: str, evidence: list[Passage], deadline: Deadline
    ) -> bool:
        if not self.settings.verify_answer or deadline.remaining() < MIN_TIME_FOR_CALL + 1:
            return True
        raw = self._generate(
            build_verify_prompt(question, answer, [p.render() for p in evidence]),
            deadline,
            max_new_tokens=20,
        )
        verdict = parse_bool(parse_reply(raw).get("supported"))
        return True if verdict is None else verdict  # unparsable: grounding already passed

    def _bridge(
        self,
        question: str,
        answer: str,
        evidence: list[Passage],
        passages: list[Passage],
        deadline: Deadline,
        trace: dict,
    ) -> list[Passage]:
        """Find a missing chain source: a passage that identifies the entry the question means.

        "The log shows an incident - which release fixed it?" names no ticket. The log names
        it, the bug table holds the answer. If the model cited only the table, the log is
        still necessary. Candidates are passages that share an identifier (ORR-1847, E7731)
        with the answer's passage but whose identifier is NOT in the question, and the model
        is asked one narrow yes/no question about each.
        """
        if deadline.remaining() < MIN_TIME_FOR_CALL + 5:
            return []
        q_low = question.lower()
        q_tokens = set(tokenize(question))
        cited = {p.chunk.file for p in evidence}
        shown_idents = [(p, {m.lower() for m in _IDENT_RE.findall(p.text)}) for p in passages]
        freq: dict[str, int] = {}
        for _, ids in shown_idents:
            for i in ids:
                freq[i] = freq.get(i, 0) + 1
        found: list[Passage] = []
        checks = 0
        for a in [p for p in evidence if contains_value(answer, p.text)]:
            a_ids = {
                i for i in {m.lower() for m in _IDENT_RE.findall(a.text)}
                if i not in q_low and 1 < freq.get(i, 0) <= 3
            }  # fmt: skip
            if not a_ids:
                continue
            for b, b_ids in shown_idents:
                if b.chunk.file in cited or b.chunk.superseded or b in found:
                    continue
                shared = a_ids & b_ids
                if not shared or len(q_tokens & set(tokenize(b.text))) < 2:
                    continue
                if checks >= MAX_BRIDGE_CHECKS or deadline.remaining() < MIN_TIME_FOR_CALL + 3:
                    return found
                checks += 1
                raw = self._generate(
                    build_bridge_prompt(
                        question, answer, ", ".join(sorted(shared)), a.render(), b.render()
                    ),
                    deadline,
                    max_new_tokens=20,
                )
                needed = parse_bool(parse_reply(raw).get("needed"))
                trace.setdefault("bridge", {})[b.chunk.file] = needed
                if needed:
                    found.append(b)
        return found

    def _prune(
        self,
        question: str,
        answer: str,
        evidence: list[Passage],
        protected: set[str],
        deadline: Deadline,
        trace: dict,
    ) -> list[Passage]:
        """Drop cited files whose removal leaves the answer unchanged."""
        if not self.settings.prune_citations:
            return evidence
        files = list(dict.fromkeys(p.chunk.file for p in evidence))
        candidates = [f for f in files if f not in protected]
        # Try the most suspicious files first: superseded, then the lowest-ranked.
        order = {f: i for i, f in enumerate(files)}
        superseded = {p.chunk.file for p in evidence if p.chunk.superseded}
        candidates.sort(key=lambda f: (f not in superseded, -order[f]))
        current = list(evidence)
        for f in candidates:
            if len({p.chunk.file for p in current}) <= 1:
                break
            if deadline.remaining() < MIN_TIME_FOR_CALL + 2:
                trace["prune_stopped"] = "deadline"
                break
            without = [p for p in current if p.chunk.file != f]
            reply = self._ask(question, without, deadline)
            again = clean_answer(reply.get("answer"))
            unchanged = bool(again) and same_answer(again, answer)
            trace.setdefault("prune", {})[f] = "dropped" if unchanged else "kept"
            if unchanged:
                current = without
        return current

    # ----------------------------------------------------------------------- main
    def answer(self, question: str, deadline: Deadline, corpus_root: Path | None = None) -> Answer:
        s = self.settings
        trace: dict[str, Any] = {"question": question}
        if not question.strip():
            return Answer.refusal("empty question", trace)
        if len(self.index) == 0:
            return Answer.refusal("empty index", trace)

        passages = [
            Passage(i + 1, c, c.text)
            for i, c in enumerate(
                self._retrieve(question, s.top_chunks, s.max_chunks_per_file, set())
            )
        ]
        trace["retrieved"] = [f"{p.chunk.file} ({p.chunk.loc})" for p in passages]
        if not passages:
            return Answer.refusal("nothing retrieved", trace)

        self._vision_focus(question, passages, deadline, corpus_root, trace)

        chain_ids: set[int] = set()
        lookup_text = ""
        reply: dict = {}
        answer = ""
        for hop in range(max(1, s.max_hops)):
            reply = self._ask(question, passages, deadline)
            answer = clean_answer(reply.get("answer"))
            trace.setdefault("hops", []).append(
                {"answer": answer, "evidence": parse_ids(reply.get("evidence")),
                 "lookup": reply.get("lookup"), "why": reply.get("why")}
            )  # fmt: skip
            if answer:
                break
            lookup = reply.get("lookup")
            if not lookup or not isinstance(lookup, str) or hop + 1 >= max(1, s.max_hops):
                break
            if deadline.remaining() < MIN_TIME_FOR_CALL + 4:
                trace["hop_stopped"] = "deadline"
                break
            lookup_text = lookup.strip()
            chain_ids = set(parse_ids(reply.get("evidence")))
            shown = {p.chunk.id for p in passages}
            fresh = self._retrieve(lookup_text, 6, 2, shown)
            if not fresh:
                break
            for chunk in fresh:
                passages.append(Passage(len(passages) + 1, chunk, chunk.text))
            trace["lookup"] = lookup_text

        if not answer:
            return Answer.refusal("no answer in the passages", trace)

        by_id = {p.pid: p for p in passages}
        evidence_ids = parse_ids(reply.get("evidence"))
        evidence = [by_id[i] for i in evidence_ids if i in by_id]

        # Chain sources (where the identifier came from) are necessary by definition.
        chain: list[Passage] = [by_id[i] for i in chain_ids if i in by_id]
        if lookup_text and not chain:
            chain = [p for p in passages if contains_value(lookup_text, p.text)][:1]
            chain = [p for p in chain if p not in evidence]
        for p in chain:
            if p not in evidence:
                evidence.append(p)
        protected = {p.chunk.file for p in chain}

        # Grounding: the value must literally occur in at least one cited passage.
        if not any(contains_value(answer, p.text) for p in evidence):
            rescue = [
                p for p in sorted(passages, key=lambda p: (p.chunk.superseded, p.pid))
                if contains_value(answer, p.text)
            ]  # fmt: skip
            if not rescue:
                return Answer.refusal("answer not found verbatim in any passage", trace)
            trace["rescued_evidence"] = rescue[0].pid
            evidence = [rescue[0], *[p for p in evidence if p in chain]]

        if not self._verify(question, answer, evidence, deadline):
            return Answer.refusal("verification rejected the answer", trace)

        bridged = self._bridge(question, answer, evidence, passages, deadline, trace)
        for p in bridged:
            evidence.append(p)
            protected.add(p.chunk.file)

        before = {p.chunk.file for p in evidence}
        evidence = self._prune(question, answer, evidence, protected, deadline, trace)
        files = list(dict.fromkeys(p.chunk.file for p in evidence))
        trace["pruned"] = sorted(before - set(files))

        confidence = 0.62
        confidence += 0.2 if s.verify_answer else 0.0
        confidence += 0.08 if len(files) == 1 else 0.0
        confidence -= 0.1 if any(p.chunk.superseded for p in evidence) else 0.0
        return Answer(answer, files, round(min(0.97, max(0.05, confidence)), 3), trace)
