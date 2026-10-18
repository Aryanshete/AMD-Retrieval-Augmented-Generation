"""Shared test helpers."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from _path import APP  # noqa: F401  (side effect: sys.path)

from sourcetrace.config import Settings
from sourcetrace.models import Chunk, ParsedDoc
from sourcetrace.store import Index

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_CORPUS = FIXTURES / "mc3-corpus"
SAMPLE_QUESTIONS = json.loads((FIXTURES / "sample-questions.json").read_text())["queries"]

_PASSAGE_RE = re.compile(r"^\[(\d+)\] file: (\S+) \(", re.MULTILINE)


def make_corpus_copy() -> Path:
    """A writable copy of the sample corpus including the empty directory."""
    tmp = Path(tempfile.mkdtemp(prefix="st-corpus-"))
    root = tmp / "corpus"
    shutil.copytree(SAMPLE_CORPUS, root)
    (root / "archive").mkdir(exist_ok=True)
    return root


def passages_in(prompt: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for pid, file in _PASSAGE_RE.findall(prompt):
        out.setdefault(file, []).append(int(pid))
    return out


def reply(answer: str, evidence: list[int] | None = None, lookup: str | None = None) -> str:
    return json.dumps(
        {"why": "test", "answer": answer, "evidence": evidence or [], "lookup": lookup}
    )


def is_verify(prompt: str) -> bool:
    return "Proposed answer:" in prompt


def verdict(ok: bool) -> str:
    return json.dumps({"supported": ok})


def make_index(files: dict[str, str], **flags) -> Index:
    """Index built from ``{relative path: text}`` (one chunk per file)."""
    docs = []
    for rel, text in files.items():
        doc = ParsedDoc(rel)
        doc.chunks.append(Chunk(rel, "lines 1-1", text, "text", flags.get("superseded", False)))
        docs.append(doc)
    return Index.build(docs, [], "/corpus")


def test_settings(**overrides) -> Settings:
    base = Settings(
        backend="mock",
        mode="standalone",
        index_dir=tempfile.mkdtemp(prefix="st-index-"),
        output_dir=tempfile.mkdtemp(prefix="st-out-"),
        socket_path=str(Path(tempfile.mkdtemp(prefix="st-sock-")) / "s.sock"),
    )
    import dataclasses

    return dataclasses.replace(base, **overrides)
