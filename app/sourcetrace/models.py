"""Data model shared by parsers, the index and the reasoner."""

from __future__ import annotations

from dataclasses import dataclass, field

KIND_TEXT = "text"
KIND_TABLE = "table"
KIND_CODE = "code"
KIND_IMAGE = "image"


@dataclass
class Chunk:
    """A retrievable passage of one file."""

    file: str  # corpus-relative path with forward slashes: this is what gets cited
    loc: str  # human-readable position: "p2", "rows 2-9", "lines 1-30", "image"
    text: str
    kind: str = KIND_TEXT
    superseded: bool = False
    id: int = -1

    def to_json(self) -> dict:
        return {
            "file": self.file,
            "loc": self.loc,
            "text": self.text,
            "kind": self.kind,
            "superseded": self.superseded,
        }

    @classmethod
    def from_json(cls, data: dict, chunk_id: int) -> Chunk:
        return cls(
            file=str(data["file"]),
            loc=str(data.get("loc", "")),
            text=str(data.get("text", "")),
            kind=str(data.get("kind", KIND_TEXT)),
            superseded=bool(data.get("superseded", False)),
            id=chunk_id,
        )


@dataclass
class ParsedDoc:
    """The result of parsing one file."""

    file: str
    chunks: list[Chunk] = field(default_factory=list)
    superseded: bool = False
    note: str = ""


class Skip(Exception):
    """Raised by a parser for a file that should be left out (encrypted, empty, ...)."""
