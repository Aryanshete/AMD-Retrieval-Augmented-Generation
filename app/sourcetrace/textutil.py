"""Text helpers shared by indexing, retrieval and answer checking."""

from __future__ import annotations

import re

_DROP = str.maketrans("", "", "-.·_")

_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "by",
        "with",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "did",
        "has",
        "have",
        "had",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "from",
        "as",
        "into",
        "than",
        "then",
        "there",
        "their",
        "about",
        "per",
        "can",
        "will",
        "would",
        "should",
        "shall",
        "any",
        "all",
        "not",
        "no",
    ]
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.:/][a-z0-9]+)*")
_SPLIT_RE = re.compile(r"[-_.:/]")
_WORD_RE = re.compile(r"[0-9A-Za-z]+")


def normalize(text: object) -> str:
    """The grader's normalisation: upper-case, no whitespace, no ``- . · _``."""
    return "".join(str(text).upper().split()).translate(_DROP)


def same_answer(a: object, b: object) -> bool:
    return normalize(a) == normalize(b)


def _fold(token: str) -> str:
    """Very light plural folding so 'temperatures' matches 'temperature'."""
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-cased tokens. Compound identifiers also contribute their parts.

    ``ORR-1847`` yields ``orr-1847``, ``orr`` and ``1847`` so both an exact identifier
    lookup and a looser keyword match work.
    """
    out: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        tok = match.group(0)
        parts = _SPLIT_RE.split(tok)
        if len(parts) > 1:
            out.append(tok)
        for part in parts:
            if part and part not in _STOP:
                out.append(_fold(part))
    return out


def query_identifiers(query: str) -> list[str]:
    """Identifier-looking tokens in a query (``ORR-1847``, ``E7731``, ``TQ-40``)."""
    found: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9_./#-]*[A-Za-z0-9#]", query):
        has_digit = any(c.isdigit() for c in raw)
        has_alpha = any(c.isalpha() for c in raw)
        if has_digit and (has_alpha or "-" in raw or "." in raw):
            found.append(raw.lower().strip(".,;:"))
    return found


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def contains_value(answer: str, text: str) -> bool:
    """True if ``answer`` occurs in ``text`` as a whole-token sequence (case-insensitive)."""
    needle = words(answer)
    if not needle:
        return False
    hay = words(text)
    n = len(needle)
    return any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))
