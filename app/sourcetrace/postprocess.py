"""Turning raw model output into a clean, gradeable value."""

from __future__ import annotations

import json
import re

_REFUSALS = {
    "", "none", "n/a", "na", "null", "nil", "unknown", "not found", "not available",
    "not specified", "no answer", "unanswerable", "-", "--", "?", "empty",
}  # fmt: skip
_REFUSAL_RE = re.compile(
    r"\b(not\s+(?:stated|specified|mentioned|provided|found|available|present|given|listed)"
    r"|cannot\s+be\s+(?:determined|found|answered)|can't\s+be\s+(?:determined|found)"
    r"|insufficient\s+information|no\s+(?:information|data|answer|relevant)"
    r"|i\s+(?:could\s*not|cannot|can't|couldn't|don't|do\s+not)"
    r"|(?:passages?|context|corpus|documents?)\s+(?:do(?:es)?\s+not|don't))\b",
    re.IGNORECASE,
)
_UNITS = (
    r"°\s*[CFK]?|º\s*[CFK]?|degrees?(?:\s+(?:celsius|fahrenheit|[CF]))?|celsius|fahrenheit|"
    r"[CFK]|W|kW|mW|V|mV|A|mA|GiB|MiB|KiB|TiB|GB|MB|KB|TB|TB/s|GB/s|MB/s|MHz|GHz|kHz|Hz|"
    r"ms|us|µs|ns|s|sec|secs|seconds?|minutes?|mins?|hours?|hrs?|days?|%|watts?|volts?|"
    r"USD|EUR|GBP|dollars?"
)
_NUM_UNIT_RE = re.compile(rf"([-+]?\d[\d,]*\.?\d*)\s*(?:{_UNITS})\.?", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"^[\$€£]\s*(\d[\d,]*\.?\d*)$")
_LEAD_RE = re.compile(r"^(?:final\s+)?answer\s*[:=-]\s*", re.IGNORECASE)
_QUALIFIERS_RE = re.compile(
    r"^(?:pin|part|ticket|version|revision|quarter)(?:\s*[:=]|\s+)\s*",
    re.IGNORECASE,
)


def clean_answer(raw: object) -> str:
    """Normalise a model answer: strip decoration, drop units, map refusals to ''."""
    if raw is None:
        return ""
    text = str(raw).strip()
    text = re.sub(r"^```(?:json|text)?|```$", "", text).strip()
    text = _LEAD_RE.sub("", text)
    text = _QUALIFIERS_RE.sub("", text)
    text = " ".join(text.split())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    text = text.rstrip(".;,").strip() if not re.search(r"\d\.$", text) else text.rstrip(".")
    if text.lower() in _REFUSALS or _REFUSAL_RE.search(text):
        return ""
    match = _NUM_UNIT_RE.fullmatch(text)
    if match:
        text = match.group(1)
    match = _CURRENCY_RE.match(text)
    if match:
        text = match.group(1)
    return text


def parse_reply(text: str) -> dict:
    """Extract the JSON object from a model reply; tolerate fences and chatter."""
    if not text:
        return {}
    start, end = text.find("{"), text.rfind("}")
    candidates = [text[start : end + 1]] if 0 <= start < end else []
    for blob in candidates:
        for attempt in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
            try:
                data = json.loads(attempt)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    answer = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if answer:
        out: dict = {"answer": answer.group(1)}
        ev = re.search(r'"evidence"\s*:\s*\[([^\]]*)\]', text)
        if ev:
            out["evidence"] = re.findall(r"\d+", ev.group(1))
        return out
    return {}


def parse_ids(value: object) -> list[int]:
    """Passage numbers from ``[1, 2]``, ``["1"]``, ``"[1],[2]"`` or ``3``."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    text = " ".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
    seen: list[int] = []
    for num in re.findall(r"\d+", text):
        n = int(num)
        if n not in seen:
            seen.append(n)
    return seen


def parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes"}:
            return True
        if low in {"false", "no"}:
            return False
    return None
