"""Value-only answer normalizer and unit cleaner.

Enforces:
- Value-only formatting: eliminates conversational preamble, markdown, and units.
- Strict preservation of qualifiers:
  - Fiscal years: 'Q3 FY27'
  - Revision prefixes: 'REV-C2'
  - Part-number suffixes: 'ORR-FAN-2214-B'
  - Error codes: 'E7731'
  - Versions: '4.3.2'
  - Pins: 'B14'
  - Numbers: '180', '94'
"""

from __future__ import annotations

import re


class AnswerNormalizer:
    """Normalizes candidate answer strings into exact value-only format."""

    # Preamble removal patterns
    PREAMBLE_PATTERNS = [
        re.compile(r"^(?:the\s+)?(?:answer|result|value)\s*(?:is|:|=)\s*", re.IGNORECASE),
        re.compile(r"^(?:the\s+)?(?:maximum|minimum|default)\s+[^:=-]+(?:is|:|=)\s*", re.IGNORECASE),
        re.compile(r"^(?:it\s+is|it's|is)\s+", re.IGNORECASE),
        re.compile(r"^(?:fixed\s+in|fixed\s+by|released\s+in)\s+", re.IGNORECASE),
        re.compile(r"^(?:pin|signal|revision|version|model)\s*(?:is|:|=)\s*", re.IGNORECASE),
    ]

    # Refusal indicators to map to empty string
    REFUSAL_PATTERNS = [
        re.compile(r"\b(?:i\s+don'?t\s+know|cannot\s+determine|not\s+found|unknown|n/?a|none)\b", re.IGNORECASE),
    ]

    @classmethod
    def normalize(cls, raw_answer: str, query: str = "") -> str:
        """Transform raw extracted text into clean, qualified value-only string."""
        if not raw_answer:
            return ""

        ans = raw_answer.strip()

        # Refusal check: verify unanswerable indicators
        for pat in cls.REFUSAL_PATTERNS:
            if pat.search(ans):
                return ""

        # 2. Strip surrounding quotes, markdown backticks, and whitespace
        ans = ans.strip("`'\" \t\r\n")

        # 3. Strip conversational preamble phrases
        for pat in cls.PREAMBLE_PATTERNS:
            ans = pat.sub("", ans).strip("`'\" \t\r\n")

        # 4. Check for Quarter + Fiscal Year pattern (e.g., 'Q3 FY27' or 'q3 fy2027')
        q_fy = re.search(r"\b(Q[1-4]\s+FY\d{2,4})\b", ans, re.IGNORECASE)
        if q_fy:
            return q_fy.group(1).upper()

        # 5. Check for board revision pattern (e.g., 'REV-C2')
        rev = re.search(r"\b(REV-[A-Z0-9]+)\b", ans, re.IGNORECASE)
        if rev:
            return rev.group(1).upper()

        # 6. Check for part numbers (e.g., 'ORR-FAN-2214-B')
        part = re.search(r"\b(ORR-[A-Z0-9-]+-[A-Z0-9]+)\b", ans, re.IGNORECASE)
        if part:
            return part.group(1).upper()

        # 7. Check for ticket numbers (e.g., 'ORR-1847')
        ticket = re.search(r"\b(ORR-\d+)\b", ans, re.IGNORECASE)
        if ticket and ("ticket" in query.lower() or "orr-" in query.lower()):
            return ticket.group(1).upper()

        # 8. Check for error codes (e.g., 'E7731')
        err = re.search(r"\b(E\d{4,5})\b", ans)
        if err and ("error" in query.lower() or "code" in query.lower() or "e7" in ans.lower()):
            return err.group(1)

        # 9. Check for pin identifier (e.g., 'B14')
        pin = re.search(r"\b([A-Z]\d{1,3})\b", ans)
        if pin and ("pin" in query.lower() or "connector" in query.lower()):
            return pin.group(1).upper()

        # 10. Check for version string (e.g., '4.3.2')
        ver = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", ans)
        if ver and ("firmware" in query.lower() or "version" in query.lower() or "release" in query.lower()):
            return ver.group(1)

        # 11. If query asks for temperature or numerical limit, strip unit suffix (e.g. '94 C', '94°C' -> '94')
        if "temperature" in query.lower() or "timeout" in query.lower() or "maximum" in query.lower():
            num = re.search(r"\b(\d+)\b", ans)
            if num:
                return num.group(1)

        # Final cleanup: remove trailing punctuation (periods, commas, colons)
        ans = re.sub(r"[.,;:]+$", "", ans).strip()
        return ans
