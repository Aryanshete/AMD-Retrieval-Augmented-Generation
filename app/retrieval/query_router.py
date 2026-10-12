"""Query classification and intent routing engine.

Classifies queries before retrieval into operational categories:
- SPECIFICATION: hardware datasheets, voltage, temperature limits, frequencies
- TICKET_BUG: defect trackers, fixed versions, ticket numbers
- IMAGE_DIAGRAM: pinouts, connector diagrams, physical asset labels
- SOURCE_CODE: python services, timeouts, class implementations, constants
- LOG_EVENT: runtime logs, error codes, throttle events
- MULTI_HOP: queries chaining incidents/logs across databases/releases
- GENERAL: general cross-document natural language queries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import List, Optional, Set


class QueryCategory(str, Enum):
    SPECIFICATION = "specification"
    TICKET_BUG = "ticket_bug"
    IMAGE_DIAGRAM = "image_diagram"
    SOURCE_CODE = "source_code"
    LOG_EVENT = "log_event"
    MULTI_HOP = "multi_hop"
    GENERAL = "general"


@dataclass
class RoutedQuery:
    original_query: str
    category: QueryCategory
    identifiers: List[str] = field(default_factory=list)
    prioritized_file_types: Set[str] = field(default_factory=set)
    is_multi_hop: bool = False
    is_historical_requested: bool = False
    clean_query: str = ""


class QueryRouter:
    """Classifies user queries to steer retrieval toward optimal document representations."""

    IDENTIFIER_REGEX = re.compile(
        r"\b([A-Z]{2,4}-[A-Z0-9-]+|E\d{4,5}|[A-Z]\d{1,3}|REV-[A-Z0-9]+|\d+\.\d+\.\d+|THERM_ALERT#?)\b",
        re.IGNORECASE,
    )

    def route(self, query: str) -> RoutedQuery:
        """Analyze query and determine retrieval intent, routing hints, and extracted entities."""
        q_raw = query.strip()
        q_lower = q_raw.lower()

        # Extract identifiers
        raw_ids = self.IDENTIFIER_REGEX.findall(q_raw)
        # Deduplicate while preserving order
        seen = set()
        identifiers = []
        for i in raw_ids:
            norm = i.upper()
            if norm not in seen:
                seen.add(norm)
                identifiers.append(norm)

        is_historical = any(w in q_lower for w in ["withdrawn", "obsolete", "preliminary", "revision 1", "r1"])

        # Detect Multi-Hop queries: Question bridges a log incident / event to a firmware release / defect
        # e.g., "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?"
        if (
            ("log" in q_lower or "incident" in q_lower or "throttle" in q_lower or "crash" in q_lower)
            and ("firmware" in q_lower or "defect" in q_lower or "fixed" in q_lower or "release" in q_lower)
            and not any("orr-" in i.lower() for i in identifiers)  # without direct ticket ID, requires hopping
        ) or ("fixed the incident" in q_lower or "underlying defect" in q_lower):
            return RoutedQuery(
                original_query=q_raw,
                category=QueryCategory.MULTI_HOP,
                identifiers=identifiers,
                prioritized_file_types={"log", "csv"},
                is_multi_hop=True,
                is_historical_requested=is_historical,
                clean_query=q_raw,
            )

        # Detect Image / Diagram questions
        if (
            "pin" in q_lower
            or "carries" in q_lower
            or "backplane" in q_lower
            or "pinout" in q_lower
            or "connector" in q_lower
            or "asset label" in q_lower
            or "label" in q_lower
            or "sticker" in q_lower
            or "photo" in q_lower
            or "image" in q_lower
            or "diagram" in q_lower
            or "printed on" in q_lower
            or "printed" in q_lower
            or "stamped" in q_lower
            or "board revision" in q_lower
            or "barcode" in q_lower
            or "serial number" in q_lower
            or "chassis" in q_lower
            or "therm_alert" in q_lower
            or ("revision" in q_lower and ("board" in q_lower or "label" in q_lower or "hardware" in q_lower))
        ):
            return RoutedQuery(
                original_query=q_raw,
                category=QueryCategory.IMAGE_DIAGRAM,
                identifiers=identifiers,
                prioritized_file_types={"image"},
                is_multi_hop=False,
                is_historical_requested=is_historical,
                clean_query=q_raw,
            )

        # Detect Source code questions
        if (
            "timeout" in q_lower
            or "function" in q_lower
            or "class" in q_lower
            or "constant" in q_lower
            or "default value" in q_lower
            or "service" in q_lower
            or ".py" in q_lower
            or "ingest" in q_lower
        ):
            return RoutedQuery(
                original_query=q_raw,
                category=QueryCategory.SOURCE_CODE,
                identifiers=identifiers,
                prioritized_file_types={"source", "text"},
                is_multi_hop=False,
                is_historical_requested=is_historical,
                clean_query=q_raw,
            )

        # Detect Log questions
        if (
            "error code" in q_lower
            or "logged" in q_lower
            or "traceback" in q_lower
            or "e7" in q_lower
            or ("log" in q_lower and "event" in q_lower)
        ):
            return RoutedQuery(
                original_query=q_raw,
                category=QueryCategory.LOG_EVENT,
                identifiers=identifiers,
                prioritized_file_types={"log", "text"},
                is_multi_hop=False,
                is_historical_requested=is_historical,
                clean_query=q_raw,
            )

        # Detect Ticket / Bug questions
        if (
            "ticket" in q_lower
            or "bug" in q_lower
            or "fixed in" in q_lower
            or "firmware version" in q_lower
            or any("orr-" in i.lower() for i in identifiers)
        ):
            return RoutedQuery(
                original_query=q_raw,
                category=QueryCategory.TICKET_BUG,
                identifiers=identifiers,
                prioritized_file_types={"csv", "xlsx"},
                is_multi_hop=False,
                is_historical_requested=is_historical,
                clean_query=q_raw,
            )

        # Detect Hardware Specification / Datasheet questions
        if (
            "junction temperature" in q_lower
            or "datasheet" in q_lower
            or "maximum" in q_lower
            or "voltage" in q_lower
            or "customer sampling" in q_lower
            or "quarter" in q_lower
            or "tq-" in q_lower
            or "part number" in q_lower
            or "fan assembly" in q_lower
            or "revision" in q_lower
        ):
            return RoutedQuery(
                original_query=q_raw,
                category=QueryCategory.SPECIFICATION,
                identifiers=identifiers,
                prioritized_file_types={"pdf", "docx", "xlsx", "image"},
                is_multi_hop=False,
                is_historical_requested=is_historical,
                clean_query=q_raw,
            )


        return RoutedQuery(
            original_query=q_raw,
            category=QueryCategory.GENERAL,
            identifiers=identifiers,
            prioritized_file_types=set(),
            is_multi_hop=False,
            is_historical_requested=is_historical,
            clean_query=q_raw,
        )
