"""Independent evidence relevance filtering."""

from __future__ import annotations

from .evidence import Evidence


class EvidenceFilter:
    def filter(self, evidence: Evidence, query: str, uncertain: str = "drop") -> Evidence:
        relevant = any(
            token in (evidence.keyword + " " + evidence.description).lower()
            for token in query.lower().split()
        )
        return (
            evidence
            if relevant
            else Evidence(
                evidence.keyword,
                evidence.source_range,
                evidence.description,
                "drop",
                "No query term matched",
            )
        )
