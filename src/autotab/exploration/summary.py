"""Keyword evidence summarization."""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import Evidence
from .prompts import keyword_summary_prompt

DROP_RESPONSE = "DROP"


@dataclass(frozen=True)
class KeywordSummary:
    """One consolidated description and its supporting ranges."""

    keyword: str
    ranges: tuple[str, ...]
    description: str


class KeywordSummarizer:
    """Consolidate aggregated evidence into query-relevant keyword summaries."""

    def __init__(self, model: object | None = None) -> None:
        self.model = model

    def summarize(self, query: str, evidence: list[Evidence]) -> list[KeywordSummary]:
        """Consolidate retained evidence without allowing keyword or range changes."""
        if self.model is None or not hasattr(self.model, "complete"):
            raise RuntimeError("An LLM is required to summarize exploration keywords")

        grouped: dict[str, list[Evidence]] = {}
        for item in evidence:
            if item.status == "keep":
                grouped.setdefault(item.keyword, []).append(item)

        summaries = []
        for keyword, items in grouped.items():
            prompt = keyword_summary_prompt(
                keyword,
                query,
                [item.description for item in items],
            )
            description = str(self.model.complete(prompt)).strip()
            if not description:
                raise RuntimeError("LLM returned an empty keyword summary")
            if description == DROP_RESPONSE:
                continue
            ranges = tuple(dict.fromkeys(item.source_range for item in items))
            summaries.append(KeywordSummary(keyword, ranges, description))
        return summaries
