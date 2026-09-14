"""Keyword extraction with deterministic fallback."""

from __future__ import annotations

import json
import re

from .prompts import keyword_extraction_prompt


def normalize_phrase(value: str) -> str:
    """Normalize whitespace and case while preserving phrase wording."""
    return re.sub(r"\s+", " ", value.strip()).lower()


class KeywordExtractor:
    """Extract a plain list of worksheet-search keywords."""

    def __init__(self, model: object | None = None) -> None:
        self.model = model

    def extract(self, query: str) -> list[str]:
        if self.model is not None and hasattr(self.model, "extract_keywords"):
            try:
                result = self.model.extract_keywords(query)
            except (AttributeError, OSError, TypeError, ValueError):
                result = None
            if result:
                return _deduplicate([str(keyword) for keyword in result])
        if self.model is not None and hasattr(self.model, "complete"):
            prompt = keyword_extraction_prompt(query)
            try:
                raw = str(self.model.complete(prompt))
                start, end = raw.find("["), raw.rfind("]")
                payload = json.loads(raw[start : end + 1])
                result = [normalize_phrase(str(item)) for item in payload if item]
                if result:
                    return _deduplicate(result)
            except (AttributeError, OSError, TypeError, ValueError):
                pass  # Model failures intentionally fall back to deterministic extraction.
        seen: set[str] = set()
        out: list[str] = []
        for match in re.finditer(r"[A-Za-z][\w$%/-]*", query):
            phrase = normalize_phrase(match.group())
            if phrase in seen or phrase in {"find", "show", "return", "every", "the", "by"}:
                continue
            seen.add(phrase)
            out.append(phrase)
        return _deduplicate(out or [normalize_phrase(query)])[:8]


def _deduplicate(keywords: list[str]) -> list[str]:
    """Remove repeated keywords while preserving query/model order."""
    seen: set[str] = set()
    result: list[str] = []
    for keyword in keywords:
        phrase = normalize_phrase(keyword)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        result.append(phrase)
    return result
