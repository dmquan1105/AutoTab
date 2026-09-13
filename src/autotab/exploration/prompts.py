"""Prompts used by the exploration models."""

from __future__ import annotations

from typing import Any


def keyword_extraction_prompt(query: str) -> str:
    """Build the structured keyword extraction prompt."""
    return (
        "Extract keywords that can be searched in Excel worksheets to find the evidence "
        "needed to answer this query. Select the important worksheet terms: entity names, "
        "column or row headers, metric names, dimensions, labels, and distinctive values. "
        "Use the worksheet wording when possible. Exclude query "
        "instructions, filler words, operations (for example, 'find' or 'calculate'), and "
        "generic terms that are unlikely to occur in a worksheet. Return JSON only as an "
        "array of keyword strings. "
        f"Query: {query}"
    )


def viewport_description_prompt(
    keyword: str, source_range: str, cells: list[dict[str, Any]]
) -> str:
    """Build the VLM prompt for a grounded viewport description."""
    return (
        "Return JSON only with one description field. Describe the keyword in the context "
        "of the viewport, including relations to other data and notable details. "
        f"Keyword: {keyword}."
    )
