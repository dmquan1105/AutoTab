"""Prompts used by the exploration models."""

from __future__ import annotations

from typing import Any


def keyword_extraction_prompt(query: str) -> str:
    """Build the structured keyword extraction prompt."""
    return f"""Extract keywords that can be searched in Excel worksheets to find the evidence
needed to answer this query. Select the important worksheet terms: entity names, column or row
headers, metric names, dimensions, labels, and distinctive values. Exclude query instructions, filler words, operations (for example, 'find' or 'calculate'),
and generic terms that are unlikely to occur in a worksheet.

DO NOT invent new keywords that do not exist in the query. Return JSON only as an array of
keyword strings.

Query: {query}"""


def viewport_description_prompt(
    keyword: str,
    source_range: str,
    cells: list[dict[str, Any]],
    *,
    query: str | None = None,
) -> str:
    """Build the VLM prompt for a grounded viewport description."""
    query_context = ""
    if query is not None:
        query_context = f"""

User query: {query}

Include concrete visible labels and values that would help a later QA step answer a question using this evidence."""

    return f"""You are examining a rendered viewport from an Excel worksheet. Return JSON only
with one string field named "description". Describe only what is visible in the worksheet viewport
and do not add outside knowledge or infer content that is not shown.

The target keyword is "{keyword}" and its source cell is {source_range}.

Explain what the keyword means specifically within this worksheet table or report. Identify whether it
is a title, row label, column header, grouped or merged header, category, measure, value, note, or another visible worksheet element.
Describe how its data extends across rows or columns and how it relates to visible parent or
child labels. Pay attention to hierarchical headers with indentation levels, merged cells,...
{query_context}

If the keyword or its relationship with others is not visible, say that it is
not visible instead of guessing."""
