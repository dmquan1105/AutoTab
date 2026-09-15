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


def keyword_summary_prompt(keyword: str, query: str, descriptions: list[str]) -> str:
    """Build the prompt for consolidating one keyword's evidence."""
    evidence = "\n".join(f"- {description}" for description in descriptions)
    return f"""Consolidate the worksheet evidence for the exact keyword "{keyword}" into one brief
description that helps answer the user query. Use the query only to judge relevance, not as evidence.

Do not rename the keyword or introduce other keywords, headings, or worksheet ranges. Do not perform
calculations. Do not add facts or values that are absent from the supplied descriptions.

Return only the consolidated description. If this keyword contributes nothing toward answering the
query, return exactly DROP.

User query: {query}

Evidence descriptions:
{evidence}"""


def viewport_description_prompt(
    keyword: str, source_range: str, cells: list[dict[str, Any]]
) -> str:
    """Build the VLM prompt for a grounded viewport description."""
    return (
        "You are examining a rendered viewport from an Excel worksheet. Return JSON only with one string field named "
        '"description". Describe only what is visible in the worksheet viewport and do not '
        "add outside knowledge or infer content that is not shown. "
        f'The target keyword is "{keyword}" and its source cell is {source_range}. '
        "Explain what the keyword means specifically within this worksheet table or report. "
        "Identify whether it is a title, row label, column header, grouped or merged header, "
        "category, measure, value, note, or another visible worksheet element. Describe how "
        "its data extends across rows or columns and how it relates to visible parent or child "
        "labels, indentation levels, adjacent headers, merged header groups, units, periods, "
        "categories, and values. Include concrete visible labels and values that would help a "
        "later QA step answer a question using this evidence. If a relationship, header, unit, "
        "or meaning is not visible, say that it is not visible instead of guessing. Do not "
        "describe colors, layout positions, icons, people, scenery, trends, correlations, or "
        "definitions unless the worksheet itself visibly supports them. The keyword may span "
        "multiple cells."
    )
