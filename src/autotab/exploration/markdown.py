"""Deterministic Markdown output."""

from __future__ import annotations

from .evidence import Evidence
from .summary import KeywordSummary


class MarkdownWriter:
    def write(self, run_id: str, evidence: list[Evidence]) -> str:
        lines = [
            "# Aggregation",
            "",
            f"Run ID: {run_id}",
            "",
        ]
        for item in evidence:
            if item.status != "keep":
                continue
            lines += [
                f"## {item.keyword}",
                "",
                f"- range: `{item.source_range}`",
                f"- description: {item.description}",
            ]
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def write_summary(self, summaries: list[KeywordSummary]) -> str:
        """Write consolidated keyword summaries as exploration Markdown."""
        lines = ["# Exploration", ""]
        for summary in summaries:
            ranges = ", ".join(f"`{source_range}`" for source_range in summary.ranges)
            lines += [
                f"## {summary.keyword}",
                "",
                f"- ranges: {ranges}",
                f"- description: {summary.description}",
                "",
            ]
        return "\n".join(lines).rstrip() + "\n"
