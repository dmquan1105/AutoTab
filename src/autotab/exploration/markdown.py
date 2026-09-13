"""Deterministic Markdown output."""

from __future__ import annotations

from .evidence import Evidence


class MarkdownWriter:
    def write(self, run_id: str, evidence: list[Evidence]) -> str:
        lines = [
            "# Exploration",
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
