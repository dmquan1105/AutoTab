"""VLM evidence extraction and grounding."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .prompts import viewport_description_prompt


@dataclass(frozen=True)
class Evidence:
    keyword: str
    source_range: str
    description: str
    status: str = "keep"
    reason: str = ""

    def to_dict(self):
        return asdict(self)


class EvidenceExtractor:
    def __init__(self, model: object | None = None) -> None:
        self.model = model
        self.last_response = ""

    def extract(
        self,
        keyword: str,
        source_range: str,
        cells: list[dict[str, Any]],
        image_path: str | Path | None = None,
    ) -> Evidence:
        """Describe a keyword using its rendered worksheet viewport."""
        if self.model is None or not hasattr(self.model, "complete"):
            raise RuntimeError("A VLM is required to generate viewport descriptions")
        prompt = viewport_description_prompt(keyword, source_range, cells)
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                retry_hint = (
                    "\nReturn valid JSON only; do not include any additional text."
                    if attempt
                    else ""
                )
                raw = str(self.model.complete(prompt + retry_hint, image_path))
                self.last_response = raw
                data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
                return Evidence(keyword, source_range, str(data["description"]))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError("VLM returned an invalid viewport description") from last_error
