"""Run AutoTab exploration followed by SheetBrain QA."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..exploration.pipeline import ExplorationPipeline
from .sheetbrain.config.settings import Config as SheetBrainConfig
from .sheetbrain.core.agent import SheetBrain


class QAPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self, workbook: str, query: str) -> Path:
        exploration_path = ExplorationPipeline(self.config).run([workbook], query)
        exploration_text = exploration_path.read_text(encoding="utf-8")
        if not exploration_text.strip():
            raise ValueError("AutoTab exploration output is empty")

        model = self.config["models"]["llm"]
        base_url = model.get("base_url")
        if not base_url:
            raise ValueError("models.llm.base_url is required for QA")
        sheetbrain_config = SheetBrainConfig(
            api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
            base_url=str(base_url),
            deployment=str(model["model"]),
            extra_body=dict(model.get("extra_body") or {}),
            max_turns=3,
            total_token_budget=5000,
            enable_validation=True,
            enable_understanding=True,
            include_diagnostics=False,
            max_retries=int(self.config["runtime"]["max_retries"]),
            timeout=int(self.config["runtime"]["request_timeout_seconds"]),
        )
        agent = SheetBrain(
            excel_path=workbook,
            config=sheetbrain_config,
            total_token_budget=sheetbrain_config.total_token_budget,
            exploration_path=str(exploration_path),
        )
        result = agent.run(query, enable_validation=True, enable_understanding=True)
        output_path = exploration_path.parent / "qa-result.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return output_path
