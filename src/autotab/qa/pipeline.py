"""Run SheetBrain QA directly against a workbook."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .sheetbrain.config.settings import Config as SheetBrainConfig
from .sheetbrain.core.agent import SheetBrain


class QAPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self, workbook: str, query: str) -> Path:
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
            enable_validation=True,
            include_diagnostics=False,
            max_retries=int(self.config["runtime"]["max_retries"]),
            timeout=int(self.config["runtime"]["request_timeout_seconds"]),
        )
        agent = SheetBrain(
            excel_path=workbook,
            config=sheetbrain_config,
        )
        result = agent.run(query, enable_validation=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
        output_dir = Path(self.config["runtime"]["artifact_root"]) / f"{timestamp}-{os.getpid()}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "qa-result.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return output_path
