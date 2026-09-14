"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import ConfigError, load_config
from .exploration.pipeline import ExplorationPipeline
from .models.client import OpenAICompatibleClient
from .qa.agent import QAAgent


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="autotab")
    sub = parser.add_subparsers(dest="command", required=True)
    explore = sub.add_parser("explore")
    explore.add_argument("--config", default="config.yaml")
    explore.add_argument("--workbook", action="append", required=True)
    explore.add_argument("--query", required=True)
    qa = sub.add_parser("qa")
    qa.add_argument("--config", default="config.yaml")
    qa.add_argument("--workbook", required=True)
    qa.add_argument("--query", required=True)
    qa.add_argument("--evidence")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "explore":
            path = ExplorationPipeline(config).run(args.workbook, args.query)
            print(path)
            return 0
        model_config = config["models"]["llm"]
        if not model_config.get("base_url"):
            raise ConfigError("models.llm.base_url is required for QA")
        model = OpenAICompatibleClient(
            model_config["base_url"],
            model_config["model"],
            config["runtime"]["request_timeout_seconds"],
            int(model_config.get("max_tokens", 4000)),
            model_config.get("extra_body"),
        )
        evidence_path = Path(args.evidence) if args.evidence else None
        if config["qa"]["exploration_enabled"]:
            evidence_path = ExplorationPipeline(config).run([args.workbook], args.query)
            run_root = evidence_path.parent.parent
        else:
            run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
            run_root = Path(config["runtime"]["artifact_root"]).resolve() / run_id
        evidence = evidence_path.read_text(encoding="utf-8") if evidence_path else None
        qa_artifacts = run_root / "qa"
        result = QAAgent(model, config).run(
            args.workbook,
            args.query,
            evidence,
            qa_artifacts,
        )
        print(result.answer)
        print(f"Artifacts: {run_root}", file=sys.stderr)
        if not result.success:
            return 1
        return 0
    except (ConfigError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
