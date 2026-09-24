"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .config import ConfigError, load_config
from .exploration.pipeline import ExplorationPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autotab")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("explore", "qa"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="config.yaml")
        command.add_argument("--workbook", action="append", required=True)
        command.add_argument("--query", required=True)
    qa = sub.choices["qa"]
    source = qa.add_mutually_exclusive_group()
    source.add_argument(
        "--exploration",
        help="reuse an exploration run directory (the one holding evidence.filtered.json)",
    )
    source.add_argument(
        "--no-exploration", action="store_true", help="answer from the workbook alone"
    )
    return parser


def _qa(args: argparse.Namespace, config: dict) -> int:
    # Imported here so `autotab explore` does not pay for the QA stack.
    from .qa import agent

    if args.no_exploration:
        config["qa"]["exploration_enabled"] = False
    elif args.exploration:
        config["qa"]["exploration_enabled"] = True
        config["qa"]["exploration_path"] = args.exploration
    outcome = agent.run(args.query, args.workbook, config)
    print(f"{outcome.status.value} after {outcome.turns} turns: {outcome.reason}")
    if outcome.answer_text is not None:
        print(f"\n{outcome.answer_text}\n")
    print(f"artifacts: {outcome.run_dir}")
    return 0 if outcome.status.value == "PASS" else 1


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "qa":
            return _qa(args, config)
        path = ExplorationPipeline(config).run(args.workbook, args.query)
        print(path)
        return 0
    except (ConfigError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
