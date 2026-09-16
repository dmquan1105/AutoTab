"""Command-line interface."""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config
from .exploration.pipeline import ExplorationPipeline
from .qa.pipeline import QAPipeline


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
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "explore":
            path = ExplorationPipeline(config).run(args.workbook, args.query)
        else:
            path = QAPipeline(config).run(args.workbook, args.query)
        print(path)
        return 0
    except (ConfigError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
