"""Command-line interface."""

from __future__ import annotations

import argparse

from .config import ConfigError, load_config
from .exploration.pipeline import ExplorationPipeline


def main() -> int:
    parser = argparse.ArgumentParser(prog="autotab")
    sub = parser.add_subparsers(dest="command", required=True)
    explore = sub.add_parser("explore")
    explore.add_argument("--config", default="config.yaml")
    explore.add_argument("--workbook", action="append", required=True)
    explore.add_argument("--query", required=True)
    args = parser.parse_args()
    try:
        path = ExplorationPipeline(load_config(args.config)).run(args.workbook, args.query)
        print(path)
        return 0
    except (ConfigError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
