from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TableAgent.configs import DEFAULT_CONFIG_PATH, load_config
from service import create_model_client
from TableAgent.stages.qa import TableQARunner


DEFAULT_STRUCTURE_PATH = ROOT / "sample" / "structure.yaml"
DEFAULT_WORKBOOK_PATH = ROOT / "sample" / "QA_sample.xlsx"
DEFAULT_TABLE_ID = "table1"
DEFAULT_QUESTION = (
    "Find every person whose last name is Pham and whose birth year is 1995. "
    "Return all available fields for each matching person as a compact Markdown table. "
    "Use the meaningful header labels from structure.yaml (for example First, Middle, "
    "Last, Birth, Admission, and Score), never internal header IDs."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real TableAgent QA flow on a workbook question.",
    )
    parser.add_argument("--structure", default=str(DEFAULT_STRUCTURE_PATH))
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK_PATH))
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--llm-profile",
        default="table_agent",
        help="Config pipeline/provider name passed to create_model_client.",
    )
    parser.add_argument(
        "--table-id",
        default=DEFAULT_TABLE_ID,
        help=f"Table id from structure.yaml (default: {DEFAULT_TABLE_ID}).",
    )
    return parser


def _subtask_to_dict(subtask: Any) -> dict[str, Any]:
    return {
        "id": subtask.id,
        "layer": subtask.layer,
        "depends_on": list(subtask.depends_on),
        "description": subtask.description,
        "status": subtask.status,
    }


def _header_contract(runner: TableQARunner, table_id: str) -> list[dict[str, str]]:
    """Expose the exact id/label pairs loaded from structure.yaml for this sample run."""
    return [
        {"id": header.id, "label": header.label}
        for header in runner.env.operators.list_headers(table_id)
    ]


def _leaked_header_ids(answer: str | None, headers: list[dict[str, str]]) -> list[str]:
    """Report internal IDs that remain as exact tokens in the user-facing answer."""
    if not answer:
        return []
    leaked = []
    for header in headers:
        header_id = header["id"]
        label = header["label"]
        if not header_id or header_id == label or header_id.isdecimal():
            continue
        if re.search(rf"(?<![\w]){re.escape(header_id)}(?![\w])", answer):
            leaked.append(header_id)
    return leaked


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.table_id:
        config.setdefault("table_agent", {})["table_id"] = args.table_id

    llm_client = create_model_client(config, kind="llm", profile=args.llm_profile)
    runner = TableQARunner(
        args.structure,
        args.workbook,
        llm_client=llm_client,
        config=config,
    )
    result = runner.run(args.question)
    headers = _header_contract(runner, args.table_id)
    leaked_header_ids = _leaked_header_ids(result.final_answer, headers)

    summary = {
        "question": result.question,
        "success": result.success,
        "final_answer": result.final_answer,
        "error": result.error,
        "execution_time": result.execution_time,
        "artifacts": result.artifacts,
        "plan": [_subtask_to_dict(subtask) for subtask in result.plan],
        "structure_headers": headers,
        "header_label_check": {
            "passed": not leaked_header_ids,
            "leaked_internal_ids": leaked_header_ids,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.success and not leaked_header_ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
