"""Bounded ReAct loop for workbook QA."""

from __future__ import annotations

import json
from pathlib import Path

from ..models.client import ModelClient
from .parser import ResponseFormatError, parse_response
from .prompts import conversation_prompt, initial_prompt, system_prompt
from .sandbox import Sandbox
from .tools import WorkbookSession
from .types import Action, FinalAnswer, QAResult

FORMAT_REMINDER = (
    "Format error: respond with exactly Thought + one Action Python block, or Thought + "
    "Final Answer. Do not include both."
)


class QAAgent:
    """Coordinate model turns and isolated workbook observations."""

    def __init__(self, model: ModelClient, config: dict) -> None:
        self._model = model
        self._config = config

    def run(
        self,
        workbook_path: str | Path,
        question: str,
        evidence: str | None = None,
        artifact_dir: str | Path | None = None,
    ) -> QAResult:
        """Answer one question about one workbook.

        Args:
            workbook_path: Host-selected workbook path.
            question: Untrusted user question.
            evidence: Optional exploration evidence supplied as context.
            artifact_dir: Optional directory for per-turn QA debugging artifacts.

        Returns:
            The final answer or a bounded-turn failure result.
        """
        qa = self._config["qa"]
        artifacts = Path(artifact_dir) if artifact_dir is not None else None
        if artifacts is not None:
            artifacts.mkdir(parents=True, exist_ok=True)
        session = WorkbookSession(workbook_path, qa["tools"]["max_range_cells"])
        try:
            system = system_prompt(session.description)
        finally:
            session.close()
        turns = [("User", initial_prompt(question, evidence))]
        sandbox = Sandbox(
            workbook_path,
            qa["sandbox"]["timeout_seconds"],
            qa["sandbox"]["memory_limit_mb"],
            qa["max_observation_chars"],
            qa["tools"]["max_range_cells"],
        )
        for turn_number in range(1, qa["max_turns"] + 1):
            prompt = conversation_prompt(system, turns)
            turn_dir = artifacts / f"turn-{turn_number:02d}" if artifacts else None
            _write_text(turn_dir, "prompt.txt", prompt)
            response = self._model.complete(prompt)
            _write_text(turn_dir, "response.txt", response)
            turns.append(("Assistant", response))
            try:
                parsed = parse_response(response)
            except ResponseFormatError:
                _write_text(turn_dir, "observation.txt", FORMAT_REMINDER)
                turns.append(("Observation", FORMAT_REMINDER))
                continue
            if isinstance(parsed, FinalAnswer):
                result = QAResult(True, parsed.answer, turn_number)
                _write_result(artifacts, result)
                return result
            assert isinstance(parsed, Action)
            if len(parsed.code) > qa["max_code_chars"]:
                observation = "Action rejected: code exceeds the configured character limit"
            else:
                sandbox_result = sandbox.execute(parsed.code)
                prefix = "Observation" if sandbox_result.success else "Execution error"
                observation = f"{prefix}: {sandbox_result.text}"
            _write_text(turn_dir, "observation.txt", observation)
            turns.append(("Observation", observation))
        result = QAResult(
            False,
            "Unable to produce a final answer within the configured turn limit.",
            qa["max_turns"],
        )
        _write_result(artifacts, result)
        return result


def _write_text(directory: Path | None, name: str, value: str) -> None:
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(value, encoding="utf-8")


def _write_result(directory: Path | None, result: QAResult) -> None:
    if directory is None:
        return
    (directory / "result.json").write_text(
        json.dumps(
            {"success": result.success, "answer": result.answer, "turns": result.turns},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
