"""Append-only history and run artifacts for one QA run.

Owns stable IDs, run-relative path safety, immutable execution artifacts, and atomic
replacement of the manifest and final answer; see ``specs/support/artifacts.md``.
It never selects prompt context or judges correctness.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from itertools import count
from pathlib import Path
from typing import Any

from .schemas import SCHEMA_VERSION, HistoryEvent, ensure_run_relative

REDACTED = "<redacted>"

# Matched against whole key names, never substrings: "token" must not hide
# max_tokens, and a pattern tied to one spelling must not miss "apiKey".
_SECRET_KEYS = frozenset(
    {"api_key", "apikey", "access_token", "token", "secret", "password", "authorization"}
)
_SECRET_SUFFIXES = ("_api_key", "_secret", "_password", "_access_token")


def _is_secret(key: str) -> bool:
    name = key.lower().replace("-", "_")
    if name == "apikey":
        return True
    return name in _SECRET_KEYS or name.endswith(_SECRET_SUFFIXES)


def redact(value: Any) -> Any:
    """Return a copy of ``value`` with every secret-named field replaced."""
    if isinstance(value, Mapping):
        return {
            key: REDACTED if _is_secret(str(key)) and item else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


class Trace:
    """The artifact root of one QA run, under ``<artifact_root>/<run_id>/qa``."""

    def __init__(self, artifact_root: str | Path, run_id: str) -> None:
        """Open the run's ``qa/`` folder beside its ``exploration/`` folder.

        Raises:
            FileExistsError: If that folder already holds a history: two runs must
                never interleave events in one append-only log.
        """
        self.run_id = run_id
        self.root = Path(artifact_root).resolve() / run_id / "qa"
        if (self.root / "history.jsonl").exists():
            raise FileExistsError(f"run {run_id} already has a QA history at {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self._events = count(1)

    def next_event_id(self) -> str:
        """Return the next stable event ID of this run."""
        return f"event_{next(self._events):04d}"

    def _path(self, relative: str) -> Path:
        ensure_run_relative(relative)
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError(f"{relative!r} escapes the run root")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def append(self, event: HistoryEvent) -> None:
        """Append one event to ``history.jsonl``; earlier lines are never rewritten."""
        line = event.model_dump_json(exclude_none=True)
        with (self.root / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def history(self) -> list[HistoryEvent]:
        """Read the history back in append order."""
        path = self.root / "history.jsonl"
        if not path.exists():
            return []
        return [
            HistoryEvent.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def write_json(self, relative: str, payload: Mapping[str, Any]) -> Path:
        """Write an immutable JSON artifact stamped with the schema version and run ID.

        Raises:
            FileExistsError: If the artifact was already written.
            ValueError: If the path leaves the run root.
        """
        path = self._path(relative)
        document = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, **payload}
        with path.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
        return path

    def write_text(self, relative: str, text: str) -> Path:
        """Write an immutable text artifact such as generated code or a prompt."""
        path = self._path(relative)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def _replace(self, relative: str, text: str) -> Path:
        path = self._path(relative)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        return path

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        """Atomically replace the manifest; secrets are redacted before writing."""
        document = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, **redact(manifest)}
        return self._replace("manifest.json", json.dumps(document, indent=2, ensure_ascii=False))

    def write_final_answer(self, answer_text: str) -> Path:
        """Atomically write the verified answer; only called after PASS."""
        return self._replace("final/answer.md", answer_text)
