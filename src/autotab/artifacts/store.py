"""Run IDs and the exploration artifact writer.

Every module writes under ``<artifact_root>/<run_id>/<module>/``, so one run's
``exploration/`` and ``qa/`` sit side by side in the same folder.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def new_run_id() -> str:
    """Return a run ID shared by every module of one run: local time, then the PID."""
    return f"{datetime.now().astimezone():%Y%m%dT%H%M%S%f}-{os.getpid()}"


class ArtifactStore:
    def __init__(self, root: str | Path, run_id: str) -> None:
        self.root = Path(root).resolve() / run_id / "exploration"
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, value: object) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path

    def write_text(self, name: str, value: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path
