"""Safe run artifact writer."""

from __future__ import annotations

import json
from pathlib import Path


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
