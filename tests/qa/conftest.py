"""Shared fakes for QA phase and agent-loop tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


class ScriptedClient:
    """A model client that replays scripted replies and records every prompt.

    A reply that is an exception is raised instead of returned, to model a failed
    request; a dict is serialized as the JSON the model would send.
    """

    def __init__(self, *replies: str | dict[str, Any] | Exception) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.last_usage: dict[str, int] | None = None
        self.last_latency_ms: float | None = None

    def complete(self, prompt: str, image_path: str | Path | None = None) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("the model was called more often than scripted")
        reply = self.replies.pop(0)
        self.last_latency_ms = 5.0
        if isinstance(reply, Exception):
            self.last_usage = None
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        self.last_usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4}
        return text


@pytest.fixture
def scripted() -> type[ScriptedClient]:
    return ScriptedClient
