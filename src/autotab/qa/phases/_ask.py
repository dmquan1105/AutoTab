"""One model interaction with validation and a bounded re-ask, shared by the phases.

A re-ask repairs a malformed reply inside the phase; it never consumes a turn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ...models.client import ModelClient
from ..prompts import repair_request

T = TypeVar("T")

_ERROR_CHARS = 600
# Numeric usage fields worth summing across attempts; providers may add others.
_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")


@dataclass(frozen=True)
class Reply(Generic[T]):
    """The validated value, every raw reply in order, and why no value came back.

    ``rejections`` pairs with ``raw``: why each reply was refused, ``None`` for the
    accepted one, so a trace can show every attempt next to its reason.

    ``usage`` sums what the endpoint reported over every attempt, re-asks included,
    with ``calls`` counting the requests made; ``latency_ms`` is their total duration.
    """

    value: T | None
    raw: tuple[str, ...]
    error: str | None
    usage: dict[str, int | float] = field(default_factory=dict)
    latency_ms: float | None = None
    rejections: tuple[str | None, ...] = ()

    @property
    def attempts(self) -> int:
        """How many replies the model produced."""
        return len(self.raw)


def _measure(client: ModelClient, usage: dict[str, int | float]) -> float | None:
    """Add the client's last-call usage to ``usage``; return that call's latency."""
    usage["calls"] = usage.get("calls", 0) + 1
    reported: Any = getattr(client, "last_usage", None)
    if isinstance(reported, dict):
        for name in _USAGE_FIELDS:
            value = reported.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[name] = usage.get(name, 0) + value
    latency: Any = getattr(client, "last_latency_ms", None)
    return float(latency) if isinstance(latency, (int, float)) else None


def ask(client: ModelClient, prompt: str, parse: Callable[[str], T], *, retries: int) -> Reply[T]:
    """Call the model, validate the reply, and re-ask up to ``retries`` times.

    A reply that fails validation is re-asked with the validation error appended. A
    failed request (``ModelResponseError`` is a ``ValueError``) is retried as sent.
    """
    raw: list[str] = []
    rejections: list[str | None] = []
    usage: dict[str, int | float] = {}
    latencies: list[float] = []
    text = prompt
    error: str | None = None

    def reply(value: T | None) -> Reply[T]:
        total = sum(latencies) if latencies else None
        return Reply(value, tuple(raw), error, usage, total, tuple(rejections))

    for _ in range(retries + 1):
        try:
            answer = client.complete(text)
        except ValueError as exc:
            error = f"the model request failed: {str(exc)[:_ERROR_CHARS]}"
            continue
        finally:
            latency = _measure(client, usage)
            if latency is not None:
                latencies.append(latency)
        raw.append(answer)
        try:
            value = parse(answer)
        except ValueError as exc:
            error = str(exc)[:_ERROR_CHARS]
            rejections.append(error)
            text = prompt + repair_request(error)
            continue
        error = None
        rejections.append(None)
        return reply(value)
    return reply(None)
