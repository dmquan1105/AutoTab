"""OpenAI-compatible local model clients."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.request import Request, urlopen


class ModelClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class OpenAICompatibleClient:
    """Call an OpenAI-compatible local endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 20,
        max_tokens: int = 512,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url, self.model, self.timeout, self.max_tokens, self.extra_body = (
            base_url.rstrip("/"),
            model,
            timeout,
            max_tokens,
            extra_body or {},
        )

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            **self.extra_body,
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            result: dict[str, Any] = json.loads(response.read().decode())
        message = result["choices"][0]["message"]
        return str(message["content"])

    def embed(self, texts: list[str] | tuple[str, ...]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        request = Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            result: dict[str, Any] = json.loads(response.read().decode())
        return [
            item["embedding"] for item in sorted(result["data"], key=lambda item: item["index"])
        ]
