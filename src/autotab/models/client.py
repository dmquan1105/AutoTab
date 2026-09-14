"""OpenAI-compatible local model clients."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen


class ModelClient(Protocol):
    def complete(self, prompt: str, image_path: str | Path | None = None) -> str: ...


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

    def complete(self, prompt: str, image_path: str | Path | None = None) -> str:
        """Complete a prompt, optionally attaching a local image to the user message."""
        content: str | list[dict[str, Any]] = prompt
        if image_path is not None:
            path = Path(image_path)
            media_type = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{encoded_image}"},
                },
            ]
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
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
