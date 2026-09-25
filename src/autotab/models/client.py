"""OpenAI-compatible model clients for local servers and hosted routers."""

from __future__ import annotations

import base64
import json
import mimetypes
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_READ_CHUNK = 8192
_ERROR_BODY_CHARS = 500


class ModelClient(Protocol):
    def complete(self, prompt: str, image_path: str | Path | None = None) -> str: ...


class ModelResponseError(ValueError):
    """Raised when an endpoint fails or answers without usable content.

    It subclasses ``ValueError`` so callers that already treat a malformed model
    answer as recoverable keep doing so.
    """


class OpenAICompatibleClient:
    """Call an OpenAI-compatible endpoint, local (vLLM) or hosted (OpenRouter)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 20,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> None:
        """Create a client.

        Args:
            base_url: Endpoint root, for example ``https://openrouter.ai/api/v1``.
            model: Model identifier understood by the endpoint.
            timeout: Per-request timeout in seconds.
            max_tokens: Output cap. ``None`` sends no cap, leaving the model's own
                limit in force.
            extra_body: Provider-specific fields merged into the request body, such
                as OpenRouter's ``reasoning`` or ``provider`` preferences.
            api_key: Sent as a bearer token when set; local servers need none.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self._api_key = api_key
        # Per thread: exploration shares one client across worker threads.
        self._last = threading.local()

    @property
    def last_usage(self) -> dict[str, Any] | None:
        """Token usage the endpoint reported for this thread's last call, if any."""
        usage: dict[str, Any] | None = getattr(self._last, "usage", None)
        return usage

    @property
    def last_latency_ms(self) -> float | None:
        """Wall-clock duration of this thread's last call, in milliseconds."""
        latency: float | None = getattr(self._last, "latency_ms", None)
        return latency

    def complete(self, prompt: str, image_path: str | Path | None = None) -> str:
        """Complete a prompt, optionally attaching a local image to the user message.

        Raises:
            ModelResponseError: If the request fails or the answer has no content.
        """
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
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        payload.update(self.extra_body)
        result = self._post("/chat/completions", payload)
        message = result["choices"][0]["message"]
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            # A reasoning model can spend its whole budget thinking and return no
            # content; str(None) would pass the literal text "None" downstream.
            raise ModelResponseError(
                f"{self.model} returned no content; if it is a reasoning model, disable "
                "reasoning in extra_body or raise max_tokens"
            )
        return text

    def embed(self, texts: list[str] | tuple[str, ...]) -> list[list[float]]:
        """Embed texts, returning vectors in input order."""
        result = self._post("/embeddings", {"model": self.model, "input": list(texts)})
        return [
            item["embedding"] for item in sorted(result["data"], key=lambda item: item["index"])
        ]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        self._last.usage = None
        started = time.perf_counter()
        # The timeout bounds the whole request. urlopen's own timeout only bounds each
        # socket read, and OpenRouter keeps a slow request alive with comment lines, so
        # a stalled provider once held a run for fourteen minutes.
        deadline = time.monotonic() + self.timeout
        try:
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    chunks: list[bytes] = []
                    # read1 returns what has arrived; read(n) would wait for n bytes.
                    while chunk := response.read1(_READ_CHUNK):
                        chunks.append(chunk)
                        if time.monotonic() > deadline:
                            raise TimeoutError
                    body = b"".join(chunks).decode(errors="replace")
                    try:
                        parsed: dict[str, Any] = json.loads(body)
                    except ValueError as exc:
                        raise ModelResponseError(
                            f"{self.model} response from {path} is not JSON: "
                            f"{body[:_ERROR_BODY_CHARS]!r}"
                        ) from exc
            except TimeoutError as exc:
                raise ModelResponseError(
                    f"{self.model} request to {path} did not finish within {self.timeout:g}s"
                ) from exc
            except HTTPError as exc:
                # The body carries the provider's reason (credits, rate limit, bad model
                # id). Headers, and so the key, are never part of the message.
                detail = exc.read().decode(errors="replace")[:_ERROR_BODY_CHARS]
                raise ModelResponseError(
                    f"{self.model} request to {path} failed with HTTP {exc.code}: {detail}"
                ) from exc
            except URLError as exc:
                raise ModelResponseError(
                    f"{self.model} request to {path} could not reach {self.base_url}: {exc.reason}"
                ) from exc
        finally:
            self._last.latency_ms = (time.perf_counter() - started) * 1000.0
        usage = parsed.get("usage")
        self._last.usage = usage if isinstance(usage, dict) else None
        return parsed


def client_from_config(
    settings: Mapping[str, Any] | None, timeout: float
) -> OpenAICompatibleClient | None:
    """Build the client for one ``models.<role>`` config section.

    Args:
        settings: The role's section, for example ``config["models"]["vlm"]``.
        timeout: Per-request timeout in seconds.

    Returns:
        A client, or ``None`` when the role has no ``base_url`` configured.
    """
    if not settings or not settings.get("base_url"):
        return None
    max_tokens = settings.get("max_tokens")
    return OpenAICompatibleClient(
        base_url=str(settings["base_url"]),
        model=str(settings["model"]),
        timeout=timeout,
        max_tokens=None if max_tokens is None else int(max_tokens),
        extra_body=settings.get("extra_body"),
        api_key=settings.get("api_key"),
    )
