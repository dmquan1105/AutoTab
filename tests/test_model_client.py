"""OpenAI-compatible client behavior against hosted endpoints such as OpenRouter."""

import io
import json
from typing import Any, Self
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from autotab.models.client import ModelResponseError, OpenAICompatibleClient

SECRET = "sk-or-v1-test-secret-value"


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read1(self, size: int = -1) -> bytes:
        body, self._body = self._body, b""
        return body


def _capture(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["body"] = json.loads(request.data.decode()) if request.data else None
        return FakeResponse(payload)

    monkeypatch.setattr("autotab.models.client.urlopen", fake_urlopen)
    return captured


def _chat(content: Any) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def test_api_key_is_sent_as_a_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch, _chat("pong"))

    OpenAICompatibleClient("https://openrouter.ai/api/v1", "m", api_key=SECRET).complete("ping")

    assert captured["headers"]["authorization"] == f"Bearer {SECRET}"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"


def test_no_authorization_header_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch, _chat("pong"))

    OpenAICompatibleClient("http://localhost:8010/v1", "m").complete("ping")

    assert "authorization" not in captured["headers"]


def test_null_max_tokens_means_no_limit_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch, _chat("pong"))

    OpenAICompatibleClient("http://x/v1", "m", max_tokens=None).complete("ping")
    assert "max_tokens" not in captured["body"]

    OpenAICompatibleClient("http://x/v1", "m", max_tokens=4000).complete("ping")
    assert captured["body"]["max_tokens"] == 4000


def test_extra_body_is_merged_into_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch, _chat("pong"))
    extra = {"reasoning": {"enabled": False}, "provider": {"order": ["AkashML"]}}

    OpenAICompatibleClient("http://x/v1", "m", extra_body=extra).complete("ping")

    assert captured["body"]["reasoning"] == {"enabled": False}
    assert captured["body"]["provider"] == {"order": ["AkashML"]}


@pytest.mark.parametrize("content", [None, "", "   "])
def test_missing_content_raises_instead_of_returning_text(
    monkeypatch: pytest.MonkeyPatch, content: Any
) -> None:
    # A reasoning model can spend its whole budget thinking and return no content.
    # Returning str(None) would hand the literal text "None" to the next stage.
    _capture(monkeypatch, _chat(content))

    with pytest.raises(ModelResponseError, match="no content"):
        OpenAICompatibleClient("http://x/v1", "m").complete("ping")


def test_missing_content_is_a_value_error_so_existing_fallbacks_still_apply() -> None:
    assert issubclass(ModelResponseError, ValueError)


def test_http_errors_report_status_and_body_without_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_urlopen(request: Request, timeout: float) -> FakeResponse:
        raise HTTPError(
            request.full_url,
            402,
            "Payment Required",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error": {"message": "Insufficient credits"}}'),
        )

    monkeypatch.setattr("autotab.models.client.urlopen", failing_urlopen)

    with pytest.raises(ModelResponseError) as caught:
        OpenAICompatibleClient("http://x/v1", "m", api_key=SECRET).complete("ping")

    message = str(caught.value)
    assert "402" in message
    assert "Insufficient credits" in message
    assert SECRET not in message


def test_embed_authenticates_and_orders_vectors_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(
        monkeypatch,
        {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]},
    )

    vectors = OpenAICompatibleClient("http://x/v1", "e", api_key=SECRET).embed(["a", "b"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["headers"]["authorization"] == f"Bearer {SECRET}"
    assert captured["url"] == "http://x/v1/embeddings"


def test_client_from_config_skips_a_role_without_base_url() -> None:
    from autotab.models.client import client_from_config

    assert client_from_config(None, timeout=5) is None
    assert client_from_config({"model": "m", "base_url": None}, timeout=5) is None


def test_client_from_config_carries_key_limit_and_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autotab.models.client import client_from_config

    captured = _capture(monkeypatch, _chat("pong"))
    client = client_from_config(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "deepseek/deepseek-v4-flash-0731",
            "api_key": SECRET,
            "max_tokens": None,
            "extra_body": {"reasoning": {"enabled": False}},
        },
        timeout=5,
    )
    assert client is not None

    client.complete("ping")

    assert captured["headers"]["authorization"] == f"Bearer {SECRET}"
    assert "max_tokens" not in captured["body"]
    assert captured["body"]["reasoning"] == {"enabled": False}
    assert client.timeout == 5


class _FakeVllm:
    """A real HTTP server on localhost that answers like vLLM and records requests."""

    def __init__(self) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        requests: list[dict[str, Any]] = []
        self.requests = requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if self.path.endswith("/embeddings"):
                    reply: dict[str, Any] = {
                        "data": [
                            {"index": i, "embedding": [float(i), 1.0]}
                            for i, _ in enumerate(body["input"])
                        ]
                    }
                else:
                    reply = {"choices": [{"message": {"content": '["score"]'}}]}
                encoded = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def test_local_vllm_path_over_real_http_is_unchanged() -> None:
    # Switching back from OpenRouter to the VPN-hosted vLLM servers must keep working:
    # no key means no Authorization header, a numeric max_tokens is sent, and
    # vLLM's chat_template_kwargs still reaches the server.
    from autotab.models.client import client_from_config

    server = _FakeVllm()
    try:
        llm = client_from_config(
            {
                "base_url": server.base_url,
                "model": "Qwen/Qwen3.6-35B-A3B-FP8",
                "max_tokens": 4000,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
            timeout=5,
        )
        embedding = client_from_config(
            {"base_url": server.base_url, "model": "Qwen/Qwen3-Embedding-4B"}, timeout=5
        )
        assert llm is not None and embedding is not None

        assert llm.complete("keywords please") == '["score"]'
        assert embedding.embed(["Score", "Điểm"]) == [[0.0, 1.0], [1.0, 1.0]]
    finally:
        server.close()

    chat, embed = server.requests
    assert "Authorization" not in chat["headers"]
    assert "Authorization" not in embed["headers"]
    assert chat["body"]["max_tokens"] == 4000
    assert chat["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert chat["path"] == "/v1/chat/completions"
    assert embed["path"] == "/v1/embeddings"


def test_usage_and_latency_of_the_last_call_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280}
    _capture(monkeypatch, {**_chat("pong"), "usage": usage})
    client = OpenAICompatibleClient("http://x/v1", "m")

    client.complete("ping")

    assert client.last_usage == usage
    assert isinstance(client.last_latency_ms, float) and client.last_latency_ms >= 0


def test_last_usage_is_kept_per_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exploration calls one VLM client from several threads at once; one thread's
    # usage must never be read back as another's.
    import threading

    _capture(monkeypatch, {**_chat("pong"), "usage": {"prompt_tokens": 1}})
    client = OpenAICompatibleClient("http://x/v1", "m")
    client.complete("ping")
    seen: list[object] = []

    worker = threading.Thread(target=lambda: seen.append(client.last_usage))
    worker.start()
    worker.join()

    assert seen == [None]
    assert client.last_usage == {"prompt_tokens": 1}


class _StallingServer:
    """Answers with headers, then either trickles keep-alive bytes or goes silent."""

    def __init__(self, trickle: bool) -> None:
        import threading
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                for _ in range(40):  # about 8 seconds, far past the client's deadline
                    if trickle:
                        # OpenRouter keeps a slow request alive with comment lines, so
                        # the socket never goes quiet long enough for its timeout.
                        self.wfile.write(b": OPENROUTER PROCESSING\n\n")
                        self.wfile.flush()
                    time.sleep(0.2)

            def log_message(self, *args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True  # closing must not wait for the stall to end
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.mark.parametrize("trickle", [True, False])
def test_a_request_never_outlives_its_timeout(trickle: bool) -> None:
    # A real run hung for fourteen minutes on one judge call: keep-alive bytes reset the
    # socket timeout forever. The timeout bounds the whole request, and a stall is a
    # ModelResponseError the phase can retry, never an uncaught socket error.
    import time

    server = _StallingServer(trickle)
    client = OpenAICompatibleClient(server.base_url, "slow-model", timeout=1)
    started = time.monotonic()
    try:
        with pytest.raises(ModelResponseError, match="1s"):
            client.complete("hello")
        elapsed = time.monotonic() - started
    finally:
        server.close()

    assert elapsed < 4


def test_an_empty_or_broken_body_is_a_model_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Broken(FakeResponse):
        def __init__(self) -> None:
            self._body = b"<html>bad gateway</html>"

    monkeypatch.setattr("autotab.models.client.urlopen", lambda request, timeout: Broken())

    with pytest.raises(ModelResponseError, match="not JSON"):
        OpenAICompatibleClient("http://localhost/v1", "m").complete("hi")
