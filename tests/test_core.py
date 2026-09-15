import json
from pathlib import Path
from typing import Any, Self
from urllib.request import Request

from PIL import Image

from autotab.config import ConfigError, load_config
from autotab.exploration.evidence import EvidenceExtractor
from autotab.exploration.filtering import EvidenceFilter
from autotab.exploration.ingestion import WorkbookLoader
from autotab.exploration.keywords import KeywordExtractor, normalize_phrase
from autotab.exploration.markdown import MarkdownWriter
from autotab.exploration.pipeline import ExplorationPipeline
from autotab.exploration.prompts import viewport_description_prompt
from autotab.exploration.retrieval import HybridCellRetriever
from autotab.models.client import OpenAICompatibleClient
from autotab.utils.ranges import parse_cell, viewport
from autotab.utils.rendering import WindowRenderer


def test_config_and_ranges() -> None:
    config = load_config("config.example.yaml")
    assert config["retrieval"]["lexical_weight"] == 0.45
    assert config["rendering"]["workers"] == 1
    assert parse_cell("C4") == (4, 3)
    assert viewport("A1", 3, 3, 9) == "A1:C3"


def test_invalid_config() -> None:
    try:
        load_config("does-not-exist.yaml")
    except ConfigError:
        raise AssertionError("missing config should use defaults")


def test_invalid_rendering_workers(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("rendering:\n  workers: 0\n", encoding="utf-8")
    try:
        load_config(config_path)
    except ConfigError as exc:
        assert "rendering.workers" in str(exc)
    else:
        raise AssertionError("non-positive rendering workers should be rejected")


def test_keywords_fallback() -> None:
    assert normalize_phrase("  Revenue   By Region ") == "revenue by region"
    values = KeywordExtractor().extract("Find revenue by region")
    assert values and values[0] == "revenue"


def test_ingestion_and_retrieval() -> None:
    snapshot = WorkbookLoader().load(Path("samples/sample.xlsx"))
    assert snapshot.path and snapshot.cells
    hits = HybridCellRetriever().search("revenue", snapshot.cells, 0.0, 2)
    assert len(hits) <= 2


def test_pipeline_and_artifacts(tmp_path: Path) -> None:
    config = load_config("config.example.yaml")
    config["runtime"]["artifact_root"] = str(tmp_path)
    output = ExplorationPipeline(config).run(["samples/sample.xlsx"], "Find revenue")
    assert output.exists() and (output.parent / "manifest.json").exists()


def test_evidence_markdown_and_rendering(tmp_path: Path) -> None:
    class FakeVLM:
        def complete(self, prompt: str, image_path: str | Path | None = None) -> str:
            assert "Excel worksheet" in prompt
            assert image_path == "viewport.png"
            return '{"description": "Revenue is shown in the viewport."}'

    evidence = EvidenceExtractor(FakeVLM()).extract(
        "revenue",
        "Sheet1!A1",
        [{"displayed_value": "Revenue"}],
        "viewport.png",
    )
    assert EvidenceFilter().filter(evidence, "revenue").status == "keep"
    markdown = MarkdownWriter().write("run", [evidence])
    assert "## revenue" in markdown
    assert "Query:" not in markdown
    assert "- relations:" not in markdown
    metadata = WindowRenderer().render(
        "samples/sample.xlsx", "Relations_Test", "A1:B2", tmp_path / "x.png", {}
    )
    assert metadata["range"] == "A1:B2"


def test_renderer_enables_isolated_pdfium_for_multiple_workers(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def fake_render(workbook, sheet, cell_range, output, *, max_workers):
        captured["max_workers"] = max_workers
        Image.new("RGB", (10, 10), "white").save(output)
        return "libreoffice"

    monkeypatch.setattr("autotab.utils.rendering._render_with_libreoffice", fake_render)

    WindowRenderer(workers=3).render("book.xlsx", "Sheet1", "A1:B2", tmp_path / "x.png", {})

    assert captured["max_workers"] == 3


def test_viewport_prompt_requires_worksheet_grounding() -> None:
    prompt = viewport_description_prompt("revenue", "Sheet1!B3", [])

    assert "source cell is Sheet1!B3" in prompt
    assert "grouped or merged header" in prompt
    assert "extends across rows or columns" in prompt
    assert "not visible instead of guessing" in prompt
    assert "User query:" not in prompt
    assert "later QA step" not in prompt

    query_prompt = viewport_description_prompt(
        "revenue",
        "Sheet1!B3",
        [],
        query="Which region has the highest revenue?",
    )
    assert "User query: Which region has the highest revenue?" in query_prompt
    assert (
        "Include concrete visible labels and values that would help a later QA step answer a "
        "question using this evidence."
    ) in query_prompt


def test_model_client_attaches_viewport_image(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "viewport.png"
    image_path.write_bytes(b"worksheet image")
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices": [{"message": {"content": "ok"}}]}'

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        captured.update(json.loads(request.data.decode()))
        return FakeResponse()

    monkeypatch.setattr("autotab.models.client.urlopen", fake_urlopen)

    result = OpenAICompatibleClient("http://localhost/v1", "vlm").complete(
        "Describe the worksheet.", image_path
    )

    content = captured["messages"][0]["content"]
    assert result == "ok"
    assert content[0] == {"type": "text", "text": "Describe the worksheet."}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
