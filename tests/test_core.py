import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Self
from urllib.request import Request

from PIL import Image

from autotab.config import ConfigError, load_config
from autotab.exploration.evidence import Evidence, EvidenceExtractor
from autotab.exploration.filtering import EvidenceFilter
from autotab.exploration.ingestion import WorkbookLoader
from autotab.exploration.keywords import KeywordExtractor, normalize_phrase
from autotab.exploration.markdown import MarkdownWriter
from autotab.exploration.pipeline import ExplorationPipeline
from autotab.exploration.prompts import keyword_summary_prompt, viewport_description_prompt
from autotab.exploration.retrieval import HybridCellRetriever
from autotab.exploration.summary import KeywordSummarizer, KeywordSummary
from autotab.models.client import OpenAICompatibleClient
from autotab.utils.pdfium_worker import bounded_scale
from autotab.utils.ranges import parse_cell, viewport
from autotab.utils.rendering import WindowRenderer


def test_config_and_ranges() -> None:
    config = load_config("config.example.yaml")
    assert config["retrieval"]["lexical_weight"] == 0.45
    assert config["exploration"]["max_concurrent_findings"] == 3
    assert config["rendering"]["workers"] == 2
    assert config["rendering"]["image_resolution"] == 300
    assert parse_cell("C4") == (4, 3)
    assert viewport("A1", 3, 3, 9) == "A1:C3"


def test_invalid_config() -> None:
    try:
        load_config("does-not-exist.yaml")
    except ConfigError:
        raise AssertionError("missing config should use defaults")


def test_invalid_max_concurrent_findings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("exploration:\n  max_concurrent_findings: 0\n", encoding="utf-8")
    try:
        load_config(config_path)
    except ConfigError as exc:
        assert "exploration.max_concurrent_findings" in str(exc)
    else:
        raise AssertionError("non-positive finding concurrency should be rejected")


def test_invalid_render_workers(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("rendering:\n  workers: 0\n", encoding="utf-8")
    try:
        load_config(config_path)
    except ConfigError as exc:
        assert "rendering.workers" in str(exc)
    else:
        raise AssertionError("non-positive render concurrency should be rejected")


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
    assert (output.parent / "aggregation.md").exists()


def test_pipeline_bounds_concurrent_findings(tmp_path: Path, monkeypatch) -> None:
    config = load_config("config.example.yaml")
    config["runtime"]["artifact_root"] = str(tmp_path)
    config["exploration"]["max_concurrent_findings"] = 5
    config["rendering"]["workers"] = 2
    config["exploration"]["similarity_threshold"] = 0.0
    config["exploration"]["max_cells_per_keyword"] = 4
    config["models"]["llm"]["base_url"] = None
    config["models"]["vlm"]["base_url"] = None
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_render(self, workbook, sheet, cell_range, output, metadata):
        nonlocal active, max_active
        assert self.workers == config["rendering"]["workers"]
        assert self.timeout_seconds == config["rendering"]["timeout_seconds"]
        assert self.image_resolution == config["rendering"]["image_resolution"]
        assert self.max_image_dimension == config["rendering"]["max_image_dimension"]
        assert self.max_image_pixels == config["rendering"]["max_image_pixels"]
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {**metadata, "sheet": sheet, "range": cell_range}

    def fake_extract(self, keyword, source_range, cells, image_path=None):
        self.last_response = '{"description": "Revenue is shown."}'
        return Evidence(keyword, source_range, "Revenue is shown.")

    monkeypatch.setattr(WindowRenderer, "render", fake_render)
    monkeypatch.setattr(EvidenceExtractor, "extract", fake_extract)
    monkeypatch.setattr(KeywordSummarizer, "summarize", lambda self, query, evidence: [])

    ExplorationPipeline(config).run(["samples/sample.xlsx"], "Find revenue")

    assert config["rendering"]["workers"] < max_active
    assert max_active <= config["exploration"]["max_concurrent_findings"]


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


def test_keyword_summary_preserves_keywords_and_ranges() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, prompt: str, image_path: str | Path | None = None) -> str:
            self.prompts.append(prompt)
            assert "Which region has the highest revenue?" in prompt
            if 'exact keyword "tháng 1"' in prompt:
                return "January revenue is listed for each region."
            return "DROP"

    model = FakeLLM()
    summaries = KeywordSummarizer(model).summarize(
        "Which region has the highest revenue?",
        [
            Evidence("tháng 1", "Data!B4", "January revenue is shown by region."),
            Evidence("tháng 1", "Other!B4", "January revenue appears on another sheet."),
            Evidence("workbook", "Data!A1", "The workbook has a title."),
        ],
    )

    assert len(model.prompts) == 2
    assert summaries == [
        KeywordSummary(
            "tháng 1",
            ("Data!B4", "Other!B4"),
            "January revenue is listed for each region.",
        )
    ]
    markdown = MarkdownWriter().write_summary(summaries)
    assert "## tháng 1" in markdown
    assert "`Data!B4`, `Other!B4`" in markdown
    assert "## workbook" not in markdown


def test_keyword_summary_prompt_requires_consolidation_and_filtering() -> None:
    prompt = keyword_summary_prompt("revenue", "Find revenue", ["Revenue is a column."])

    assert 'exact keyword "revenue"' in prompt
    assert "Do not rename the keyword" in prompt
    assert "Do not perform" in prompt
    assert "return exactly DROP" in prompt


def test_renderer_passes_config_to_libreoffice(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_render(workbook, sheet, cell_range, output, **options):
        captured.update(options)
        Image.new("RGB", (10, 10), "white").save(output)
        return "libreoffice"

    monkeypatch.setattr("autotab.utils.rendering._render_with_libreoffice", fake_render)

    WindowRenderer(
        workers=3,
        libreoffice_path="soffice.exe",
        timeout_seconds=17,
        image_resolution=240,
        max_image_dimension=4096,
        max_image_pixels=8_000_000,
    ).render("book.xlsx", "Sheet1", "A1:B2", tmp_path / "x.png", {})

    assert captured["max_workers"] == 3
    assert captured["libreoffice_path"] == "soffice.exe"
    assert captured["timeout_seconds"] == 17
    assert captured["image_resolution"] == 240
    assert captured["max_image_dimension"] == 4096
    assert captured["max_image_pixels"] == 8_000_000


def test_renderer_bounds_concurrent_work(tmp_path: Path, monkeypatch) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_render(workbook, sheet, cell_range, output, **options):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        Image.new("RGB", (10, 10), "white").save(output)
        with lock:
            active -= 1
        return "libreoffice"

    monkeypatch.setattr("autotab.utils.rendering._render_with_libreoffice", fake_render)
    renderer = WindowRenderer(workers=2)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(
                renderer.render,
                "book.xlsx",
                "Sheet1",
                "A1:B2",
                tmp_path / f"{index}.png",
                {},
            )
            for index in range(6)
        ]
        for future in futures:
            future.result()

    assert max_active == 2


def test_renderer_records_timeout_and_falls_back(tmp_path: Path, monkeypatch) -> None:
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("soffice", 7)

    def fallback(workbook, sheet, cell_range, output):
        Image.new("RGB", (10, 10), "white").save(output)
        return "pillow_fallback"

    monkeypatch.setattr("autotab.utils.rendering._render_with_libreoffice", timed_out)
    monkeypatch.setattr("autotab.utils.rendering._render_with_pillow", fallback)

    metadata = WindowRenderer(timeout_seconds=7, max_image_dimension=5).render(
        "book.xlsx", "Sheet1", "A1:B2", tmp_path / "x.png", {}
    )

    assert metadata["renderer"] == "pillow_fallback"
    assert metadata["render_warning"] == "Viewport rendering timed out after 7 seconds"
    assert metadata["image_width"] == 5
    assert metadata["image_height"] == 5


def test_pdf_scale_respects_dimension_and_pixel_limits() -> None:
    scale = bounded_scale(1200, 900, 600, 1000, 500_000)

    assert 1200 * scale <= 1000
    assert 900 * scale <= 1000
    assert (1200 * scale) * (900 * scale) <= 500_000


def test_viewport_prompt_requires_worksheet_grounding() -> None:
    prompt = viewport_description_prompt("revenue", "Sheet1!B3", [])

    assert "source cell is Sheet1!B3" in prompt
    assert "grouped or merged header" in prompt
    assert "extends across rows or columns" in prompt
    assert "not visible instead of guessing" in prompt
    assert "later QA step" in prompt


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
