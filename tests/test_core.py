from pathlib import Path

from autotab.config import ConfigError, load_config
from autotab.exploration.evidence import EvidenceExtractor
from autotab.exploration.filtering import EvidenceFilter
from autotab.exploration.ingestion import WorkbookLoader
from autotab.exploration.keywords import KeywordExtractor, normalize_phrase
from autotab.exploration.markdown import MarkdownWriter
from autotab.exploration.pipeline import ExplorationPipeline
from autotab.exploration.retrieval import HybridCellRetriever
from autotab.utils.ranges import parse_cell, viewport
from autotab.utils.rendering import WindowRenderer


def test_config_and_ranges() -> None:
    config = load_config("config.example.yaml")
    assert config["retrieval"]["lexical_weight"] == 0.45
    assert parse_cell("C4") == (4, 3)
    assert viewport("A1", 3, 3, 9) == "A1:C3"


def test_invalid_config() -> None:
    try:
        load_config("does-not-exist.yaml")
    except ConfigError:
        raise AssertionError("missing config should use defaults")


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
        def complete(self, prompt: str) -> str:
            assert "viewport" in prompt
            return '{"description": "Revenue is shown in the viewport."}'

    evidence = EvidenceExtractor(FakeVLM()).extract(
        "revenue", "Sheet1!A1:B2", [{"displayed_value": "Revenue"}]
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
