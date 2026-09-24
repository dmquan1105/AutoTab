"""Lexical-only fallback applies when embedding fails, never instead of it.

``PLAN.md``: if the embedding provider fails, fail retrieval by default; an explicit
config fallback may permit lexical-only runs and must be recorded in the manifest.
"""

from collections.abc import Sequence

import pytest

from autotab.exploration.ingestion import CellDocument
from autotab.exploration.retrieval import HybridCellRetriever
from autotab.models.client import ModelResponseError

DOCS = [
    CellDocument("book.xlsx", "Sheet", "A1", "Revenue", "Revenue", "Revenue", None, "General"),
    CellDocument("book.xlsx", "Sheet", "A2", "Cost", "Cost", "Cost", None, "General"),
]


class WorkingEmbedding:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str] | tuple[str, ...]) -> Sequence[Sequence[float]]:
        self.calls += 1
        return [[1.0, 0.0] if "revenue" in text.lower() else [0.0, 1.0] for text in texts]


class DeadEmbedding:
    def embed(self, texts: list[str] | tuple[str, ...]) -> Sequence[Sequence[float]]:
        raise ModelResponseError("embedding request could not reach http://localhost:8001/v1")


def test_a_working_embedding_is_used_even_when_fallback_is_allowed() -> None:
    embedding = WorkingEmbedding()
    retriever = HybridCellRetriever(embedding, allow_lexical_fallback=True)

    findings = retriever.search("revenue", DOCS, threshold=0.0, limit=5)

    assert embedding.calls == 1
    assert findings[0].semantic_score > 0
    assert retriever.fallback_reason is None


def test_a_failing_embedding_falls_back_to_lexical_and_says_why() -> None:
    retriever = HybridCellRetriever(DeadEmbedding(), allow_lexical_fallback=True)

    findings = retriever.search("revenue", DOCS, threshold=0.5, limit=5)

    assert [finding.document.coordinate for finding in findings] == ["A1"]
    assert all(finding.semantic_score == 0.0 for finding in findings)
    assert retriever.fallback_reason is not None
    assert "localhost:8001" in retriever.fallback_reason


def test_a_failing_embedding_is_fatal_unless_fallback_is_allowed() -> None:
    retriever = HybridCellRetriever(DeadEmbedding(), allow_lexical_fallback=False)

    with pytest.raises(ModelResponseError):
        retriever.search("revenue", DOCS, threshold=0.0, limit=5)
