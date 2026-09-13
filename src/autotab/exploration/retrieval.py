"""Lexical, semantic, and hybrid cell retrieval."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from .ingestion import CellDocument


class EmbeddingModel(Protocol):
    def embed(self, texts: list[str] | tuple[str, ...]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class CellFinding:
    keyword: str
    document: CellDocument
    lexical_score: float
    semantic_score: float
    score: float

    def to_dict(self):
        return {**asdict(self), "document": self.document.to_dict()}


class LexicalIndex:
    def search(self, keyword: str, docs: Sequence[CellDocument]) -> dict[str, float]:
        query = set(re.findall(r"\w+", keyword.lower()))
        raw = {}
        for d in docs:
            tokens = set(re.findall(r"\w+", d.text.lower()))
            raw[d.coordinate + "|" + d.sheet] = len(query & tokens) / max(1, len(query))
        return raw


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = max(len(a), len(b))
    aa = list(a) + [0.0] * (n - len(a))
    bb = list(b) + [0.0] * (n - len(b))
    den = math.sqrt(sum(x * x for x in aa) * sum(x * x for x in bb))
    return sum(x * y for x, y in zip(aa, bb)) / den if den else 0.0


class HybridCellRetriever:
    def __init__(
        self,
        embedding: EmbeddingModel | None = None,
        lexical_weight: float = 0.45,
        semantic_weight: float = 0.55,
    ) -> None:
        self.embedding, self.lw, self.sw = (
            embedding,
            lexical_weight,
            semantic_weight,
        )

    def search(
        self, keyword: str, docs: Sequence[CellDocument], threshold: float, limit: int
    ) -> list[CellFinding]:
        lex = LexicalIndex().search(keyword, docs)
        if self.embedding is None:
            findings = [
                CellFinding(keyword, d, score, 0.0, score)
                for d in docs
                for score in [lex[d.coordinate + "|" + d.sheet]]
            ]
        else:
            vectors = self.embedding.embed([keyword] + [d.text for d in docs])
            sem = [cosine(vectors[0], v) for v in vectors[1:]]
            lo, hi = min(lex.values(), default=0), max(lex.values(), default=1)
            lrange = max(hi - lo, 1e-9)
            findings = [
                CellFinding(
                    keyword,
                    d,
                    (lex[d.coordinate + "|" + d.sheet] - lo) / lrange,
                    max(0.0, (s + 1) / 2),
                    self.lw * ((lex[d.coordinate + "|" + d.sheet] - lo) / lrange)
                    + self.sw * max(0.0, (s + 1) / 2),
                )
                for d, s in zip(docs, sem)
            ]
        return sorted(
            (f for f in findings if f.score >= threshold),
            key=lambda f: (-f.score, f.document.sheet, f.document.coordinate),
        )[:limit]
