"""Exploration orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from openpyxl.utils.cell import range_boundaries

from ..artifacts.store import ArtifactStore
from ..models.client import OpenAICompatibleClient
from ..utils.ranges import viewport
from ..utils.rendering import WindowRenderer
from .evidence import Evidence, EvidenceExtractor
from .filtering import EvidenceFilter
from .ingestion import WorkbookLoader
from .keywords import KeywordExtractor
from .markdown import MarkdownWriter
from .prompts import viewport_description_prompt
from .retrieval import HybridCellRetriever


class ExplorationPipeline:
    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self, workbooks: list[str], query: str) -> Path:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        store = ArtifactStore(self.config["runtime"]["artifact_root"], run_id)
        snapshots = [
            WorkbookLoader().load(p, self.config["exploration"]["include_hidden_sheets"])
            for p in workbooks
        ]
        models = self.config.get("models", {})
        llm = (
            OpenAICompatibleClient(
                models["llm"]["base_url"],
                models["llm"]["model"],
                8,
                256,
                models["llm"].get("extra_body"),
            )
            if models.get("llm", {}).get("base_url")
            else None
        )
        vlm = (
            OpenAICompatibleClient(
                models["vlm"]["base_url"],
                models["vlm"]["model"],
                8,
                int(models["vlm"].get("max_tokens", 512)),
                models["vlm"].get("extra_body"),
            )
            if models.get("vlm", {}).get("base_url")
            else None
        )
        keywords = KeywordExtractor(llm).extract(query)
        store.write_json("keywords.json", keywords)
        evidence: list[Evidence] = []
        renderer = WindowRenderer()
        evidence_extractor = EvidenceExtractor(vlm)
        retriever = HybridCellRetriever(
            embedding=(
                OpenAICompatibleClient(
                    models["embedding"]["base_url"], models["embedding"]["model"]
                )
                if models.get("embedding", {}).get("base_url")
                and not self.config["retrieval"].get("allow_lexical_only_fallback", False)
                else None
            ),
            lexical_weight=self.config["retrieval"]["lexical_weight"],
            semantic_weight=self.config["retrieval"]["semantic_weight"],
        )
        if retriever.embedding is None and not self.config["retrieval"].get(
            "allow_lexical_only_fallback", False
        ):
            raise RuntimeError("An embedding provider is required for retrieval")
        for snap in snapshots:
            for keyword in keywords:
                hits = retriever.search(
                    keyword,
                    snap.cells,
                    self.config["exploration"]["similarity_threshold"],
                    self.config["exploration"]["max_cells_per_keyword"],
                )
                for index, hit in enumerate(hits):
                    sheet_info = next(s for s in snap.sheets if s["name"] == hit.document.sheet)
                    rng = viewport(
                        hit.document.coordinate,
                        sheet_info["max_row"],
                        sheet_info["max_column"],
                        self.config["exploration"]["window_size"],
                    )
                    finding_id = f"{len(evidence):04d}"
                    keyword_id = "-".join(
                        part for part in keyword.lower().replace("_", " ").split() if part
                    )
                    folder_name = f"{keyword_id}-{finding_id}"
                    folder = store.root / "viewports" / folder_name
                    meta = renderer.render(
                        snap.path,
                        hit.document.sheet,
                        rng,
                        folder / "image.png",
                        {"anchor": hit.document.coordinate},
                    )
                    c1, r1, c2, r2 = range_boundaries(rng)
                    cells = [
                        d.to_dict()
                        for d in snap.cells
                        if d.sheet == hit.document.sheet
                        and r1 <= int("".join(filter(str.isdigit, d.coordinate))) <= r2
                        and c1
                        <= __import__("openpyxl").utils.column_index_from_string(
                            "".join(filter(str.isalpha, d.coordinate))
                        )
                        <= c2
                    ]
                    prompt = viewport_description_prompt(
                        keyword,
                        f"{hit.document.sheet}!{hit.document.coordinate}",
                        cells,
                    )
                    store.write_text(f"viewports/{folder_name}/prompt.txt", prompt)
                    store.write_json(f"viewports/{folder_name}/metadata.json", meta)
                    evidence.append(
                        evidence_extractor.extract(
                            keyword,
                            f"{hit.document.sheet}!{hit.document.coordinate}",
                            cells,
                            folder / "image.png",
                        )
                    )
                    store.write_text(
                        f"viewports/{folder_name}/response.txt", evidence_extractor.last_response
                    )
        filtered = [
            EvidenceFilter().filter(
                e, query, self.config["exploration"].get("uncertain_evidence", "drop")
            )
            for e in evidence
        ]
        store.write_json("evidence.filtered.json", [e.to_dict() for e in filtered])
        store.write_text(
            "exploration.md",
            MarkdownWriter().write(run_id, filtered),
        )
        store.write_json(
            "manifest.json",
            {
                "run_id": run_id,
                "query": query,
                "send_query_to_vlm": self.config["exploration"]["send_query_to_vlm"],
            },
        )
        return store.root / "exploration.md"
