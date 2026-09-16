"""Exploration orchestration."""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

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
from .summary import KeywordSummarizer


class ExplorationPipeline:
    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self, workbooks: list[str], query: str) -> Path:
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
        run_id = f"{timestamp}-{os.getpid()}"
        store = ArtifactStore(self.config["runtime"]["artifact_root"], run_id)
        snapshots = [
            WorkbookLoader().load(p, self.config["exploration"]["include_hidden_sheets"])
            for p in workbooks
        ]
        models = self.config.get("models", {})
        request_timeout = self.config["runtime"]["request_timeout_seconds"]
        llm = (
            OpenAICompatibleClient(
                base_url=models["llm"]["base_url"],
                model=models["llm"]["model"],
                timeout=request_timeout,
                max_tokens=int(models["llm"]["max_tokens"]),
                extra_body=models["llm"].get("extra_body"),
            )
            if models.get("llm", {}).get("base_url")
            else None
        )
        vlm = (
            OpenAICompatibleClient(
                base_url=models["vlm"]["base_url"],
                model=models["vlm"]["model"],
                timeout=request_timeout,
                max_tokens=int(models["vlm"]["max_tokens"]),
                extra_body=models["vlm"].get("extra_body"),
            )
            if models.get("vlm", {}).get("base_url")
            else None
        )
        keywords = KeywordExtractor(llm).extract(query)
        store.write_json("keywords.json", keywords)
        vlm_query = query if self.config["exploration"]["send_query_to_vlm"] else None
        max_concurrent_findings = self.config["exploration"]["max_concurrent_findings"]
        rendering = self.config["rendering"]
        renderer = WindowRenderer(
            workers=int(rendering["workers"]),
            libreoffice_path=rendering.get("libreoffice_path"),
            timeout_seconds=float(rendering["timeout_seconds"]),
            image_resolution=int(rendering["image_resolution"]),
            max_image_dimension=int(rendering["max_image_dimension"]),
            max_image_pixels=int(rendering["max_image_pixels"]),
        )
        retriever = HybridCellRetriever(
            embedding=(
                OpenAICompatibleClient(
                    base_url=models["embedding"]["base_url"],
                    model=models["embedding"]["model"],
                    timeout=request_timeout,
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
        futures: list[Future[Evidence]] = []
        with ThreadPoolExecutor(max_workers=max_concurrent_findings) as executor:
            for snap in snapshots:
                for keyword in keywords:
                    hits = retriever.search(
                        keyword,
                        snap.cells,
                        self.config["exploration"]["similarity_threshold"],
                        self.config["exploration"]["max_cells_per_keyword"],
                    )
                    for hit in hits:
                        sheet_info = next(s for s in snap.sheets if s["name"] == hit.document.sheet)
                        rng = viewport(
                            hit.document.coordinate,
                            sheet_info["max_row"],
                            sheet_info["max_column"],
                            self.config["exploration"]["window_size"],
                        )
                        finding_id = f"{len(futures):04d}"
                        keyword_id = "-".join(
                            part for part in keyword.lower().replace("_", " ").split() if part
                        )
                        folder_name = f"{keyword_id}-{finding_id}"
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
                        futures.append(
                            executor.submit(
                                _extract_finding,
                                renderer,
                                store,
                                vlm,
                                snap.path,
                                folder_name,
                                rng,
                                keyword,
                                hit.document.sheet,
                                hit.document.coordinate,
                                cells,
                                vlm_query,
                            )
                        )
            evidence = [future.result() for future in futures]
        filtered = [
            EvidenceFilter().filter(
                e, query, self.config["exploration"].get("uncertain_evidence", "drop")
            )
            for e in evidence
        ]
        store.write_json("evidence.filtered.json", [e.to_dict() for e in filtered])
        aggregation = MarkdownWriter().write(run_id, filtered)
        store.write_text("aggregation.md", aggregation)
        summaries = KeywordSummarizer(llm).summarize(query, filtered)
        store.write_text(
            "exploration.md",
            MarkdownWriter().write_summary(summaries),
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


def _extract_finding(
    renderer: WindowRenderer,
    store: ArtifactStore,
    vlm: object | None,
    workbook: str,
    folder_name: str,
    cell_range: str,
    keyword: str,
    sheet: str,
    coordinate: str,
    cells: list[dict[str, Any]],
    query: str | None,
) -> Evidence:
    """Render and describe one independent cell finding."""
    folder = store.root / "viewports" / folder_name
    image_path = folder / "image.png"
    source_range = f"{sheet}!{coordinate}"
    meta = renderer.render(
        workbook,
        sheet,
        cell_range,
        image_path,
        {"anchor": coordinate},
    )
    prompt = viewport_description_prompt(keyword, source_range, cells, query=query)
    store.write_text(f"viewports/{folder_name}/prompt.txt", prompt)
    store.write_json(f"viewports/{folder_name}/metadata.json", meta)
    evidence_extractor = EvidenceExtractor(vlm)
    evidence = evidence_extractor.extract(keyword, source_range, cells, image_path, query=query)
    store.write_text(f"viewports/{folder_name}/response.txt", evidence_extractor.last_response)
    return evidence
