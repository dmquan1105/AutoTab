"""Exploration orchestration."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openpyxl.utils.cell import range_boundaries

from ..artifacts.store import ArtifactStore, new_run_id
from ..models.client import client_from_config
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

    def run(self, workbooks: list[str], query: str, run_id: str | None = None) -> Path:
        """Explore ``workbooks`` for ``query`` and return the path of exploration.md.

        Args:
            workbooks: Source workbooks; never modified.
            query: The user question.
            run_id: The run folder to write into; QA passes its own so both modules
                land in one ``<artifact_root>/<run_id>/``. A new ID when omitted.
        """
        run_id = run_id or new_run_id()
        store = ArtifactStore(self.config["runtime"]["artifact_root"], run_id)
        snapshots = [
            WorkbookLoader().load(p, self.config["exploration"]["include_hidden_sheets"])
            for p in workbooks
        ]
        models = self.config.get("models", {})
        request_timeout = self.config["runtime"]["request_timeout_seconds"]
        llm = client_from_config(models.get("llm"), request_timeout)
        vlm = client_from_config(models.get("vlm"), request_timeout)
        keywords = KeywordExtractor(llm).extract(query)
        store.write_json("keywords.json", keywords)
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
        allow_lexical_fallback = bool(
            self.config["retrieval"].get("allow_lexical_only_fallback", False)
        )
        embedding = client_from_config(models.get("embedding"), request_timeout)
        if embedding is None and not allow_lexical_fallback:
            raise RuntimeError("An embedding provider is required for retrieval")
        retriever = HybridCellRetriever(
            embedding=embedding,
            lexical_weight=self.config["retrieval"]["lexical_weight"],
            semantic_weight=self.config["retrieval"]["semantic_weight"],
            allow_lexical_fallback=allow_lexical_fallback,
        )
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
                "retrieval": {
                    "semantic": embedding is not None and retriever.fallback_reason is None,
                    "lexical_only_fallback_reason": (
                        "no embedding provider configured"
                        if embedding is None
                        else retriever.fallback_reason
                    ),
                },
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
    prompt = viewport_description_prompt(keyword, source_range, cells)
    store.write_text(f"viewports/{folder_name}/prompt.txt", prompt)
    store.write_json(f"viewports/{folder_name}/metadata.json", meta)
    evidence_extractor = EvidenceExtractor(vlm)
    evidence = evidence_extractor.extract(keyword, source_range, cells, image_path)
    store.write_text(f"viewports/{folder_name}/response.txt", evidence_extractor.last_response)
    return evidence
