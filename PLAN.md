# AutoTab Exploration Plan

## Objective

Implement only the Exploration phase first. Given `.xlsx` workbooks and a natural-language query, produce a grounded `exploration.md` containing relevant keywords, keyword source ranges, and one combined viewport-grounded description per finding. QA, agent execution, operators, Observe, and validation loops are deferred.

## Constraints

- Python 3.11+, managed and run with `uv`.
- Follow `CODING_STANDARDS.md`: SOLID boundaries, KISS/YAGNI, typed Python, Google-style docstrings, Black (100 columns), Ruff, isort, mypy, and 80% coverage.
- Start as a local modular package; avoid services and a persistent vector database.
- Treat workbook content and model output as untrusted data. Secrets come from environment variables, never YAML or artifacts.

## Flow

```text
query + workbooks -> ingest -> keywords -> cell search (>= t, max k)
                  -> one clipped n x n viewport per cell finding -> VLM evidence
                  -> LLM relevance filter -> exploration.md
```

The final output is Markdown. Each finding has one natural-language description grounded in the visible viewport; it describes the keyword in context, relations to other data, and notable details. The displayed range is the keyword source range, while the viewport is rendering context. The original query is not included in `exploration.md`.

## Step-by-Step Plan

### 1. `uv` project and CLI

Create `pyproject.toml`, package code under `src/autotab/`, and tests under `tests/`. Define an `autotab` CLI with one initial command:

```bash
uv sync
uv run autotab explore --config config.yaml \
  --workbook samples/sample.xlsx \
  --query "Find revenue by region"
uv run pytest
uv run ruff check .
uv run black --check .
uv run isort --check-only .
uv run mypy src
```

The command writes `outputs/<run_id>/exploration/exploration.md` and its related artifacts. `outputs/` is gitignored, and the `exploration/` phase folder keeps these artifacts separate from future QA output. Intermediate JSON snapshots may be stored for debugging, but `exploration.md` is the only user-facing exploration output. Invalid arguments/configuration must fail with actionable messages.

### 2. Workbook ingestion

Implement a `WorkbookLoader` abstraction with an `openpyxl` backend. Preserve workbook hash, sheet visibility/dimensions, merged ranges, formulas, raw/displayed values, number formats, and optional `structure.yaml` metadata. Emit typed `CellDocument` records with location, searchable text, inherited headers, and table metadata. Keep blank cells in snapshots but do not index them independently.

Add focused unit tests for workbook loading and viewport range handling using one small sample workbook.

### 3. Keyword extraction

Add a versioned LLM prompt that returns only a list of worksheet-search keywords covering entities, metrics, dimensions, time expressions, filters, and operations. Normalize case/whitespace, preserve meaningful phrases, and deduplicate keywords. Provide deterministic phrase/token fallback when the model is unavailable.

Add focused unit tests for keyword normalization and the deterministic fallback.

### 4. Hybrid cell retrieval

Use both retrieval signals for every keyword:

1. **Lexical search** handles exact labels, identifiers, numbers, dates, and formula tokens.
2. **Semantic search** embeds each keyword and every `CellDocument`, then ranks by cosine similarity.
3. **Score fusion** normalizes both scores to `[0, 1]` and combines them with configured weights. Store component scores and the final score for debugging and calibration.

Define separate adapters for the lexical index and embedding model, plus a `HybridCellRetriever` that owns fusion and ranking. The initial embedding backend may be a hosted embedding model or a local sentence-transformer selected in config; changing providers must not change the pipeline.

For every keyword, search all eligible sheets, keep only final scores `>= similarity_threshold`, and return at most `max_cells_per_keyword` findings with stable tie-breaking. If the embedding provider fails, fail the retrieval stage by default rather than silently changing ranking semantics; an explicit config fallback may permit lexical-only runs and must be recorded in the manifest.

Add focused unit tests for hybrid score fusion, threshold filtering, and top-`k` selection.

### 5. Window planning and rendering

Create exactly one clipped odd-sized viewport around each hit; never cross sheet boundaries. Treat every viewport independently; never combine findings. Each cell finding keeps its own image, snapshot, metadata, and VLM request.

Adapt the rendering pipeline from [TableAgent](https://github.com/HmmOrange/TableAgent):

1. Copy the source workbook into a temporary directory; never modify the original.
2. Open the copy with `openpyxl`, activate the finding's sheet, hide other sheets, and set the viewport range as the worksheet `print_area`.
3. Enable row/column coordinates and gridlines, remove page margins, and fit the selected range to one page.
4. Ensure every row and column in the viewport has an explicit visible dimension. Estimate display width using Unicode East Asian width rules, add a small width padding, clamp widths between configured minimum and maximum values, and preserve existing workbook widths when they are already sufficient.
5. Preserve spreadsheet formatting from the source copy: fonts, fills, borders, alignment, merged cells, row heights, column widths, number formats, dates, percentages, currencies, and formula display values. Do not flatten the viewport into plain text before LibreOffice renders it.
6. For long content, honor existing `wrap_text` and explicit line breaks. Otherwise enable wrapping when the display width exceeds the available cell width. Calculate row height from wrapped line count and font size, including the width of merged-cell spans, so text remains visible without changing the source workbook.
7. Ensure merged ranges are preserved. Read the value from the merged range's top-left cell, use the full merged width when calculating wrapping, and never treat covered `MergedCell` objects as independent values.
8. Use workbook-native number formats and alignment during rendering; do not replace formatted values with Python string representations in the temporary workbook.
9. Use LibreOffice headless with a separate temporary user profile to export the prepared workbook to PDF.
10. Render the first PDF page to PNG with `pypdfium2`. Isolate PDFium work in a subprocess because its process-global API is not thread-safe.
11. Trim outer white space with Pillow, then resize only when the configured maximum dimension or pixel count is exceeded.
12. Write render metadata containing workbook hash, sheet, range, anchor cell, image dimensions, renderer, resolution, and coordinate visibility.

Do not tile, split, or overlap images. One cell finding always produces one complete PNG viewport. Store canonical parallel snapshots:

```text
viewports/<keyword>-<finding_id>/image.png
viewports/<keyword>-<finding_id>/metadata.json
viewports/<keyword>-<finding_id>/prompt.txt
```

Add one rendering fixture test that checks a viewport image is produced and its metadata contains the expected sheet and range.

The CLI must perform a rendering preflight and return an actionable error when LibreOffice is unavailable. Rendering dependencies are `openpyxl`, LibreOffice, `pypdfium2`, and Pillow.

### Concurrency strategy

Use bounded concurrency only for independent cell findings. The pipeline must preserve deterministic output ordering even when work completes out of order.

- **Workbook ingestion:** load each workbook once per run. Do not let concurrent tasks mutate a shared `openpyxl` workbook; use read-only snapshots or one isolated workbook copy per render.
- **Keyword retrieval:** run lexical and semantic searches for different keywords concurrently when the index/model backend supports it. Build the index once and treat it as immutable during a run.
- **Viewport rendering:** render findings concurrently only through a semaphore. Each render gets its own temporary workbook copy, LibreOffice user profile, PDF path, PNG path, and metadata path. Never share a mutable workbook, temporary profile, or output filename.
- **PDFium:** keep the TableAgent safety rule: one PDFium call per subprocess. The parent process limits the number of render subprocesses with `render_workers` so CPU and memory usage remain bounded.
- **LLM/VLM calls:** use async requests bounded by `exploration.max_concurrent_findings` and the provider's rate limits. Embedding requests may additionally use `embedding_concurrency`. Apply request timeouts and bounded retries with jitter.
- **Artifact writes:** workers write only to their unique finding directory. A single coordinator writes the final `exploration.md`, manifest, and ordered trace after all results are collected.
- **Failure behavior:** one finding failure should be recorded and should not cancel unrelated findings. Cancel remaining work only when the run-level timeout or explicit cancellation is reached.

Use `asyncio` for model/network-bound work and subprocess/process isolation for rendering. Do not use unbounded `asyncio.gather`, threads around a shared `openpyxl` workbook, or concurrent writes to the same artifact file. Add one focused test that the concurrency limit is respected.

### 6. VLM evidence extraction

By default, send only the keyword and image to the VLM. Do not serialize the source range or viewport cell records into the prompt. Sending the original user query is configurable and defaults to disabled to reduce prompt leakage and keep viewport analysis focused. When enabled, include the query explicitly and record that choice in the manifest. The keyword and source range already belong to the cell finding and must be attached by the pipeline, not regenerated by the VLM. Ask the VLM to return one natural-language description of the keyword in the context of the visible viewport, including relations to other data and notable details. Cache successful responses by input/model/prompt hashes. Validate the returned text and grounding against the in-memory viewport cell records; retry one malformed response with the validation error, then continue with a typed warning.

Use one mocked VLM test for a valid response and one for a rejected response.

### 7. Deduplication and relevance filtering

Evaluate every finding independently; do not combine findings, even when their source ranges are equal or adjacent. Ask an LLM to classify each finding as `keep`, `drop`, or `uncertain` relative to the original query. Treat `uncertain` as `drop` by default, so uncertain findings do not appear in `exploration.md`. Still record every uncertain and dropped finding, its reason, source range, and model response in the run trace and filtered snapshot. The pruning model cannot create new facts, ranges, or relations; retained descriptions and relations must remain natural-language statements tied to that finding's source range.

Add one focused test that dropped findings do not appear in Markdown but remain in the trace.

### 8. Deterministic Markdown output

Render Markdown deterministically without an LLM:

```markdown
## revenue

- range: `Data_Synthesis!B4`
- description: The image contains monthly revenue columns.
```

Include run ID, workbook hashes, config snapshot, and model/prompt versions. Do not include the original query in `exploration.md`. Order keywords by query order, then evidence by relevance and source location. Repeated inputs must produce identical Markdown apart from explicitly marked timestamps.

## `config.yaml` Proposal

Commit `config.example.yaml`; users copy it to `config.yaml`. Load credentials from environment variables.

```yaml
project:
  name: autotab
  python_version: "3.11"

models:
  llm:
    profile: default_llm
    model: gpt-5-mini
    temperature: 0
    max_tokens: 4000
  vlm:
    profile: default_vlm
    model: gpt-5-mini
    temperature: 0
    max_tokens: 4000
  embedding:
    profile: default_embedding
    model: text-embedding-3-small
    dimensions: 1536

exploration:
  similarity_threshold: 0.72
  max_cells_per_keyword: 8
  max_concurrent_findings: 8
  window_size: 9
  send_query_to_vlm: false
  include_hidden_sheets: false
  uncertain_evidence: drop

retrieval:
  lexical_weight: 0.45
  semantic_weight: 0.55
  allow_lexical_only_fallback: false

runtime:
  artifact_root: outputs
  request_timeout_seconds: 60
  max_retries: 1
  max_input_file_mb: 100

rendering:
  backend: libreoffice
  libreoffice_path: null       # use PATH or standard install locations
  timeout_seconds: 60
  image_resolution: 600
  show_coordinates: true
  min_cell_width: 10
  max_cell_width: 50
  cell_width_padding: 3
  max_image_dimension: 8192
  max_image_pixels: 33554432
  trim_padding: 6
```

The pipeline maps responsibilities to these shared providers: keyword extraction and relevance filtering use `models.llm`, viewport evidence extraction uses `models.vlm`, and semantic retrieval uses `models.embedding`. Provider adapters remain generic and do not encode feature-specific configuration keys.

Validate threshold `[0, 1]`, positive `k`, positive odd `window_size`, positive `exploration.max_concurrent_findings`, boolean `exploration.send_query_to_vlm`, retrieval weights in `[0, 1]` whose sum is `1`, positive rendering/concurrency limits and timeouts, and `uncertain_evidence` as either `keep` or `drop` (default: `drop`). Resolve LibreOffice from `libreoffice_path`, `soffice`/`libreoffice` on `PATH`, then standard OS install locations. Document environment overrides such as `AUTOTAB_MODELS__EMBEDDING__MODEL` and `AUTOTAB_RENDERING__LIBREOFFICE_PATH`.

## Planned Structure

### Repository structure

```text
AutoTab/
  pyproject.toml
  uv.lock
  config.yaml                 # local, ignored; never commit secrets
  config.example.yaml        # safe template
  README.md
  CODING_STANDARDS.md
  src/
    autotab/
      __init__.py
      cli.py                  # uv entry point: autotab explore
      config.py               # YAML loading and validation
      models/                 # LLM/VLM provider adapters
        __init__.py
        client.py
        prompts.py
      exploration/            # Exploration phase only
        __init__.py
        pipeline.py            # phase orchestration
        ingestion.py           # workbook and structure.yaml loading
        keywords.py            # query keyword extraction and fallback
        retrieval.py           # cell documents and lexical index
      windows.py             # one n x n viewport per cell finding
        evidence.py            # VLM extraction and grounding checks
        filtering.py           # deduplication and LLM relevance filter
        markdown.py            # deterministic Markdown writer
      qa/                     # reserved for the later QA phase
        __init__.py
      utils/                  # reusable, phase-neutral helpers
        __init__.py
        rendering.py           # spreadsheet/image rendering utilities
        ranges.py              # A1 parsing and viewport clipping helpers
        hashing.py             # content and cache-key hashes
        files.py               # safe paths and artifact file operations
      artifacts/              # run directories, manifests, snapshots
        __init__.py
        store.py               # writes only under configured outputs/ root
  tests/
    exploration/
      unit/
      integration/
    qa/                      # reserved for the later QA phase
  samples/
  outputs/                    # generated artifacts, ignored by git
```

Responsibilities stay separate: `exploration/pipeline.py` coordinates; it should not contain workbook parsing, model-provider details, rendering logic, or Markdown formatting. `utils/` contains reusable mechanics only and must not depend on Exploration or QA business rules. Keep each module cohesive, public functions typed/documented, exceptions specific, and comments limited to non-obvious reasoning.

### Exploration runtime

```text
ExploreCommand
  -> exploration.pipeline.ExplorationPipeline
       -> exploration.ingestion.WorkbookLoader
       -> exploration.keywords.KeywordExtractor
       -> exploration.retrieval.CellRetriever
       -> exploration.windows.WindowPlanner
       -> utils.rendering.WindowRenderer
       -> exploration.evidence.EvidenceExtractor (VLM)
       -> exploration.filtering.EvidenceFilter (LLM)
       -> exploration.markdown.MarkdownWriter
       -> artifacts.store.ArtifactStore
```

Each arrow is a small Python interface. Initial implementations are local and deterministic where possible; model calls are injected adapters so tests can use fakes and future providers can be added without changing orchestration.

### Run artifacts (gitignored output)

```text
  outputs/<run_id>/
    exploration/              # Exploration phase artifacts
      request.yaml
      config.snapshot.yaml
      manifest.json
      keywords.json
      hits.json
      viewports/<keyword>-<finding_id>/
        image.png
        metadata.json
        prompt.txt
      evidence.raw.json
      evidence.filtered.json
      evidence.dropped.json   # dropped/uncertain findings with reasons
      events.jsonl             # model calls, decisions, warnings, and failures
      exploration.md           # final user-facing output
```

`artifacts.store.ArtifactStore` is the only module allowed to create generated output. It resolves and validates paths beneath `outputs/<run_id>/exploration/`; future phases will receive separate subfolders such as `outputs/<run_id>/qa/`. No generated files are written into `src/`, `samples/`, or the repository root. JSON files are internal debugging/replay artifacts and must not replace or alter the requested Markdown output.

### Markdown output

```markdown
# Exploration

## revenue

- range: `Data_Synthesis!B4`
- description: Revenue is organized by month and region, with the quarterly total highlighted.

## region

- range: `Data_Synthesis!A5:A7`
- description: The range lists the regions represented in the table and relates each region to adjacent revenue values.
```

`description` is a complete natural-language sentence (or sentences) covering the keyword in viewport context, related data, and notable details. Ranges use workbook-qualified sheet notation and identify the keyword source cell/range; every description must be supported by the rendered viewport. Omit a keyword section when no hit reaches the threshold; record that omission in the manifest.

## Delivery and Exit Criteria

1. Foundation/config/CLI/artifact manifest.
2. Ingestion and one rendering fixture.
3. Keywords and fallback.
4. Retrieval and one-viewport-per-finding planning.
5. VLM validation and cache.
6. Pruning and Markdown.
7. One end-to-end CLI test and quality checks.

Complete Exploration when the sample workbooks produce grounded deterministic Markdown with natural-language descriptions and relations, and these commands pass:

```bash
uv run pytest --cov=src/autotab --cov-fail-under=80
uv run ruff check .
uv run black --check .
uv run isort --check-only .
uv run mypy src
```

## Deferred

QA agent, context management, code execution, operators, Observe, validation feedback loops, multi-agent orchestration, distributed workers, persistent vector storage, and write-capable workbook operations.

