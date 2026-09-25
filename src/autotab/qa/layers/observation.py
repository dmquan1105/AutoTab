"""Pure normalization of raw tool, sandbox, and exploration results.

The output is bounded evidence for the next turn: exact values with coordinates,
provenance, uncertainties, and the smallest question needed to make progress. It
never judges correctness; see ``specs/layers/observation.md``.

A read that fits ``max_inline_cells`` is shown whole, as a grid with one line per
worksheet row. A larger read is shown as a deterministic preview -- header rows, the
first and last rows, column profiles over every row, and the structural markers a
spreadsheet hides in its middle -- because the prompt is not where bulk data lives:
code loads the full range in the sandbox, and only small results come back.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas import (
    JsonValue,
    Observation,
    ObservationSource,
    ProvenanceKind,
    ProvenanceReference,
    ResultState,
    SandboxResult,
    ToolResult,
)

DEFAULT_MAX_CHARS = 20_000
DEFAULT_MAX_INLINE_CELLS = 200
EXPLORATION_FILE = "evidence.filtered.json"
EXPLORATION_CAVEAT = (
    "Exploration evidence is a lead from a small window around one cell and needs an "
    "exact read; the table may extend beyond the rows it mentions."
)
_HEADER_ROWS = 3  # a range read's first rows are shown, as they usually hold headers
_HEAD_ROWS = 5
_TAIL_ROWS = 5  # totals and footnotes live at the bottom, so the tail is always shown
_MARKERS_SHOWN = 20
_SAMPLES_SHOWN = 3
_SOURCE_RANGE = re.compile(
    r"^(?:'(?P<quoted>(?:[^']|'')+)'|(?P<plain>[^!]+))!(?P<range>\$?[A-Za-z]+\$?\d+"
    r"(?::\$?[A-Za-z]+\$?\d+)?)$"
)


class ExplorationError(ValueError):
    """Raised when the exploration artifact is missing or malformed."""


class _Unknown:
    """A formula whose cached value is missing: unknown, never zero."""

    def __repr__(self) -> str:
        return "?"


_UNKNOWN = _Unknown()


@dataclass(frozen=True)
class _Table:
    """A read reduced to rows of values, whatever tool produced it."""

    title: str
    columns: list[str]
    rows: list[tuple[int, list[Any]]]  # (worksheet row number, values)
    header_rows: int
    merged: list[str]
    formulas: list[str]
    formula_rows: list[int]


def _bounded(facts: list[str], max_chars: int) -> tuple[list[str], int]:
    """Keep whole facts, in order, until the character budget is spent."""
    kept: list[str] = []
    used = 0
    for fact in facts:
        if used + len(fact) > max_chars:
            break
        kept.append(fact)
        used += len(fact)
    return kept, len(facts) - len(kept)


def _text(value: Any) -> str:
    """Render one value for a grid line; blank stays empty, separators are escaped."""
    if value is None:
        return ""
    if value is _UNKNOWN:
        return "?"
    text = value if isinstance(value, str) else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n")


def _line(row: int, values: list[Any]) -> str:
    return f"{row} | " + " | ".join(_text(value) for value in values)


def _capped(items: list[Any]) -> str:
    shown = ", ".join(str(item) for item in items[:_MARKERS_SHOWN])
    hidden = len(items) - _MARKERS_SHOWN
    return shown + (f" (+{hidden} more)" if hidden > 0 else "")


def _table_from_cells(payload: dict[str, Any]) -> _Table:
    sheet = payload.get("sheet", "?")
    grid = [row for row in payload.get("cells") or [] if row]
    columns = [re.sub(r"\d+", "", str(cell["coordinate"])) for cell in grid[0]] if grid else []
    rows: list[tuple[int, list[Any]]] = []
    merged: list[str] = []
    formulas: list[str] = []
    formula_rows: list[int] = []
    for cells in grid:
        number = int(re.sub(r"[A-Za-z]+", "", str(cells[0]["coordinate"])))
        values: list[Any] = []
        for cell in cells:
            label = cell.get("merged_with")
            if label and label not in merged:
                merged.append(label)
            formula = cell.get("formula")
            if formula is None:
                values.append(cell.get("value"))
                continue
            cached = cell.get("displayed")
            known = "no cached value" if cached is None else f"cached {cached!r}"
            formulas.append(f"{sheet}!{cell['coordinate']}: formula {formula}, {known}")
            if number not in formula_rows:
                formula_rows.append(number)
            values.append(_UNKNOWN if cached is None else cached)
        rows.append((number, values))
    title = f"{sheet}!{payload.get('range', '')}"
    return _Table(title, columns, rows, _HEADER_ROWS, merged, formulas, formula_rows)


def _table_from_frame(payload: dict[str, Any]) -> _Table:
    header_row = payload.get("header_row")
    first = 1 if header_row is None else int(header_row) + 1
    rows = [(first + offset, list(row)) for offset, row in enumerate(payload.get("rows") or [])]
    title = f"{payload.get('sheet', '?')}!{payload.get('range', '')}"
    columns = [str(column) for column in payload.get("columns") or []]
    return _Table(title, columns, rows, 0, [], [], [])


def _profile(column: str, values: list[Any]) -> str:
    """Summarize one column over every data row, including the ones not shown."""
    present = [v for v in values if v is not None and v != "" and v is not _UNKNOWN]
    numbers = [v for v in present if isinstance(v, (int, float)) and not isinstance(v, bool)]
    texts = [str(v) for v in present if not isinstance(v, (int, float)) or isinstance(v, bool)]
    parts = [f"{len(present)} non-blank of {len(values)}"]
    if numbers:
        parts.append(f"{len(numbers)} numbers, min {min(numbers)} max {max(numbers)}")
    if texts:
        distinct = list(dict.fromkeys(texts))
        samples = ", ".join(repr(text) for text in distinct[:_SAMPLES_SHOWN])
        parts.append(f"{len(distinct)} distinct texts, e.g. {samples}")
    return f"column {column}: " + "; ".join(parts)


def _render(table: _Table, max_inline_cells: int) -> tuple[list[str], str | None]:
    """Return the facts for a table, and a preview note when rows were left out."""
    if not table.rows:
        return [], None
    header = f"{table.title} (row | {' | '.join(table.columns)})"
    merged = [f"merged: {', '.join(table.merged)}"] if table.merged else []
    if len(table.rows) * max(1, len(table.columns)) <= max_inline_cells:
        lines = [_line(number, values) for number, values in table.rows]
        return [header, *lines, *merged, *table.formulas], None

    lead = table.rows[: table.header_rows + _HEAD_ROWS]
    tail = table.rows[max(len(lead), len(table.rows) - _TAIL_ROWS) :]
    omitted = len(table.rows) - len(lead) - len(tail)
    facts = [
        (
            f"{table.title}: {len(table.rows)} rows × {len(table.columns)} columns; "
            f"preview only, {omitted} rows omitted"
        ),
        header,
        *(_line(number, values) for number, values in lead),
        f"… {omitted} rows omitted …",
        *(_line(number, values) for number, values in tail),
        *merged,
    ]
    blank = [n for n, values in table.rows if all(v is None or v == "" for v in values)]
    if blank:
        facts.append(f"blank rows: {_capped(blank)}")
    if table.formula_rows:
        facts.append(f"rows with formulas: {_capped(table.formula_rows)}")
    facts.extend(table.formulas[:_MARKERS_SHOWN])
    # Profiles cover the whole range read, header rows included: how many header rows
    # a table has is not known here, and skipping a guessed number would silently drop
    # data rows. Text headers do not move a numeric column's min or max.
    facts.append("Profiles over every row of the range, header rows included:")
    rows = [values for _, values in table.rows]
    for index, column in enumerate(table.columns):
        facts.append(_profile(column, [row[index] for row in rows if index < len(row)]))
    note = (
        f"Preview only: {omitted} of {len(table.rows)} rows are not shown. To use every "
        "row, load the range in a code action (wb.range(...) or wb.sheet(...)) and compute "
        "there; print only small results."
    )
    return facts, note


def _attribute_facts(payload: dict[str, Any], max_inline_cells: int) -> list[str]:
    sheet = payload.get("sheet", "?")
    cells = [cell for row in payload.get("cells") or [] for cell in row]
    facts = [
        f"{sheet}!{cell['coordinate']}: "
        + json.dumps({k: v for k, v in cell.items() if k != "coordinate"}, ensure_ascii=False)
        for cell in cells[:max_inline_cells]
    ]
    if len(cells) > max_inline_cells:
        facts.append(f"… {len(cells) - max_inline_cells} more cells; request fewer at a time")
    return facts


def _search_facts(payload: dict[str, Any], max_inline_cells: int) -> list[str]:
    matches = payload.get("matches") or []
    facts = []
    for match in matches[:max_inline_cells]:
        text = f"{match.get('reference')}: raw {match.get('value')!r}"
        if match.get("displayed") != match.get("value"):
            text += f", displayed {match.get('displayed')!r}"
        facts.append(text + " (workbook value)")
    if len(matches) > max_inline_cells:
        facts.append(
            f"… {len(matches) - max_inline_cells} more matches; narrow the search or count "
            "them in code"
        )
    return facts


def _tool_facts(result: JsonValue, max_inline_cells: int) -> tuple[list[str], str | None]:
    if not isinstance(result, dict):
        return ([] if result is None else [json.dumps(result, ensure_ascii=False)]), None
    if "cells" in result:
        grid: Any = result.get("cells") or []  # the tool's own shape: rows of cell dicts
        first = next((row[0] for row in grid if row), None)
        if isinstance(first, dict) and "displayed" not in first:
            return _attribute_facts(result, max_inline_cells), None
        return _render(_table_from_cells(result), max_inline_cells)
    if "rows" in result:
        return _render(_table_from_frame(result), max_inline_cells)
    if "matches" in result:
        return _search_facts(result, max_inline_cells), None
    return [json.dumps(result, ensure_ascii=False)], None


def _where(result: ToolResult) -> str:
    payload = result.result if isinstance(result.result, dict) else {}
    sheet, rng = payload.get("sheet"), payload.get("range")
    if sheet and rng:
        return f"{sheet}!{rng}"
    return str(result.arguments.get("range_ref") or result.arguments.get("sheet_name") or "")


def normalize_tool(
    result: ToolResult,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_inline_cells: int = DEFAULT_MAX_INLINE_CELLS,
) -> Observation:
    """Normalize one tool result without changing any value it reports.

    Args:
        result: The raw tool result.
        max_chars: Hard cap on the characters of all facts together.
        max_inline_cells: Reads up to this many cells are shown whole; larger reads
            become a preview and the observation is marked truncated.
    """
    where = _where(result)
    summary = f"{result.tool_name} {where}".strip()
    if result.state is ResultState.ERROR:
        return Observation(
            source=ObservationSource.TOOL,
            status=ResultState.ERROR,
            summary=summary,
            error=result.error,
            provenance=list(result.provenance),
            uncertainties=[f"{result.tool_name} failed: {result.error}"],
            next_question=f"How must the {result.tool_name} call change so that it succeeds?",
        )
    rendered, preview = _tool_facts(result.result, max_inline_cells)
    facts, dropped = _bounded(rendered, max_chars)
    uncertainties: list[str] = []
    status = result.state
    next_question = None
    if preview is not None:
        status = ResultState.TRUNCATED
        uncertainties.append(preview)
        next_question = "Which computation over the full range, run in code, answers this?"
    if dropped or result.state is ResultState.TRUNCATED:
        status = ResultState.TRUNCATED
        uncertainties.append(
            f"Observation truncated: {len(facts)} facts shown, {dropped} omitted; the "
            "full result is in the action artifact. Read a narrower range if needed."
        )
        next_question = next_question or "Which narrower range contains the values still needed?"
    if status is ResultState.EMPTY:
        uncertainties.append(f"{summary} returned no values.")
        next_question = (
            f"The read of {where or 'this target'} was empty; which range holds the data?"
        )
    return Observation(
        source=ObservationSource.TOOL,
        status=status,
        summary=summary,
        facts=facts,
        provenance=list(result.provenance),
        uncertainties=uncertainties,
        next_question=next_question,
    )


def normalize_sandbox(
    result: SandboxResult, *, max_chars: int = DEFAULT_MAX_CHARS, code_path: str | None = None
) -> Observation:
    """Normalize one sandbox run; every value it produced is agent-computed.

    ``code_path`` is where the run's trace keeps the code, cited as provenance.
    """
    summary = f"code run {result.execution_id}"
    provenance = [
        *(
            [ProvenanceReference(kind=ProvenanceKind.ARTIFACT, artifact_path=code_path)]
            if code_path
            else []
        ),
        *(
            ProvenanceReference(kind=ProvenanceKind.COMPUTATION, computation_id=computation_id)
            for computation_id in result.computation_ids
        ),
        # Ranges the code read through wb, so an answer citing them can find this turn.
        *(
            ProvenanceReference(
                kind=ProvenanceKind.WORKBOOK,
                workbook_id=str(call["workbook_id"]),
                sheet=str(call["sheet_name"]),
                range_ref=str(call["range_ref"]),
            )
            for call in result.facade_calls
            if call.get("method") in ("range", "attributes")
            and call.get("workbook_id")
            and call.get("sheet_name")
            and call.get("range_ref")
        ),
    ]
    reads = [
        f"workbook read: {call.get('method')}("
        + ", ".join(f"{k}={v!r}" for k, v in call.items() if k not in {"method"} and v is not None)
        + ")"
        for call in result.facade_calls
    ]
    if result.state is ResultState.ERROR:
        return Observation(
            source=ObservationSource.SANDBOX,
            status=ResultState.ERROR,
            summary=summary,
            facts=reads,
            error=result.error,
            provenance=provenance,
            uncertainties=[f"The code failed: {result.error}"],
            next_question="How must the code change so that it runs?",
        )
    facts = list(reads)
    if result.stdout.strip():
        facts.append(f"stdout (agent-computed): {result.stdout.strip()}")
    if result.result is not None:
        facts.append(f"result (agent-computed): {json.dumps(result.result, ensure_ascii=False)}")
    if result.computation_ids:
        facts.append(f"recorded computations: {', '.join(result.computation_ids)}")
    if result.stderr.strip():
        facts.append(f"stderr: {result.stderr.strip()}")
    facts, dropped = _bounded(facts, max_chars)
    uncertainties: list[str] = []
    status = result.state
    if dropped or result.state is ResultState.TRUNCATED:
        status = ResultState.TRUNCATED
        uncertainties.append("Output truncated; print or bind a smaller `result`.")
    if isinstance(result.result, dict) and result.result.get("preview") is True:
        # The run completed; only its display is partial, so the raw state stays as is.
        status = ResultState.TRUNCATED
        uncertainties.append(
            "The result is a preview of both ends; the full value is still bound in the "
            "session. Compute on it in code and return only what you need."
        )
    next_question = None
    if status is ResultState.EMPTY:
        uncertainties.append("The code produced no output.")
        next_question = "What value should the code bind to `result` or print?"
    return Observation(
        source=ObservationSource.SANDBOX,
        status=status,
        summary=summary,
        facts=facts,
        result=result.result,
        provenance=provenance,
        uncertainties=uncertainties,
        next_question=next_question,
    )


def _parse_source_range(source_range: str) -> tuple[str, str]:
    match = _SOURCE_RANGE.match(source_range.strip())
    if match is None:
        raise ExplorationError(f"exploration source_range {source_range!r} is not Sheet!A1")
    sheet = match.group("quoted")
    sheet = sheet.replace("''", "'") if sheet is not None else match.group("plain")
    return sheet, match.group("range").replace("$", "").upper()


def load_exploration(directory: str | Path, workbook_id: str) -> list[Observation]:
    """Turn an exploration run's kept evidence into leads for QA.

    Only ``evidence.filtered.json`` is read; ``exploration.md`` is prose for people.

    Raises:
        ExplorationError: If the artifact is missing or malformed.
    """
    path = Path(directory) / EXPLORATION_FILE
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExplorationError(f"exploration artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExplorationError(f"exploration artifact is not JSON: {path}") from exc
    if not isinstance(records, list):
        raise ExplorationError(f"exploration artifact must be a list of records: {path}")
    leads: list[Observation] = []
    for record in records:
        if not isinstance(record, dict) or record.get("status") != "keep":
            continue
        try:
            keyword, source_range = str(record["keyword"]), str(record["source_range"])
            description = str(record["description"])
        except KeyError as exc:
            raise ExplorationError(f"exploration record lacks {exc.args[0]!r}: {path}") from exc
        sheet, range_ref = _parse_source_range(source_range)
        leads.append(
            Observation(
                source=ObservationSource.EXPLORATION,
                status=ResultState.SUCCESS,
                summary=keyword,
                facts=[description],
                provenance=[
                    ProvenanceReference(
                        kind=ProvenanceKind.WORKBOOK,
                        workbook_id=workbook_id,
                        sheet=sheet,
                        range_ref=range_ref,
                    )
                ],
                uncertainties=[EXPLORATION_CAVEAT],
            )
        )
    return leads
