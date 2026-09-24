"""Read-only workbook registry and the single implementation of every QA read.

The sandbox facade wraps these same callables, so tool semantics and limits cannot
drift between the two surfaces; see ``specs/layers/tools.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import openpyxl
from openpyxl.utils.cell import get_column_letter, range_boundaries
from openpyxl.worksheet.worksheet import Worksheet
from pydantic import Field, ValidationError, model_validator

from ..schemas import (
    JsonValue,
    NonEmptyStr,
    ProvenanceKind,
    ProvenanceReference,
    ResultState,
    StrictModel,
    ToolResult,
)

DEFAULT_MAX_RANGE_CELLS = 10_000


class ToolError(ValueError):
    """Raised when a read is rejected before it touches the workbook."""


class AttributeName(str, Enum):
    FORMULA = "formula"
    FONT = "font"
    FILL = "fill"
    NUMBER_FORMAT = "number_format"
    DATA_TYPE = "data_type"


class SearchType(str, Enum):
    PARTIAL = "partial"
    WHOLE = "whole"
    STRIP = "strip"


class _ToolArgs(StrictModel):
    workbook_id: NonEmptyStr


class _SheetArgs(_ToolArgs):
    sheet_name: NonEmptyStr | None = None


class InspectRangeArgs(_SheetArgs):
    """Read raw and cached values of one A1 cell or rectangular range."""

    range_ref: NonEmptyStr


class InspectAttributesArgs(_SheetArgs):
    """Read registered cell attributes of one A1 cell or rectangular range."""

    range_ref: NonEmptyStr
    attributes: Annotated[list[AttributeName], Field(min_length=1)]

    @model_validator(mode="after")
    def _reject_duplicates(self) -> InspectAttributesArgs:
        if len(set(self.attributes)) != len(self.attributes):
            raise ValueError("attributes must not repeat")
        return self


class SearchArgs(_SheetArgs):
    """Find cells whose displayed text matches a value."""

    value: str | int | float | bool
    case_sensitive: bool = False
    search_type: SearchType = SearchType.PARTIAL


class SheetDataframeArgs(_SheetArgs):
    """Read one sheet as rows and columns of cached values."""

    header_row: Annotated[int, Field(ge=1)] | None = 1
    max_rows: Annotated[int, Field(ge=1)] | None = None


class WorkbookReader:
    """Read-only access to one workbook, keeping formulas and caches separate."""

    def __init__(
        self,
        workbook_id: str,
        path: str | Path,
        *,
        include_hidden: bool = False,
        max_range_cells: int = DEFAULT_MAX_RANGE_CELLS,
    ) -> None:
        self.workbook_id = workbook_id
        self.path = Path(path)
        self.include_hidden = include_hidden
        self.max_range_cells = max_range_cells
        self._formulas = openpyxl.load_workbook(self.path, data_only=False)
        self._values = openpyxl.load_workbook(self.path, data_only=True)

    def sheet_names(self) -> list[str]:
        """Return the sheets this reader is allowed to open, in workbook order."""
        return [ws.title for ws in self._formulas.worksheets if self._is_readable(ws)]

    def describe(self) -> list[dict[str, Any]]:
        """Return each readable sheet's name, used range, and merged ranges.

        This is the workbook manifest the model sees before it reads anything.
        """
        return [
            {
                "name": sheet.title,
                "dimensions": sheet.dimensions,
                "merged": sorted(str(merged) for merged in sheet.merged_cells.ranges),
            }
            for sheet in self._formulas.worksheets
            if self._is_readable(sheet)
        ]

    def _is_readable(self, sheet: Worksheet) -> bool:
        return self.include_hidden or sheet.sheet_state == "visible"

    def resolve_sheet(self, sheet_name: str | None) -> str:
        """Return the sheet a read applies to, defaulting to the active sheet."""
        if sheet_name is None:
            active = self._formulas.active
            if active is None:
                raise ToolError(f"workbook {self.workbook_id!r} has no active sheet")
            sheet_name = str(active.title)
        if sheet_name not in self._formulas.sheetnames:
            raise ToolError(
                f"sheet {sheet_name!r} does not exist in workbook {self.workbook_id!r}; "
                f"available sheets: {', '.join(self.sheet_names())}"
            )
        sheet = self._formulas[sheet_name]
        if not self._is_readable(sheet):
            raise ToolError(f"sheet {sheet_name!r} is hidden and hidden sheets are disabled")
        return sheet_name

    def _bounds(self, range_ref: str) -> tuple[int, int, int, int]:
        try:
            min_col, min_row, max_col, max_row = range_boundaries(range_ref.upper())
        except (ValueError, TypeError) as exc:
            raise ToolError(f"{range_ref!r} is not valid A1 syntax") from exc
        if None in (min_col, min_row, max_col, max_row):
            raise ToolError(f"{range_ref!r} must name a cell or a rectangular range")
        cells = (max_row - min_row + 1) * (max_col - min_col + 1)
        if cells > self.max_range_cells:
            raise ToolError(
                f"{range_ref!r} covers {cells} cells, above the limit of "
                f"{self.max_range_cells}; request a smaller range"
            )
        return min_row, min_col, max_row, max_col

    def _merged_anchors(self, sheet_name: str) -> dict[str, tuple[str, str]]:
        anchors: dict[str, tuple[str, str]] = {}
        for merged in self._formulas[sheet_name].merged_cells.ranges:
            label = str(merged)
            anchor = f"{get_column_letter(merged.min_col)}{merged.min_row}"
            for row in range(merged.min_row, merged.max_row + 1):
                for col in range(merged.min_col, merged.max_col + 1):
                    anchors[f"{get_column_letter(col)}{row}"] = (anchor, label)
        return anchors

    def read_range(self, range_ref: str, sheet_name: str | None = None) -> dict[str, JsonValue]:
        """Return coordinates with raw values, caches, and formulas."""
        resolved = self.resolve_sheet(sheet_name)
        min_row, min_col, max_row, max_col = self._bounds(range_ref)
        merged = self._merged_anchors(resolved)
        rows: list[JsonValue] = []
        for row in range(min_row, max_row + 1):
            cells: list[JsonValue] = []
            for col in range(min_col, max_col + 1):
                coordinate = f"{get_column_letter(col)}{row}"
                anchor, label = merged.get(coordinate, (coordinate, None))
                cell = self._formulas[resolved][anchor]
                raw = cell.value
                formula = raw if isinstance(raw, str) and raw.startswith("=") else None
                payload: dict[str, JsonValue] = {
                    "coordinate": coordinate,
                    "value": None if formula is not None else _jsonable(raw),
                    "displayed": _jsonable(self._values[resolved][anchor].value),
                    "formula": formula,
                    "data_type": str(cell.data_type),
                    "number_format": str(cell.number_format),
                }
                if label is not None:
                    payload["merged_with"] = label
                cells.append(payload)
            rows.append(cells)
        return {"sheet": resolved, "range": range_ref.upper(), "cells": rows}

    def read_attributes(
        self,
        range_ref: str,
        attributes: Sequence[AttributeName],
        sheet_name: str | None = None,
    ) -> dict[str, JsonValue]:
        """Return only the registered attributes requested for each cell."""
        resolved = self.resolve_sheet(sheet_name)
        min_row, min_col, max_row, max_col = self._bounds(range_ref)
        rows: list[JsonValue] = []
        for row in range(min_row, max_row + 1):
            cells: list[JsonValue] = []
            for col in range(min_col, max_col + 1):
                coordinate = f"{get_column_letter(col)}{row}"
                cell = self._formulas[resolved][coordinate]
                payload: dict[str, JsonValue] = {"coordinate": coordinate}
                for attribute in attributes:
                    payload[attribute.value] = _attribute(cell, attribute)
                cells.append(payload)
            rows.append(cells)
        return {"sheet": resolved, "range": range_ref.upper(), "cells": rows}

    def search(
        self,
        # int and float stay distinct: the needle is reported back as the model sent it.
        value: str | int | float | bool,  # noqa: PYI041
        sheet_name: str | None = None,
        *,
        case_sensitive: bool = False,
        search_type: SearchType = SearchType.PARTIAL,
    ) -> tuple[dict[str, JsonValue], bool]:
        """Return matches ordered by sheet name, row, and column."""
        sheets = [self.resolve_sheet(sheet_name)] if sheet_name else self.sheet_names()
        needle = _comparable(value, case_sensitive)
        matches: list[tuple[str, int, int, dict[str, JsonValue]]] = []
        scanned = 0
        truncated = False
        for sheet in sorted(sheets):
            for row in self._values[sheet].iter_rows():
                for cell in row:
                    scanned += 1
                    if scanned > self.max_range_cells:
                        truncated = True
                        break
                    shown = cell.value
                    if shown is None:
                        shown = self._formulas[sheet][cell.coordinate].value
                    if shown is None or not _matches(
                        _comparable(shown, case_sensitive), needle, search_type
                    ):
                        continue
                    matches.append(
                        (
                            sheet,
                            int(cell.row),
                            int(cell.column),
                            {
                                "reference": f"{sheet}!{cell.coordinate}",
                                "sheet": sheet,
                                "coordinate": str(cell.coordinate),
                                "value": _jsonable(self._formulas[sheet][cell.coordinate].value),
                                "displayed": _jsonable(cell.value),
                            },
                        )
                    )
                if truncated:
                    break
            if truncated:
                break
        matches.sort(key=lambda item: (item[0], item[1], item[2]))
        payload: dict[str, JsonValue] = {"matches": [match[3] for match in matches]}
        return payload, truncated

    def read_sheet(
        self,
        sheet_name: str | None = None,
        *,
        header_row: int | None = 1,
        max_rows: int | None = None,
    ) -> tuple[dict[str, JsonValue], bool]:
        """Return one sheet as columns and rows of cached values."""
        resolved = self.resolve_sheet(sheet_name)
        sheet = self._values[resolved]
        width = int(sheet.max_column)
        height = int(sheet.max_row)
        if header_row is not None and header_row > height:
            raise ToolError(
                f"header_row {header_row} is past the last row ({height}) of {resolved!r}"
            )
        if header_row is None:
            columns = [get_column_letter(col) for col in range(1, width + 1)]
            first_data_row = 1
        else:
            columns = [
                _label(sheet.cell(row=header_row, column=col).value, col)
                for col in range(1, width + 1)
            ]
            first_data_row = header_row + 1
        rows: list[JsonValue] = []
        truncated = False
        for row in range(first_data_row, height + 1):
            if max_rows is not None and len(rows) >= max_rows:
                truncated = True
                break
            rows.append(
                [_jsonable(sheet.cell(row=row, column=col).value) for col in range(1, width + 1)]
            )
        used = f"A{first_data_row}:{get_column_letter(width)}{height}"
        return (
            {
                "sheet": resolved,
                "header_row": header_row,
                "range": used,
                "columns": columns,
                "rows": rows,
                "shape": [len(rows), width],
            },
            truncated,
        )


def _jsonable(value: Any) -> JsonValue:
    """Return a JSON-safe copy of a workbook value without inventing one."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _label(value: Any, column: int) -> str:
    text = "" if value is None else str(value).strip()
    return text or get_column_letter(column)


def _comparable(value: Any, case_sensitive: bool) -> str:
    text = str(value)
    return text if case_sensitive else text.casefold()


def _matches(haystack: str, needle: str, search_type: SearchType) -> bool:
    if search_type is SearchType.WHOLE:
        return haystack == needle
    if search_type is SearchType.STRIP:
        return haystack.strip() == needle.strip()
    return needle in haystack


def _attribute(cell: Any, attribute: AttributeName) -> JsonValue:
    if attribute is AttributeName.FORMULA:
        raw = cell.value
        return raw if isinstance(raw, str) and raw.startswith("=") else None
    if attribute is AttributeName.NUMBER_FORMAT:
        return str(cell.number_format)
    if attribute is AttributeName.DATA_TYPE:
        return str(cell.data_type)
    if attribute is AttributeName.FONT:
        font = cell.font
        return {
            "name": font.name,
            "size": float(font.size) if font.size is not None else None,
            "bold": bool(font.bold),
            "italic": bool(font.italic),
        }
    fill = cell.fill
    return {
        "pattern": fill.patternType,
        "color": getattr(fill.fgColor, "rgb", None) if fill.fgColor is not None else None,
    }


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool: its contract, its description, and its callable."""

    name: str
    description: str
    args_model: type[_ToolArgs]
    handler: Callable[[WorkbookReader, Any], tuple[dict[str, JsonValue], ResultState]]


def _inspect_range(
    reader: WorkbookReader, args: InspectRangeArgs
) -> tuple[dict[str, JsonValue], ResultState]:
    payload = reader.read_range(args.range_ref, args.sheet_name)
    return payload, ResultState.EMPTY if _is_blank(payload) else ResultState.SUCCESS


def _inspect_attributes(
    reader: WorkbookReader, args: InspectAttributesArgs
) -> tuple[dict[str, JsonValue], ResultState]:
    return reader.read_attributes(args.range_ref, args.attributes, args.sheet_name), (
        ResultState.SUCCESS
    )


def _search(reader: WorkbookReader, args: SearchArgs) -> tuple[dict[str, JsonValue], ResultState]:
    payload, truncated = reader.search(
        args.value,
        args.sheet_name,
        case_sensitive=args.case_sensitive,
        search_type=args.search_type,
    )
    if truncated:
        return payload, ResultState.TRUNCATED
    matches = payload["matches"]
    empty = isinstance(matches, list) and not matches
    return payload, ResultState.EMPTY if empty else ResultState.SUCCESS


def _sheet_dataframe(
    reader: WorkbookReader, args: SheetDataframeArgs
) -> tuple[dict[str, JsonValue], ResultState]:
    payload, truncated = reader.read_sheet(
        args.sheet_name, header_row=args.header_row, max_rows=args.max_rows
    )
    if truncated:
        return payload, ResultState.TRUNCATED
    rows = payload["rows"]
    empty = isinstance(rows, list) and not rows
    return payload, ResultState.EMPTY if empty else ResultState.SUCCESS


def _is_blank(payload: dict[str, JsonValue]) -> bool:
    rows = payload.get("cells")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, list):
            continue
        for cell in row:
            if not isinstance(cell, dict):
                continue
            if any(cell.get(key) is not None for key in ("value", "displayed", "formula")):
                return False
    return True


SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "inspect_range",
        "Read one A1 cell or rectangular range. Returns each coordinate with its raw "
        "value, its cached/displayed value, its formula text when present, its data "
        "type, and its number format. Formula text and cache stay separate and a "
        "missing cache is never reported as zero. Example: "
        '{"workbook_id": "workbook_1", "sheet_name": "Revenue", "range_ref": "B2:B4"}.',
        InspectRangeArgs,
        _inspect_range,
    ),
    ToolSpec(
        "inspect_attributes",
        "Read registered attributes (formula, font, fill, number_format, data_type) "
        "for one A1 cell or rectangular range. The attribute list must be non-empty "
        "and free of duplicates. Example: "
        '{"workbook_id": "workbook_1", "range_ref": "B2", "attributes": ["number_format"]}.',
        InspectAttributesArgs,
        _inspect_attributes,
    ),
    ToolSpec(
        "search",
        "Find cells whose displayed text matches a value, with partial, whole, or "
        "strip matching and optional case sensitivity. Results are ordered by sheet "
        "name, row, then column. Example: "
        '{"workbook_id": "workbook_1", "value": "Revenue", "search_type": "whole"}.',
        SearchArgs,
        _search,
    ),
    ToolSpec(
        "get_sheet_as_dataframe",
        "Read one sheet as columns and rows of cached values. header_row is the "
        "1-based header row, or null when the sheet has no header and columns are "
        "positional. max_rows caps the data rows and reports a truncated result. "
        'Example: {"workbook_id": "workbook_1", "sheet_name": "Revenue", "max_rows": 50}.',
        SheetDataframeArgs,
        _sheet_dataframe,
    ),
)


class ToolRegistry:
    """Validated, read-only dispatch for the v1 tool set."""

    def __init__(self, readers: Iterable[WorkbookReader]) -> None:
        self._readers = {reader.workbook_id: reader for reader in readers}
        self._specs = {spec.name: spec for spec in SPECS}

    def readers(self) -> list[WorkbookReader]:
        """Return the registered workbook readers, in registration order."""
        return list(self._readers.values())

    def sheet_names(self, workbook_id: str) -> list[str]:
        """Return the readable sheets of one workbook."""
        return self._reader(workbook_id).sheet_names()

    def descriptions(self) -> list[dict[str, Any]]:
        """Return the prompt-facing tool descriptions and generated schemas."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "schema": spec.args_model.model_json_schema(),
            }
            for spec in SPECS
        ]

    def _reader(self, workbook_id: str) -> WorkbookReader:
        reader = self._readers.get(workbook_id)
        if reader is None:
            known = ", ".join(sorted(self._readers)) or "none"
            raise ToolError(f"unknown workbook_id {workbook_id!r}; known workbooks: {known}")
        return reader

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Validate one model-generated action and run it, or reject it."""
        spec = self._specs.get(tool_name)
        if spec is None:
            return self._error(
                tool_name,
                arguments,
                f"unknown tool {tool_name!r}; available tools: "
                f"{', '.join(sorted(self._specs))}",
            )
        try:
            # Actions reach us as a JSON object, so validate in JSON mode: enum values
            # arrive as strings, while strict mode still refuses "1" for an integer.
            args = spec.args_model.model_validate_json(json.dumps(arguments))
            reader = self._reader(args.workbook_id)
            payload, state = spec.handler(reader, args)
        except (ValidationError, TypeError) as exc:
            if isinstance(exc, TypeError):
                return self._error(tool_name, arguments, f"arguments must be JSON: {exc}")
            return self._error(tool_name, arguments, _first_message(exc))
        except ToolError as exc:
            return self._error(tool_name, arguments, str(exc))
        return ToolResult(
            state=state,
            tool_name=tool_name,
            arguments=arguments,
            result=payload,
            provenance=[_provenance(args, payload)],
        )

    @staticmethod
    def _error(tool_name: str, arguments: dict[str, Any], message: str) -> ToolResult:
        return ToolResult(
            state=ResultState.ERROR,
            tool_name=tool_name or "<unnamed>",
            arguments=arguments,
            error=message,
        )


def _provenance(args: _ToolArgs, payload: dict[str, JsonValue]) -> ProvenanceReference:
    sheet = payload.get("sheet")
    range_ref = payload.get("range")
    if isinstance(sheet, str) and isinstance(range_ref, str):
        return ProvenanceReference(
            kind=ProvenanceKind.WORKBOOK,
            workbook_id=args.workbook_id,
            sheet=sheet,
            range_ref=range_ref,
        )
    return ProvenanceReference(kind=ProvenanceKind.WORKBOOK, workbook_id=args.workbook_id)


def _first_message(exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"]) or "arguments"
    return f"{location}: {error['msg']}"
