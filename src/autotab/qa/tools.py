"""Read-only, workbook-scoped QA tools."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries
from openpyxl.workbook.workbook import Workbook


class WorkbookToolError(ValueError):
    """Raised for invalid, safely reportable workbook tool input."""


_A1_RANGE = re.compile(r"\A[A-Za-z]{1,3}[1-9]\d*(?::[A-Za-z]{1,3}[1-9]\d*)?\Z")


class WorkbookSession:
    """Own one read-only workbook and expose its approved QA tools."""

    def __init__(self, workbook_path: str | Path, max_range_cells: int) -> None:
        path = Path(workbook_path)
        if not path.is_file():
            raise WorkbookToolError("Workbook does not exist")
        if path.suffix.lower() not in {".xlsx", ".xlsm"}:
            raise WorkbookToolError("Workbook must be an .xlsx or .xlsm file")
        if not isinstance(max_range_cells, int) or isinstance(max_range_cells, bool):
            raise WorkbookToolError("max_range_cells must be a positive integer")
        if max_range_cells <= 0:
            raise WorkbookToolError("max_range_cells must be a positive integer")
        self._workbook: Workbook = load_workbook(path, read_only=True, data_only=True)
        self._max_range_cells = max_range_cells

    @property
    def tools(self) -> dict[str, Callable[..., object]]:
        """Return the fixed tool registry for this session."""
        return {
            "load_dataframe": self.load_dataframe,
            "inspect_range": self.inspect_range,
        }

    @property
    def description(self) -> str:
        """Return safe worksheet names and dimensions for the QA prompt."""
        return ", ".join(
            f"{sheet.title} ({rows} rows x {columns} columns)"
            for sheet in self._workbook.worksheets
            for rows, columns in [_worksheet_dimensions(sheet)]
        )

    def close(self) -> None:
        """Close the workbook without saving it."""
        self._workbook.close()

    def load_dataframe(self, sheet_name: str | None = None) -> pd.DataFrame:
        """Load the active or named worksheet as a DataFrame.

        Args:
            sheet_name: Exact worksheet name, or None for the active worksheet.

        Returns:
            Worksheet records with the first row used as column labels.

        Raises:
            WorkbookToolError: If the worksheet name is unknown.
        """
        sheet = self._sheet(sheet_name)
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows[1:], columns=list(rows[0]))

    def inspect_range(
        self, range_ref: str, sheet_name: str | None = None
    ) -> list[list[object | None]]:
        """Read values from an A1 range on the active or named worksheet.

        Args:
            range_ref: A single cell or rectangular A1 range.
            sheet_name: Exact worksheet name, or None for the active worksheet.

        Returns:
            Cell values as a two-dimensional row-major list.

        Raises:
            WorkbookToolError: If the sheet, range, bounds, or range size is invalid.
        """
        if not isinstance(range_ref, str) or not _A1_RANGE.fullmatch(range_ref):
            raise WorkbookToolError("range_ref must be a valid A1 range")
        sheet = self._sheet(sheet_name)
        min_col, min_row, max_col, max_row = range_boundaries(range_ref.upper())
        if min_col > max_col or min_row > max_row:
            raise WorkbookToolError("range_ref must start at its top-left cell")
        sheet_rows, sheet_columns = _worksheet_dimensions(sheet)
        if max_row > sheet_rows or max_col > sheet_columns:
            raise WorkbookToolError("range_ref is outside the worksheet bounds")
        cell_count = (max_row - min_row + 1) * (max_col - min_col + 1)
        if cell_count > self._max_range_cells:
            raise WorkbookToolError(f"range_ref exceeds the {self._max_range_cells}-cell limit")
        return [
            list(row)
            for row in sheet.iter_rows(
                min_row=min_row,
                max_row=max_row,
                min_col=min_col,
                max_col=max_col,
                values_only=True,
            )
        ]

    def _sheet(self, sheet_name: str | None):
        if sheet_name is None:
            return self._workbook.active
        if not isinstance(sheet_name, str) or sheet_name not in self._workbook.sheetnames:
            raise WorkbookToolError(f"Unknown worksheet: {sheet_name!r}")
        return self._workbook[sheet_name]


def _worksheet_dimensions(sheet) -> tuple[int, int]:
    rows, columns = sheet.max_row or 0, sheet.max_column or 0
    if rows == columns == 1 and sheet["A1"].value is None:
        return 0, 0
    return rows, columns
