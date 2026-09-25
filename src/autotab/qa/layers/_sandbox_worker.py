"""The subprocess that owns one QA run's code session.

The parent speaks JSON lines over stdin/stdout; see ``specs/layers/sandbox.md``.
This module is harness code, not model code: the generated snippet itself runs with
the restricted builtins and injected objects assembled here.
"""

from __future__ import annotations

import ast
import datetime as datetime_module
import io
import json
import math
import re as re_module
import statistics
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.utils.cell import range_boundaries

from autotab.qa.layers.code_policy import ALLOWED_BUILTINS, INJECTED_NAMES, refusal
from autotab.qa.layers.tools import (
    AttributeName,
    SearchType,
    ToolError,
    ToolRegistry,
    WorkbookReader,
)
from autotab.qa.layers.verification import (
    CONDITIONS,
    NUMERIC_OPERATIONS,
    REPLAYABLE_OPERATIONS,
    ROW_SELECTION,
    parse_citation,
)
from autotab.qa.schemas import ResultState

RESULT_NAME = "result"


class FacadeError(RuntimeError):
    """Raised inside generated code when a facade read is rejected."""


class ComputationError(ValueError):
    """Raised inside generated code when a computation cannot be recorded."""


_COMPUTATION_ID = re_module.compile(r"^calc_\d+$")
_NEEDS_QUOTES = re_module.compile(r"[^A-Za-z0-9_]")


class WorkbookFacade:
    """Read-only workbook access for generated code, backed by the tool registry."""

    def __init__(self, registry: ToolRegistry, workbook_id: str, calls: list[dict[str, Any]]):
        self._registry = registry
        self._workbook_id = workbook_id
        self._calls = calls

    def _dispatch(self, method: str, tool: str, arguments: dict[str, Any]) -> Any:
        payload = {"workbook_id": self._workbook_id, **arguments}
        self._calls.append({"method": method, **payload})
        result = self._registry.dispatch(tool, payload)
        if result.state is ResultState.ERROR:
            raise FacadeError(result.error or f"{tool} failed")
        return result.result

    def sheets(self, workbook_id: str | None = None) -> list[str]:
        """Return the readable sheet names."""
        self._calls.append({"method": "sheets", "workbook_id": workbook_id or self._workbook_id})
        return self._registry.sheet_names(workbook_id or self._workbook_id)

    def sheet(
        self,
        sheet_name: str | None = None,
        header_row: int | None = 1,
        max_rows: int | None = None,
        workbook_id: str | None = None,
    ) -> pd.DataFrame:
        """Return one sheet as a pandas DataFrame of cached values.

        The argument is ``sheet_name``, as on every other wb method and every tool.
        """
        payload = self._dispatch(
            "sheet",
            "get_sheet_as_dataframe",
            {
                "workbook_id": workbook_id or self._workbook_id,
                "sheet_name": sheet_name,
                "header_row": header_row,
                "max_rows": max_rows,
            },
        )
        # Indexed by worksheet row, so df.loc[n] is row n and a filter's index names the
        # rows to cite: no position-to-row arithmetic for the model to get wrong.
        first = 1 if header_row is None else header_row + 1
        rows = payload["rows"]
        return pd.DataFrame(rows, columns=payload["columns"], index=range(first, first + len(rows)))

    def range(
        self,
        range_ref: str,
        sheet_name: str | None = None,
        workbook_id: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Return the cells of one A1 range."""
        payload = self._dispatch(
            "range",
            "inspect_range",
            {
                "workbook_id": workbook_id or self._workbook_id,
                "range_ref": range_ref,
                "sheet_name": sheet_name,
            },
        )
        cells: list[list[dict[str, Any]]] = payload["cells"]
        return cells

    def attributes(
        self,
        range_ref: str,
        attributes: list[str],
        sheet_name: str | None = None,
        workbook_id: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Return registered attributes of one A1 range."""
        payload = self._dispatch(
            "attributes",
            "inspect_attributes",
            {
                "workbook_id": workbook_id or self._workbook_id,
                "range_ref": range_ref,
                "attributes": [AttributeName(item).value for item in attributes],
                "sheet_name": sheet_name,
            },
        )
        cells: list[list[dict[str, Any]]] = payload["cells"]
        return cells

    def search(
        self,
        value: Any,
        sheet_name: str | None = None,
        case_sensitive: bool = False,
        search_type: str = "partial",
        workbook_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return matches ordered by sheet name, row, then column."""
        payload = self._dispatch(
            "search",
            "search",
            {
                "workbook_id": workbook_id or self._workbook_id,
                "value": value,
                "sheet_name": sheet_name,
                "case_sensitive": case_sensitive,
                "search_type": SearchType(search_type).value,
            },
        )
        matches: list[dict[str, Any]] = payload["matches"]
        return matches


def _frame_rows(frame: pd.DataFrame) -> list[list[Any]]:
    """A frame's rows with pandas' missing markers (NaN, NaT, NA) as None.

    Inside a frame NaN means a blank cell, not a computed infinity: refusing it would
    fail almost every real table, whose blanks all become NaN on load.
    """
    rows: list[list[Any]] = frame.astype(object).where(frame.notna(), None).values.tolist()
    return rows


def _positional(index: pd.Index) -> bool:
    """Whether an index is just positions 0..n-1, which carry nothing worth showing."""
    return isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1


def _labels(value: pd.DataFrame | pd.Series) -> list[Any]:
    """A frame's or series' index labels -- worksheet rows for a sheet -- JSON-safe."""
    labels: list[Any] = _jsonable(value.index.tolist())
    return labels


def _index_ends(value: pd.DataFrame | pd.Series) -> dict[str, Any]:
    """The index labels of a preview's two ends, when they are worth showing."""
    if _positional(value.index):
        return {}
    labels = _labels(value)
    return {"index_head": labels[:_PREVIEW_ITEMS], "index_tail": labels[-_PREVIEW_ITEMS:]}


def _series_items(series: pd.Series) -> list[Any]:
    items: list[Any] = series.astype(object).where(series.notna(), None).tolist()
    return items


def _jsonable(value: Any) -> Any:
    """Return a deterministic, JSON-safe view of a value produced by generated code.

    Raises:
        ValueError: If the value contains a non-finite float.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("a non-finite float cannot be serialized")
        return value
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        frame: dict[str, Any] = {} if _positional(value.index) else {"index": _labels(value)}
        return {
            **frame,
            "columns": [str(column) for column in value.columns],
            "rows": _jsonable(_frame_rows(value)),
        }
    if isinstance(value, pd.Series):
        items = _jsonable(_series_items(value))
        return items if _positional(value.index) else {"index": _labels(value), "values": items}
    if isinstance(value, (pd.Index, np.ndarray)):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


# Import statements resolve to the objects already bound in the session, and to
# nothing else; the parent refuses any other module statically as well.
_IMPORTABLE = {
    "pandas": pd,
    "math": math,
    "statistics": statistics,
    "datetime": datetime_module,
    "re": re_module,
    "json": json,
}


def _safe_import(
    name: str,
    globals: Any = None,
    locals: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    """Serve ``import pandas as pd`` and friends from the session's own objects."""
    if level != 0 or name not in _IMPORTABLE:
        raise ImportError(f"import of {name!r} is not available")
    return _IMPORTABLE[name]


def _restricted_builtins() -> dict[str, Any]:
    import builtins

    allowed = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS}
    allowed["__import__"] = _safe_import
    return allowed


_PREVIEW_ITEMS = 5


def _preview(kind: str, items: list[Any], size: dict[str, Any]) -> dict[str, Any]:
    return {
        "preview": True,
        "type": kind,
        **size,
        "head": _jsonable(items[:_PREVIEW_ITEMS]),
        "tail": _jsonable(items[-_PREVIEW_ITEMS:]),
        "note": "only both ends are shown; the full value stays bound in the session",
    }


def _result_view(value: Any, limit: int) -> Any:
    """Return ``value`` JSON-safe, or a preview of both ends when it is large.

    The prompt is not where bulk data lives: a large frame or list stays bound in the
    session for the next action, and only its shape and ends come back.
    """
    if isinstance(value, pd.DataFrame):
        rows, columns = value.shape
        if rows * max(1, columns) > limit:
            data = _frame_rows(value)
            size = {"shape": [rows, columns], "columns": [str(c) for c in value.columns]}
            return _preview("DataFrame", data, {**size, **_index_ends(value)})
    elif isinstance(value, pd.Series):
        if len(value) > limit:
            length: dict[str, Any] = {"length": len(value)}
            return _preview("Series", _series_items(value), {**length, **_index_ends(value)})
    elif isinstance(value, (set, frozenset)):
        if len(value) > limit:
            return _preview("set", sorted(value, key=repr), {"length": len(value)})
    elif isinstance(value, (list, tuple)):
        if len(value) > limit:
            return _preview(type(value).__name__, list(value), {"length": len(value)})
    elif isinstance(value, dict) and len(value) > limit:
        items = [[key, item] for key, item in value.items()]
        return _preview("dict", items, {"length": len(value)})
    return _jsonable(value)


def _both_ends(text: str, limit: int) -> str:
    """Cut the middle out of ``text`` so it fits ``limit``; the end often holds the answer."""
    template = "\n… {} characters omitted …\n"
    keep = max(0, limit - len(template.format(len(text))))
    head, tail = keep // 2, keep - keep // 2
    marker = template.format(len(text) - head - tail)
    return text[:head] + marker + (text[-tail:] if tail else "")


class Session:
    """One persistent namespace plus the objects generated code may touch."""

    def __init__(self, config: dict[str, Any]) -> None:
        readers = [
            WorkbookReader(
                book["workbook_id"],
                Path(book["path"]),
                include_hidden=bool(book.get("include_hidden", False)),
                max_range_cells=int(book.get("max_range_cells", 10_000)),
            )
            for book in config["workbooks"]
        ]
        self.registry = ToolRegistry(readers)
        self.max_output_chars = int(config["max_output_chars"])
        self.max_inline_cells = int(config.get("max_inline_cells", 200))
        self.primary = readers[0].workbook_id if readers else ""
        self.calls: list[dict[str, Any]] = []
        self.computations: list[dict[str, Any]] = []
        self._counter = 0
        self.namespace: dict[str, Any] = {
            "__builtins__": _restricted_builtins(),
            "pd": pd,
            "math": math,
            "statistics": statistics,
            "datetime": datetime_module,
            "re": re_module,
            "json": json,
            "wb": WorkbookFacade(self.registry, self.primary, self.calls),
            "record_computation": self._record_computation,
        }

    def _record_computation(
        self,
        operation: str,
        inputs: list[Any],
        output: Any,
        unit: str | None = None,
        tolerance: float = 0.0,
        condition: dict[str, Any] | None = None,
    ) -> str:
        """Persist one replayable computation and return its ID.

        ``rows_where`` records a row selection instead of a number: over one column
        range, the worksheet rows whose value meets ``condition`` (one of ``equals``,
        ``not_equals``, ``gt``, ``ge``, ``lt``, ``le``, ``contains``, against a constant
        or an earlier computation ID), with those row numbers as ``output``.

        The model names where each input comes from; the values are read here, from
        the workbook or from earlier computations, so a recorded input cannot differ
        from its source. Each input is a ``'Sheet!A1'`` or ``'Sheet!A1:B9'`` reference
        (ranges expand to their cells, blanks skipped), an earlier computation ID such
        as ``'calc_1'``, or ``{'constant': value}``.

        Raises:
            ComputationError: If an input is malformed, unreadable, blank, or a
                formula with no cached value, or if nothing is sourced at all.
        """
        name = str(operation).strip().lower()
        if name not in REPLAYABLE_OPERATIONS:
            raise ComputationError(
                f"operation {operation!r} cannot be replayed; use one of "
                f"{', '.join(sorted(REPLAYABLE_OPERATIONS))}. To record which rows meet a "
                "condition, use rows_where; a label on those rows is cited by its cells."
            )
        selection: dict[str, Any] | None = None
        if name == ROW_SELECTION:
            output = _row_numbers(output)
            selection = self._condition(condition)
            _require_one_column(inputs)
        elif condition is not None:
            raise ComputationError(
                f"condition applies only to {ROW_SELECTION!r}, which records the rows that "
                "meet it"
            )
        if isinstance(output, np.generic):
            output = output.item()  # a numpy scalar, e.g. what a pandas max returns
        numeric = isinstance(output, (int, float)) and not isinstance(output, bool)
        if name in NUMERIC_OPERATIONS and not numeric:
            raise ComputationError(
                f"{operation!r} produces a number, but the output given is {output!r}. "
                "Record the number here; put a looked-up name or label in a separate claim "
                "backed by citations."
            )
        if not isinstance(inputs, (list, tuple)) or not inputs:
            raise ComputationError("inputs must be a non-empty list")
        resolved: list[dict[str, Any]] = []
        skipped: list[str] = []
        sources: list[Any] = []
        for item in inputs:
            if isinstance(item, str) and _COMPUTATION_ID.match(item):
                resolved.append(self._from_computation(item))
                sources.append(item)
            elif isinstance(item, str):
                cells, blanks = self._from_workbook(item)
                resolved.extend(cells)
                skipped.extend(blanks)
                sources.append(item)
            elif isinstance(item, dict) and set(item) == {"constant"}:
                resolved.append({"reference": None, "value": _jsonable(item["constant"])})
                sources.append({"constant": _jsonable(item["constant"])})
            else:
                raise ComputationError(
                    "each input is a 'Sheet!A1' or 'Sheet!A1:B9' reference, an earlier "
                    "computation ID such as 'calc_1', or {'constant': value}; "
                    "record_computation reads workbook values itself, so never pass "
                    f"{{'reference': ..., 'value': ...}} (got {item!r})"
                )
        if all(entry["reference"] is None for entry in resolved):
            raise ComputationError(
                "a computation needs at least one workbook reference or earlier "
                "computation; constants alone prove nothing"
            )
        metadata: dict[str, Any] = {"sources": sources}
        if selection is not None:
            metadata["condition"] = selection
        if unit is not None:
            metadata["unit"] = str(unit)
        if skipped:
            metadata["skipped_blank"] = skipped
        self._counter += 1
        computation_id = f"calc_{self._counter}"
        self.computations.append(
            {
                "schema_version": "1.0",
                "id": computation_id,
                "operation": str(operation),
                "inputs": resolved,
                "output": _jsonable(output),
                "metadata": metadata,
                "tolerance": float(tolerance),
            }
        )
        return computation_id

    def _condition(self, condition: Any) -> dict[str, Any]:
        """Validate a row-selection condition and resolve its target to a value."""
        allowed = ", ".join(CONDITIONS)
        if not isinstance(condition, dict) or not condition:
            raise ComputationError(
                f"{ROW_SELECTION} needs condition={{op: target}} with op one of {allowed}, "
                "e.g. condition={'equals': 'calc_1'}"
            )
        if len(condition) != 1:
            raise ComputationError(f"give exactly one condition, one of {allowed}")
        ((op, target),) = condition.items()
        op = _CONDITION_SPELLINGS.get(str(op).strip().lower(), op)
        if op not in CONDITIONS:
            raise ComputationError(f"unknown condition {op!r}; use one of {allowed}")
        if isinstance(target, str) and _COMPUTATION_ID.match(target):
            value = self._from_computation(target)["value"]
            if not isinstance(value, (int, float, str, bool)):
                raise ComputationError(
                    f"a condition compares each cell with a single value, but {target} is "
                    f"{value!r}; compare with a number, a text, or a computation of one value"
                )
            return {"op": op, "value": value, "source": target}
        if isinstance(target, np.generic):
            target = target.item()
        if not isinstance(target, (int, float, str, bool)):
            raise ComputationError(
                "a condition's target is a number, a text, or an earlier computation ID"
            )
        return {"op": op, "value": target, "source": "constant"}

    def _from_computation(self, computation_id: str) -> dict[str, Any]:
        for record in self.computations:
            if record["id"] == computation_id:
                return {"reference": computation_id, "value": record["output"]}
        raise ComputationError(f"computation {computation_id} does not exist")

    def _from_workbook(self, reference: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Read a reference's cells in row order; return (inputs, skipped blanks)."""
        parsed = parse_citation(reference)
        if parsed is None:
            raise ComputationError(
                f"{reference!r} is not a Sheet!A1 or Sheet!A1:B9 reference "
                "(quote sheet names with spaces: 'My Sheet'!A1)"
            )
        sheet, range_ref = parsed
        owners = [
            reader.workbook_id
            for reader in self.registry.readers()
            if sheet in reader.sheet_names()
        ]
        if len(owners) != 1:
            where = "no workbook" if not owners else f"several workbooks ({', '.join(owners)})"
            raise ComputationError(f"sheet {sheet!r} is in {where}")
        facade: WorkbookFacade = self.namespace["wb"]
        label = f"'{sheet}'" if _NEEDS_QUOTES.search(sheet) else sheet
        single = ":" not in range_ref
        inputs: list[dict[str, Any]] = []
        blanks: list[str] = []
        for row in facade.range(range_ref, sheet_name=sheet, workbook_id=owners[0]):
            for cell in row:
                coordinate = f"{label}!{cell['coordinate']}"
                merged = cell.get("merged_with")
                if merged and not merged.startswith(f"{cell['coordinate']}:"):
                    continue  # a covered merged cell repeats its anchor's value
                if cell.get("formula") is not None:
                    if cell.get("displayed") is None:
                        raise ComputationError(
                            f"{coordinate} is a formula with no cached value, so its value "
                            "is unknown"
                        )
                    value = cell["displayed"]
                else:
                    value = cell.get("value")
                if value is None:
                    if single:
                        raise ComputationError(f"{coordinate} is blank")
                    blanks.append(coordinate)
                    continue
                inputs.append({"reference": coordinate, "value": value})
        return inputs, blanks

    def run(self, code: str) -> dict[str, Any]:
        """Execute one snippet and return its JSON-safe outcome."""
        self.calls.clear()
        before = len(self.computations)
        stdout, stderr = io.StringIO(), io.StringIO()
        error: str | None = None
        result: Any = None
        self.namespace.pop(RESULT_NAME, None)
        try:
            tree = ast.parse(code)
            # Checked against the live namespace, before any line runs.
            error = refusal(tree, self.namespace)
            if error is None:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exec(compile(tree, "<agent_code>", "exec"), self.namespace)  # noqa: S102
                result = _result_view(self.namespace.get(RESULT_NAME), self.max_inline_cells)
        except (FacadeError, ToolError, ComputationError) as exc:
            error = str(exc)
        except MemoryError:
            error = "the action exceeded its memory limit"
        except SyntaxError as exc:
            error = f"SyntaxError: {exc.msg} (line {exc.lineno})"
        # Nothing generated code raises may kill the worker, and only the exception
        # type and message cross the boundary -- never the stack.
        except BaseException as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        captured = stdout.getvalue()
        truncated = len(captured) > self.max_output_chars
        recorded = self.computations[before:]
        state = (
            ResultState.ERROR
            if error is not None
            else (
                ResultState.TRUNCATED
                if truncated
                else ResultState.EMPTY if not captured and result is None else ResultState.SUCCESS
            )
        )
        return {
            "state": state.value,
            "stdout": _both_ends(captured, self.max_output_chars) if truncated else captured,
            "stderr": stderr.getvalue()[: self.max_output_chars],
            "result": None if error is not None else result,
            "error": error,
            "facade_calls": list(self.calls),
            "computation_ids": [record["id"] for record in recorded],
            "computations": recorded,
        }


# Common spellings of the row-selection conditions, mapped to their names.
_CONDITION_SPELLINGS = {
    "eq": "equals", "==": "equals", "=": "equals", "equal": "equals",
    "ne": "not_equals", "!=": "not_equals", "neq": "not_equals",
    ">": "gt", ">=": "ge", "gte": "ge", "<": "lt", "<=": "le", "lte": "le",
    "in": "contains",
}  # fmt: skip


def _row_numbers(output: Any) -> list[int]:
    """A row selection's output as sorted worksheet row numbers, numpy ones included."""
    if isinstance(output, (pd.Index, np.ndarray)):
        output = output.tolist()
    if isinstance(output, (list, tuple)):
        rows = [item.item() if isinstance(item, np.generic) else item for item in output]
        if all(isinstance(row, int) and not isinstance(row, bool) for row in rows):
            return sorted(rows)
    raise ComputationError(
        f"the output of {ROW_SELECTION} is the list of worksheet row numbers that meet the "
        f"condition, e.g. [4, 17] (got {output!r})"
    )


def _require_one_column(inputs: Any) -> None:
    """A row selection reads exactly one column range, so each cell names one row."""
    ranges = [item for item in inputs] if isinstance(inputs, (list, tuple)) else []
    parsed = parse_citation(ranges[0]) if len(ranges) == 1 and isinstance(ranges[0], str) else None
    if parsed is not None:
        min_col, _, max_col, _ = range_boundaries(parsed[1])
        if min_col == max_col:
            return
    raise ComputationError(
        f"{ROW_SELECTION} selects rows of one column: give exactly one single-column range, "
        "e.g. ['Sheet!C2:C41']"
    )


def _limit_memory(megabytes: int) -> bool:
    """Lower the address-space limit when the platform allows it.

    ``resource`` is POSIX-only, so this import stays local: a top-level import would
    stop the worker from starting at all on Windows.

    Returns:
        Whether the limit is actually in force.
    """
    try:
        import resource
    except ImportError:
        return False
    limit = megabytes * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, resource.RLIM_INFINITY))
    except (ValueError, OSError, AttributeError):
        return False
    return True


def main() -> int:
    """Serve one session until stdin closes."""
    config = json.loads(sys.stdin.readline())
    enforced = _limit_memory(int(config["memory_limit_mb"]))
    session = Session(config)
    sys.stdout.write(json.dumps({"ready": True, "memory_limit_enforced": enforced}) + "\n")
    sys.stdout.flush()
    while True:
        # readline, not iteration: iterating a pipe uses a read-ahead buffer that can
        # withhold a complete command until more input arrives.
        line = sys.stdin.readline()
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        command = json.loads(line)
        if command.get("command") == "names":
            payload: dict[str, Any] = {
                "names": sorted(
                    name
                    for name in session.namespace
                    if not name.startswith("__") and name not in INJECTED_NAMES
                )
            }
        else:
            payload = session.run(command["code"])
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
