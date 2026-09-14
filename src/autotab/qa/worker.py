"""Child-process implementation for the QA sandbox."""

from __future__ import annotations

import ast
import io
import json
import sys
from typing import Any

from .tools import WorkbookSession

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
TOOL_NAMES = {"load_dataframe", "inspect_range"}
SAFE_ATTRIBUTES = {
    "T",
    "agg",
    "all",
    "any",
    "at",
    "astype",
    "columns",
    "contains",
    "count",
    "describe",
    "drop_duplicates",
    "dropna",
    "dt",
    "dtypes",
    "fillna",
    "first",
    "get",
    "groupby",
    "head",
    "iat",
    "idxmax",
    "idxmin",
    "iloc",
    "index",
    "isin",
    "isna",
    "item",
    "items",
    "last",
    "loc",
    "lower",
    "max",
    "mean",
    "median",
    "min",
    "notna",
    "nunique",
    "reset_index",
    "round",
    "shape",
    "sort_index",
    "sort_values",
    "str",
    "strip",
    "sum",
    "tail",
    "to_dict",
    "tolist",
    "unique",
    "upper",
    "value_counts",
    "values",
}
ALLOWED_NODES = {
    ast.Add,
    ast.And,
    ast.Assign,
    ast.Attribute,
    ast.BinOp,
    ast.BitAnd,
    ast.BitOr,
    ast.BoolOp,
    ast.Call,
    ast.Compare,
    ast.comprehension,
    ast.Constant,
    ast.Dict,
    ast.DictComp,
    ast.Div,
    ast.Eq,
    ast.Expr,
    ast.FloorDiv,
    ast.For,
    ast.FormattedValue,
    ast.Gt,
    ast.GtE,
    ast.If,
    ast.IfExp,
    ast.In,
    ast.Is,
    ast.IsNot,
    ast.JoinedStr,
    ast.keyword,
    ast.List,
    ast.ListComp,
    ast.Load,
    ast.Lt,
    ast.LtE,
    ast.Mod,
    ast.Module,
    ast.Mult,
    ast.Name,
    ast.Not,
    ast.NotEq,
    ast.NotIn,
    ast.Or,
    ast.Pow,
    ast.Set,
    ast.SetComp,
    ast.Slice,
    ast.Store,
    ast.Sub,
    ast.Subscript,
    ast.Tuple,
    ast.UAdd,
    ast.UnaryOp,
    ast.USub,
}


class UnsafeCodeError(ValueError):
    """Raised when generated Python uses a non-allowlisted construct."""


class CodeValidator(ast.NodeVisitor):
    """Reject syntax and calls not needed for in-memory workbook analysis."""

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in ALLOWED_NODES:
            raise UnsafeCodeError(f"{type(node).__name__} is not allowed")
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr not in SAFE_ATTRIBUTES or not isinstance(node.ctx, ast.Load):
            raise UnsafeCodeError(f"Attribute {node.attr!r} is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_BUILTINS and node.func.id not in TOOL_NAMES:
                raise UnsafeCodeError(f"Call to {node.func.id!r} is not allowed")
        elif not isinstance(node.func, ast.Attribute):
            raise UnsafeCodeError("Indirect calls are not allowed")
        self.generic_visit(node)


class LimitedWriter(io.TextIOBase):
    """Capture at most a fixed number of output characters."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._parts: list[str] = []
        self._length = 0

    def write(self, text: str) -> int:
        remaining = self._limit - self._length
        if remaining > 0:
            value = text[:remaining]
            self._parts.append(value)
            self._length += len(value)
        return len(text)

    def value(self) -> str:
        return "".join(self._parts)


def execute(payload: dict[str, Any]) -> str:
    """Validate and execute one sandbox payload."""
    tree = ast.parse(str(payload["code"]), mode="exec")
    CodeValidator().visit(tree)
    session = WorkbookSession(payload["workbook_path"], int(payload["max_range_cells"]))
    writer = LimitedWriter(int(payload["max_observation_chars"]))
    globals_: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, **session.tools}
    previous_stdout = sys.stdout
    try:
        sys.stdout = writer
        exec(compile(tree, "<qa-action>", "exec"), globals_, {})  # noqa: S102
    finally:
        sys.stdout = previous_stdout
        session.close()
    return writer.value() or "No output. Print values needed for analysis."


def main() -> int:
    """Read one JSON payload, execute it, and emit only safe output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    payload: dict[str, Any] = json.loads(sys.stdin.read())
    try:
        print(execute(payload), end="")
        return 0
    except Exception as exc:  # noqa: BLE001 - child boundary must return safe errors
        path = str(payload.get("workbook_path", ""))
        message = str(exc).replace(path, "<workbook>")
        print(f"Execution error: {type(exc).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
