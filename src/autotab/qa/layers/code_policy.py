"""The allowlist a generated snippet must satisfy before any of it runs.

Pure and namespace-aware: the sandbox worker calls ``refusal`` with its live session
namespace, so a module imported on an earlier turn keeps its attribute allowlist and a
name bound earlier counts as defined. Everything not listed here is refused, with a
message the model can act on; see ``specs/layers/sandbox.md``.

This is a guardrail against mistakes by our own model, not a boundary against hostile
code (see the threat model in ``layers/sandbox.py``). Refusing before execution also
means a snippet never half-runs and leaves a partial namespace behind.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import string
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Any

ALLOWED_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "float", "int",
    "isinstance", "len", "list", "max", "min", "print", "range", "round", "set",
    "sorted", "str", "sum", "tuple", "zip",
)  # fmt: skip

# Names the harness binds; code may use them but never rebind them.
INJECTED_NAMES = frozenset(
    {"pd", "math", "statistics", "datetime", "re", "json", "wb", "record_computation"}
)
WORKBOOK_METHODS = frozenset({"sheets", "sheet", "range", "attributes", "search"})

# Module attributes code may use. ``None`` means every public attribute that is not
# itself a module: those modules hold pure functions, but they also re-export modules
# (``statistics.sys``) that lead to the rest of the interpreter.
PANDAS_ATTRIBUTES = frozenset(
    {
        "DataFrame", "Series", "Index", "MultiIndex", "Categorical", "NA", "NaT",
        "Timestamp", "Timedelta", "Period", "DateOffset", "Grouper", "NamedAgg",
        "IndexSlice", "concat", "merge", "crosstab", "pivot_table", "melt", "cut",
        "qcut", "unique", "factorize", "get_dummies", "isna", "isnull", "notna",
        "notnull", "to_numeric", "to_datetime", "to_timedelta", "date_range",
        "period_range",
    }
)  # fmt: skip
MODULE_ATTRIBUTES: dict[str, frozenset[str] | None] = {
    "pandas": PANDAS_ATTRIBUTES,
    "math": None,
    "statistics": None,
    "re": None,
    "datetime": frozenset({"date", "datetime", "time", "timedelta", "timezone"}),
    "json": frozenset({"dumps", "loads"}),
}
IMPORTABLE_MODULES = frozenset(MODULE_ATTRIBUTES)

# Statements and expressions a snippet may contain. Loops over data, conditionals,
# assignments, comprehensions, and calls cover the analyses we need; everything else
# either makes cost independent of the data or has no use in straight-line analysis.
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Pass,
    ast.Break, ast.Continue, ast.Import, ast.ImportFrom, ast.alias,
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.keyword, ast.IfExp,
    ast.Attribute, ast.Subscript, ast.Slice, ast.Starred, ast.Name, ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.comprehension, ast.JoinedStr, ast.FormattedValue,
    ast.operator, ast.boolop, ast.cmpop, ast.unaryop, ast.Load, ast.Store,
)  # fmt: skip
_CONSTRUCT_HINTS: dict[type[ast.AST], str] = {
    ast.While: "while loops are not available; iterate over the data you read",
    ast.FunctionDef: "function definitions are not available; write straight-line code",
    ast.AsyncFunctionDef: "function definitions are not available; write straight-line code",
    ast.Lambda: "lambdas are not available; write straight-line code or a comprehension",
    ast.ClassDef: "class definitions are not available; write straight-line code",
    ast.Try: "try/except is not available; check values with if instead",
}

# pandas reads and writes arbitrary paths; the workbook is read through `wb` only.
_FILE_ATTRIBUTES = frozenset({"ExcelWriter", "ExcelFile", "HDFStore", "io"})
_SAFE_CONVERSIONS = frozenset(
    {
        "to_dict", "to_list", "to_numpy", "to_string", "to_frame", "to_records",
        "to_numeric", "to_datetime", "to_timedelta", "to_period", "to_timestamp",
        "to_pydatetime", "to_flat_index", "to_series",
    }
)  # fmt: skip
# Methods that evaluate strings as code (eval, query), plot through matplotlib, expose
# raw memory (ctypes), write files (tofile, dump), or walk frames and code objects.
_DENIED_ATTRIBUTES = frozenset(
    {"eval", "query", "style", "plot", "hist", "boxplot", "ctypes", "tofile", "dump", "mro"}
)
_DENIED_PREFIXES = ("gi_", "cr_", "ag_", "f_", "tb_", "co_")
# Introspection builtins models reach for; refused with the way to look instead.
_INTROSPECTION = frozenset(
    {"type", "dir", "vars", "getattr", "hasattr", "setattr", "repr", "callable", "id", "help"}
)


def refusal(tree: ast.Module, namespace: Mapping[str, Any]) -> str | None:
    """Return why ``tree`` must not run in ``namespace``, or None when it may."""
    return _Policy(tree, namespace).check()


def _is_file_attribute(name: str) -> bool:
    if name.startswith("read_") or name in _FILE_ATTRIBUTES:
        return True
    return name.startswith("to_") and name not in _SAFE_CONVERSIONS


def _module_allows(module: ModuleType, attribute: str) -> bool:
    allowed = MODULE_ATTRIBUTES.get(module.__name__, frozenset())
    if allowed is not None:
        return attribute in allowed
    value = getattr(module, attribute, None)
    return not attribute.startswith("_") and not isinstance(value, ModuleType)


def _introspection_hint(name: str) -> str:
    return (
        f"{name}() is not available; to see what a value holds, print(x) it or test "
        "isinstance(x, dict), isinstance(x, list), ..."
    )


def _plain_format(template: str) -> bool:
    """Whether a format string only fills fields, without attribute or index lookups."""
    try:
        fields = [name for _, name, _, _ in string.Formatter().parse(template) if name]
    except ValueError:
        return False
    return all(name.isdigit() or name.isidentifier() for name in fields)


class _Policy:
    def __init__(self, tree: ast.Module, namespace: Mapping[str, Any]) -> None:
        self.tree = tree
        self.namespace = namespace
        self.builtins = frozenset(ALLOWED_BUILTINS)
        # Names this snippet binds, and what its imports bind them to.
        self.bound: set[str] = set()
        self.imported_modules: dict[str, ModuleType] = {}
        self.imported_objects: dict[str, Any] = {}

    # --- resolution ------------------------------------------------------------------

    def _module(self, name: str) -> ModuleType | None:
        if name in self.imported_modules:
            return self.imported_modules[name]
        value = self.namespace.get(name)
        return value if isinstance(value, ModuleType) and name not in self.bound else None

    def _allowed_callables(self) -> Iterable[Any]:
        for module_name, allowed in MODULE_ATTRIBUTES.items():
            module = self._importable(module_name)
            names = allowed if allowed is not None else dir(module)
            for attribute in names:
                if _module_allows(module, attribute):
                    yield getattr(module, attribute, None)

    @staticmethod
    def _importable(name: str) -> ModuleType:
        return importlib.import_module(name)

    def _target(self, func: ast.expr) -> Any:
        """The object a call resolves to, when it is known without running anything."""
        if isinstance(func, ast.Name):
            if func.id in self.imported_objects:
                return self.imported_objects[func.id]
            if func.id in self.bound:
                return None
            return self.namespace.get(func.id)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = self._module(func.value.id)
            if module is not None:
                return getattr(module, func.attr, None)
            if func.value.id == "wb" and "wb" in self.namespace:
                return getattr(self.namespace["wb"], func.attr, None)
        return None

    # --- checks ----------------------------------------------------------------------

    def check(self) -> str | None:
        nodes = list(ast.walk(self.tree))
        for node in nodes:
            refused = self._construct(node)
            if refused is not None:
                return refused
        for node in nodes:
            refused = self._import(node) if isinstance(node, (ast.Import, ast.ImportFrom)) else None
            if refused is not None:
                return refused
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.bound.add(node.id)
        module_values = {id(node.value) for node in nodes if isinstance(node, ast.Attribute)}
        for node in nodes:
            refused = self._node(node, module_values)
            if refused is not None:
                return refused
        return None

    def _construct(self, node: ast.AST) -> str | None:
        if isinstance(node, _ALLOWED_NODES):
            if isinstance(node, ast.comprehension) and node.is_async:
                return "async comprehensions are not available"
            return None
        hint = _CONSTRUCT_HINTS.get(type(node))
        if hint is not None:
            return hint
        label = type(node).__name__
        return (
            f"{label} is not available; use assignments, for loops over data, if, "
            "comprehensions, and calls"
        )

    def _import(self, node: ast.Import | ast.ImportFrom) -> str | None:
        allowed = ", ".join(sorted(IMPORTABLE_MODULES))
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in IMPORTABLE_MODULES:
                    return (
                        f"import of {alias.name!r} is not available; only {allowed} can be "
                        "imported, and they are already bound (pd, math, statistics, "
                        "datetime, re, json)"
                    )
                module = self._importable(alias.name)
                name = alias.asname or alias.name
                refused = self._rebinds(name, module)
                if refused is not None:
                    return refused
                self.imported_modules[name] = module
            return None
        if node.level != 0 or node.module not in IMPORTABLE_MODULES:
            return (
                f"import from {node.module or '.' * node.level!r} is not available; only "
                f"{allowed} can be imported"
            )
        module = self._importable(node.module)
        for alias in node.names:
            if alias.name == "*" or not _module_allows(module, alias.name):
                return f"from {node.module} import {alias.name} is not available"
            name = alias.asname or alias.name
            value = getattr(module, alias.name)
            refused = self._rebinds(name, value)
            if refused is not None:
                return refused
            self.imported_objects[name] = value
        return None

    def _rebinds(self, name: str, value: Any) -> str | None:
        """Refuse binding a harness or builtin name to something else."""
        if name in self.builtins or (
            name in INJECTED_NAMES and self.namespace.get(name) is not value
        ):
            hint = " (write datetime.datetime(...) instead)" if name == "datetime" else ""
            return f"do not rebind {name!r}; it is provided by the session{hint}"
        return None

    def _node(self, node: ast.AST, module_values: set[int]) -> str | None:
        if isinstance(node, ast.Name):
            return self._name(node, module_values)
        if isinstance(node, ast.Attribute):
            return self._attribute(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        return None

    def _name(self, node: ast.Name, module_values: set[int]) -> str | None:
        if node.id.startswith("_"):
            return f"name {node.id!r} is not available"
        if node.id in _INTROSPECTION and node.id not in self.bound:
            return _introspection_hint(node.id)
        if isinstance(node.ctx, ast.Store):
            if node.id in self.builtins or node.id in INJECTED_NAMES:
                return self._rebinds(node.id, object())
            return None
        known = (
            node.id in self.bound
            or node.id in self.builtins
            or node.id in self.imported_modules
            or node.id in self.imported_objects
            or (node.id in self.namespace and not node.id.startswith("_"))
        )
        if not known:
            return f"NameError: name {node.id!r} is not defined in this session"
        if self._module(node.id) is not None and id(node) not in module_values:
            return (
                f"the module {node.id!r} can only be used as {node.id}.<name>; it cannot "
                "be assigned, passed, or stored"
            )
        return None

    def _attribute(self, node: ast.Attribute) -> str | None:
        name = node.attr
        if name.startswith("_") or name.startswith(_DENIED_PREFIXES):
            return f"attribute {name!r} is not available"
        if _is_file_attribute(name):
            return (
                f"file access through {name!r} is not available; read the workbook with "
                "wb.sheet(...) or wb.range(...)"
            )
        if name in _DENIED_ATTRIBUTES:
            return f"attribute {name!r} is not available; compute with ordinary pandas instead"
        if name in {"format", "format_map"}:
            receiver = node.value
            if not (
                isinstance(receiver, ast.Constant)
                and isinstance(receiver.value, str)
                and _plain_format(receiver.value)
            ):
                return (
                    f"{name!r} is not available here; use an f-string, or .format on a "
                    "literal string with plain {} fields"
                )
        if isinstance(node.value, ast.Name):
            owner = node.value.id
            module = self._module(owner)
            if module is not None:
                if isinstance(node.ctx, ast.Store) or not _module_allows(module, name):
                    return f"{owner}.{name} is not available"
            elif (
                owner == "wb"
                and owner not in self.bound
                and (isinstance(node.ctx, ast.Store) or name not in WORKBOOK_METHODS)
            ):
                methods = ", ".join(sorted(WORKBOOK_METHODS))
                return f"wb.{name} is not available; wb has {methods}"
        return None

    def _call(self, node: ast.Call) -> str | None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _INTROSPECTION and func.id not in self.bound:
                return _introspection_hint(func.id)
            if not self._callable_name(func.id):
                return (
                    f"calling {func.id!r} is not available; call builtins, "
                    "record_computation, module functions, or methods directly"
                )
        elif not isinstance(func, ast.Attribute):
            return (
                "call a function by its name or as a method; calling an expression is not available"
            )
        return self._arguments(node)

    def _callable_name(self, name: str) -> bool:
        if name in self.builtins or name == "record_computation" or name in self.imported_objects:
            return True
        if name in self.bound or name not in self.namespace:
            return False
        value = self.namespace[name]
        return any(value is allowed for allowed in self._allowed_callables())

    def _arguments(self, node: ast.Call) -> str | None:
        """Refuse a call whose arguments cannot bind to a known signature."""
        target = self._target(node.func)
        if target is None or isinstance(target, type) and target.__module__ == "builtins":
            return None
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            return None
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return None
        placeholder = object()
        try:
            signature.bind(
                *[placeholder] * len(node.args),
                **{keyword.arg: placeholder for keyword in node.keywords if keyword.arg},
            )
        except TypeError as exc:
            return f"{ast.unparse(node.func)}(): {exc}"
        return None
