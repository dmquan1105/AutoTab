"""Bounded code session for generated Python.

One isolated worker subprocess per run holds the namespace, the injected safe
objects, and the read-only workbook facade; see ``specs/layers/sandbox.md``.

Threat model: the code comes from our own model reading a spreadsheet, so these are
guardrails against mistakes -- static checks, restricted builtins, a scratch working
directory, a timeout -- not a boundary against hostile code. Code that can reach
attributes of pandas objects could still find a way out; a real boundary would be a
container or seccomp, not more static checks.
"""

from __future__ import annotations

import ast
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from ..schemas import ComputationRecord, ResultState, SandboxResult
from .tools import ToolRegistry, WorkbookReader

WORKER_MODULE = "autotab.qa.layers._sandbox_worker"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MEMORY_LIMIT_MB = 512
DEFAULT_MAX_CODE_CHARS = 10_000
DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_MAX_NODES = 2_000
DEFAULT_MAX_INLINE_CELLS = 200  # larger `result` values come back as a preview

_STARTUP_TIMEOUT_SECONDS = 60.0
_SHUTDOWN_GRACE_SECONDS = 2.0


def memory_limit_supported() -> bool:
    """Return whether this platform lets an unprivileged process lower RLIMIT_AS.

    macOS refuses to lower the address-space limit, so the sandbox reports the
    limit as unenforced there instead of pretending it applies. The timeout and
    process isolation still bound a runaway action on every platform.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - POSIX only
        return False
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        return False
    return sys.platform.startswith("linux")


class Sandbox:
    """A persistent, bounded code session for one QA run."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        max_code_chars: int = DEFAULT_MAX_CODE_CHARS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_inline_cells: int = DEFAULT_MAX_INLINE_CELLS,
    ) -> None:
        self._registry = registry
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.max_code_chars = max_code_chars
        self.max_output_chars = max_output_chars
        self.max_nodes = max_nodes
        self.max_inline_cells = max_inline_cells
        self.computations: dict[str, ComputationRecord] = {}
        self.memory_limit_enforced = False
        self._process: subprocess.Popen[str] | None = None
        self.workdir: Path | None = None
        self._replies: queue.Queue[str] = queue.Queue()
        self._start()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _config(self) -> dict[str, Any]:
        return {
            "workbooks": [
                {
                    "workbook_id": reader.workbook_id,
                    # Absolute: the worker runs in a scratch directory, not here.
                    "path": str(reader.path.resolve()),
                    "include_hidden": reader.include_hidden,
                    "max_range_cells": reader.max_range_cells,
                }
                for reader in self._readers()
            ],
            "memory_limit_mb": self.memory_limit_mb,
            "max_output_chars": self.max_output_chars,
            "max_inline_cells": self.max_inline_cells,
        }

    def _readers(self) -> list[WorkbookReader]:
        return self._registry.readers()

    def _start(self) -> None:
        environment = dict(os.environ)
        package_root = str(Path(__file__).resolve().parents[3])
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_root if not existing else os.pathsep.join([package_root, existing])
        )
        # Pin both ends to UTF-8: the default console encoding on Windows is cp1252,
        # which cannot carry a sheet name like "Doanh thu quý 1".
        environment["PYTHONIOENCODING"] = "utf-8"
        # A scratch working directory: a relative write that slips past the static
        # checks lands here and is deleted, never in the repository.
        self.workdir = Path(tempfile.mkdtemp(prefix="autotab_sandbox_"))
        self._process = subprocess.Popen(
            [sys.executable, "-m", WORKER_MODULE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=environment,
            cwd=self.workdir,
        )
        self._replies = queue.Queue()
        # The queue is passed, never read from self: after a restart replaces it, the
        # previous worker's pump thread must not be able to answer for the new one.
        threading.Thread(
            target=self._pump, args=(self._process, self._replies), daemon=True
        ).start()
        self._send(self._config())
        ready = self._receive(_STARTUP_TIMEOUT_SECONDS)
        if ready is None:
            raise RuntimeError("the sandbox worker did not start")
        self.memory_limit_enforced = bool(ready.get("memory_limit_enforced", False))

    @staticmethod
    def _pump(process: subprocess.Popen[str], replies: queue.Queue[str]) -> None:
        """Drain one worker's stdout into the queue that belongs to that worker."""
        if process.stdout is None:  # pragma: no cover - always a pipe here
            return
        try:
            for line in process.stdout:
                replies.put(line)
        except (ValueError, OSError):
            # close() shuts the stream under this thread on purpose; that is the
            # signal to stop draining, not an error worth reporting.
            return

    def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("the sandbox session is closed")
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def _receive(self, timeout: float) -> dict[str, Any] | None:
        try:
            line = self._replies.get(timeout=timeout)
        except queue.Empty:
            return None
        parsed: dict[str, Any] = json.loads(line)
        return parsed

    def _restart(self) -> None:
        # The worker is wedged in a loop it will never leave, so do not wait politely.
        self.close(force=True)
        self._start()

    def bound_names(self) -> list[str]:
        """Return the names generated code has bound in this session."""
        self._send({"command": "names"})
        reply = self._receive(self.timeout_seconds)
        if reply is None:
            self._restart()
            return []
        names: list[str] = reply.get("names", [])
        return names

    def execute(self, code: str, execution_id: str) -> SandboxResult:
        """Run one snippet in the session and return its bounded result."""
        rejection = self._reject(code)
        if rejection is not None:
            return SandboxResult(
                state=ResultState.ERROR, execution_id=execution_id, error=rejection
            )
        self._send({"code": code})
        reply = self._receive(self.timeout_seconds)
        if reply is None:
            self._restart()
            return SandboxResult(
                state=ResultState.ERROR,
                execution_id=execution_id,
                error=(
                    f"the action timed out after {self.timeout_seconds:g}s; the session was "
                    "restarted and previously bound names are gone"
                ),
            )
        for record in reply.pop("computations", []):
            computation = ComputationRecord.model_validate_json(json.dumps(record))
            self.computations[computation.id] = computation
        return SandboxResult.model_validate_json(
            json.dumps({**reply, "execution_id": execution_id})
        )

    def _reject(self, code: str) -> str | None:
        """Return why this snippet must not run, or None when it may."""
        if not code.strip():
            return "the code action is empty"
        if len(code) > self.max_code_chars:
            return (
                f"the code is {len(code)} chars, above the limit of " f"{self.max_code_chars} chars"
            )
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"SyntaxError: {exc.msg} (line {exc.lineno})"
        nodes = list(ast.walk(tree))
        if len(nodes) > self.max_nodes:
            return (
                f"the code has {len(nodes)} syntax nodes, above the limit of "
                f"{self.max_nodes}; split the work across turns"
            )
        # Everything else -- constructs, names, attributes, calls -- is checked by
        # layers/code_policy.py inside the worker, against the live session namespace.
        return None

    def close(self, *, force: bool = False) -> None:
        """Stop the worker; the namespace does not survive.

        Args:
            force: Kill immediately instead of asking the worker to finish. Used when
                the worker is known to be stuck, where the polite path only costs
                the shutdown grace period.
        """
        process, self._process = self._process, None
        if process is None:
            return
        if force:
            process.kill()
        elif process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:  # pragma: no cover - the worker may already be gone
                pass
        try:
            process.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:  # pragma: no cover - already torn down
                    pass
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None
