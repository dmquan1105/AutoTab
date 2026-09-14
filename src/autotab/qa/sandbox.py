"""Isolated execution boundary for model-generated QA code."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .types import Observation


class Sandbox:
    """Run one generated action at a time in a constrained child process."""

    def __init__(
        self,
        workbook_path: str | Path,
        timeout_seconds: int,
        memory_limit_mb: int,
        max_observation_chars: int,
        max_range_cells: int,
    ) -> None:
        self._payload = {
            "workbook_path": str(Path(workbook_path).resolve()),
            "max_observation_chars": max_observation_chars,
            "max_range_cells": max_range_cells,
        }
        self._timeout_seconds = timeout_seconds
        self._memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self._max_observation_chars = max_observation_chars

    def execute(self, code: str) -> Observation:
        """Execute approved code and return bounded, safe output.

        Args:
            code: Model-generated Python that has already passed the size limit.

        Returns:
            A successful output or safe execution error observation.
        """
        payload = json.dumps({**self._payload, "code": code})
        process = subprocess.Popen(
            [sys.executable, "-I", "-m", "autotab.qa.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        assert process.stdin is not None
        process.stdin.write(payload)
        process.stdin.close()
        process.stdin = None
        deadline = time.monotonic() + self._timeout_seconds
        error = ""
        while process.poll() is None:
            if time.monotonic() >= deadline:
                error = "Execution timed out"
                process.kill()
                break
            memory = _memory_bytes(process.pid)
            if memory is not None and memory > self._memory_limit_bytes:
                error = "Execution exceeded the memory limit"
                process.kill()
                break
            time.sleep(0.01)
        stdout, stderr = process.communicate()
        if error:
            return Observation(error, False)
        if process.returncode != 0:
            message = stderr.strip() or "Sandbox process failed"
            return Observation(message[: self._max_observation_chars], False)
        return Observation(stdout.rstrip(), True)


def _memory_bytes(process_id: int) -> int | None:
    """Return child resident memory when the operating system exposes it."""
    if os.name == "nt":
        return _windows_memory_bytes(process_id)
    status = Path(f"/proc/{process_id}/status")
    if not status.exists():
        return None
    for line in status.read_text(encoding="ascii").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _windows_memory_bytes(process_id: int) -> int | None:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    query_information = 0x0400
    handle = ctypes.windll.kernel32.OpenProcess(query_information, False, process_id)
    if not handle:
        return None
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else None
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
