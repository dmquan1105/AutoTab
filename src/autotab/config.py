"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


DEFAULTS: dict[str, Any] = {
    "project": {"name": "autotab", "python_version": "3.11"},
    "models": {
        "llm": {"model": "gpt-5-mini"},
        "vlm": {"model": "gpt-5-mini"},
        "embedding": {"model": "text-embedding-3-small"},
    },
    "exploration": {
        "similarity_threshold": 0.72,
        "max_cells_per_keyword": 8,
        "max_concurrent_findings": 8,
        "window_size": 9,
        "send_query_to_vlm": False,
        "include_hidden_sheets": False,
        "uncertain_evidence": "drop",
    },
    "retrieval": {
        "lexical_weight": 0.45,
        "semantic_weight": 0.55,
        "allow_lexical_only_fallback": False,
    },
    "runtime": {
        "artifact_root": "outputs",
        "request_timeout_seconds": 60,
        "max_retries": 1,
        "max_input_file_mb": 100,
    },
    "rendering": {
        "backend": "libreoffice",
        "timeout_seconds": 60,
        "image_resolution": 600,
        "show_coordinates": True,
        "min_cell_width": 10,
        "max_cell_width": 50,
        "cell_width_padding": 3,
        "max_image_dimension": 8192,
        "max_image_pixels": 33554432,
        "trim_padding": 6,
    },
    "qa": {
        "exploration_enabled": True,
        "max_turns": 10,
        "max_code_chars": 10000,
        "max_observation_chars": 20000,
        "sandbox": {"timeout_seconds": 10, "memory_limit_mb": 512},
        "tools": {"max_range_cells": 10000},
    },
}

QA_KEYS = {
    "exploration_enabled",
    "max_turns",
    "max_code_chars",
    "max_observation_chars",
    "sandbox",
    "tools",
}
QA_SANDBOX_KEYS = {"timeout_seconds", "memory_limit_mb"}
QA_TOOL_KEYS = {"max_range_cells"}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _env_overrides(config: dict[str, Any]) -> None:
    prefix = "AUTOTAB_"
    for key, value in os.environ.items():
        if not key.startswith(prefix) or "__" not in key:
            continue
        parts = key[len(prefix) :].lower().split("__")
        target: dict[str, Any] = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        raw = value
        parsed: Any = yaml.safe_load(raw)
        target[parts[-1]] = parsed


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, merge, validate, and return YAML configuration."""
    source = Path(path)
    data = (
        yaml.safe_load(source.read_text(encoding="utf-8"))
        if source.exists() and source.read_text(encoding="utf-8").strip()
        else {}
    )
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping")
    config = _merge(DEFAULTS, data)
    _env_overrides(config)
    validate_config(config)
    return config


def validate_config(c: dict[str, Any]) -> None:
    """Validate user-facing constraints."""
    e, r = c["exploration"], c["retrieval"]
    if not 0 <= float(e["similarity_threshold"]) <= 1:
        raise ConfigError("exploration.similarity_threshold must be in [0, 1]")
    if int(e["max_cells_per_keyword"]) <= 0:
        raise ConfigError("exploration.max_cells_per_keyword must be positive")
    if int(e["window_size"]) <= 0 or int(e["window_size"]) % 2 == 0:
        raise ConfigError("exploration.window_size must be a positive odd integer")
    if int(e["max_concurrent_findings"]) <= 0:
        raise ConfigError("exploration.max_concurrent_findings must be positive")
    if not isinstance(e["send_query_to_vlm"], bool):
        raise ConfigError("exploration.send_query_to_vlm must be boolean")
    lw, sw = float(r["lexical_weight"]), float(r["semantic_weight"])
    if not (0 <= lw <= 1 and 0 <= sw <= 1 and abs(lw + sw - 1) < 1e-6):
        raise ConfigError("retrieval weights must be in [0,1] and sum to 1")
    if e.get("uncertain_evidence", "drop") not in {"keep", "drop"}:
        raise ConfigError("uncertain_evidence must be keep or drop")
    qa = c["qa"]
    _reject_unknown_keys("qa", qa, QA_KEYS)
    _reject_unknown_keys("qa.sandbox", qa["sandbox"], QA_SANDBOX_KEYS)
    _reject_unknown_keys("qa.tools", qa["tools"], QA_TOOL_KEYS)
    if not isinstance(qa["exploration_enabled"], bool):
        raise ConfigError("qa.exploration_enabled must be boolean")
    limits = {
        "qa.max_turns": qa["max_turns"],
        "qa.max_code_chars": qa["max_code_chars"],
        "qa.max_observation_chars": qa["max_observation_chars"],
        "qa.sandbox.timeout_seconds": qa["sandbox"]["timeout_seconds"],
        "qa.sandbox.memory_limit_mb": qa["sandbox"]["memory_limit_mb"],
        "qa.tools.max_range_cells": qa["tools"]["max_range_cells"],
    }
    for name, value in limits.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{name} must be a positive integer")


def _reject_unknown_keys(section: str, values: object, allowed: set[str]) -> None:
    if not isinstance(values, dict):
        raise ConfigError(f"{section} must be a mapping")
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(f"Unknown {section} key: {min(unknown)}")
