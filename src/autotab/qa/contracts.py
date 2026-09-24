"""The mutable in-process state of one QA run.

This module defines exactly one type. Every other value crossing a layer or phase
boundary is a strict Pydantic model from ``schemas.py``; see ``specs/contracts.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .schemas import CandidateAnswer, Observation, VerificationResult


@dataclass
class AgentState:
    """Run state carried across phases; transitions live in ``layers/agent_state``."""

    query: str
    max_turns: int
    phase: str = "INITIALIZE"
    turn: int = 0
    objective: str | None = None
    candidate: CandidateAnswer | None = None
    latest_observation: Observation | None = None
    latest_verification: VerificationResult | None = None
    history_refs: list[str] = field(default_factory=list)
