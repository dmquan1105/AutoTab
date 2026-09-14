"""Small records shared by the QA modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """A parsed model action."""

    thought: str
    code: str


@dataclass(frozen=True)
class FinalAnswer:
    """A parsed final model answer."""

    thought: str
    answer: str


@dataclass(frozen=True)
class Observation:
    """The safe output from one sandbox action."""

    text: str
    success: bool


@dataclass(frozen=True)
class QAResult:
    """The outcome of a bounded QA run."""

    success: bool
    answer: str
    turns: int
