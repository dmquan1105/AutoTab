"""State-machine shape and turn-budget helpers.

This module does not call models, tools, phases, or the lifecycle loop. It imports
the shared ``AgentState`` dataclass and only answers questions about legal
transitions and turn accounting; see ``specs/layers/agent_state.md``.
"""

from __future__ import annotations

from ..contracts import AgentState

__all__ = [
    "TRANSITIONS",
    "AgentState",
    "advance_turn",
    "can_transition",
    "max_turns_reached",
]

TERMINAL_PHASES = frozenset({"FINALIZE", "FAIL"})

TRANSITIONS: dict[str, frozenset[str]] = {
    "INITIALIZE": frozenset({"OBSERVE", "FAIL"}),
    "OBSERVE": frozenset({"EXECUTE", "FAIL"}),
    "EXECUTE": frozenset({"VERIFY", "OBSERVE", "FAIL"}),
    "VERIFY": frozenset({"OBSERVE", "FINALIZE", "FAIL"}),
    "FINALIZE": frozenset(),
    "FAIL": frozenset(),
}


def can_transition(current: str, target: str, *, action_type: str | None = None) -> bool:
    """Return whether the state machine permits ``current`` -> ``target``.

    Args:
        current: Phase the run is leaving.
        target: Phase the run would enter.
        action_type: The action of the turn, when leaving EXECUTE or VERIFY. Every
            valid action is verified; an EXECUTE that produced no valid action returns
            to OBSERVE, since there is nothing to verify; and only a verified
            ``answer`` can finalize.

    Returns:
        True when the transition is legal.
    """
    if target not in TRANSITIONS.get(current, frozenset()):
        return False
    if current == "EXECUTE" and target == "VERIFY":
        return action_type in {"tool", "code", "answer"}
    if current == "EXECUTE" and target == "OBSERVE":
        return action_type is None
    if current == "VERIFY" and target == "FINALIZE":
        return action_type == "answer"
    return True


def advance_turn(state: AgentState) -> AgentState:
    """Count one dispatched EXECUTE action without selecting the next phase."""
    state.turn += 1
    return state


def max_turns_reached(state: AgentState) -> bool:
    """Return whether the execution budget is exhausted."""
    return state.turn >= state.max_turns
