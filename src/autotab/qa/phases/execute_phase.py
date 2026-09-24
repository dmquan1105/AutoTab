"""One EXECUTE interaction: call the model, validate one AgentAction, dispatch it.

Dispatches exactly one tool call, one sandbox run, or one answer per turn, then
normalizes the result and runs the action-scope checks. It never judges the answer
or transitions the FSM; see ``specs/phases/execute_phase.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import TypeAdapter

from ...models.client import ModelClient
from ..layers.context_management import AssembledPrompt, ContextRequest, assemble_context
from ..layers.observation import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_INLINE_CELLS,
    normalize_sandbox,
    normalize_tool,
)
from ..layers.verification import extract_json_object, run_action_checks
from ..schemas import (
    AgentAction,
    CodeAction,
    ComputationRecord,
    DeterministicCheckReport,
    Observation,
    Phase,
    SandboxResult,
    ToolAction,
    ToolResult,
)
from ._ask import Reply, ask

_ACTION: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)


class ToolDispatcher(Protocol):
    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult: ...


class CodeRunner(Protocol):
    @property
    def computations(self) -> Mapping[str, ComputationRecord]: ...

    def execute(self, code: str, execution_id: str) -> SandboxResult: ...


@dataclass(frozen=True)
class ExecuteOutcome:
    """The action, what running it produced, and its action-scope checks.

    ``raw_result``, ``observation``, and ``report`` are None for an answer action and
    when no valid action came back.
    """

    prompt: AssembledPrompt
    reply: Reply[AgentAction]
    raw_result: ToolResult | SandboxResult | None = None
    observation: Observation | None = None
    report: DeterministicCheckReport | None = None
    computations: tuple[ComputationRecord, ...] = ()

    @property
    def action(self) -> AgentAction | None:
        return self.reply.value


def _parse(raw: str) -> AgentAction:
    return _ACTION.validate_json(extract_json_object(raw))


def _effect(action: AgentAction) -> tuple[str, ...]:
    """What an action does, ignoring how its rationale is worded."""
    if isinstance(action, ToolAction):
        return ("tool", action.tool_name, json.dumps(action.arguments, sort_keys=True))
    if isinstance(action, CodeAction):
        return ("code", action.code.strip())
    return ("answer", action.answer.model_dump_json())


def _parser(previous: AgentAction | None) -> Callable[[str], AgentAction]:
    """Parse a reply, refusing an unchanged repeat of the previous action.

    specs/phases/agent.md: never repeat an unchanged action -- its result is already
    in the context. Raising here reuses the phase's re-ask, so a repeat costs no turn.
    """

    def parse(raw: str) -> AgentAction:
        action = _parse(raw)
        if previous is not None and _effect(action) == _effect(previous):
            raise ValueError(
                "this repeats the previous action unchanged, and its result is already "
                "above; choose a different action that makes progress"
            )
        return action

    return parse


def run(
    request: ContextRequest,
    client: ModelClient,
    *,
    registry: ToolDispatcher,
    sandbox: CodeRunner,
    execution_id: str,
    retries: int = 1,
    max_observation_chars: int = DEFAULT_MAX_CHARS,
    max_inline_cells: int = DEFAULT_MAX_INLINE_CELLS,
    previous: AgentAction | None = None,
) -> ExecuteOutcome:
    """Choose and run exactly one action.

    Raises:
        ValueError: If the request is not for the EXECUTE phase.
        ContextBudgetError: If required content cannot fit the context budget.
    """
    if request.phase is not Phase.EXECUTE:
        raise ValueError(f"execute_phase cannot build a {request.phase.value} prompt")
    prompt = assemble_context(request)
    reply = ask(client, prompt.text, _parser(previous), retries=retries)
    action = reply.value
    if isinstance(action, ToolAction):
        tool_result = registry.dispatch(action.tool_name, dict(action.arguments))
        return ExecuteOutcome(
            prompt,
            reply,
            raw_result=tool_result,
            observation=normalize_tool(
                tool_result, max_chars=max_observation_chars, max_inline_cells=max_inline_cells
            ),
            report=run_action_checks(tool_result.state, []),
        )
    if isinstance(action, CodeAction):
        sandbox_result = sandbox.execute(action.code, execution_id)
        computations = tuple(
            sandbox.computations[computation_id]
            for computation_id in sandbox_result.computation_ids
            if computation_id in sandbox.computations
        )
        return ExecuteOutcome(
            prompt,
            reply,
            raw_result=sandbox_result,
            observation=normalize_sandbox(sandbox_result, max_chars=max_observation_chars),
            report=run_action_checks(sandbox_result.state, computations),
            computations=computations,
        )
    # An answer runs nothing, and no valid action means nothing may run.
    return ExecuteOutcome(prompt, reply)
